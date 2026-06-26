set -x

TASK_TYPE="${TASK_TYPE:-master}"
MODEL_DOWNLOAD_PATH="${MODEL_DOWNLOAD_PATH:-s3://llm_snapshot/Qwen2.5-32B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen2.5-32B-Instruct}"
DATA_PATH_PREFIX=/workspace/data/kk/instruct/3ppl

ARG_PACKS="${ARG_PACKS:-}"
USE_HYDRA=1

#### config args, 
if [ -z "$ARG_PACKS" ]; then
    echo "ARG_PACKS is not set, no extra parameter file provided."
    arg_pack_files=()
else
    IFS=',' read -r -a arg_pack_files <<< "$ARG_PACKS"
fi

_EXTRA_ARGS=()
for file in "${arg_pack_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Args pack file not found: $file"
        continue
    fi
    while IFS= read -r line || [ -n "$line" ]; do
        if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
            continue
        fi
        if [ "${USE_HYDRA}" -ne "0" ] && [[ $line == --* ]]; then
            _EXTRA_ARGS+=( "++${line:2}" )
        else
            _EXTRA_ARGS+=( "$line" )
        fi
    done < "$file"
done

for arg in "$@"; do
    if [ "${USE_HYDRA}" -ne "0" ] && [[ $arg == --* ]]; then
        _EXTRA_ARGS+=( "++${arg:2}" )
    else
        _EXTRA_ARGS+=( "$arg" )
    fi
done

echo "Parsed _EXTRA_ARGS :"
for arg in "${_EXTRA_ARGS[@]}"; do
    echo "  $arg"
done
echo "================================="

if [ -n "$INSTALL_JAVIS_REQS" ] && [ -f "$INSTALL_JAVIS_REQS" ]; then
    echo "Installing requirements in $INSTALL_JAVIS_REQS"
    pip install -r $INSTALL_JAVIS_REQS
fi

bash setup_ray.sh

if [ "${TASK_TYPE}" == "master" ]; then

    export WANDB_MODE="offline"
    export VLLM_ATTENTION_BACKEND=XFORMERS

    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files="${DATA_PATH_PREFIX}"/train.parquet \
        data.val_files="${DATA_PATH_PREFIX}"/test.parquet \
        data.train_batch_size=16 \
        data.val_batch_size=16 \
        data.max_prompt_length=2048 \
        data.max_response_length=8192 \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.actor.optim.lr=3e-7 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size=64 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.grad_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=160 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=64 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
        trainer.critic_warmup=0 \
        trainer.logger=['console','tensorboard'] \
        trainer.project_name='GRPO_logic_KK' \
        trainer.experiment_name='Qwen2.5-7B-Instruct-1M-step1' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=2 \
        trainer.save_freq=20 \
        trainer.test_freq=10 \
        trainer.total_epochs=1 "${_EXTRA_ARGS[@]}" \
        +hydra.job.config.allow_unknown_args=True 2>&1 | tee grpo.log
else
    sleep infinity
fi