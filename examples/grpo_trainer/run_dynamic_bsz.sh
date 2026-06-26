set -x
MODEL_PATH=/workspace/models/Qwen2.5-7B-Instruct
DATA_PATH_PREFIX=/workspace/data/R1-dataset/

export HF_ENDPOINT=https://hf-mirror.com
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_PATH_PREFIX}"/train_kod10k_codeforce5k_dapo17k.parquet \
    data.val_files="${DATA_PATH_PREFIX}"/aime2024_math_format_with_sft.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=20480 \
    data.shuffle=True \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=21504 \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_token_level_loss=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=21504 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=21504 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=21504 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.overlong_buffer.enable=True \
    custom_reward_function.overlong_buffer.len=4096 \
    custom_reward_function.overlong_buffer.penalty_factor=1.0 \
    reward_model.reward_manager=prime \
    reward_model.prime_num_processes=32 \
    reward_model.prime_batch_size=512 \
    trainer.logger=['console','tensorboard'] \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    trainer.val_before_train=True \
    trainer.project_name='GRPO_TRAINING' \
    trainer.experiment_name='Qwen2.5-7B-INST-ZERO' \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=checkpoints/GRPO_MIXTURE \
    trainer.val_generations_to_log_to_txt=100 $@ 2>&1 | tee grpo.log
    

# ps -ef | grep verl.trainer.main_ppo | grep -v grep | awk -F' ' '{print $2}' | xargs kill -9
# ps -ef | grep run.sh | grep -v grep | awk -F' ' '{print $2}' | xargs kill -9
# ps -ef | grep run_qwen2_7b_zero.sh | grep -v grep | awk -F' ' '{print $2}' | xargs kill -9