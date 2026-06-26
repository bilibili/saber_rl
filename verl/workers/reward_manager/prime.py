# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import asyncio
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
import torch
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl.bl_utils.global_config import get_config_obj
import time
from collections import defaultdict


async def single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=120.):
    loop = asyncio.get_running_loop()
    # print("origin completion: \n", completion)
    # print("origin reference: \n", reference)
    # print("origin task_extra_info: \n", task_extra_info)
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                partial(evaluation_func, task, completion, reference, task_extra_info)
            ),
            timeout=timeout
        )

        # 处理结果
        score, logs = result
        if logs:
            print(f"[Evaluation Logs]\n{logs}")

        return score
    except asyncio.TimeoutError:
        print(f"Timeout occurred for completion: {completion}")
        return {"score": 0.0}
    except Exception as e:
        print(f"Error processing completion: {completion}, Error: {e}, Exception type: {type(e)}")
        return {"score": 0.0}


async def parallel_compute_score_async(evaluation_func,
                                       completions,
                                       references,
                                       tasks,
                                       extra_info=None,
                                       num_processes=64):
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        if extra_info is None:
            extra_info = [None] * len(tasks)
        # Create tasks for all rows
        tasks_async = [
            single_compute_score(evaluation_func, completion, reference, task, task_extra_info, executor, timeout=3600.)
            for completion, reference, task, task_extra_info in zip(completions, references, tasks, extra_info)
        ]
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        try:
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except:
            for pid, proc in executor._processes.items():
                try:
                    proc.kill()
                except Exception as kill_err:
                    print('shut down failed: ' + str(kill_err))
            raise

    for result in results:
        if result is None or isinstance(result, Exception):
            scores.append({"score": 0.0})
        else:
            scores.append(result)
    return scores


class PrimeRewardManager:
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = _default_compute_score
        whole_config = get_config_obj()
        self.prime_num_processes = whole_config.reward_model.prime_num_processes
        self.overlong_buffer_cfg = whole_config.custom_reward_function.overlong_buffer
        self.max_resp_len = whole_config.data.max_response_length
        self.use_llm_verify = whole_config.reward_model.use_llm_verify
        self.llm_server_ip = whole_config.reward_model.llm_server_ip
        self.llm_server_port = whole_config.reward_model.llm_server_port

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if overlong_buffer_cfg={self.overlong_buffer_cfg}, but got None"

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # data.save_to_disk(f"/workspace/verl/prime_data_proto_{time.time()}.pkl")

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        response_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch['data_source']
        extra_info = data.non_tensor_batch.get('extra_info', [None] * len(data_sources))

        assert len(response_str) == len(ground_truth) == len(data_sources) == len(extra_info)
        print("response_str length: \n", len(response_str))

        extra_info_process = []
        for info, response in zip(extra_info, response_str):
            if isinstance(info, dict):
                copied_info = copy.deepcopy(info)
                copied_info["response_id"] = torch.tensor(self.tokenizer.encode(response, add_special_tokens=False))
                copied_info["use_llm_verify"] = self.use_llm_verify
                copied_info["llm_server_ip"] = self.llm_server_ip
                copied_info["llm_server_port"] = self.llm_server_port
                if "token_upper" not in copied_info:
                    copied_info["token_upper"] = 32768
                extra_info_process.append(copied_info)
            else:
                tmp = {
                    "token_upper": 32768,
                    "use_llm_verify": self.use_llm_verify,
                    "llm_server_ip": self.llm_server_ip,
                    "llm_server_port": self.llm_server_port,
                    "response_id": torch.tensor(self.tokenizer.encode(response, add_special_tokens=False)),
                }
                extra_info_process.append(tmp)

        assert len(extra_info_process) == len(response_str)

        try:
            scores = asyncio.run(
                parallel_compute_score_async(self.compute_score,
                                             response_str,
                                             ground_truth,
                                             data_sources,
                                             extra_info_process,
                                             num_processes=self.prime_num_processes))
        except asyncio.TimeoutError as e:
            print('Global timeout in reward computing! Setting all as 0.')
            scores = [{"score": 0.0} for _ in range(len(response_str))]
        except Exception as e:
            print(f"Unexpected error in batched reward computing. Setting all as 0.: {e}")
            scores = [{"score": 0.0} for _ in range(len(response_str))]

        assert len(scores) == len(data)

        final_reward_list = []
        reward_extra_info = defaultdict(list)
        for i in range(len(scores)):
            data_source = data_sources[i]
            info = extra_info[i]
            think_token_budget = info["token_upper"]
            answer_token_origin = info["answer_token_origin"]
            final_reward = scores[i]['score']
            if 'extra_info' in scores[i]:
                error_ = 0
                answer_score = scores[i]['extra_info']['answer_score']
                format_score = scores[i]['extra_info']['format_score']
                length_score = scores[i]['extra_info']['length_score']
                rm_response = scores[i]['extra_info']['rm_response']
                think_length_success = scores[i]['extra_info']['think_length_success']
                answer_length_success = scores[i]['extra_info']['answer_length_success']
                think_token_count = scores[i]['extra_info']['think_token_count']
                answer_token_count = scores[i]['extra_info']['answer_token_count']
                llm_equal = scores[i]['extra_info'].get("llm_equal", 0)
                if_llm_equal = scores[i]['extra_info'].get("if_llm_equal", 0)
                llm_response = scores[i]['extra_info'].get("llm_response", "")

                overlong_reward = 0.0
                if self.overlong_buffer_cfg.enable:
                    overlong_buffer_len = self.overlong_buffer_cfg.len
                    expected_len = self.max_resp_len - overlong_buffer_len
                    exceed_len = valid_response_length[i].item() - expected_len
                    overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                    overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                    final_reward += overlong_reward

            else:
                error_ = 1
                answer_score = 0.0
                format_score = 0.0
                length_score = 0.0
                rm_response = ""
                think_length_success = False
                answer_length_success = False
                think_token_count = 0
                answer_token_count = 0
                llm_equal = 0
                if_llm_equal = 0
                overlong_reward = 0.0
                llm_response = ""

            reward_extra_info["data_source"].append(data_source)
            reward_extra_info["error"].append(error_)
            reward_extra_info["answer_score"].append(answer_score)
            reward_extra_info["format_score"].append(format_score)
            reward_extra_info["length_score"].append(length_score)
            reward_extra_info["rm_response"].append(rm_response)
            reward_extra_info["think_length_success"].append(int(think_length_success))
            reward_extra_info["answer_length_success"].append(int(answer_length_success))
            reward_extra_info["think_token_count"].append(think_token_count)
            reward_extra_info["think_token_budget"].append(think_token_budget)
            reward_extra_info["answer_token_count"].append(answer_token_count)
            reward_extra_info["answer_token_origin"].append(answer_token_origin)
            reward_extra_info["llm_equal"].append(llm_equal)
            reward_extra_info["if_llm_equal"].append(if_llm_equal)
            reward_extra_info["overlong_reward"].append(overlong_reward)
            reward_extra_info["llm_response"].append(llm_response)

            # if self.overlong_buffer_cfg.enable:
            #     overlong_buffer_len = self.overlong_buffer_cfg.len
            #     expected_len = self.max_resp_len - overlong_buffer_len
            #     exceed_len = valid_response_length[i].item() - expected_len
            #     overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
            #     overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
            #     final_reward += overlong_reward
            #     if self.overlong_buffer_cfg.log:
            #         reward_extra_info["overlong_reward"].append(overlong_reward)
            #         reward_extra_info["overlong"].append(overlong_reward < 0)

            final_reward_list.append(final_reward)

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = final_reward_list[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[Response]", response_str[i])

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
