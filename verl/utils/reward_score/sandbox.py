import os
import re
import sys
import time
import math
import json
import base64
import logging
from typing import Dict, Tuple, Optional

from sandbox_fusion import set_endpoint, run_code, RunCodeRequest
from sandbox_fusion import (run_code, 
                            run_concurrent,
                            RunCodeRequest, 
                            RunCodeResponse,
                            RunStatus)

from verl.utils.logging_utils import LogCollector
from verl.utils.reward_score.saber import saber_compute_score

_SANDBOX_APP_ADDRESS = ""
_SELECTOR_CONFIG = {}
_sandbox_selector = ""


def extract_markdown_code_blocks_with_language(text):
    if not text:
        return []
    pattern = r"```(\w+)?\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    code_blocks = []
    for language, code in matches:
        code_blocks.append({
            "language": language.strip() if language else "python",
            "code": f"\n{code.strip()}\n"
        })
    
    return code_blocks


def extract_solution(solution_str: str, logger: LogCollector) -> Tuple[Optional[str], str]:
    processed_str = solution_str
    if len(solution_str.split("</think>")) <= 1:
        return None, processed_str
    final_answer = solution_str.split("</think>")[1]

    return final_answer, processed_str


def get_ret_reward(result: RunCodeResponse, ground_truth: str, logger: LogCollector, tol=1e-9):
    """
    return: 1, 0, -1, -2
    """
    if result.status == RunStatus.Success:
        ret_str = result.run_result.stdout
        logger.log(f"  Predicted: {ret_str}")
        if not ret_str or not ret_str.strip():
            return 0

        ret_tokens = ret_str.strip().split()
        gt_tokens = ground_truth.strip().split()

        if len(ret_tokens) != len(gt_tokens):
            return 0

        for ret_token, gt_token in zip(ret_tokens, gt_tokens):
            try:
                ret_num = float(ret_token)
                gt_num = float(gt_token)
                if not math.isclose(ret_num, gt_num, rel_tol=tol, abs_tol=tol):
                    return 0
            except ValueError:
                if ret_token != gt_token:
                    return 0
        return 1
    else:
        logger.log(f"  Predicted: {result}")
        return -1


def validate_response_structure(processed_str: str, logger: LogCollector) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    logger.log("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        logger.log(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            logger.log(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    if positions['think_start'] > positions['think_end']:
        logger.log("  [Error] Incorrect tag order: Expected <think>...</think>")
        validation_passed = False
    else:
        logger.log("  Tag sequence validation passed")
        
    if validation_passed:
        return True
    else:
        return False


def compute_score(completion, test_cases_str, extra_info):
    try:
        return compute_score_inner(completion, test_cases_str, extra_info)
    except Exception as e:
        print(f"[CODE] completion: \n", completion)
        print(f"[CODE] test_cases_str: \n", test_cases_str)
        print(f"[CODE] extra_info: \n", extra_info)
        print(f"[CODE] Error: {e}")


def compute_score_inner(completion, test_cases_str, extra_info):
    node = _sandbox_selector.select()
    set_endpoint(f"http://{node.ip}:{node.port}")
    tic = time.time()

    logger = LogCollector(prefix="CODE")
    logger.clear()
    
    logger.log("=" * 80)
    logger.log(" Processing New Sample ".center(80, "="))

    if int(extra_info["token_upper"]) == 0:
        completion = "<think>\n\n</think>\n\n" + completion
    else:
        completion = "<think>\n" + completion

    answer_text, processed_str = extract_solution(completion, logger)
    logger.log(f"\n[Model Response]\n{processed_str}")
    logger.log(f"\n[Model Answer]\n{answer_text}")

    # Validate response structure
    format_correct = validate_response_structure(processed_str, logger)
    format_score = 0.0 if format_correct else -0.2
    logger.log(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    
    code_blocks = extract_markdown_code_blocks_with_language(answer_text)
    markdown_correct = bool(code_blocks and code_blocks[-1]['code'].strip())
    
    answer_score = 0.0
    length_score = 0.0
    if format_correct and markdown_correct:
        solution_code, language = code_blocks[-1]['code'], code_blocks[-1]['language']
        
        if extra_info["task"] == "stdin":
            test_cases = json.loads(test_cases_str)
            
            results = run_concurrent(run_code, args=[
                [RunCodeRequest(code=solution_code, language=language, stdin=test_input)] for test_input in test_cases["inputs"]])
            scores = []
            logger.log(f"\n[Content Validation]")
            for result, gt, test_input in zip(results, test_cases["outputs"], test_cases["inputs"]):
                literal_input = test_input.replace("\n", r"\n")
                logger.log(f"  Test Case: {literal_input}")
                logger.log(f"  Expected: {gt}")
                score = get_ret_reward(result, gt, logger)
                if score == -1:
                    logger.log("  Content validation: FAILED TO RUN")
                    scores = [0.0]
                    break
                else:  # normal
                    scores.append(score)
            sum_score = float(sum(scores))
            if abs(sum_score - len(test_cases["inputs"])) < 1e-7:
                answer_score = 1.0
            else:
                answer_score = 0.0
        elif extra_info["task"] == "pytest":
            if isinstance(solution_code, str):
                solution_code = solution_code.encode('utf-8')
            base64_solution = base64.b64encode(solution_code).decode('utf-8')
            
            if isinstance(test_cases_str, str):
                test_cases_str = test_cases_str.encode('utf-8')
            base64_test_code = base64.b64encode(test_cases_str).decode('utf-8')
            
            logger.log(f"\n[Content Validation]")
            result = run_code(RunCodeRequest(code="pytest test_code.py", language='bash', files = {'solution.py': base64_solution, "test_code.py": base64_test_code}), max_attempts=10)
            
            if result.status == RunStatus.Success:
                logger.log("  Content validation: SUCCESS TO RUN")
                answer_score = 1.0
            else:
                logger.log("  Content validation: FAILED TO RUN")
                answer_score = 0.0    
        else:
            logger.log(f"[WARNING] extra_info.task is not valid. task can only be selected from [\"stdin\", \"pytest\"], please check your dataset")
        
        length_score, think_length_success, answer_length_success, think_token_count, answer_token_count = saber_compute_score(extra_info, logger, think_buffer=1000, answer_buffer=1000)
    else:
        length_score = 0.0
        answer_score = 0.0
        think_length_success = False
        answer_length_success = False
        think_token_count = 0
        answer_token_count = 0
        logger.log("\n[Content Validation] Skipped due to format errors or missing answer")

    total_score = answer_score + format_score + length_score
    logger.log("\n" + "-"*80)
    logger.log(f" Final Score ".center(80, "-"))
    logger.log(f"  Format: {format_score}")
    logger.log(f"  Answer: {answer_score}")
    logger.log(f"  Length: {length_score}")
    logger.log(f"  think_length_success: {think_length_success}")
    logger.log(f"  answer_length_success: {answer_length_success}")
    logger.log("=" * 80 + "\n")
    cost = time.time() - tic
    _sandbox_selector.report(node, cost, True)  
    
    return {
        "score": total_score,
        "extra_info": {
            "format_score": format_score,
            "answer_score": answer_score,
            "length_score": length_score,
            "rm_response": "",
            "think_length_success": think_length_success,
            "answer_length_success": answer_length_success,
            "think_token_count": think_token_count,
            "answer_token_count": answer_token_count
            }
    }, logger.get_logs()
