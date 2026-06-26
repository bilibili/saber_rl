# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
import re
from typing import Dict, Tuple, Optional
from itertools import islice, zip_longest
from sympy.parsing.latex import parse_latex
from verl.utils.logging_utils import LogCollector
from verl.utils.reward_score.saber import saber_compute_score
try:
    from math_verify import parse, verify
except ImportError:
    print("math_verify is not installed in this environment")
    parse = None
    verify = None

def repeatness(s: str):
    def ranks(l):
        index = {v: i for i, v in enumerate(sorted(set(l)))}
        return [index[v] for v in l]

    def suffixArray(s):
        line = ranks(s)
        n, k, ans, sa = len(s), 1, line, [0] * len(s)
        while k < n - 1:
            line = ranks(list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
            ans, k = line, k << 1
        for i, k in enumerate(ans):
            sa[k] = i
        return ans, sa

    def lcp(arr, suffixArr, inv_suff):
        n, ans, k = len(arr), [0] * len(arr), 0

        for i in range(n):
            if inv_suff[i] == n - 1:
                k = 0
                continue

            j = suffixArr[inv_suff[i] + 1]
            while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
                k += 1

            ans[inv_suff[i]] = k
            if k > 0:
                k -= 1

        return ans

    arr = [ord(i) for i in s]
    n = len(arr)
    if n <= 1:
        return 0
    c, sa = suffixArray(arr)
    cnt = sum(lcp(arr, sa, c))

    return (cnt * 2 / (n * (n + 1))) > 0.2


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]


REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.
    This code comes from https://arxiv.org/pdf/2206.14858.pdf, page18.
    """
    # final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


def latex_eval(latex):
    sym = parse_latex(latex)
    val = sym.evalf()
    return sym, val


def _is_latex_equal(str1, str2):
    try:
        sym1, val1 = latex_eval(str1)
        sym2, val2 = latex_eval(str2)
        if sym1 == sym2 or val1 == val2:
            return True
        else:
            raise ValueError
    except Exception:  # noqa
        try:
            norm1, norm2 = normalize_final_answer(str1), normalize_final_answer(str2)
            sym1, val1 = latex_eval(norm1)
            sym2, val2 = latex_eval(norm2)
            if sym1 == sym2 or val1 == val2:
                return True
        except Exception:  # noqa
            return norm1 == norm2
    return False


def is_latex_equal(str1, str2, math_mode="legacy"):
    if math_mode == "legacy":
        if (len(str1) > 128 and repeatness(str1)) or (len(str2) > 128 and repeatness(str2)):
            return False

        try:
            result = _is_latex_equal(str1, str2)
            return result
        except Exception as e:
            return False
    elif math_mode == "math_verify":
        try:
            result = verify(parse(str1), parse(str2))
            return result
        except Exception as e:
            return False
    else:
        raise NotImplementedError(f"Math mode {math_mode} is not implemented")


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:  # noqa
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:  # noqa
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace(",", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=False) -> bool:
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        # if verbose:
        #     print(ss1, ss2)
        try:
            return float(ss1) == (float(ss2))
        except Exception:  # noqa
            return ss1 == ss2
    except Exception:  # noqa
        return str1 == str2


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def get_answer_str(s: str) -> str:
    res = remove_boxed(last_boxed_only_string(s))
    if res is not None:
        return res
    return s


def is_equal(str1, str2, math_mode="legacy"):
    first_equal = is_equiv(str1, str2)
    if first_equal:
        return True
    return is_latex_equal(str1, str2, math_mode)


def solution2answer(solution: str, math_mode="eval_peeking") -> str:
    answer = solution
    if math_mode == "eval_peeking":
        answer = get_answer_str(solution)
    else:
        raise ValueError(f"Invalid math_mode: {math_mode}")
    return answer


def get_final_answer(output: str) -> str:
    output = output.replace("is:", "is").replace("answer:", "answer is").strip()
    if output.endswith("."):
        output = output[:-1]
    if ".$" in output:
        output = output.replace(".$", "$")
    pattern_list = [
        r"answer is (-?\d+\.?\d*)$",
        r"answer is (.+?)$",
    ]
    matches = []
    for pat in pattern_list:
        matches = re.findall(pat, output, re.S)
        if matches:
            return get_answer_str(matches[0])

    return get_answer_str(output)


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


def extract_solution(solution_str: str, logger: LogCollector) -> Tuple[Optional[str], str]:
    processed_str = solution_str
    if len(solution_str.split("</think>")) <= 1:
        return None, processed_str
    final_answer = solution_str.split("</think>")[1]

    return final_answer, processed_str


def compute_score(solution_str, ground_truth, extra_info):
    try:
        return compute_score_inner(solution_str, ground_truth, extra_info)
    except Exception as e:
        print(f"[MATH] completion: \n", solution_str)
        print(f"[MATH] ground_truth: \n", ground_truth)
        print(f"[MATH] extra_info: \n", extra_info)
        print(f"[MATH] Error: {e}")


def compute_score_inner(solution_str, ground_truth, extra_info):
    logger = LogCollector(prefix="MATH")
    logger.clear()
    
    logger.log("\n" + "="*80)
    logger.log(" Processing New Sample ".center(80, '='))
    logger.log(f"[Ground Truth]: {ground_truth}")

    if int(extra_info["token_upper"]) == 0:
        solution_str = "<think>\n\n</think>\n\n" + solution_str
    else:
        solution_str = "<think>\n" + solution_str

    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str, logger)
    logger.log(f"\n[Model Response]\n{processed_str}")
    logger.log(f"\n[Model Answer]\n{answer_text}")
    
    # Validate response structure
    format_correct = validate_response_structure(processed_str, logger)
    format_score = 0.0 if format_correct else -0.2
    logger.log(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")

    answer_score = 0.0
    length_score = 0.0
    if_llm_equal = 0
    llm_equal = 0
    llm_response = ""
    if format_correct and answer_text:
        answer = remove_boxed(last_boxed_only_string(answer_text))
        if answer is not None:
            logger.log(f"\n[Content Validation]")
            logger.log(f"  Expected: {ground_truth}")
            logger.log(f"  Predicted: {answer}")

            first_equal = is_equal(answer, ground_truth, math_mode="math_verify")

            if first_equal:
                answer_score = 1.0
                logger.log("  Content validation: FULL MATCH")
            else:
                answer_score = 0.0
                logger.log("  Content validation: MISMATCH")
        else:
            answer_score = 0.0
            logger.log("Fail to parse answer")

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
    logger.log("\n" + "-" * 80)
    logger.log(f" Final Score ".center(80, '-'))
    logger.log(f"  Format: {format_score}")
    logger.log(f"  Answer: {answer_score}")
    logger.log(f"  Length: {length_score}")
    logger.log(f"  think_length_success: {think_length_success}")
    logger.log(f"  answer_length_success: {answer_length_success}")
    logger.log(f"  llm_response: \n{llm_response}")
    logger.log("=" * 80 + "\n")

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
            "answer_token_count": answer_token_count,
            "llm_equal": llm_equal,
            "if_llm_equal": if_llm_equal,
            "llm_response": llm_response
        }
    }, logger.get_logs()
