import re
import sys
import jsonlines
from fraction import Fraction


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

#--------------------------------- old function -----------------------------------------
# def extract_answer_number(completion):
#     text = completion.split('The answer is: ')
#     if len(text) > 1:
#         extract_ans = text[-1].strip()
#         match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
#         if match:
#             if '/' in match.group():
#                 denominator = match.group().split('/')[1]
#                 numerator = match.group().split('/')[0]
#                 if is_number(denominator) == True and is_number(numerator) == True:
#                     if denominator == '0':
#                         return round(float(numerator.replace(',', '')))
#                     else:
#                         frac = Fraction(match.group().replace(',', ''))
#                         num_numerator = frac.numerator
#                         num_denominator = frac.denominator
#                         return round(float(num_numerator / num_denominator))
#                 else:
#                     return None
#             else:
#                 if float(match.group().replace(',', '')) == float('inf'):
#                     return None
#                 return round(float(match.group().replace(',', '')))
#         else:
#             return None
#     else:
#         return None

#-----------------------------------------------------------------------------------------


#--------------------------------- new function -----------------------------------------
import re
from fractions import Fraction
from typing import Optional, List, Tuple, Dict

def _parse_number_str(num_str: str) -> Optional[int]:
    s = num_str.strip().replace(",", "")
    if s.lower() in {"inf", "+inf", "-inf", "infinity"}:
        return None
    try:
        if "/" in s:
            val = float(Fraction(s))
        else:
            val = float(s)
        return int(round(val))
    except Exception:
        return None

def _pick_candidate(cands: List[Tuple[str, int]], policy: str, text_len: int, tail_window: int) -> Optional[int]:
    """
    cands: [(num_str, start_idx_in_text), ...]
    policy: 'last' | 'first' | 'majority' | 'tail'
    """
    if not cands:
        return None

    if policy == "first":
        num = _parse_number_str(cands[0][0])
        return num

    if policy == "majority":
        counter: Dict[int, int] = {}
        last_pos: Dict[int, int] = {}
        for s, pos in cands:
            n = _parse_number_str(s)
            if n is None: 
                continue
            counter[n] = counter.get(n, 0) + 1
            last_pos[n] = pos
        if not counter:
            return None
        max_cnt = max(counter.values())
        # 票数相同，取“出现位置更靠后的”
        best_vals = [v for v,c in counter.items() if c == max_cnt]
        best_vals.sort(key=lambda v: last_pos[v])
        return best_vals[-1]

    if policy == "tail":
        # 优先在尾部窗口内找最后一次
        cutoff = max(0, text_len - tail_window)
        tail_cands = [(s, pos) for s, pos in cands if pos >= cutoff]
        if tail_cands:
            s, _ = tail_cands[-1]
            return _parse_number_str(s)
        # 否则退化为 'last'
        s, _ = cands[-1]
        return _parse_number_str(s)

    # 默认 'last'
    s, _ = cands[-1]
    return _parse_number_str(s)

def extract_answer_number(
    text: str, 
    policy: str = "majority",         # 'last' | 'first' | 'majority' | 'tail'
    tail_window: int = 1000        # policy='tail' 时的尾部窗口大小
) -> Optional[int]:
    """
    提取顺序（每步用 policy 选一条）：
      1) '#### <number>'（默认取最后一次/按 policy）
      2) '\boxed{<number>}'（可选，常见于 LaTeX）
      3) '(the) answer is|final answer|answer' 后的数字
      4) 文本尾部兜底（最后 1000 字符内的最后一个数字）
    """
    # 记录匹配到的 [字符串, 起始位置]，以便做策略选择
    def find_all_with_pos(pat: re.Pattern, text: str) -> List[Tuple[str,int]]:
        return [(m.group(1), m.start()) for m in pat.finditer(text)]

    # 1) #### number
    pat_hash = re.compile(r"#\s*#\s*#\s*#\s*([-+]?\d+(?:[.,]\d+)?(?:/\d+)?)", re.IGNORECASE)
    cands = find_all_with_pos(pat_hash, text)
    n = _pick_candidate(cands, policy, len(text), tail_window)
    if n is not None:
        return n

    # 2) \boxed{number} (可选，但很常见于数学答案)
    pat_box = re.compile(r"\\boxed\{\s*([-+]?\d+(?:[.,]\d+)?(?:/\d+)?)\s*\}")
    cands = find_all_with_pos(pat_box, text)
    n = _pick_candidate(cands, policy, len(text), tail_window)
    if n is not None:
        return n

    # 3) "answer is / final answer / answer:"
    pat_ans = re.compile(
        r"(?:the\s+answer\s+is|final\s+answer|answer)[:\s]*([-+]?\d+(?:[.,]\d+)?(?:/\d+)?)",
        re.IGNORECASE,
    )
    cands = find_all_with_pos(pat_ans, text)
    n = _pick_candidate(cands, policy, len(text), tail_window)
    if n is not None:
        return n

    # 4) 兜底：看文本尾部，找最后一个数字
    tail = text[-tail_window:]
    m_all = re.findall(r"([-+]?\d+(?:[.,]\d+)?(?:/\d+)?)", tail)
    if m_all:
        return _parse_number_str(m_all[-1])

    return None


#-----------------------------------------------------------------------------------------


def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n - 1):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_data.append(data_list[start:end])

    last_start = (n - 1) * batch_size
    last_end = sys.maxsize
    batch_data.append(data_list[last_start:last_end])
    return batch_data


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def process_results(doc, completion, answer, invalid_outputs):
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        if is_equiv(extract_ans, answer):
            return True
        else:
            return False
    else:
        temp = {'question': doc, 'output': completion, 'answer': answer}
        invalid_outputs.append(temp)
        return False


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

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def fix_fracs(string):
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
                except AssertionError:
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


def fix_a_slash_b(string):
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
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
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


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

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
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        # pdb.set_trace()
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def generate_instruction_following_task_prompt(instruction, is_chat_model=True):
    if is_chat_model:
        prompt = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {instruction} ASSISTANT:"""
    else:
        prompt = f"""{instruction}

### Response:
"""
    return prompt

def get_math_task_prompt():
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    return problem_prompt


def generate_code_task_prompt(input_text):
    INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{input_text}

### Response:"""
    return INSTRUCTION


def read_mbpp(path):
    mbpp_problems = {}
    with jsonlines.open(path, "r") as fin:
        for obj in fin:
            mbpp_problems[obj["task_id"]] = obj
    return mbpp_problems
