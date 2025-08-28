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

# # --------------------------------- old function -----------------------------------------
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

# # -----------------------------------------------------------------------------------------


#--------------------------------- new function -----------------------------------------
import re
import string
from fractions import Fraction
from typing import Optional, List, Tuple, Dict, Union

Number = Union[int, float]          # 返回类型：可能是 int，也可能是 float
_EPS = 1e-9                         # 判断“是否是整数”的容差

def _coerce_int_if_clean(val: float) -> Number:
    """若 val 非常接近整数，则返回 int；否则返回原 float。"""
    # round(val) 得到最接近的整数；若差值在极小容差内，就当作“干净整数”
    iv = round(val)
    return int(iv) if abs(val - iv) <= _EPS else val

def _parse_number_str(num_str: str) -> Optional[Number]:
    """将字符串解析为数字：
       - 支持 1,234.56、-3.5、7/2 等
       - 去掉末尾标点（解决句号/逗号黏在数字后面）
       - 返回 int 或 float（整数干净则 int）"""
    s = num_str.strip()                     # 去掉首尾空白
    s = s.replace(",", "")                  # 去掉千位分隔逗号
    s = s.replace("−", "-")                 # 统一全角/数学减号为普通减号
    s = s.rstrip(string.punctuation).strip()# 去掉末尾标点，再次去空白

    # 排除无穷等无效数字
    if s.lower() in {"inf", "+inf", "-inf", "infinity"}:
        return None

    try:
        # 支持分数，如 "7/2"
        if "/" in s:
            val = float(Fraction(s))        # Fraction 更稳健，能处理“3/10”
        else:
            val = float(s)                  # 其余按浮点解析
        return _coerce_int_if_clean(val)    # 整数干净就转 int，否则保留 float
    except Exception:
        return None                         # 解析失败返回 None   



def _pick_candidate(
    cands: List[Tuple[str, int]],
    policy: str,
    text_len: int,
    tail_window: int
) -> Optional[Number]:
    """在若干候选 (字符串数字, 起始位置) 中按策略挑一个：
       - policy: 'last' | 'first' | 'majority' | 'tail'"""
    if not cands:
        return None

    if policy == "first":
        # 取第一个匹配到的候选
        return _parse_number_str(cands[0][0])

    if policy == "majority":
        # 统计哪一个数值出现次数最多，平票时取“出现位置更靠后的”
        counter: Dict[Number, int] = {}
        last_pos: Dict[Number, int] = {}
        for s, pos in cands:
            n = _parse_number_str(s)
            if n is None:
                continue
            counter[n] = counter.get(n, 0) + 1
            last_pos[n] = pos
        if not counter:
            return None
        max_cnt = max(counter.values())
        best_vals = [v for v, c in counter.items() if c == max_cnt]
        best_vals.sort(key=lambda v: last_pos[v])
        return best_vals[-1]

    if policy == "tail":
        # 优先在文本尾部窗口（tail_window 字符内）里选最后一次出现
        cutoff = max(0, text_len - tail_window)
        tail_cands = [(s, pos) for s, pos in cands if pos >= cutoff]
        if tail_cands:
            return _parse_number_str(tail_cands[-1][0])
        # 否则退化为 'last'
        return _parse_number_str(cands[-1][0])

    # 默认策略：取最后一次出现
    return _parse_number_str(cands[-1][0])

def extract_answer_number(
    text: str,
    policy: str = "majority",   # 'last' | 'first' | 'majority' | 'tail'
    tail_window: int = 1000     # policy='tail' 时的尾部窗口大小
) -> Optional[Number]:
    """
    依次尝试以下模式（每步用 policy 选一条）：
      1) '#### <number>'
      2) '\\boxed{<number>}'（LaTeX 常见）
      3) '(the) answer is|final answer|final answer is|the final answer is' 后的数字
      4) 兜底：在文本尾部窗口内寻找最后一个数字
    匹配到数字后：
      - 若是“干净整数”，返回 int
      - 否则返回 float（保留小数）
    """
    # 小工具：返回 [(匹配到的“数字字符串”, 起始位置), ...]
    def find_all_with_pos(pat: re.Pattern, txt: str) -> List[Tuple[str, int]]:
        return [(m.group(1), m.start()) for m in pat.finditer(txt)]

    # 1) #### number
    pat_hash = re.compile(
        r"#\s*#\s*#\s*#\s*([-+]?\d+(?:[.,]\d+)?(?:/\d+)?)",
        re.IGNORECASE
    )
    cands = find_all_with_pos(pat_hash, text)
    n = _pick_candidate(cands, policy, len(text), tail_window)
    if n is not None:
        return n

    # 2) \boxed{number}
    pat_box = re.compile(
        r"\\boxed\{\s*([-+]?\d+(?:[.,]\d+)?(?:/\d+)?)\s*\}"
    )
    cands = find_all_with_pos(pat_box, text)
    n = _pick_candidate(cands, policy, len(text), tail_window)
    if n is not None:
        return n

    # 3) "answer" 系列：包含
    #    "the answer is", "final answer", "final answer is", "the final answer is", 以及简单 "answer"
    pat_ans = re.compile(
        r"(?:the\s+final\s+answer\s+is|final\s+answer\s+is|the\s+answer\s+is|final\s+answer|answer)"
        r"[:\s]*([-+]?\d+(?:[.,]\d+)?(?:/\d+)?)",
        re.IGNORECASE,
    )
    cands = find_all_with_pos(pat_ans, text)
    n = _pick_candidate(cands, policy, len(text), tail_window)
    if n is not None:
        return n

    # 4) 兜底：看文本尾部，找最后一个数字（支持整数/小数/分数）
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

# # --------------------------------- old function -----------------------------------------
# def process_results(doc, completion, answer, invalid_outputs):
#     split_ans = completion.split('The answer is: ')
#     if len(split_ans) > 1:
#         ans = split_ans[-1]
#         extract_ans_temp = ans.split('.\n')[0]
#         extract_ans_temp = extract_ans_temp.strip()
#         if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == '.':
#             extract_ans = extract_ans_temp[0:-1]
#         else:
#             extract_ans = extract_ans_temp
#         extract_ans = extract_ans.strip()
#         if is_equiv(extract_ans, answer):
#             return True
#         else:
#             return False
#     else:
#         temp = {'question': doc, 'output': completion, 'answer': answer}
#         invalid_outputs.append(temp)
#         return False
# # -----------------------------------------------------------------------------------------


# --------------------------------- new function -----------------------------------------
def _clean_answer_text(s: str) -> str:
    """清理答案字符串：去末尾标点/空白，去 LaTeX 包裹。"""
    s = s.strip()
    # 去掉最外层 \boxed{...} / $...$
    m = re.fullmatch(r"\\boxed\{(.+)\}", s)
    if m:
        s = m.group(1).strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    # 去掉末尾标点（句号、逗号等）
    s = s.rstrip(string.punctuation + "，。；；…")
    return s.strip()

def extract_final_answer_text(text: str) -> str | None:
    """
    从文本中提取“最终答案”字符串：
      - 依次尝试以下模式的“最后一次出现”：
        1) #### <ans>
        2) \boxed{<ans>}
        3) (the) answer is / final answer / final answer is / the final answer is
    """
    # 1) #### <ans>
    m = None
    for m in re.finditer(r"#\s*#\s*#\s*#\s*(.+)", text, flags=re.IGNORECASE):
        pass
    if m:
        return _clean_answer_text(m.group(1))

    # 2) \boxed{<ans>}
    for m in re.finditer(r"\\boxed\{\s*(.+?)\s*\}", text):
        pass
    if m:
        return _clean_answer_text(m.group(1))

    # 3) answer 提示词（取最后一次）
    pat = re.compile(
        r"(?:the\s+final\s+answer\s+is|final\s+answer\s+is|the\s+answer\s+is|final\s+answer|answer)"
        r"[:\s]*([^\n\r]+)",
        flags=re.IGNORECASE,
    )
    for m in pat.finditer(text):
        pass
    if m:
        return _clean_answer_text(m.group(1))

    return None

def process_results(doc, completion, answer, invalid_outputs):
    # 提取模型给出的“最终答案文本”
    pred = extract_final_answer_text(completion)
    if pred is None:
        invalid_outputs.append({'question': doc, 'output': completion, 'answer': answer})
        return False

    # 规范化一下两边（去空格、大小写）
    pred_norm = _clean_answer_text(pred).replace(" ", "").lower()
    gold_norm = _clean_answer_text(answer).replace(" ", "").lower()

    # 如果你有更强的等价判断（比如符号化比较），在 is_equiv 里实现即可
    if is_equiv(pred_norm, gold_norm):
        return True
    else:
        # 这条样例里 pred = "h+c/2" 与 gold = "2k" 不等价，应判错
        return False
# -----------------------------------------------------------------------------------------



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
