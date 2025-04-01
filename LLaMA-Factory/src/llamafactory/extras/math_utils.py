# Copyright 2025 the LlamaFactory team.
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

import re
import math
import time
import threading
import signal
import platform
import os
from typing import Optional, Union, Any, Tuple, Callable

try:
    import sympy
    from sympy.parsing.latex import parse_latex
    from sympy.core.basic import Basic as SympyBasic
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    parse_latex = None
    SympyBasic = None

from ..extras import logging


logger = logging.get_logger(__name__)

# 全局超时设置
SYMPY_TIMEOUT_SECONDS = 5  # 设置sympy计算的最大超时时间
MAX_EXPRESSION_LENGTH = 500  # 设置表达式最大长度
MAX_EXPRESSION_COMPLEXITY = 100  # 设置表达式最大复杂度 (括号、运算符、函数数量)


class TimeoutError(Exception):
    """超时异常类"""
    pass


def timeout_handler(signum, frame):
    """处理信号超时的回调函数"""
    raise TimeoutError(f"Sympy computation timed out after {SYMPY_TIMEOUT_SECONDS} seconds")


def with_timeout(func: Callable, args: Tuple = None, kwargs: dict = None, timeout: int = SYMPY_TIMEOUT_SECONDS) -> Tuple[Any, Optional[Exception]]:
    """
    使用超时机制运行函数，适用于所有平台
    
    Args:
        func: 要执行的函数
        args: 函数参数
        kwargs: 函数关键字参数
        timeout: 超时时间(秒)
        
    Returns:
        (result, exception): 结果或异常
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
        
    result = None
    exception = None
    
    # 使用线程方法实现超时（Windows兼容）
    def target():
        nonlocal result, exception
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            exception = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        return None, TimeoutError(f"Function execution timed out after {timeout} seconds")
    
    return result, exception


def estimate_expression_complexity(expr: str) -> int:
    """
    估计表达式的复杂度，基于特殊字符、运算符和括号的数量
    
    Args:
        expr: 要评估的表达式字符串
        
    Returns:
        复杂度分数 (越高越复杂)
    """
    if not expr:
        return 0
        
    # 计算基本运算符数量
    operators = expr.count('+') + expr.count('-') + expr.count('*') + expr.count('/') + expr.count('^')
    
    # 计算特殊函数数量
    special_funcs = len(re.findall(r'\\(sqrt|frac|cdot|sin|cos|tan|log|ln)', expr))
    
    # 计算括号层级
    max_paren_depth = 0
    current_depth = 0
    for char in expr:
        if char in '({[':
            current_depth += 1
            max_paren_depth = max(max_paren_depth, current_depth)
        elif char in ')}]':
            current_depth = max(0, current_depth - 1)
    
    # 计算嵌套命令数量 (如 \sqrt 内的 \frac)
    nested_commands = len(re.findall(r'\\[a-z]+\{[^{}]*\\[a-z]+\{', expr))
    
    # 综合复杂度分数
    complexity_score = operators + (special_funcs * 2) + (max_paren_depth * 3) + (nested_commands * 5)
    
    return complexity_score


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    提取文本中最后一个\\boxed{...}或boxed{...}中的内容，
    使用大括号匹配来处理嵌套情况。
    
    Args:
        text: 包含boxed答案的文本
        
    Returns:
        提取的答案内容，如果没有找到则返回None
    """
    if not text or not isinstance(text, str):
        return None
    
    last_boxed_content = None
    # 查找最后出现的标记
    last_idx_bs = text.rfind("\\boxed{")
    last_idx_no_bs = text.rfind("boxed{")

    start_index = -1
    start_token = None

    if last_idx_bs != -1 and last_idx_no_bs != -1:
        start_index = max(last_idx_bs, last_idx_no_bs)
        start_token = "\\boxed{" if start_index == last_idx_bs else "boxed{"
    elif last_idx_bs != -1:
        start_index = last_idx_bs
        start_token = "\\boxed{"
    elif last_idx_no_bs != -1:
        start_index = last_idx_no_bs
        start_token = "boxed{"

    if start_index != -1:
        content_start_index = start_index + len(start_token)
        brace_level = 1
        current_index = content_start_index
        content_end_index = -1

        # 查找匹配的右括号，处理嵌套情况
        while current_index < len(text):
            if text[current_index] == '{':
                brace_level += 1
            elif text[current_index] == '}':
                brace_level -= 1
                if brace_level == 0:
                    content_end_index = current_index
                    break  # 找到匹配的括号
            current_index += 1

        if content_end_index != -1:
            # 找到完整的box
            last_boxed_content = text[content_start_index:content_end_index]

    if last_boxed_content is not None:
        return last_boxed_content.strip()
    else:
        return None


def normalize_math_expression(expr: str, normalize: bool = True) -> str:
    """
    使用sympy将数学表达式规范化为标准形式。
    如果规范化失败或禁用，则返回原始字符串。
    
    Args:
        expr: 要规范化的表达式
        normalize: 是否进行规范化
        
    Returns:
        规范化后的表达式或原始表达式
    """
    if not isinstance(expr, str): 
        expr = str(expr)  # 确保输入是字符串
    
    expr = expr.strip()
    
    # 快速检查: 如果禁用规范化或没有sympy模块，直接返回
    if not normalize or not HAS_SYMPY:
        return expr
        
    # 检查表达式长度
    if len(expr) > MAX_EXPRESSION_LENGTH:
        logger.warning_rank0(f"Expression too long ({len(expr)} chars), skipping normalization")
        return expr
        
    # 检查表达式复杂度
    complexity = estimate_expression_complexity(expr)
    if complexity > MAX_EXPRESSION_COMPLEXITY:
        logger.warning_rank0(f"Expression too complex (score: {complexity}), skipping normalization")
        return expr
    
    # 预处理常见的LaTeX问题
    processed_expr = expr.replace("\\\\", "\\")  # 处理双反斜杠
    processed_expr = processed_expr.replace("\\%", "").replace("%", "")  # 移除百分号
    processed_expr = re.sub(r"\\left\s*([(\[{|])", r"\1", processed_expr)  # 移除\left
    processed_expr = re.sub(r"\\right\s*([)\]}|])", r"\1", processed_expr)  # 移除\right
    processed_expr = re.sub(r"\\text\{([^}]+)\}", r"\1", processed_expr)  # 移除\text{...}包装
    processed_expr = processed_expr.replace("^{\\circ}", "").replace("^\\circ", "").replace("°", "")  # 移除度数符号
    
    # 尝试使用sympy.sympify解析和简化
    try:
        # 使用超时机制运行sympify
        parsed_expr, exception = with_timeout(
            func=sympy.sympify,
            args=(processed_expr,),
            timeout=SYMPY_TIMEOUT_SECONDS
        )
        
        if exception is not None:
            raise exception
            
        if not isinstance(parsed_expr, SympyBasic):
            raise TypeError(f"Sympify result is not a Sympy expression, but {type(parsed_expr)}")
        
        # 用超时机制简化表达式
        simplified_expr, exception = with_timeout(
            func=sympy.simplify,
            args=(parsed_expr,),
            timeout=SYMPY_TIMEOUT_SECONDS
        )
        
        if exception is not None:
            raise exception
            
        normalized = str(simplified_expr)
        return normalized
        
    except Exception as e_sympify:
        # 如果常规sympify失败，尝试使用LaTeX解析器
        if parse_latex:
            try:
                # 对LaTeX解析器进行更多预处理
                processed_expr_latex = processed_expr.replace("\\infty", "oo")
                
                # 使用超时机制运行LaTeX解析
                parsed_expr_latex, exception = with_timeout(
                    func=parse_latex,
                    args=(processed_expr_latex,),
                    timeout=SYMPY_TIMEOUT_SECONDS
                )
                
                if exception is not None:
                    raise exception
                
                if not isinstance(parsed_expr_latex, SympyBasic):
                    raise TypeError(f"Parse_latex result is not a Sympy expression, but {type(parsed_expr_latex)}")
                
                # 使用超时机制简化LaTeX表达式
                simplified_latex, exception = with_timeout(
                    func=sympy.simplify,
                    args=(parsed_expr_latex,),
                    timeout=SYMPY_TIMEOUT_SECONDS
                )
                
                if exception is not None:
                    raise exception
                    
                normalized_latex = str(simplified_latex)
                return normalized_latex
                
            except Exception as e_latex:
                # 两种解析方法都失败时记录警告
                logger.warning_rank0(
                    f"Failed to normalize expression '{expr}': "
                    f"Sympify error: {type(e_sympify).__name__}('{e_sympify}'). "
                    f"LaTeX parse error: {type(e_latex).__name__}('{e_latex}')"
                )
                return expr  # 失败时返回原始字符串
        else:
            # 没有LaTeX解析器时的警告
            logger.warning_rank0(
                f"Failed to normalize expression '{expr}' using sympify: "
                f"{type(e_sympify).__name__}('{e_sympify}'). LaTeX parser not available."
            )
            return expr  # 返回原始字符串


def math_equal(answer1: Union[str, None], answer2: Union[str, None], normalize: bool = True) -> bool:
    """
    比较两个数学答案是否相等，使用改进的规范化和比较方法。
    增加了超时处理和复杂度检测。
    
    Args:
        answer1: 第一个答案
        answer2: 第二个答案
        normalize: 是否使用规范化比较
        
    Returns:
        如果两个答案相等则返回True，否则返回False
    """
    if answer1 is None or answer2 is None:
        return False
    
    # 转换成字符串并去除空白
    ans1_str = str(answer1).strip()
    ans2_str = str(answer2).strip()
    
    # 1. 直接字符串比较（快速检查）
    if ans1_str == ans2_str:
        return True
    
    # 2. 如果任一字符串过长或过复杂，则跳过sympy比较
    if len(ans1_str) > MAX_EXPRESSION_LENGTH or len(ans2_str) > MAX_EXPRESSION_LENGTH:
        logger.warning_rank0(f"Skipping sympy comparison due to expression length: "
                           f"{len(ans1_str)} and {len(ans2_str)} chars")
        # 跳过sympy比较，直接进行数值和简单字符串比较
        pass
    
    elif estimate_expression_complexity(ans1_str) > MAX_EXPRESSION_COMPLEXITY or \
         estimate_expression_complexity(ans2_str) > MAX_EXPRESSION_COMPLEXITY:
        logger.warning_rank0(f"Skipping sympy comparison due to expression complexity")
        # 跳过sympy比较，直接进行数值和简单字符串比较
        pass
    
    # 3. 规范化和Sympy比较（如果启用并可用）
    elif normalize and HAS_SYMPY:
        norm_ans1 = normalize_math_expression(ans1_str, normalize=True)
        norm_ans2 = normalize_math_expression(ans2_str, normalize=True)
        
        # 先比较规范化后的字符串
        if norm_ans1 == norm_ans2:
            # 检查规范化是否对两个表达式都失败，返回原始字符串
            if norm_ans1 == ans1_str and norm_ans2 == ans2_str and ans1_str != ans2_str:
                pass  # 如果返回了原始字符串，继续进行sympy比较
            else:
                return True  # 如果规范化字符串匹配，则返回True
        
        # 尝试sympy比较（如果规范化后的表达式有效）
        try:
            # 使用超时机制进行sympy比较
            expr1, exception1 = with_timeout(
                func=sympy.sympify, 
                args=(norm_ans1,), 
                timeout=SYMPY_TIMEOUT_SECONDS
            )
            
            if exception1 is not None:
                raise exception1
                
            expr2, exception2 = with_timeout(
                func=sympy.sympify, 
                args=(norm_ans2,), 
                timeout=SYMPY_TIMEOUT_SECONDS
            )
            
            if exception2 is not None:
                raise exception2
            
            # 检查类型是否正确
            if not isinstance(expr1, SympyBasic) or not isinstance(expr2, SympyBasic):
                raise TypeError("One or both sympify results are not Sympy expressions.")
            
            # 使用equals()进行结构比较
            equals_result, exception3 = with_timeout(
                func=expr1.equals, 
                args=(expr2,), 
                timeout=SYMPY_TIMEOUT_SECONDS
            )
            
            if exception3 is not None:
                raise exception3
                
            if equals_result:
                return True
            
            # 备用方法：检查差值简化（可能较慢）
            try:
                # 设置差值简化的超时
                diff_expr = expr1 - expr2
                simplified_diff, exception4 = with_timeout(
                    func=sympy.simplify, 
                    args=(diff_expr,), 
                    timeout=SYMPY_TIMEOUT_SECONDS
                )
                
                if exception4 is not None:
                    raise exception4
                    
                if simplified_diff == 0:
                    return True
                    
            except Exception as simp_e:
                logger.debug(f"Sympy difference simplification failed: {simp_e}")
                
        except Exception as e:
            # 如果sympy比较失败，继续使用备用方法
            logger.debug(f"Sympy comparison failed: {e}")
    
    # 4. 备用：数值比较（如果两者都像数字）
    try:
        num1 = float(ans1_str)
        num2 = float(ans2_str)
        
        # 适当处理NaN/inf
        if math.isnan(num1) and math.isnan(num2): 
            return True
            
        if math.isinf(num1) and math.isinf(num2) and (num1 > 0) == (num2 > 0): 
            return True
            
        # 使用math.isclose进行浮点比较
        if math.isclose(num1, num2, rel_tol=1e-4, abs_tol=1e-6):
            return True
            
    except (ValueError, TypeError):
        pass  # 不是数字，继续下一步
    
    # 5. 备用：简单字符串清理和比较
    def clean_str(s):
        s = re.sub(r"\\boxed\{([^}]+)\}", r"\1", s)  # 移除\boxed
        s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)  # 基本分数转换
        s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)  # 基本平方根转换
        s = re.sub(r"\\left|\\right", "", s)  # 移除\left, \right
        s = re.sub(r"[{}]", "", s)  # 移除大括号
        s = re.sub(r"\s+", "", s)  # 移除所有空白
        return s.lower()
    
    clean1 = clean_str(ans1_str)
    clean2 = clean_str(ans2_str)
    
    if clean1 == clean2:
        return True
    
    # 如果所有检查都失败
    return False


# 初始化超时模块
def initialize_timeout_module():
    """初始化超时模块，设置适当的超时处理机制"""
    global SYMPY_TIMEOUT_SECONDS
    
    # 如果有环境变量设置超时值，则使用环境变量
    env_timeout = os.environ.get('SYMPY_TIMEOUT_SECONDS')
    if env_timeout:
        try:
            SYMPY_TIMEOUT_SECONDS = int(env_timeout)
            logger.info_rank0(f"Setting sympy timeout to {SYMPY_TIMEOUT_SECONDS} seconds from environment")
        except ValueError:
            pass


# 初始化超时模块
initialize_timeout_module()