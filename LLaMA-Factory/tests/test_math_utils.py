import sys
import os
import pytest

# Add project root to sys.path to allow importing llamafactory modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from llamafactory.extras.math_utils import extract_boxed_answer, math_equal, normalize_math_expression, HAS_SYMPY
except ImportError as e:
    print(f"Error importing llamafactory modules: {e}")
    print(f"sys.path: {sys.path}")
    pytest.skip("Skipping tests due to import error.", allow_module_level=True)


def test_extract_boxed_answer():
    assert extract_boxed_answer("The answer is \\boxed{42}") == "42"
    assert extract_boxed_answer("Equation: \\boxed{x=5}") == "x=5"
    assert extract_boxed_answer("Multiple boxes \\boxed{a} and \\boxed{b}") == "b"
    assert extract_boxed_answer("No box here") is None
    assert extract_boxed_answer("Invalid box \\boxed {test}") is None # Space after command
    assert extract_boxed_answer("Nested \\boxed{outer \\boxed{inner}}") == "outer \\boxed{inner}" # Extracts outer
    assert extract_boxed_answer("\\boxed{}") == "" # Empty box

@pytest.mark.skipif(not HAS_SYMPY, reason="sympy not installed")
def test_normalize_math_expression_sympy():
    assert normalize_math_expression(" 3 + 2 ") == "5"
    assert normalize_math_expression("x*y") == "x*y" # Order might depend on sympy version
    assert normalize_math_expression("y * x") == "x*y" # Should normalize order
    assert normalize_math_expression("1/2 + 1/2") == "1"
    assert normalize_math_expression("a + b - a") == "b"
    assert normalize_math_expression("invalid expression [") == "invalid expression [" # Should return original on error

def test_normalize_math_expression_no_sympy():
    # Test basic stripping without sympy
    assert normalize_math_expression(" 3 + 2 ", normalize=False) == "3 + 2"
    assert normalize_math_expression("  test  ", normalize=False) == "test"

def test_math_equal_basic():
    assert math_equal("42", "42") is True
    assert math_equal("abc", "abc") is True
    assert math_equal("42", "43") is False
    assert math_equal("abc", "def") is False
    assert math_equal(None, "42") is False
    assert math_equal("42", None) is False
    assert math_equal(None, None) is False

@pytest.mark.skipif(not HAS_SYMPY, reason="sympy not installed")
def test_math_equal_sympy():
    assert math_equal("2+3", "5") is True
    assert math_equal("x*y", "y*x") is True
    assert math_equal("1/2", "0.5") is True
    assert math_equal("(a+b)**2", "a**2 + 2*a*b + b**2") is True
    assert math_equal("sin(pi/2)", "1") is True
    assert math_equal("2+3", "6") is False
    assert math_equal("x", "y") is False
    assert math_equal("invalid [", "invalid [") is True # Equal if normalization fails identically
    assert math_equal("invalid [", "other ]") is False

def test_math_equal_no_normalize():
    assert math_equal("2+3", "5", normalize=False) is False # Not identical strings
    assert math_equal(" 5 ", "5", normalize=False) is False # Whitespace differs
    assert math_equal("5", "5", normalize=False) is True
