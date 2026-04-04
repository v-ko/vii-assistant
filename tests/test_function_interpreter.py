from __future__ import annotations

from assistant.inference.function_interpreter import HybridFunctionInterpreter


def test_basic_registration():
    """Test basic function registration."""
    hfi = HybridFunctionInterpreter()

    @hfi.function()
    def add(a: int, b: int) -> int:
        return a + b

    assert "add" in hfi.list_functions()
    func_def = hfi.get_function("add")
    assert func_def is not None
    assert func_def.name == "add"
    assert func_def.parameters == ["a", "b"]


def test_parse_line():
    """Test parsing function calls from text."""
    hfi = HybridFunctionInterpreter()

    @hfi.function()
    def locate(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
        return (x1, y1, x2, y2)

    # Test basic call
    result = hfi.parse_line("locate(10, 20, 100, 200)")
    assert result is not None
    lhs, func_name, args = result
    assert lhs is None
    assert func_name == "locate"
    assert args == ["10", "20", "100", "200"]

    # Test with assignment
    result = hfi.parse_line("output.bbox = locate(10, 20, 100, 200)")
    assert result is not None
    lhs, func_name, args = result
    assert lhs == "output.bbox"
    assert func_name == "locate"
    assert args == ["10", "20", "100", "200"]

    # Test non-matching line
    result = hfi.parse_line("This is not a function call")
    assert result is None


def test_execute_call():
    """Test executing registered functions."""
    hfi = HybridFunctionInterpreter()

    @hfi.function()
    def multiply(a: int, b: int) -> int:
        return a * b

    result = hfi.execute_call("multiply", ["5", "7"])
    assert result == 35


def test_parse_and_execute():
    """Test full parsing and execution flow."""
    hfi = HybridFunctionInterpreter()

    @hfi.function()
    def add(a: int, b: int) -> int:
        return a + b

    @hfi.function(name="subtract")
    def sub(a: int, b: int) -> int:
        return a - b

    text = """
    result1 = add(10, 20)
    add(5, 3)
    result2 = subtract(100, 30)
    some random text
    subtract(50, 10)
    """

    results = hfi.parse_and_execute(text)
    assert len(results) == 4

    # Check results
    assert results[0] == ("result1", 30)
    assert results[1] == (None, 8)
    assert results[2] == ("result2", 70)
    assert results[3] == (None, 40)


def test_custom_name():
    hfi = HybridFunctionInterpreter()

    @hfi.function(name="double")
    def _double(x: int) -> int:
        return x * 2

    assert "double" in hfi.list_functions()
    assert hfi.execute_call("double", ["7"]) == 14


def test_multiple_functions():
    hfi = HybridFunctionInterpreter()

    @hfi.function()
    def add(a: int, b: int) -> int:
        return a + b

    @hfi.function()
    def negate(x: int) -> int:
        return -x

    assert set(hfi.list_functions()) == {"add", "negate"}

    text = "x = add(3, 4)\nnegate(10)"
    results = hfi.parse_and_execute(text)
    assert len(results) == 2
    assert results[0] == ("x", 7)
    assert results[1] == (None, -10)
