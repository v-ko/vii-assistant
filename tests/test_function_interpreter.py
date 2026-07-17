"""Tests for HybridFunctionInterpreter."""

from __future__ import annotations

from assistant.inference.function_interpreter import HybridFunctionInterpreter


def _make_hfi() -> HybridFunctionInterpreter:
    hfi = HybridFunctionInterpreter()

    @hfi.function()
    def crop_text(description: str) -> str:
        return f"text from: {description}"

    @hfi.function()
    def crop_image(description: str) -> str:
        return f"image from: {description}"

    @hfi.function(name="curator_push")
    def curator_push(feed: str, content: dict, metadata: dict | None = None) -> str:
        return f"pushed to {feed}"

    return hfi


# --- Registration ---


def test_basic_registration():
    hfi = HybridFunctionInterpreter()

    @hfi.function()
    def add(a: int, b: int) -> int:
        return a + b

    assert "add" in hfi.list_functions()
    func_def = hfi.get_function("add")
    assert func_def is not None
    assert func_def.name == "add"
    assert func_def.parameters == ["a", "b"]


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


# --- parse_line ---


def test_parse_line():
    hfi = HybridFunctionInterpreter()

    @hfi.function()
    def locate(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
        return (x1, y1, x2, y2)

    result = hfi.parse_line("locate(10, 20, 100, 200)")
    assert result is not None
    lhs, func_name, args = result
    assert lhs is None
    assert func_name == "locate"
    assert args == ["10", "20", "100", "200"]

    result = hfi.parse_line("output.bbox = locate(10, 20, 100, 200)")
    assert result is not None
    lhs, func_name, args = result
    assert lhs == "output.bbox"
    assert func_name == "locate"
    assert args == ["10", "20", "100", "200"]

    result = hfi.parse_line("This is not a function call")
    assert result is None


# --- execute_call ---


def test_execute_call():
    hfi = HybridFunctionInterpreter()

    @hfi.function()
    def multiply(a: int, b: int) -> int:
        return a * b

    result = hfi.execute_call("multiply", ["5", "7"])
    assert result == 35


def test_parse_line_and_execute():
    hfi = HybridFunctionInterpreter()

    @hfi.function()
    def add(a: int, b: int) -> int:
        return a + b

    @hfi.function(name="subtract")
    def sub(a: int, b: int) -> int:
        return a - b

    lines = [
        "result1 = add(10, 20)",
        "add(5, 3)",
        "result2 = subtract(100, 30)",
        "some random text",
        "subtract(50, 10)",
    ]

    results = []
    for line in lines:
        parsed = hfi.parse_line(line)
        if parsed is None:
            continue
        lhs, func_name, args = parsed
        result = hfi.execute_call(func_name, args)
        results.append((lhs, result))

    assert len(results) == 4
    assert results[0] == ("result1", 30)
    assert results[1] == (None, 8)
    assert results[2] == ("result2", 70)
    assert results[3] == (None, 40)


# --- resolve_arg ---


def test_resolve_arg_string_literal():
    hfi = _make_hfi()
    assert hfi.resolve_arg("'hello world'") == "hello world"
    assert hfi.resolve_arg('"double quotes"') == "double quotes"


def test_resolve_arg_variable():
    hfi = _make_hfi()
    hfi.variables["my_var"] = "resolved_value"
    assert hfi.resolve_arg("my_var") == "resolved_value"
    hfi.variables["my_dict"] = {"key": "val"}
    result = hfi.resolve_arg("my_dict")
    assert isinstance(result, dict)
    assert result == {"key": "val"}


def test_resolve_arg_dict_access():
    hfi = _make_hfi()
    hfi.variables["data"] = {"key": "val"}
    assert hfi.resolve_arg("data['key']") == "val"


# --- execute_tool_calls ---


def test_execute_tool_calls_simple_assignment():
    hfi = _make_hfi()
    results = hfi.execute_tool_calls("title = crop_text('the album title')")
    assert len(results) == 1
    assert results[0][1] == "crop_text"
    assert results[0][2] == "text from: the album title"
    assert hfi.variables["title"] == "text from: the album title"


def test_execute_tool_calls_dict_ops():
    hfi = _make_hfi()
    code = """item = {}
item['title'] = crop_text('the album title')
item['cover'] = crop_image('the album cover')"""
    results = hfi.execute_tool_calls(code)
    assert len(results) == 2
    assert hfi.variables["item"]["title"] == "text from: the album title"
    assert hfi.variables["item"]["cover"] == "image from: the album cover"


def test_execute_tool_calls_curator_push():
    hfi = _make_hfi()
    code = """content = {}
content['text'] = crop_text('the title')
curator_push('new_stuff', content)"""
    results = hfi.execute_tool_calls(code)
    assert len(results) == 2  # crop_text + curator_push
    assert results[1][1] == "curator_push"
    _, _, _, resolved_args = results[1]
    assert resolved_args[0] == "new_stuff"
    assert isinstance(resolved_args[1], dict)
    assert resolved_args[1]["text"] == "text from: the title"


def test_execute_tool_calls_curator_push_with_metadata():
    hfi = _make_hfi()
    code = """curator_push('my_feed', {'text': 'hello'}, {'title': 'A greeting'})"""
    results = hfi.execute_tool_calls(code)
    assert len(results) == 1
    _, func_name, _, resolved_args = results[0]
    assert func_name == "curator_push"
    assert resolved_args[0] == "my_feed"
    assert resolved_args[1] == {"text": "hello"}
    assert resolved_args[2] == {"title": "A greeting"}


def test_execute_tool_calls_returns_resolved_args():
    hfi = _make_hfi()
    results = hfi.execute_tool_calls("x = crop_text('hello world')")
    assert len(results) == 1
    lhs, func_name, result, resolved_args = results[0]
    assert lhs == "x"
    assert func_name == "crop_text"
    assert resolved_args == ["hello world"]


def test_execute_tool_calls_preserves_state():
    hfi = _make_hfi()
    hfi.execute_tool_calls("x = crop_text('first')")
    assert hfi.variables["x"] == "text from: first"

    hfi2 = _make_hfi()
    hfi2.variables = dict(hfi.variables)
    assert hfi2.variables["x"] == "text from: first"


def test_execute_tool_calls_comments_and_blanks():
    hfi = _make_hfi()
    code = """# This is a comment

x = crop_text('test')
# another comment
"""
    results = hfi.execute_tool_calls(code)
    assert len(results) == 1
    assert hfi.variables["x"] == "text from: test"


def test_execute_tool_calls_ignores_prose():
    """Raw model output with prose + code lines — only valid ops execute."""
    hfi = _make_hfi()
    text = """I'll extract the album information for you.

item = {}
item['title'] = crop_text('the album title')

Here is the cover image:

item['cover'] = crop_image('the album cover')
curator_push('new_stuff', item)

That should be everything!
"""
    results = hfi.execute_tool_calls(text)
    assert len(results) == 3
    assert results[0][1] == "crop_text"
    assert results[1][1] == "crop_image"
    assert results[2][1] == "curator_push"
    assert hfi.variables["item"]["title"] == "text from: the album title"


def test_execute_tool_calls_does_not_coerce_subscript_as_call():
    """Subscript syntax is not treated as a function call."""
    hfi = _make_hfi()
    code = """\
content = {}
content['title'] = crop_text['The song title "Words"']
content['artist'] = crop_text['The artist name "Feint"']
content['image'] = crop_image['The album cover for "Words"']

curator_push('new_stuff', content)
"""
    results = hfi.execute_tool_calls(code)
    assert len(results) == 1
    assert results[0][1] == "curator_push"
    assert hfi.variables["content"]["title"] is None
    assert hfi.variables["content"]["artist"] is None
    assert hfi.variables["content"]["image"] is None
    _, _, _, push_args = results[0]
    assert push_args[0] == "new_stuff"
    assert push_args[1]["title"] is None
