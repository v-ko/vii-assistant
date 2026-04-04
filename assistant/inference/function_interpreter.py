from __future__ import annotations

import inspect
import re
from typing import Any, Callable, TypeVar, get_type_hints

from fusion import get_logger

log = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class FunctionDefinition:
    def __init__(
        self,
        name: str,
        func: Callable[..., Any],
        signature: inspect.Signature,
        description: str | None = None,
        resolved_hints: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.func = func
        self.signature = signature
        self.description = description or ""
        self.parameters = list(signature.parameters.keys())
        self.resolved_hints = resolved_hints or {}


class HybridFunctionInterpreter:
    """Decorator-based function call parser inspired by FastAPI.

    Example:
        hfi = HybridFunctionInterpreter()

        @hfi.function()
        def locate(x1: int, y1: int, x2: int, y2: int) -> Rectangle:
            ...

        # Parse text containing function calls
        results = hfi.parse("locate(10, 20, 100, 200)")
    """

    def __init__(self) -> None:
        self._functions: dict[str, FunctionDefinition] = {}
        self._call_pattern: re.Pattern[str] | None = None

    def function(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[[F], F]:
        """Decorator to register a function for parsing.

        Args:
            name: Optional custom name for the function. Defaults to function.__name__
            description: Optional description of what the function does
        """

        def decorator(func: F) -> F:
            func_name = name or func.__name__
            sig = inspect.signature(func)
            try:
                hints = get_type_hints(func)
            except Exception:
                hints = {}

            func_def = FunctionDefinition(
                name=func_name,
                func=func,
                signature=sig,
                description=description,
                resolved_hints=hints,
            )

            self._functions[func_name] = func_def
            self._call_pattern = None  # Invalidate cached pattern

            log.debug(
                f"Registered function '{func_name}' with params: {func_def.parameters}"
            )

            return func

        return decorator

    def _build_call_pattern(self) -> re.Pattern[str]:
        """Build regex pattern matching all registered function names."""
        if not self._functions:
            # Match nothing if no functions registered
            return re.compile(r"(?!.*)")

        func_names = "|".join(re.escape(name) for name in self._functions.keys())
        # Pattern: optional assignment, function name, args in parens
        pattern = (
            r"^\s*(?:"
            r"(?P<lhs>[A-Za-z0-9_.]+)\s*=\s*"
            r")?(?P<func>" + func_names + r")\s*\((?P<args>[^)]*)\)\s*$"
        )
        return re.compile(pattern)

    def parse_line(self, line: str) -> tuple[str | None, str, list[str]] | None:
        """Parse a single line for a function call.

        Returns:
            Tuple of (lhs, func_name, args_list) if match found, None otherwise
            lhs is the assignment target (if present), func_name is the function,
            args_list is the comma-separated arguments as strings
        """
        if self._call_pattern is None:
            self._call_pattern = self._build_call_pattern()

        match = self._call_pattern.match(line.strip())
        if not match:
            return None

        lhs = match.group("lhs")
        func_name = match.group("func")
        args_raw = match.group("args")

        args_list = [arg.strip() for arg in args_raw.split(",") if arg.strip()]

        return (lhs, func_name, args_list)

    def execute_call(
        self,
        func_name: str,
        args: list[str],
        type_converters: dict[type, Callable[[str], Any]] | None = None,
    ) -> Any:
        """Execute a registered function with string arguments.

        Args:
            func_name: Name of the registered function
            args: List of string arguments to convert and pass
            type_converters: Optional dict mapping types to converter functions

        Returns:
            The return value of the function call

        Raises:
            KeyError: If function not registered
            TypeError: If argument conversion fails or count mismatch
        """
        if func_name not in self._functions:
            raise KeyError(f"Function '{func_name}' not registered")

        func_def = self._functions[func_name]
        sig = func_def.signature

        # Build type converter lookup: custom overrides default
        default_converters = {
            int: int,
            float: float,
            str: str,
            bool: lambda s: s.lower() in ("true", "1", "yes"),
        }
        converters = default_converters.copy()
        if type_converters:
            converters.update(type_converters)

        # Convert arguments based on function signature
        converted_args = []
        params = list(sig.parameters.values())

        if len(args) != len(params):
            raise TypeError(
                f"Function '{func_name}' expects {len(params)} arguments, got"
                f" {len(args)}"
            )

        hints = func_def.resolved_hints
        for arg_str, param in zip(args, params):
            param_type = hints.get(param.name, param.annotation)
            if param_type is inspect.Parameter.empty:
                # No type hint, use as string
                converted_args.append(arg_str)
                continue

            # Try custom converter first, then defaults
            converter = converters.get(param_type) or default_converters.get(param_type)
            if converter is None:
                log.warning(
                    f"No converter for type {param_type}, using string for param"
                    f" '{param.name}'"
                )
                converted_args.append(arg_str)
            else:
                try:
                    converted_args.append(converter(arg_str))
                except (ValueError, TypeError) as e:
                    raise TypeError(
                        f"Failed to convert argument '{arg_str}' to {param_type} for"
                        f" param '{param.name}': {e}"
                    ) from e

        return func_def.func(*converted_args)

    def parse_and_execute(
        self,
        text: str,
        type_converters: dict[type, Callable[[str], Any]] | None = None,
    ) -> list[tuple[str | None, Any]]:
        """Parse text for function calls and execute them.

        Args:
            text: Multi-line text potentially containing function calls
            type_converters: Optional dict mapping types to converter functions

        Returns:
            List of (lhs, result) tuples for each successful execution.
            lhs is the assignment target if specified, result is the return value
        """
        results: list[tuple[str | None, Any]] = []

        for line in text.splitlines():
            parsed = self.parse_line(line)
            if parsed is None:
                continue

            lhs, func_name, args = parsed
            try:
                result = self.execute_call(func_name, args, type_converters)
                results.append((lhs, result))
                log.debug(
                    f"Executed {func_name}({', '.join(args)}) -> {result}"
                    + (f" (assigned to {lhs})" if lhs else "")
                )
            except Exception as e:
                log.error(f"Failed to execute {func_name}({', '.join(args)}): {e}")

        return results

    def get_function(self, name: str) -> FunctionDefinition | None:
        """Get a registered function by name."""
        return self._functions.get(name)

    def list_functions(self) -> list[str]:
        """List all registered function names."""
        return list(self._functions.keys())

    def generate_tool_prompt(self) -> str:
        if not self._functions:
            return ""

        lines = [
            "You have access to function calling. Call functions on a new line"
            " using python-like syntax. Available tools:\n"
        ]
        for func_def in self._functions.values():
            params = []
            for p in func_def.signature.parameters.values():
                hint = func_def.resolved_hints.get(p.name)
                type_name = hint.__name__ if hint and hasattr(hint, "__name__") else ""
                params.append(f"{p.name}: {type_name}" if type_name else p.name)
            sig = f"def {func_def.name}({', '.join(params)})"
            ret = func_def.resolved_hints.get("return")
            if ret and hasattr(ret, "__name__"):
                sig += f" -> {ret.__name__}"
            sig += ":"
            if func_def.description:
                lines.append(f"{sig}\n    # {func_def.description}\n    ...\n")
            else:
                lines.append(f"{sig}\n    ...\n")
        return "\n".join(lines)


__all__ = ["HybridFunctionInterpreter", "FunctionDefinition"]
