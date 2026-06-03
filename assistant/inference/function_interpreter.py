from __future__ import annotations

import ast
import inspect
from typing import Any, Callable, TypeVar, get_type_hints

from sivkit import get_logger

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
        self.variables: dict[str, Any] = {}

    def resolve_arg(self, arg_str: str) -> Any:
        """Resolve a string expression to a Python value using ast."""
        try:
            node = ast.parse(arg_str.strip(), mode="eval").body
            return self._resolve_node(node)
        except SyntaxError:
            return arg_str.strip()

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

            log.debug(
                f"Registered function '{func_name}' with params: {func_def.parameters}"
            )

            return func

        return decorator

    def parse_line(self, line: str) -> tuple[str | None, str, list[str]] | None:
        """Parse a single line for a registered function call using ast.

        Returns:
            Tuple of (lhs, func_name, args_list) if match found, None otherwise.
            args_list contains string representations of each argument.
        """
        try:
            tree = ast.parse(line.strip(), mode="exec")
        except SyntaxError:
            return None

        if not tree.body:
            return None

        stmt = tree.body[0]

        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            if not isinstance(stmt.value, ast.Call):
                return None
            fname = self._func_name_from_node(stmt.value.func)
            if fname is None or fname not in self._functions:
                return None
            lhs = self._target_str(stmt.targets[0])
            args = [ast.unparse(a) for a in stmt.value.args]
            return (lhs, fname, args)

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            fname = self._func_name_from_node(stmt.value.func)
            if fname is None or fname not in self._functions:
                return None
            args = [ast.unparse(a) for a in stmt.value.args]
            return (None, fname, args)

        return None

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

    def execute_tool_calls(
        self,
        text: str,
    ) -> list[tuple[str | None, str, Any, list[Any]]]:
        """Parse and execute a code block with variable tracking.

        Uses ast.parse for robust Python syntax handling. Processes:
        - Assignments with function calls: ``var = func(args)``
        - Dict key assignments: ``var['key'] = func(args)``
        - Bare function calls: ``func(args)``
        - Variable/dict assignments: ``var = expr``, ``var = {}``

        Returns list of (lhs, func_name, result, resolved_args) for each
        executed function call.
        """
        results: list[tuple[str | None, str, Any, list[Any]]] = []

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                tree = ast.parse(line, mode="exec")
            except SyntaxError:
                continue

            for stmt in tree.body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    target = stmt.targets[0]
                    value = stmt.value

                    if isinstance(value, ast.Call):
                        fname = self._func_name_from_node(value.func)
                        if fname and fname in self._functions:
                            resolved_args = [self._resolve_node(a) for a in value.args]
                            try:
                                result = self._call_function(fname, resolved_args)
                            except Exception as e:
                                log.error(f"Failed to execute {fname}: {e}")
                                continue
                            lhs_str = self._target_str(target)
                            self._assign_to_target(target, result)
                            results.append((lhs_str, fname, result, resolved_args))
                            log.debug(f"Executed {lhs_str} = {fname}(...) -> {result}")
                            continue

                    # Plain assignment: var = expr
                    resolved = self._resolve_node(value)
                    self._assign_to_target(target, resolved)
                    log.debug(f"Assigned {self._target_str(target)} = {resolved!r}")

                elif isinstance(stmt, ast.Expr):
                    if isinstance(stmt.value, ast.Call):
                        fname = self._func_name_from_node(stmt.value.func)
                        if fname and fname in self._functions:
                            resolved_args = [
                                self._resolve_node(a) for a in stmt.value.args
                            ]
                            try:
                                result = self._call_function(fname, resolved_args)
                            except Exception as e:
                                log.error(f"Failed to execute {fname}: {e}")
                                continue
                            results.append((None, fname, result, resolved_args))
                            log.debug(f"Executed {fname}(...) -> {result}")

        return results

    def _call_function(self, func_name: str, resolved_args: list[Any]) -> Any:
        """Call a registered function with already-resolved Python values."""
        if func_name not in self._functions:
            raise KeyError(f"Function '{func_name}' not registered")
        func_def = self._functions[func_name]
        return func_def.func(*resolved_args)

    def _resolve_node(self, node: ast.expr) -> Any:
        """Resolve an AST expression node to a Python value."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.variables:
                return self.variables[node.id]
            return node.id
        if isinstance(node, ast.Subscript):
            obj = self._resolve_node(node.value)
            key = self._resolve_node(node.slice)
            if isinstance(obj, dict):
                return obj.get(key)
            return None
        if isinstance(node, ast.Dict):
            return {
                self._resolve_node(k): self._resolve_node(v)
                for k, v in zip(node.keys, node.values)
                if k is not None
            }
        if isinstance(node, ast.List):
            return [self._resolve_node(el) for el in node.elts]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            val = self._resolve_node(node.operand)
            if isinstance(val, (int, float)):
                return -val
        return ast.unparse(node)

    def _func_name_from_node(self, node: ast.expr) -> str | None:
        """Extract dotted function name (e.g. 'curator.push') from a Call's func."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts: list[str] = []
            n: ast.expr = node
            while isinstance(n, ast.Attribute):
                parts.append(n.attr)
                n = n.value
            if isinstance(n, ast.Name):
                parts.append(n.id)
                return ".".join(reversed(parts))
        return None

    def _target_str(self, node: ast.expr) -> str:
        """Convert an assignment target to a display string."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Subscript):
            base = self._target_str(node.value)
            key = self._resolve_node(node.slice)
            return f"{base}['{key}']"
        if isinstance(node, ast.Attribute):
            base = self._target_str(node.value)
            return f"{base}.{node.attr}"
        return ast.unparse(node)

    def _assign_to_target(self, target: ast.expr, value: Any) -> None:
        """Assign a value into self.variables based on AST target shape."""
        if isinstance(target, ast.Name):
            self.variables[target.id] = value
        elif isinstance(target, ast.Subscript):
            obj = self._resolve_node(target.value)
            key = self._resolve_node(target.slice)
            if isinstance(obj, dict):
                obj[key] = value

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
