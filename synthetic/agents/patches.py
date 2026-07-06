"""
Patches and compatibility shims for smolagents + vLLM interoperability.

These patches address known issues in smolagents 1.26.0 when used with
newer versions of vLLM (>= 0.11).
"""

import functools
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any

from smolagents.models import VLLMModel


class _PatchedVLLMModel(VLLMModel):
    """VLLMModel patch that fixes generation-time sampling-parameter handling.

    smolagents 1.26.0 forwards unrecognized kwargs (such as `max_tokens`)
    straight into `vllm.LLM.generate()`, which rejects them and crashes
    the run.

    This subclass keeps the requested generation settings working by:

    * storing `max_tokens`, `temperature`, `top_p` and `top_k`
      separately (so they never land in `self.kwargs` and leak into
      `LLM.generate()`),
    * injecting `max_tokens` and `temperature` into each `generate()`
      call so the `SamplingParams` built by smolagents honours them,
    * temporarily wrapping `vllm.SamplingParams` to inject
      `top_p`/`top_k` (which smolagents never forwards), and
    * temporarily wrapping the underlying `LLM.generate()` to drop the
      leaked sampling kwargs before vLLM sees them.

    You can see the discussion here:
    - https://github.com/huggingface/smolagents/issues/2417
    """

    def __init__(
        self,
        *args: Any,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._generation_max_tokens = max_tokens
        self._generation_temperature = temperature
        self._generation_top_p = top_p
        self._generation_top_k = top_k

    def generate(self, messages, **kwargs):  # type: ignore[override]
        if self._generation_max_tokens is not None:
            kwargs.setdefault("max_tokens", self._generation_max_tokens)
        # `temperature` is the only sampling param read from the generate
        # kwargs by smolagents' SamplingParams construction.
        if self._generation_temperature is not None:
            kwargs.setdefault("temperature", self._generation_temperature)

        # smolagents never forwards `top_p`/`top_k` to SamplingParams,
        # so we inject them by wrapping the SamplingParams constructor for
        # the duration of this call.
        import vllm  # type: ignore

        original_sampling_params = vllm.SamplingParams
        extra_sampling: dict[str, Any] = {}
        if self._generation_top_p is not None:
            extra_sampling["top_p"] = self._generation_top_p
        if self._generation_top_k is not None:
            extra_sampling["top_k"] = self._generation_top_k

        def _sampling_params_with_extra(*sp_args: Any, **sp_kwargs: Any):
            for key, value in extra_sampling.items():
                sp_kwargs.setdefault(key, value)
            return original_sampling_params(*sp_args, **sp_kwargs)

        original_generate = self.model.generate

        def _generate_without_leaked_kwargs(*gen_args: Any, **gen_kwargs: Any):
            # smolagents leaks sampling kwargs into LLM.generate(); strip the
            # ones the vLLM offline engine does not accept (they are already
            # applied via SamplingParams).
            for leaked in ("max_tokens", "temperature", "top_p", "top_k"):
                gen_kwargs.pop(leaked, None)
            # Suppress vLLM's per-call "Processed prompts" tqdm progress bar,
            # which otherwise floods stderr (and leaks into captured error
            # messages).
            gen_kwargs.setdefault("use_tqdm", False)
            return original_generate(*gen_args, **gen_kwargs)

        vllm.SamplingParams = _sampling_params_with_extra
        self.model.generate = _generate_without_leaked_kwargs
        try:
            return super().generate(messages, **kwargs)
        finally:
            self.model.generate = original_generate
            vllm.SamplingParams = original_sampling_params


def _ensure_vllm_tokenizer_compat() -> None:
    """Restore the legacy `vllm.transformers_utils.tokenizer` import path.

    smolagents 1.26.0 imports `get_tokenizer` from
    `vllm.transformers_utils.tokenizer`, but vLLM >= ~0.11 moved that
    function to `vllm.tokenizers`. When running against the newer vLLM we
    register an alias module so the import keeps working without pinning
    an older vLLM build.

    You can see the discussion here:
    - https://github.com/huggingface/smolagents/issues/2417
    """
    import importlib
    import sys
    import types

    legacy_path = "vllm.transformers_utils.tokenizer"
    if legacy_path in sys.modules:
        return
    try:
        importlib.import_module(legacy_path)
        return  # Legacy path already exists on this vLLM version.
    except ModuleNotFoundError:
        pass

    try:
        from vllm.tokenizers import cached_get_tokenizer, get_tokenizer
    except Exception as exc:
        print(f"Could not set up vLLM tokenizer compatibility shim: {exc}")
        return

    shim = types.ModuleType(legacy_path)
    shim.get_tokenizer = get_tokenizer
    shim.cached_get_tokenizer = cached_get_tokenizer
    sys.modules[legacy_path] = shim
    print(f"🩹 Applied vLLM tokenizer compatibility shim ({legacy_path} -> vllm.tokenizers).")


def _patch_smolagents_execution_timeout() -> None:
    """Patch smolagents' code-execution timeout to never deadlock the run.

    `smolagents.local_python_executor.timeout()` (1.26.0) wraps each code
    step in `with ThreadPoolExecutor(max_workers=1) as executor:` and calls
    `future.result(timeout=timeout_seconds)`.  If the wrapped call never
    returns — e.g. a hung network request from `DuckDuckGoSearchTool` /
    `WikipediaSearchTool` (no internet egress on some compute nodes, DNS
    hang, upstream rate limiting, ...) or literally an LLM-generated
    infinite loop — `future.result()` correctly raises after
    *timeout_seconds*, but the surrounding `with` block still calls
    `executor.shutdown(wait=True)` on exit. Since the leaked worker thread
    can never be killed and never returns, `shutdown(wait=True)` blocks
    *forever*, silently freezing the whole job with zero further log
    output.

    This patches the module-level `timeout` decorator (referenced by name
    at call time inside `LocalPythonExecutor.__call__`, so replacing the
    module attribute here takes effect immediately) to call
    `executor.shutdown(wait=False)` instead, so a stuck call can no longer
    block per-example progress.

    See: https://github.com/huggingface/smolagents/blob/v1.26.0/src/smolagents/local_python_executor.py
    Issue: https://github.com/huggingface/smolagents/issues/2464
    """
    import smolagents.local_python_executor as _lpe

    def _non_deadlocking_timeout(timeout_seconds: int):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any):
                executor = ThreadPoolExecutor(max_workers=1)
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_seconds)
                except FuturesTimeoutError:
                    # Do NOT block on shutdown(wait=True): the worker thread
                    # may be stuck forever on a hung network call.
                    executor.shutdown(wait=False)
                    raise _lpe.ExecutionTimeoutError(
                        f"Code execution exceeded the maximum execution time "
                        f"of {timeout_seconds} seconds"
                    ) from None

            return wrapper

        return decorator

    _lpe.timeout = _non_deadlocking_timeout


# Maximum bit-length allowed for the result of a single integer operation
# inside the sandboxed executor.  ~10 Mbit (≈ 1.25 MB integer) is orders of
# magnitude beyond any legitimate calculation while still being instant to
# estimate and reject.
_MAX_INT_RESULT_BITS = 10_000_000


def _patch_smolagents_binop_guard() -> None:
    """Patch smolagents' `evaluate_binop` to reject unbounded-size integer ops.

    This closes a second deadlock that the execution-timeout patch
    (`_patch_smolagents_execution_timeout`) cannot: a thread-based timeout
    is unable to interrupt a *single* CPU-bound big-integer operation.

    When the model generates code such as `10 ** 10 ** 8` (or an equally
    explosive `<<` / `*`), CPython computes the astronomically large
    integer entirely in C while holding the GIL and never crossing a
    bytecode boundary.  `future.result()` still raises after the timeout,
    but the leaked worker thread keeps churning and monopolises the GIL
    forever.  On the next agent step the main thread calls
    `ThreadPoolExecutor.submit()` -> `Thread.start()`, which blocks
    waiting for the new worker to signal startup — impossible while the
    leaked thread holds the GIL.  The whole job then freezes with zero
    further output.

    smolagents' own `MAX_OPERATIONS` guard does not help here: a giant
    `**` counts as a single operation.  This patch estimates the result
    magnitude of the explosive integer operators (`**`, `<<`, `*`)
    *before* computing them and raises `InterpreterError` when the result
    would exceed `_MAX_INT_RESULT_BITS` bits, so the agent simply sees a
    normal code error and can recover.

    Like the timeout patch, this replaces a module-level function that
    `evaluate_ast` looks up by name at call time, so swapping the module
    attribute takes effect immediately.
    """
    import ast

    import smolagents.local_python_executor as _lpe

    InterpreterError = _lpe.InterpreterError
    evaluate_ast = _lpe.evaluate_ast

    def _guarded_evaluate_binop(binop, state, static_tools, custom_tools, authorized_imports):
        left_val = evaluate_ast(binop.left, state, static_tools, custom_tools, authorized_imports)
        right_val = evaluate_ast(binop.right, state, static_tools, custom_tools, authorized_imports)
        op = binop.op

        # Size guards for the operators that can turn tiny operands into a
        # gigantic integer (and thus hang the interpreter uninterruptibly).
        if isinstance(left_val, int) and isinstance(right_val, int):
            est_bits = 0
            if isinstance(op, ast.Pow):
                if right_val > 1 and left_val not in (-1, 0, 1):
                    est_bits = right_val * max(left_val.bit_length(), 1)
            elif isinstance(op, ast.LShift):
                if right_val > 0 and left_val != 0:
                    est_bits = left_val.bit_length() + right_val
            elif isinstance(op, ast.Mult):
                est_bits = left_val.bit_length() + right_val.bit_length()
            if est_bits > _MAX_INT_RESULT_BITS:
                raise InterpreterError(
                    f"Refusing to evaluate {type(op).__name__} on integers: "
                    f"the result would be ~{est_bits} bits "
                    f"(limit {_MAX_INT_RESULT_BITS}).  Aborting to avoid an "
                    "uninterruptible big-integer computation that would hang "
                    "the interpreter."
                )

        # Dispatch the operator (mirrors smolagents' evaluate_binop).
        if isinstance(op, ast.Add):
            return left_val + right_val
        elif isinstance(op, ast.Sub):
            return left_val - right_val
        elif isinstance(op, ast.Mult):
            return left_val * right_val
        elif isinstance(op, ast.Div):
            return left_val / right_val
        elif isinstance(op, ast.Mod):
            return left_val % right_val
        elif isinstance(op, ast.Pow):
            return left_val**right_val
        elif isinstance(op, ast.FloorDiv):
            return left_val // right_val
        elif isinstance(op, ast.BitAnd):
            return left_val & right_val
        elif isinstance(op, ast.BitOr):
            return left_val | right_val
        elif isinstance(op, ast.BitXor):
            return left_val ^ right_val
        elif isinstance(op, ast.LShift):
            return left_val << right_val
        elif isinstance(op, ast.RShift):
            return left_val >> right_val
        else:
            raise NotImplementedError(f"Binary operation {type(op).__name__} is not implemented.")

    _lpe.evaluate_binop = _guarded_evaluate_binop
