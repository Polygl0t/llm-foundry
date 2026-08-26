"""
Custom tools for the CodeAgent.

Three region-aware tools are defined here:
- `RegionDuckDuckGoSearchTool`: free DuckDuckGo search with a `region` code.
- `RegionGoogleSearchTool`: Google search (SerpAPI/Serper) with country,
  language and domain parameters.
- `RegionWikipediaSearchTool`: Wikipedia search with exact-title guidance
  and a title fallback.

The mathematical problem-solving tool (`MathTool`, sympy-backed) and a set
of local filesystem tools are also defined here.  All other tools
(PythonInterpreterTool, FinalAnswerTool, VisitWebpageTool) come from
smolagents.default_tools.
"""

import contextlib
import io
import multiprocessing as mp
import re
import time
from pathlib import Path
from typing import Any

from smolagents import Tool
from smolagents.default_tools import (
    DuckDuckGoSearchTool,
    GoogleSearchTool,
    WikipediaSearchTool,
)


class ReadFileTool(Tool):
    """Read the contents of a file, optionally limited to a line range.

    This is the primary tool for inspecting file contents.  If the file is
    large you can narrow the view with *start_line* / *end_line* to avoid
    blowing up the context window.
    """

    name = "read_file"
    description = (
        "Read the contents of a file on the local file system. "
        "Use *start_line* and *end_line* (1-based, inclusive) to read a "
        "subset of a large file.  If the file does not exist an error is "
        "returned."
    )
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Absolute or relative path to the file to read.",
        },
        "start_line": {
            "type": "integer",
            "description": "First line to read (1-based).  Omit to start at the beginning.",
            "nullable": True,
        },
        "end_line": {
            "type": "integer",
            "description": "Last line to read (1-based, inclusive).  Omit to read to end of file.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        file_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        p = Path(file_path).expanduser().resolve()
        if not p.exists():
            return f"Error: file not found: '{p}'"
        if not p.is_file():
            return f"Error: '{p}' is not a regular file."

        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except Exception as exc:
            return f"Error reading '{p}': {exc}"

        total = len(lines)
        # Clamp and convert to 0-based indices
        sl = max(1, start_line or 1) - 1
        el = min(end_line or total, total)  # end_line is inclusive -> slice end is exclusive

        if sl >= total:
            return f"Error: start_line {sl + 1} exceeds file length ({total} lines)."
        if sl >= el:
            return f"Error: start_line {sl + 1} is after end_line {el}."

        snippet = "".join(lines[sl:el])
        header = f"# {p}  (lines {sl + 1}-{el} of {total})\n"
        return header + snippet


class WriteFileTool(Tool):
    """Create a new file (or overwrite an existing one) with the given content."""

    name = "write_file"
    description = (
        "Create a new file or overwrite an existing file with the given "
        "*content*.  Parent directories are created automatically if needed. "
        "Use this to persist results, intermediate data, or scripts."
    )
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to create or overwrite.",
        },
        "content": {
            "type": "string",
            "description": "The full text content to write to the file.",
        },
    }
    output_type = "string"

    def forward(self, file_path: str, content: str) -> str:
        p = Path(file_path).expanduser().resolve()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(content)
        except Exception as exc:
            return f"Error writing '{p}': {exc}"

        size = p.stat().st_size
        nlines = content.count("\n") + (0 if content.endswith("\n") else 1)
        return f"✅ Wrote {size} byte(s) / {nlines} line(s) to '{p}'."


class EditFileTool(Tool):
    """Replace an exact string in a file with a new string (like sed)."""

    name = "edit_file"
    description = (
        "Replace an exact literal string (*old_string*) with *new_string* "
        "inside an existing file.  The *old_string* must appear exactly once "
        "in the file — if it appears zero or multiple times an error is "
        "returned so you can refine your match.  This is the safest way to "
        "perform surgical edits."
    )
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to edit in-place.",
        },
        "old_string": {
            "type": "string",
            "description": "Exact literal string to replace (must appear exactly once).",
        },
        "new_string": {
            "type": "string",
            "description": "Replacement string.",
        },
    }
    output_type = "string"

    def forward(self, file_path: str, old_string: str, new_string: str) -> str:
        p = Path(file_path).expanduser().resolve()
        if not p.is_file():
            return f"Error: file not found: '{p}'."

        try:
            original = p.read_text(encoding="utf-8")
        except Exception as exc:
            return f"Error reading '{p}': {exc}"

        count = original.count(old_string)
        if count == 0:
            return f"Error: old_string not found in '{p}'."
        if count > 1:
            return (
                f"Error: old_string appears {count} times in '{p}'. "
                f"Please provide more surrounding context to make it unique."
            )

        edited = original.replace(old_string, new_string, 1)
        try:
            p.write_text(edited, encoding="utf-8")
        except Exception as exc:
            return f"Error writing '{p}': {exc}"

        return f"✅ Replaced 1 occurrence in '{p}'."


class ListDirectoryTool(Tool):
    """List the contents of a directory (like `ls -la`)."""

    name = "list_directory"
    description = (
        "List the files and subdirectories inside a directory.  "
        "Returns one entry per line with type indicator ('[DIR]' or '[FILE]'), "
        "size in bytes, and the name.  Entries are sorted alphabetically with "
        "directories first.  Hidden entries (starting with '.') are excluded "
        "by default; set *show_hidden* to True to include them."
    )
    inputs = {
        "directory_path": {
            "type": "string",
            "description": "Path to the directory to list.  Defaults to current working directory.",
            "nullable": True,
        },
        "show_hidden": {
            "type": "boolean",
            "description": "Whether to include hidden files/directories (default false).",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        directory_path: str | None = None,
        show_hidden: bool = False,
    ) -> str:
        p = Path(directory_path).expanduser().resolve() if directory_path else Path.cwd()
        if not p.exists():
            return f"Error: directory not found: '{p}'."
        if not p.is_dir():
            return f"Error: '{p}' is not a directory."

        entries: list[str] = []
        try:
            for child in sorted(p.iterdir()):
                name = child.name
                if not show_hidden and name.startswith("."):
                    continue
                tag = "[DIR]" if child.is_dir() else "[FILE]"
                size = ""
                if child.is_file():
                    try:
                        size = f"  {child.stat().st_size:>10,} B"
                    except OSError:
                        size = "            ? B"
                elif child.is_dir():
                    size = " " * 13  # align with file size column
                if child.is_symlink():
                    tag = "[LINK]"
                entries.append(f"  {tag}{size}  {name}")
        except PermissionError:
            return f"Error: permission denied reading '{p}'."

        if not entries:
            return f"'{p}' is empty."
        return f"# {p}/\n" + "\n".join(entries)


class SearchFilesTool(Tool):
    """Find files matching a glob pattern (like `find` or `fd`)."""

    name = "search_files"
    description = (
        "Search for files and directories matching a glob pattern "
        "(e.g. '**/*.py', '*.jsonl', 'data/**/*.parquet').  "
        "The search is rooted at *base_dir* (defaults to the current "
        "working directory).  Results are returned as absolute paths, one "
        "per line.  Set *max_results* to limit output (default 100)."
    )
    inputs = {
        "pattern": {
            "type": "string",
            "description": "Glob pattern to match (supports ** for recursive matching).",
        },
        "base_dir": {
            "type": "string",
            "description": "Root directory for the search (defaults to cwd).",
            "nullable": True,
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return (default 100).",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        pattern: str,
        base_dir: str | None = None,
        max_results: int = 100,
    ) -> str:
        root = Path(base_dir).expanduser().resolve() if base_dir else Path.cwd()
        if not root.is_dir():
            return f"Error: base directory not found: '{root}'."

        results: list[str] = []
        for p in root.glob(pattern):
            if len(results) >= max_results:
                break
            results.append(str(p))

        if not results:
            return f"No files matched pattern '{pattern}' under '{root}'."
        truncated = len(results) >= max_results
        out = f"# {len(results)} match(es) for '{pattern}' under '{root}'"
        if truncated:
            out += "  (truncated)"
        return out + "\n" + "\n".join(results)


class GrepFilesTool(Tool):
    """Search inside files using a regex pattern (like grep).

    Searches files matching *file_pattern* (glob) for lines that match
    *regex*.  Returns matching lines with file path and line number.
    """

    name = "grep_files"
    description = (
        "Search inside files for lines matching a regular expression. "
        "Provide a *regex* pattern and a *file_pattern* glob to select "
        "which files to search (e.g. '**/*.py', '*.txt', 'docs/**/*.md'). "
        "Set *ignore_case* (default false) for case-insensitive matching. "
        "Results include the file path, line number, and matching line text."
    )
    inputs = {
        "regex": {
            "type": "string",
            "description": "Regular expression pattern to search for.",
        },
        "file_pattern": {
            "type": "string",
            "description": "Glob pattern for files to search (e.g. '**/*.py', '*.jsonl').",
        },
        "base_dir": {
            "type": "string",
            "description": "Root directory for the search (defaults to cwd).",
            "nullable": True,
        },
        "ignore_case": {
            "type": "boolean",
            "description": "Case-insensitive matching if true (default false).",
            "nullable": True,
        },
        "max_matches": {
            "type": "integer",
            "description": "Maximum total matches to return (default 200).",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        regex: str,
        file_pattern: str,
        base_dir: str | None = None,
        ignore_case: bool = False,
        max_matches: int = 200,
    ) -> str:
        root = Path(base_dir).expanduser().resolve() if base_dir else Path.cwd()
        if not root.is_dir():
            return f"Error: base directory not found: '{root}'."

        # Compile the regex
        flags = re.IGNORECASE if ignore_case else 0
        try:
            compiled = re.compile(regex, flags)
        except re.error as exc:
            return f"Error: invalid regex '{regex}': {exc}"

        results: list[str] = []
        files_searched = 0

        for p in root.glob(file_pattern):
            if not p.is_file():
                continue
            files_searched += 1

            # Skip binary files
            try:
                with open(p, "rb") as fh:
                    head = fh.read(8192)
                if b"\x00" in head:
                    continue
            except (OSError, PermissionError):
                continue

            try:
                with open(p, encoding="utf-8", errors="replace") as fh:
                    for line_no, line in enumerate(fh, start=1):
                        if compiled.search(line):
                            results.append(f"{p}:{line_no}: {line.rstrip()}")
                            if len(results) >= max_matches:
                                break
            except (OSError, PermissionError):
                continue

            if len(results) >= max_matches:
                break

        if not results:
            return (
                f"No matches for regex '{regex}' in files matching "
                f"'{file_pattern}' under '{root}' "
                f"({files_searched} file(s) searched)."
            )

        truncated = len(results) >= max_matches
        header = (
            f"# {len(results)} match(es) for '{regex}' in '{file_pattern}' "
            f"({files_searched} file(s) searched)"
        )
        if truncated:
            header += "  (truncated)"
        return header + "\n" + "\n".join(results)


class FileInfoTool(Tool):
    """Get metadata about a file or directory (size, mtime, type, line count)."""

    name = "file_info"
    description = (
        "Return metadata about a file or directory: absolute path, type "
        "(file/directory/symlink), size in bytes, last-modified timestamp, "
        "and for text files an approximate line count."
    )
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Path to the file or directory to inspect.",
        },
    }
    output_type = "string"

    def forward(self, file_path: str) -> str:
        p = Path(file_path).expanduser().resolve()
        if not p.exists():
            return f"Error: path not found: '{p}'."

        lines: list[str] = [f"Path      : {p}"]
        try:
            st = p.stat()
        except OSError as exc:
            return f"Error stat-ing '{p}': {exc}"

        if p.is_symlink():
            lines.append("Type      : symlink")
            lines.append(f"Target    : {p.readlink()}")
        elif p.is_dir():
            lines.append("Type      : directory")
        elif p.is_file():
            lines.append("Type      : file")
        else:
            lines.append("Type      : special")

        lines.append(f"Size      : {st.st_size:,} bytes")
        lines.append(
            f"Modified  : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}"
        )

        if p.is_file():
            # Try to count lines for text files
            try:
                with open(p, "rb") as fh:
                    head = fh.read(8192)
                if b"\x00" not in head:
                    with open(p, encoding="utf-8", errors="replace") as fh:
                        lc = sum(1 for _ in fh)
                    lines.append(f"Lines     : {lc:,}")
            except OSError:
                pass

        return "\n".join(lines)


# Timeout for math tool subprocesses.  This is a hard wall-clock timeout
# that kills the subprocess if it exceeds this duration.  This is necessary
# because sympy can get stuck in big-integer arithmetic or other symbolic
# computations that cannot be preempted by a thread-based timeout.
MATH_TOOL_TIMEOUT_SECONDS = 30


def _math_worker(code: str, problem: str, result_queue: "mp.Queue") -> None:
    """Execute *code* against a sympy-backed namespace in a fresh process.

    Runs in a `multiprocessing` **spawn** subprocess (started fresh via
    `spawn`, i.e. NOT a `fork()` of the parent — forking would be unsafe
    here since the parent process has an active CUDA/vLLM context). Puts a
    `(status, payload)` tuple on *result_queue*, where `status` is
    `"ok"` or `"error"`.
    """
    try:
        import math as _math
        from decimal import Decimal
        from fractions import Fraction

        import sympy

        namespace: dict[str, Any] = {
            "sp": sympy,
            "sympy": sympy,
            "math": _math,
            "Fraction": Fraction,
            "Decimal": Decimal,
            "Symbol": sympy.Symbol,
            "symbols": sympy.symbols,
            "solve": sympy.solve,
            "Eq": sympy.Eq,
            "integrate": sympy.integrate,
            "diff": sympy.diff,
            "limit": sympy.limit,
            "Matrix": sympy.Matrix,
            "simplify": sympy.simplify,
            "expand": sympy.expand,
            "factor": sympy.factor,
            "apart": sympy.apart,
            "together": sympy.together,
            "solve_poly_system": sympy.solve_poly_system,
            "dsolve": sympy.dsolve,
            "rsolve": sympy.rsolve,
            "summation": sympy.summation,
            "product": sympy.product,
            "oo": sympy.oo,
            "pi": sympy.pi,
            "E": sympy.E,
            "I": sympy.I,
            "sqrt": sympy.sqrt,
            "log": sympy.log,
            "exp": sympy.exp,
            "sin": sympy.sin,
            "cos": sympy.cos,
            "tan": sympy.tan,
            "asin": sympy.asin,
            "acos": sympy.acos,
            "atan": sympy.atan,
            "sinh": sympy.sinh,
            "cosh": sympy.cosh,
            "tanh": sympy.tanh,
            "Rational": sympy.Rational,
            "N": sympy.N,
        }

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            try:
                exec(code, namespace)
            except Exception as exc:
                result_queue.put(("error", f"{type(exc).__name__}: {exc}"))
                return

        captured = stdout.getvalue().strip()
        result = namespace.get("result")
        if result is not None and not captured:
            captured = str(result)

        if not captured:
            result_queue.put(
                (
                    "error",
                    "The code ran without printing anything. Make sure to print() the answer.",
                )
            )
            return
        result_queue.put(("ok", f"Problem: {problem}\nResult: {captured}"))
    except Exception as exc:  # pragma: no cover - defensive catch-all
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


class MathTool(Tool):
    """Solve advanced mathematical problems using sympy.

    This tool uses Python's sympy library for symbolic mathematics:
    equation solving, calculus, linear algebra, simplification, etc.
    The agent writes a sympy code snippet that is executed in this tool's
    own sandbox.

    Available imports: sympy (as sp), math, fractions, decimal.
    """

    name = "solve_math"
    description = (
        "Solve a mathematical problem using sympy (symbolic mathematics).  "
        "Write a short Python code snippet that uses sympy (available as "
        "'sp') to compute the answer and PRINT it.  "
        "Examples:\n"
        "  - 'sp.solve(sp.Eq(x**2 + 2*x + 1, 0), x)'\n"
        "  - 'sp.integrate(sp.sin(x), x)'\n"
        "  - 'sp.limit(sp.sin(x)/x, x, 0)'\n"
        "  - 'sp.Matrix([[1,2],[3,4]]).eigenvals()'\n"
        "The *problem* description is for context; the actual computation "
        "happens inside *code*."
    )
    inputs = {
        "problem": {
            "type": "string",
            "description": "Natural-language description of the mathematical problem.",
        },
        "code": {
            "type": "string",
            "description": (
                "Python/sympy code that solves the problem and prints the answer. "
                "sympy is pre-imported as 'sp'."
            ),
        },
    }
    output_type = "string"

    def __init__(self, timeout_seconds: int = MATH_TOOL_TIMEOUT_SECONDS, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.timeout_seconds = timeout_seconds

    def forward(self, problem: str, code: str) -> str:
        try:
            import sympy  # noqa: F401  (fail fast if sympy is missing)
        except ImportError:
            return "Error: sympy is not installed in this environment. Run: pip install sympy"

        # Run the actual computation in an isolated *subprocess* (spawn, not
        # fork) with a hard wall-clock timeout. This guarantees the call can
        # be killed even if it gets stuck holding the GIL inside sympy/CPython
        # C-level big-integer arithmetic, which a thread-based timeout cannot
        # preempt. See MATH_TOOL_TIMEOUT_SECONDS docstring for details.
        ctx = mp.get_context("spawn")
        result_queue: mp.Queue = ctx.Queue()
        proc = ctx.Process(target=_math_worker, args=(code, problem, result_queue), daemon=True)
        proc.start()
        proc.join(self.timeout_seconds)

        if proc.is_alive():
            proc.kill()
            proc.join(5)
            return (
                f"Error: computation exceeded the {self.timeout_seconds}s time limit and was "
                "terminated. Try a faster approach (e.g. numeric solving with sp.nsolve() "
                "instead of symbolic sp.solve(), or simplify the equation first)."
            )

        try:
            status, payload = result_queue.get_nowait()
        except Exception:
            exit_code = proc.exitcode
            return (
                f"Error: math subprocess exited unexpectedly (exit code {exit_code}) "
                "without returning a result."
            )

        if status == "error":
            return f"Math execution error: {payload}"
        return payload


def get_custom_tools() -> list[Tool]:
    """Return the complete list of custom tools defined in this module.

    These tools complement (and go beyond) smolagents.default_tools.
    Use them when constructing a CodeAgent that needs local file system
    access, code editing, grep searching, or advanced math capabilities.

    Returns:
        List of instantiated Tools
    """
    return [
        # Comment or Uncomment the tools you want to include in your agent ...
        # ReadFileTool(),
        # WriteFileTool(),
        # EditFileTool(),
        # ListDirectoryTool(),
        # SearchFilesTool(),
        # GrepFilesTool(),
        # FileInfoTool(),
        MathTool(),
    ]


class RegionDuckDuckGoSearchTool(DuckDuckGoSearchTool):
    """DuckDuckGo search tool with configurable region and language.

    Extends DuckDuckGoSearchTool to allow searching with a specific
    region code (e.g. "pt-br" for Brazilian Portuguese, "pt-pt" for
    European Portuguese).  This biases search results toward content
    relevant to that region.  Output strings (header and error messages)
    are localized according to `language`.

    Args:
        region:      DuckDuckGo region code (default `"wt-wt"` = worldwide).
        language:    Language code for output strings. `"pt"` or `"en"`
                     (default `"pt"`).
        max_results: Maximum number of search results to return
                     (default `15`).  DuckDuckGo's free API caps around
                     ~20 per query.
        rate_limit:  Maximum queries per second (default `1.0`, i.e. one
                     query per second).
        **kwargs:    Forwarded to DuckDuckGoSearchTool.
    """

    name = "web_search"

    def __init__(
        self,
        region: str = "wt-wt",
        language: str = "pt",
        max_results: int = 15,
        rate_limit: float | None = 1.0,
        **kwargs,
    ):
        super().__init__(max_results=max_results, rate_limit=rate_limit, **kwargs)
        self.region = region
        self.language = language

    def forward(self, query: str) -> str:
        self._enforce_rate_limit()
        results = self.ddgs.text(query, region=self.region, max_results=self.max_results)
        is_pt = self.language.startswith("pt")
        if len(results) == 0:
            raise Exception(
                "Nenhum resultado encontrado! Tente uma consulta menos restritiva/mais curta, "
                "ou use wikipedia_search com o título exato do artigo."
                if is_pt
                else "No results found! Try a less restrictive/shorter query, "
                "or use wikipedia_search with the exact article title."
            )
        postprocessed_results = [
            f"[{result['title']}]({result['href']})\n{result['body']}" for result in results
        ]
        header = "## Resultados da Pesquisa" if is_pt else "## Search Results"
        return header + "\n\n" + "\n\n".join(postprocessed_results)


class RegionGoogleSearchTool(GoogleSearchTool):
    """Google search tool with configurable region/language parameters.

    Extends :class:`GoogleSearchTool` to bias results toward a specific
    country (`gl`), interface language (`hl`) and (for the SerpAPI
    provider) Google domain (e.g. `"google.com.br"`).  The stock
    smolagents tool hardcodes `google_domain="google.com"` and exposes no
    region control.

    Requires an API key in the environment: `SERPAPI_API_KEY` (provider
    `"serpapi"`) or `SERPER_API_KEY` (provider `"serper"`).

    Args:
        provider:      Search API backend. `"serpapi"` or `"serper"`
                       (default `"serpapi"`).
        region:        Country/geolocation code for the `gl` parameter
                       (e.g. `"br"`, `"us"`).
        language:      Interface language for the `hl` parameter
                       (e.g. `"pt"`, `"en"`).
        google_domain: Google domain for the SerpAPI provider (e.g.
                       `"google.com.br"`).  Ignored by `"serper"`.
    """

    name = "web_search"

    def __init__(
        self,
        provider: str = "serpapi",
        region: str = "br",
        language: str = "pt",
        google_domain: str = "google.com.br",
    ):
        super().__init__(provider=provider)
        self.region = region
        self.language = language
        self.google_domain = google_domain

    def forward(self, query: str, filter_year: int | None = None) -> str:
        import requests

        if self.provider == "serpapi":
            params = {
                "q": query,
                "api_key": self.api_key,
                "engine": "google",
                "google_domain": self.google_domain,
                "gl": self.region,
                "hl": self.language,
            }
            base_url = "https://serpapi.com/search.json"
        else:
            params = {
                "q": query,
                "api_key": self.api_key,
                "gl": self.region,
                "hl": self.language,
            }
            base_url = "https://google.serper.dev/search"
        if filter_year is not None:
            params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            results = response.json()
        else:
            raise ValueError(response.json())

        is_pt = self.language.startswith("pt")
        header = "## Resultados da Pesquisa" if is_pt else "## Search Results"

        if self.organic_key not in results:
            if filter_year is not None:
                if is_pt:
                    raise Exception(
                        f"Nenhum resultado encontrado para a consulta: '{query}' com filtro de "
                        f"ano={filter_year}. Use uma consulta menos restritiva ou não filtre por ano."
                    )
                raise Exception(
                    f"No results found for query: '{query}' with filtering on year={filter_year}. "
                    "Use a less restrictive query or do not filter on year."
                )
            if is_pt:
                raise Exception(
                    f"Nenhum resultado encontrado para a consulta: '{query}'. "
                    "Use uma consulta menos restritiva."
                )
            raise Exception(f"No results found for query: '{query}'. Use a less restrictive query.")
        if len(results[self.organic_key]) == 0:
            if filter_year is not None:
                year_filter_message = (
                    f" com filtro de ano={filter_year}"
                    if is_pt
                    else f" with filter year={filter_year}"
                )
            else:
                year_filter_message = ""
            if is_pt:
                return (
                    f"Nenhum resultado encontrado para '{query}'{year_filter_message}. "
                    "Tente com uma consulta mais geral, ou remova o filtro de ano."
                )
            return (
                f"No results found for '{query}'{year_filter_message}. "
                "Try with a more general query, or remove the year filter."
            )

        date_label = "Data de publicação: " if is_pt else "Date published: "
        source_label = "Fonte: " if is_pt else "Source: "

        web_snippets = []
        for idx, page in enumerate(results[self.organic_key]):
            date_published = ""
            if "date" in page:
                date_published = "\n" + date_label + page["date"]

            source = ""
            if "source" in page:
                source = "\n" + source_label + page["source"]

            snippet = ""
            if "snippet" in page:
                snippet = "\n" + page["snippet"]

            redacted_version = (
                f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
            )
            web_snippets.append(redacted_version)

        return header + "\n\n" + "\n\n".join(web_snippets)


class RegionWikipediaSearchTool(WikipediaSearchTool):
    """Wikipedia search tool with exact-title guidance and a title fallback.

    Extends :class:`WikipediaSearchTool` in two ways:

    1. The tool description tells the agent that `query` is matched against
       Wikipedia article *titles* (not free text), so it should pass a short,
       title-like query (e.g. ``"Ayrton Senna"``) instead of a sentence or
       question.
    2. When the exact title does not exist, instead of dead-ending with
       "No Wikipedia page found", the tool runs a Wikipedia opensearch and
       returns the closest matching titles so the agent can retry with one
       of them.

    Args:
        user_agent:     Custom user-agent string (required by Wikipedia's API
                        policy).
        language:       Wikipedia language code. `"pt"` or `"en"`.
        content_type:   `"summary"` or `"text"` (default `"text"`).
        extract_format: `"WIKI"` or `"HTML"` (default `"WIKI"`).
    """

    name = "wikipedia_search"
    description = (
        "Searches Wikipedia and returns the full text (or summary) of the requested "
        "article along with its URL. IMPORTANT: the query is matched against Wikipedia "
        "article titles, so pass a short title-like query (e.g. 'Ayrton Senna' or "
        "'Great Pyramid of Giza'), not a sentence or a question."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "The Wikipedia article title to fetch. Use a short, title-like phrase; "
                "if the exact title is not found, close title matches are suggested automatically."
            ),
        }
    }

    def __init__(
        self,
        user_agent: str = "Smolagents (myemail@example.com)",
        language: str = "en",
        content_type: str = "text",
        extract_format: str = "WIKI",
    ):
        super().__init__(
            user_agent=user_agent,
            language=language,
            content_type=content_type,
            extract_format=extract_format,
        )

    def forward(self, query: str) -> str:
        try:
            page = self.wiki.page(query)

            if not page.exists():
                return self._no_page_found(query)

            title = page.title
            url = page.fullurl

            if self.content_type == "summary":
                text = page.summary
            elif self.content_type == "text":
                text = page.text
            else:
                return "⚠️ Invalid `content_type`. Use either 'summary' or 'text'."

            return (
                f"✅ **Wikipedia Page:** {title}\n\n**Content:** {text}\n\n🔗 **Read more:** {url}"
            )

        except Exception as e:
            return f"Error fetching Wikipedia summary: {str(e)}"

    def _suggest_titles(self, query: str, limit: int = 8) -> list[str]:
        """Return up to `limit` closest Wikipedia article titles for `query`.

        Uses the MediaWiki opensearch API, which performs a fuzzy *title*
        search, so descriptive queries such as "Ayrton Senna Birthday" still
        surface the exact title "Ayrton Senna".
        """
        import requests

        url = f"https://{self.language}.wikipedia.org/w/api.php"
        params = {
            "action": "opensearch",
            "search": query,
            "limit": limit,
            "namespace": 0,
            "format": "json",
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception:
            return []
        # MediaWiki opensearch returns [query, [titles], [descriptions], [urls]].
        if not isinstance(data, list) or len(data) < 2 or not isinstance(data[1], list):
            return []
        return [title for title in data[1] if title]

    def _no_page_found(self, query: str) -> str:
        """Build the localized not-found message, with title suggestions when available."""
        is_pt = self.language.startswith("pt")
        suggestions = self._suggest_titles(query)
        if suggestions:
            joined = ", ".join(repr(title) for title in suggestions)
            if is_pt:
                return (
                    f"Nenhuma página da Wikipédia encontrada para '{query}'. "
                    f"Tente um destes títulos exatos: {joined}."
                )
            return (
                f"No Wikipedia page found for '{query}'. "
                f"Try one of these exact article titles: {joined}."
            )
        if is_pt:
            return (
                f"Nenhuma página da Wikipédia encontrada para '{query}'. "
                "Use um título de artigo exato e curto (ex.: 'Ayrton Senna'), não uma frase."
            )
        return (
            f"No Wikipedia page found for '{query}'. "
            "Use a short, exact article title (e.g. 'Ayrton Senna'), not a sentence."
        )
