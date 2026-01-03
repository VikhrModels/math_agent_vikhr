import sys
import os
import re
import subprocess
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
import requests
from opentelemetry import trace
import brotli
import json
from configs.config_loader import MINIF2F_DIR, LEAN_TIMEOUT, TMP_DIR, LOG_DIR, SEARCH_CONFIG
import tempfile
import shutil

# Add the parent directory to Python path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))

from smolagents import Tool
from configs.config_loader import MINIF2F_DIR, LEAN_TIMEOUT

# Set up logging for the tool
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "tools.log"),
        logging.StreamHandler()
    ],
    force=True  # Force reconfiguration even if already configured
)
logger = logging.getLogger(__name__)
tracer = trace.get_tracer("math_prover.tools")

# ---------------------------------------------------------------------------
# LeanVerifier helper (adapted from verify_task.py)
# ---------------------------------------------------------------------------

class LeanVerifier:
    """Light-weight wrapper around the Lean 4 compiler (via Lake) that verifies on-the-fly code snippets.

    The verifier writes the provided Lean code to a temporary file inside the
    `miniF2F-lean4` project, calls `lake env lean` on that file, captures the
    compiler output, and returns a structured result the agent can consume.

    Parameters
    ----------
    lake_binary : str | None, optional
        Path to the `lake` executable.  If *None*, we attempt to locate it via
        the `LAKE_BINARY` environment variable and finally fall back to
        ``shutil.which('lake')``.
    project_root : pathlib.Path | None, optional
        Absolute path to the Lake project root that contains either
        `lakefile.lean` *or* `lake-manifest.json`.  Defaults to
        ``<repo-root>/miniF2F-lean4``.
    timeout : int, optional
        Seconds to wait for Lean to finish before aborting the process.
    """

    def __init__(self,
                 lake_binary: str | None = None,
                 project_root: Path | None = None,
                 timeout: int = LEAN_TIMEOUT) -> None:
        self.timeout = timeout

        # --- Determine Lake binary ------------------------------------------------
        env_binary = os.getenv("LAKE_BINARY")
        self.lake_binary = lake_binary or env_binary or shutil.which("lake")
        if not self.lake_binary or not Path(self.lake_binary).exists():
            # Fall-back to common elan path if not found
            default_path = Path.home() / ".elan" / "bin" / "lake"
            if default_path.exists():
                self.lake_binary = str(default_path)
            else:
                raise FileNotFoundError(
                    "Could not locate `lake` executable. Set LAKE_BINARY env or install Lean tooling.")

        # --- Determine project root ----------------------------------------------
        repo_root = Path(__file__).resolve().parent.parent
        default_root = repo_root / "miniF2F-lean4"
        self.project_root = Path(project_root) if project_root else default_root

        if not ((self.project_root / "lakefile.lean").exists() or (self.project_root / "lake-manifest.json").exists()):
            raise FileNotFoundError(
                f"Expected Lake project root at {self.project_root} (no lakefile.lean or lake-manifest.json found)" )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def verify_lean_code(self, lean_code: str, label: str | None = None) -> Dict[str, Any]:
        """Compile *lean_code* inside the project and return a structured result.

        Parameters
        ----------
        lean_code : str
            Complete Lean code (will be prepended with the required import if
            missing).
        label : str | None
            Optional label used only for logging; ignored otherwise.
        """
        label_str = label or "anonymous"
        logger.info(f"[LeanVerifier] Verifying Lean snippet '{label_str}' (length={len(lean_code)} chars)")
        with tracer.start_as_current_span(
            "lean_verify",
            attributes={
                "lean.label": label_str,
                "lean.length": len(lean_code),
                "lean.project_root": str(self.project_root),
            },
        ):
            # Ensure required import is present
            if "MiniF2F.Minif2fImport" not in lean_code:
                lean_code = "import MiniF2F.Minif2fImport\n\n" + lean_code.lstrip()

            # Write to temporary file in project root
            tmp_file = tempfile.NamedTemporaryFile("w", suffix=".lean", dir=self.project_root, delete=False, encoding="utf-8")
            try:
                tmp_file.write(lean_code)
                tmp_file.flush()
                tmp_path = Path(tmp_file.name)
            finally:
                tmp_file.close()

            # Call Lean compiler via Lake
            cmd = [self.lake_binary, "env", "lean", str(tmp_path)]
            logger.info(f"[LeanVerifier] Running command: {' '.join(cmd)}  (cwd={self.project_root})")
            try:
                with tracer.start_as_current_span(
                    "lean_compile",
                    attributes={
                        "cmd": " ".join(cmd),
                        "timeout": self.timeout,
                    },
                ):
                    proc = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                logger.error("[LeanVerifier] Lean compilation timed-out")
                tmp_path.unlink(missing_ok=True)
                return {"success": False, "output": "Lean compilation timed-out", "exit_code": -1}

            stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
            output_combined = (stdout or "") + (stderr or "")

            # Clean compiler paths to reduce noise
            clean_output = self._clean_output(output_combined, str(tmp_path))

            # Remove temporary file
            tmp_path.unlink(missing_ok=True)

            success = exit_code == 0
            logger.info(f"[LeanVerifier] Compilation finished – success={success}, exit_code={exit_code}")

            return {
                "success": success,
                "exit_code": exit_code,
                "output": clean_output,
            }

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    @staticmethod
    def _clean_output(raw: str, tmp_path: str) -> str:
        """Strip occurrences of *tmp_path* from Lean diagnostics for readability."""
        cleaned_lines: list[str] = []
        for line in raw.splitlines():
            if tmp_path in line:
                # Keep only the part after the filename so we still have line/col info
                parts = line.split(tmp_path)
                cleaned_lines.append(parts[1].lstrip(":") if len(parts) > 1 else line)
            else:
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

# ---------------------------------------------------------------------------
# Instantiate global Lean verifier (single process reuse)
# ---------------------------------------------------------------------------
lean_verifier = LeanVerifier()

class VerifyLeanProof(Tool):
    """
    A tool for verifying Lean 4 mathematical proofs by compiling them within the MiniF2F project environment
    using Lake. It takes a complete Lean theorem, injects the required import, compiles it with Lake through
    the LeanVerifier helper, and returns whether compilation succeeded together with compiler output.
    """
    
    name = "verify_lean_proof"
    description = """
    Verifies a Lean 4 mathematical proof by compiling it within the miniF2F project environment.
    
    This tool creates a temporary Lean file with the provided theorem and attempts to compile it
    using the Lean 4 compiler. It ensures all necessary dependencies are available by working
    within the miniF2F project structure.
    
    The theorem should be a complete Lean statement that may contain a `sorry` placeholder.
    The tool will attempt to compile the theorem and return whether it was successful.
    
    Args:
        theorem_statement: The complete Lean theorem statement to verify. This should be a full
                          theorem definition that may contain `:= sorry` or `:= begin sorry end`.
    
    Returns:
        A dictionary with:
        - success: boolean indicating whether the Lean compilation was successful
        - output: string containing the full compiler output (stdout + stderr)
    """
    
    inputs = {
        "theorem_statement": {
            "type": "string",
            "description": "The complete Lean theorem statement to verify. This should be a full theorem definition that may contain `:= sorry` or `:= begin sorry end`."
        }
    }
    
    output_type = "object"
    
    def forward(self, theorem_statement: str) -> Dict[str, Any]:
        """
        Verifies a Lean 4 proof by compiling it with Lake.
        
        Args:
            theorem_statement: Complete Lean theorem statement to verify
        Returns:
            Dictionary with keys 'success', 'output', and optionally 'error'.
        """
        logger.info("Starting Lean 4 proof verification")
        logger.info(f"Theorem statement length: {len(theorem_statement)} characters")
        
        # Validate input
        if not theorem_statement or not theorem_statement.strip():
            logger.warning("Empty theorem statement provided")
            return {
                "success": False,
                "output": "Error: Empty theorem statement provided"
            }
        
        # Check if the proof contains 'sorry' - this means it's incomplete
        if "sorry" in theorem_statement.lower():
            # Check if sorry is the only content between begin and end
            begin_end_match = re.search(r'begin\s*(.*?)\s*end', theorem_statement, re.DOTALL | re.IGNORECASE)
            if begin_end_match:
                proof_content = begin_end_match.group(1).strip().lower()
                # If the only content is 'sorry' (with possible whitespace), then it's incomplete
                if proof_content == 'sorry':
                    logger.warning("Proof contains only 'sorry' - marking as incomplete")
                    return {
                        "success": False,
                        "output": "Error: Proof contains only 'sorry' and is incomplete"
                    }
                else:
                    logger.info("Proof contains 'sorry' but also other tactics - proceeding with verification")
            else:
                # If we can't find begin/end structure, check if sorry is the only content
                if theorem_statement.lower().strip() == 'sorry':
                    logger.warning("Theorem contains only 'sorry' - marking as incomplete")
                    return {
                        "success": False,
                        "output": "Error: Theorem contains only 'sorry' and is incomplete"
                    }
        
        # === Lean 4 verification path ===
        stmt = theorem_statement.lstrip()
        
        # Clean up the theorem statement - remove any problematic characters or syntax
        cleaned_statement = theorem_statement.strip()
        
        # Check for basic syntax issues
        if cleaned_statement.count(":=") == 0 and "theorem" in cleaned_statement.lower():
            # Add missing := if theorem doesn't have it
            if "sorry" in cleaned_statement.lower():
                cleaned_statement = cleaned_statement.replace("sorry", ":= sorry")
            else:
                cleaned_statement = cleaned_statement.replace("theorem", "theorem test_theorem")
                if ":=" not in cleaned_statement:
                    cleaned_statement += " := sorry"
        
        has_any_import = stmt.startswith("import ") or "\nimport " in stmt
        needs_minif2f = "MiniF2F.Minif2fImport" not in cleaned_statement
        if has_any_import:
            # Preserve user imports; ensure MiniF2F import is present once
            if needs_minif2f:
                lean_code = "import MiniF2F.Minif2fImport\n" + cleaned_statement
            else:
                lean_code = cleaned_statement
        else:
            # No imports provided; add required import
            lean_code = "import MiniF2F.Minif2fImport\n\n" + cleaned_statement

        theorem_name = f"TempProof_{os.getpid()}_{time.time_ns()}"

        try:
            result = lean_verifier.verify_lean_code(lean_code, theorem_name)
            return result
        except Exception as e:
            logger.error(f"Unexpected error during Lean 4 verification: {e}")
            return {
                "success": False,
                "output": f"Error: {e}"
            }
    
    def _clean_output(self, output: str, temp_file_name: str) -> str:
        """
        Clean up the Lean output by removing temporary file paths and formatting errors properly.
        
        Args:
            output: Raw output from Lean compiler
            temp_file_name: Name of the temporary file to remove from output
            
        Returns:
            Cleaned output string
        """
        # Remove the temporary file path from all lines
        lines = output.splitlines()
        cleaned_lines = []
        
        for line in lines:
            # Remove the full path to the temporary file, keeping only line:column info
            if temp_file_name in line:
                # Extract line:column:message format
                parts = line.split(temp_file_name)
                if len(parts) >= 2:
                    # Keep everything after the filename (line:column:message)
                    cleaned_line = parts[1].lstrip(':')
                    cleaned_lines.append(cleaned_line)
                else:
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

class MoogleSemanticSearch(Tool):
    """
    A tool for semantic search of theorems, lemmas, structures, and more via moogle.ai.

    This tool sends a query to moogle.ai's semantic search API and returns a filtered JSON response
    containing relevant mathematical declarations (theorems, definitions, etc.).

    Args:
        query: The search query string (e.g., theorem name, concept, or keywords).

    Returns:
        A dictionary with a 'data' field containing a list of relevant declarations. Each declaration is a dict with the following keys:
            - declarationName: str (e.g., 'Real.tan_pi_div_six')
            - declarationCode: str (Lean code for the declaration)
            - declarationDocstring: str (docstring or description, may be empty)
            - declarationType: str (e.g., 'theorem', 'definition', etc.)
            - sourceCodeUrl: str (URL to the source code in mathlib)
            - mathlibPath: str (path to the file in mathlib)
        Example output:
        {
            'data': [
                {
                    'declarationName': 'Real.tan_pi_div_six',
                    'declarationCode': 'theorem tan_pi_div_six : tan (π / 6) = 1 / sqrt 3 := by ...',
                    'declarationDocstring': '/-- The tangent of π/6 is 1/√3. -/',
                    'declarationType': 'theorem',
                    'sourceCodeUrl': 'https://github.com/leanprover-community/mathlib4/...',
                    'mathlibPath': 'Mathlib/Analysis/SpecialFunctions/Trigonometric/Basic.lean'
                },
                ...
            ]
        }
    """
    name = "moogle_semantic_search"
    description = (
        "Performs a semantic search for theorems, lemmas, structures, etc. using moogle.ai's API. "
        "Returns a dictionary with a 'data' field containing a list of relevant declarations. "
        "Each declaration is a dict with: declarationName, declarationCode, declarationDocstring, declarationType, sourceCodeUrl, mathlibPath."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query string (e.g., theorem name, concept, or keywords)."
        }
    }
    output_type = "object"

    def forward(self, query: str) -> dict:
        """
        Performs a semantic search request to moogle.ai and returns filtered results as JSON.
        Logs all steps and errors. Handles brotli decoding. Does not write to disk.
        """
        logger.info(f"[MoogleSemanticSearch] Starting semantic search for query: '{query}'")
        url = 'https://www.moogle.ai/api/search'
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:139.0) Gecko/20100101 Firefox/139.0',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Referer': 'https://www.moogle.ai/search/raw?q=cursor',
            'Content-Type': 'application/json',
            'Origin': 'https://www.moogle.ai',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'DNT': '1',
            'Sec-GPC': '1',
            'Priority': 'u=4',
        }
        data = [{"isFind": False, "contents": query}]
        try:
            logger.info("[MoogleSemanticSearch] Sending POST request to moogle.ai API...")
            with tracer.start_as_current_span(
                "semantic_search",
                attributes={
                    "provider": "moogle.ai",
                    "query.length": len(query),
                },
            ):
                resp = requests.post(url, headers=headers, json=data)
            resp.raise_for_status()
            encoding = resp.headers.get('content-encoding', '').lower()
            raw = resp.content
            text = None
            if 'br' in encoding:
                try:
                    raw = brotli.decompress(raw)
                    text = raw.decode('utf-8')
                    logger.info("[MoogleSemanticSearch] Brotli decoding successful.")
                except Exception as e:
                    logger.error(f"[MoogleSemanticSearch] Brotli decode error: {e}")
                    logger.debug(f"[MoogleSemanticSearch] Raw response bytes: {raw}")
                    try:
                        text = resp.text
                    except Exception as e2:
                        logger.error(f"[MoogleSemanticSearch] Error getting resp.text: {e2}")
                        text = None
            else:
                text = resp.text
            if text is not None:
                try:
                    resp_json = json.loads(text)
                    allowed = {"declarationName", "declarationCode", "declarationDocstring", "declarationType", "sourceCodeUrl", "mathlibPath"}
                    if 'data' in resp_json and isinstance(resp_json['data'], list):
                        for i, item in enumerate(resp_json['data']):
                            if isinstance(item, dict):
                                resp_json['data'][i] = {k: v for k, v in item.items() if k in allowed}
                    logger.info(f"[MoogleSemanticSearch] Successfully parsed and filtered response. Returned {len(resp_json.get('data', []))} items.")
                    return resp_json
                except Exception as e:
                    logger.error(f"[MoogleSemanticSearch] JSON parse error: {e}")
                    logger.debug(f"[MoogleSemanticSearch] Raw response text: {text}")
                    return {"error": f"JSON parse error: {e}", "raw": text}
            else:
                logger.error("[MoogleSemanticSearch] No text response to parse.")
                return {"error": "No text response to parse."}
        except Exception as e:
            logger.error(f"[MoogleSemanticSearch] Request error: {e}")
            return {"error": str(e)}


class BatchMoogleSemanticSearch(Tool):
    """
    Batched semantic search over moogle.ai.

    Accepts multiple query strings at once and returns grouped, size-limited results
    to minimize the number of search tool invocations (token- and step-efficient).

    Inputs:
      - theorem_key: str — stable key for this theorem (e.g., theorem name), used for budget tracking
      - queries: list[str]  — a small batch of rephrasings/abstractions of the same goal
      - max_per_query: int (optional, default 10) — cap for results per query to reduce noise
      - max_batches_global: int (optional, default 4) — cap on number of times this tool can be called per theorem_key

    Output:
      - { "results": { query: [ items... ] }, "total_items": int }
    """
    name = "batch_semantic_search"
    description = (
        "Runs a single batched semantic search request against moogle.ai for multiple queries. "
        "Returns grouped results per input query. Always prefer this over multiple single-query calls."
    )
    inputs = {
        "theorem_key": {
            "type": "string",
            "description": "Stable identifier for this theorem (budgeted per key)."
        },
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Batch of related search queries (rephrasings, equivalent forms)."
        },
        "max_per_query": {
            "type": "integer",
            "description": "Maximum number of items to keep per query (default 10).",
            "default": 10,
            "nullable": True
        },
        "max_batches_global": {
            "type": "integer",
            "description": "Maximum allowed calls per theorem_key (default 4).",
            "default": 4,
            "nullable": True
        }
    }
    output_type = "object"

    # Global in-memory trackers
    _budget_usage: Dict[str, int] = {}
    _cache: Dict[str, list] = {}
    _seen_queries: Dict[str, List[str]] = {}  # theorem_key -> list of normalized queries
    _seen_declarations: Dict[str, Set[Tuple[str, str]]] = {}  # theorem_key -> {(name, path)}
    _plateau_streak: Dict[str, int] = {}  # theorem_key -> consecutive low-growth count

    def _single_search(self, query: str) -> dict:
        # Reuse the single-search logic directly to avoid code duplication
        # (copy of MoogleSemanticSearch.forward body with minor adjustments)
        logger.info(f"[BatchMoogleSemanticSearch] Single semantic search for: '{query}'")
        url = 'https://www.moogle.ai/api/search'
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:139.0) Gecko/20100101 Firefox/139.0',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'Referer': 'https://www.moogle.ai/search/raw?q=cursor',
            'Content-Type': 'application/json',
            'Origin': 'https://www.moogle.ai',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'DNT': '1',
            'Sec-GPC': '1',
            'Priority': 'u=4',
        }
        data = [{"isFind": False, "contents": query}]
        try:
            logger.info("[BatchMoogleSemanticSearch] POST moogle.ai API")
            with tracer.start_as_current_span(
                "semantic_search.batch",
                attributes={
                    "provider": "moogle.ai",
                    "query.length": len(query),
                },
            ):
                resp = requests.post(url, headers=headers, json=data)
            resp.raise_for_status()
            encoding = resp.headers.get('content-encoding', '').lower()
            raw = resp.content
            text = None
            if 'br' in encoding:
                try:
                    raw = brotli.decompress(raw)
                    text = raw.decode('utf-8')
                    logger.info("[BatchMoogleSemanticSearch] Brotli decoding successful.")
                except Exception as e:
                    logger.error(f"[BatchMoogleSemanticSearch] Brotli decode error: {e}")
                    try:
                        text = resp.text
                    except Exception as e2:
                        logger.error(f"[BatchMoogleSemanticSearch] Error getting resp.text: {e2}")
                        text = None
            else:
                text = resp.text
            if text is not None:
                try:
                    resp_json = json.loads(text)
                    allowed = {"declarationName", "declarationCode", "declarationDocstring", "declarationType", "sourceCodeUrl", "mathlibPath"}
                    if 'data' in resp_json and isinstance(resp_json['data'], list):
                        for i, item in enumerate(resp_json['data']):
                            if isinstance(item, dict):
                                resp_json['data'][i] = {k: v for k, v in item.items() if k in allowed}
                    logger.info(f"[BatchMoogleSemanticSearch] Parsed response. Items={len(resp_json.get('data', []))}")
                    return resp_json
                except Exception as e:
                    logger.error(f"[BatchMoogleSemanticSearch] JSON parse error: {e}")
                    return {"error": f"JSON parse error: {e}", "raw": text}
            else:
                logger.error("[BatchMoogleSemanticSearch] No text response to parse.")
                return {"error": "No text response to parse."}
        except Exception as e:
            logger.error(f"[BatchMoogleSemanticSearch] Request error: {e}")
            return {"error": str(e)}

    @staticmethod
    def _normalize_query(q: str) -> str:
        s = (q or "").lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def _jaccard_similarity(a: str, b: str, n: int = 3) -> float:
        def ngrams(s: str, n: int) -> Set[str]:
            if len(s) < n:
                return {s}
            return {s[i:i+n] for i in range(len(s)-n+1)}
        A = ngrams(a, n)
        B = ngrams(b, n)
        if not A and not B:
            return 1.0
        inter = len(A & B)
        union = len(A | B) or 1
        return inter / union

    def _is_novel(self, theorem_key: str, q_norm: str) -> bool:
        min_novelty = float(SEARCH_CONFIG['novelty'].get('min_query_novelty', 0.15))
        seen = self._seen_queries.get(theorem_key, [])
        if not seen:
            return True
        # novelty ~ 1 - max_similarity
        max_sim = 0.0
        for prev in seen:
            sim = self._jaccard_similarity(prev, q_norm, n=3)
            if sim > max_sim:
                max_sim = sim
            if max_sim >= 1 - min_novelty:
                break
        novelty = 1.0 - max_sim
        return novelty >= min_novelty

    def _update_seen_queries(self, theorem_key: str, q_norms: List[str]) -> None:
        cur = self._seen_queries.get(theorem_key, [])
        cur.extend(q_norms)
        # keep last 100 for memory bound
        self._seen_queries[theorem_key] = cur[-100:]

    def _collect_decls(self, items: List[dict]) -> Set[Tuple[str, str]]:
        decls: Set[Tuple[str, str]] = set()
        for it in items or []:
            name = str(it.get('declarationName', '') or '')
            path = str(it.get('mathlibPath', '') or '')
            if name or path:
                decls.add((name, path))
        return decls

    def _apply_rerank_and_trim(self, theorem_key: str, q: str, items: List[dict], batch_index: int, max_per_query: int) -> List[dict]:
        if not isinstance(items, list) or not items:
            return []
        schedule = SEARCH_CONFIG['rerank'].get('max_per_query_schedule', [10, 6, 3])
        # pick schedule value by batch index if available
        scheduled_cap = schedule[batch_index] if batch_index < len(schedule) else schedule[-1]
        cap = min(max_per_query or scheduled_cap, scheduled_cap)
        enable_mmr = bool(SEARCH_CONFIG['rerank'].get('enable_mmr', True))
        if not enable_mmr:
            return items[:cap]
        # Simple diversity heuristic: prefer unique mathlibPath then name
        seen_paths: Set[str] = set()
        seen_names: Set[str] = set()
        diversified: List[dict] = []
        for it in items:
            path = it.get('mathlibPath') or ''
            name = it.get('declarationName') or ''
            key = (path, name)
            if path in seen_paths and name in seen_names:
                continue
            seen_paths.add(path)
            seen_names.add(name)
            diversified.append(it)
            if len(diversified) >= cap:
                break
        if len(diversified) < cap:
            # top up with remaining items
            for it in items:
                if it not in diversified:
                    diversified.append(it)
                    if len(diversified) >= cap:
                        break
        return diversified[:cap]

    def forward(self, theorem_key: str, queries: List[str], max_per_query: int = 10, max_batches_global: int = 4) -> dict:
        logger.info(f"[BatchMoogleSemanticSearch] Starting batched search. Theorem={theorem_key}, Queries={len(queries)}, max_per_query={max_per_query}, max_batches_global={max_batches_global}")
        if not isinstance(queries, list) or not queries or not isinstance(theorem_key, str) or not theorem_key:
            return {"results": {}, "total_items": 0}

        # Centralized caps from config
        cfg_cap = int(SEARCH_CONFIG['caps'].get('max_batches_global_default', 2))
        # Hard safety cap of 5 regardless of what the caller requests
        effective_cap = min(max_batches_global or cfg_cap, cfg_cap, 5)
        used = self._budget_usage.get(theorem_key, 0)
        if used >= effective_cap:
            logger.warning(f"[BatchMoogleSemanticSearch] Budget exhausted for theorem_key='{theorem_key}' (used={used} ≥ cap={max_batches_global})")
            out = {"results": {}, "total_items": 0, "warning": "search_budget_exhausted"}
            if SEARCH_CONFIG.get('finalize_hint', {}).get('enabled', True):
                out["hint"] = "finalize_next_step"
            return out

        grouped: Dict[str, list] = {}
        total_items = 0
        # Prepare novelty filtering
        norm_queries = [self._normalize_query(q) for q in queries]
        kept_indices: List[int] = []
        for idx, (q, qn) in enumerate(zip(queries, norm_queries)):
            if self._is_novel(theorem_key, qn):
                kept_indices.append(idx)
            else:
                logger.info(f"[BatchMoogleSemanticSearch] Skipping non-novel query: '{q}'")
                grouped[q] = []
        # If all queries are non-novel
        if not kept_indices:
            used_cap = used
            logger.info(f"[BatchMoogleSemanticSearch] All queries non-novel for theorem_key='{theorem_key}'")
            out = {"results": grouped, "total_items": 0, "warning": "not_novel"}
            if SEARCH_CONFIG.get('finalize_hint', {}).get('enabled', True):
                out["hint"] = "finalize_next_step"
            return out

        # Determine batch index for schedule
        batch_index = used  # 0-based

        for idx in kept_indices:
            q = queries[idx]
            qn = norm_queries[idx]
            # Cache lookup
            if q in self._cache:
                items = list(self._cache[q])
                logger.info(f"[BatchMoogleSemanticSearch] Cache hit for query '{q}' -> {len(items)} items")
            else:
                res = self._single_search(q)
                items = res.get('data', []) if isinstance(res, dict) else []
                # cache raw list (untrimmed) to allow different max_per_query later
                self._cache[q] = list(items)
            # Rerank and schedule-based trim
            # prefer config default if max_per_query is None
            mpq_default = int(SEARCH_CONFIG['caps'].get('max_per_query_default', 10))
            eff_max_per_query = max_per_query if max_per_query is not None else mpq_default
            items = self._apply_rerank_and_trim(theorem_key, q, items, batch_index, eff_max_per_query)
            grouped[q] = items
            total_items += len(items)

        # Plateau detection
        new_decls = set()
        for lst in grouped.values():
            new_decls |= self._collect_decls(lst)
        seen_decls = self._seen_declarations.get(theorem_key, set())
        added = new_decls - seen_decls
        growth_ratio = (len(added) / max(1, len(new_decls))) if new_decls else 0.0
        min_growth = float(SEARCH_CONFIG['plateau'].get('min_new_items_ratio', 0.05))
        patience = int(SEARCH_CONFIG['plateau'].get('patience', 1))
        streak = self._plateau_streak.get(theorem_key, 0)

        plateau_triggered = False
        if growth_ratio < min_growth:
            streak += 1
            self._plateau_streak[theorem_key] = streak
            if streak >= patience:
                plateau_triggered = True
        else:
            self._plateau_streak[theorem_key] = 0

        # Update seen state
        self._seen_declarations[theorem_key] = seen_decls | new_decls
        self._update_seen_queries(theorem_key, [norm_queries[i] for i in kept_indices])

        # Update budget
        self._budget_usage[theorem_key] = used + 1

        # Compose output
        out: Dict[str, Any] = {"results": grouped, "total_items": total_items}
        if plateau_triggered:
            out["warning"] = "plateau_reached"
        if SEARCH_CONFIG.get('finalize_hint', {}).get('enabled', True):
            # Give a hint to finalize if budget low, plateau, or low novelty
            if plateau_triggered or self._budget_usage[theorem_key] >= effective_cap:
                out["hint"] = "finalize_next_step"

        # Telemetry attributes
        try:
            with tracer.start_as_current_span(
                "search_metrics",
                attributes={
                    "search.budget_used": self._budget_usage[theorem_key],
                    "search.budget_cap": effective_cap,
                    "search.kept_queries": len(kept_indices),
                    "search.total_items": total_items,
                    "search.growth_ratio": growth_ratio,
                    "search.plateau_streak": self._plateau_streak.get(theorem_key, 0),
                },
            ):
                pass
        except Exception:
            pass

        logger.info(f"[BatchMoogleSemanticSearch] Finished batched search. Kept queries={len(kept_indices)} Total kept items={total_items}. Budget now used={self._budget_usage[theorem_key]}/{effective_cap}")
        return out

# Create an instance of the tool
verify_lean_proof = VerifyLeanProof()

# Register the tool instance
moogle_semantic_search = MoogleSemanticSearch()

# Batched search instance (preferred to reduce steps/tokens)
batch_semantic_search = BatchMoogleSemanticSearch()