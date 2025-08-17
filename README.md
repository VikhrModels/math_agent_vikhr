## Math Agent Vikhr

Agent-based and prompt-based Lean 4 theorem proving on the MiniF2F dataset. This repository bundles a ready-to-run Lean project (`miniF2F-lean4`) and multiple Python entrypoints to parse tasks, call LLMs, verify generated proofs with Lake, and checkpoint progress.

### Key features
- Multi-agent pipeline using `smolagents` to plan, search lemmas, generate Lean code, and verify with `lake`.
- Prompt-based benchmarks with either OpenRouter or OpenAI APIs.
- Robust Lean verification tooling with clear logs and timeouts.
- Checkpointing per stage and per run to resume work and analyze results.

### Table of contents
- Installation
  - Python dependencies
  - Lean 4 toolchain and building `miniF2F-lean4`
- Quickstart
- Scripts and CLI flags
  - agents/math_prover_agent.py (multi-agent benchmark)
  - benchmark_openrouter.py (prompt baseline via OpenRouter)
  - benchmark_openai.py (prompt baseline via OpenAI Responses API)
  - process_lean.py (build `valid.json` from Lean sources)
  - verify_task.py (compile/verify ad-hoc Lean code)
- Configuration
- Checkpoints, logs, and outputs
- Project structure
- Troubleshooting
- Attribution and licenses

## Installation

### Python (3.10+ recommended)
Create and activate a virtual environment in your preferred way, then install requirements:

```bash
pip install -r requirements.txt
```

Environment variables for providers (set at least one depending on which scripts you run):

```bash
# For OpenRouter-based scripts
export OPENROUTER_API_KEY=...   # required for agents/math_prover_agent.py and benchmark_openrouter.py

# For OpenAI-based scripts
export OPENAI_API_KEY=...       # required for benchmark_openai.py
```

Optional environment variables:

```bash
# Increase default HTTP timeout for LLM calls (seconds)
export LLM_REQUEST_TIMEOUT=180

# Select a non-default model globally (some scripts also accept --model)
export MATH_AGENT_MODEL="anthropic/claude-sonnet-4"

# Switch default provider for future extensibility (not used by all entrypoints)
export MATH_AGENT_PROVIDER=openrouter  # or openai

# Use a custom lake binary if not found on PATH
export LAKE_BINARY="$HOME/.elan/bin/lake"
```

### Lean 4 tooling and dataset
This repo vendors the `miniF2F-lean4` project under `miniF2F-lean4/`. You must have the Lean toolchain and `lake` to build and verify code.

1) Install elan (Lean toolchain manager):

```bash
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh
```

2) Build the included MiniF2F Lean project:

```bash
cd miniF2F-lean4
lake exe cache get
lake build
cd -
```

For more details, see the upstream documentation referenced in `miniF2F-lean4/README.md`.

## Quickstart

1) Generate or refresh the task file (`valid.json`) from Lean sources:

```bash
python process_lean.py
```

2) Run the multi-agent benchmark (OpenRouter provider):

```bash
export OPENROUTER_API_KEY=...
python agents/math_prover_agent.py \
  --subset_size 20 \
  --json_file valid.json \
  --model anthropic/claude-sonnet-4 \
  --concurrency 4 \
  --stages 2
```

3) List available checkpoints and resume from a previous run:

```bash
# List all available checkpoints
python agents/math_prover_agent.py --list_checkpoints

# Resume from a specific run (recommended for stage-based checkpoints)
python agents/math_prover_agent.py --resume_run 20250816-204214

# Resume from a specific stage within a run
python agents/math_prover_agent.py --resume_run 20250816-204214 --resume_stage 1
```

4) Or try the prompt-only baselines:

```bash
# OpenRouter baseline
export OPENROUTER_API_KEY=...
python benchmark_openrouter.py --subset_size 20 --json_file valid.json --model anthropic/claude-sonnet-4 --concurrency 4 --stages 1

# OpenAI Responses baseline
export OPENAI_API_KEY=...
python benchmark_openai.py --subset_size 20 --json_file valid.json --model gpt-4.1 --concurrency 4 --effort low --max_output_tokens 4096 --stages 1
```

Checkpoints and logs are written under `tmp/` and `log/` respectively (details below).

## Scripts and CLI flags

### agents/math_prover_agent.py
Multi-agent system powered by `smolagents`:
- Idea generator agent: searches for relevant lemmas via `moogle_semantic_search`, plans a strategy, and delegates code generation.
- Code generator agent: produces Lean 4 code and verifies it with the `verify_lean_proof` tool (calls Lean via Lake).

Flags:
- `--subset_size int` (default from config): Number of tasks to run. Use `0` or `-1` for the full dataset.
- `--json_file Path` (default `valid.json`): Path to the theorems JSON.
- `--log_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`: Logging verbosity.
- `--max_steps int` (default from config): Max agent steps per theorem.
- `--planning_interval int` (default from config): How often to run the planning phase.
- `--concurrency int` (default from config): Number of theorems processed in parallel.
- `--model str` (default from config): Model ID (OpenRouter-compatible). Available examples are listed in `config.py`.
- `--checkpoint str` Name to resume from (legacy single-file checkpoint in `tmp/checkpoints/<name>.json`).
- `--resume_run str` Resume from a specific run ID (e.g., '20250816-204214'). New stage-based format.
- `--resume_stage int` Specific stage to resume from (used with --resume_run). If not specified, resumes from the latest stage.
- `--save_checkpoint str` Name to save final legacy checkpoint as.
- `--checkpoint_interval int` Save legacy checkpoint every N tasks.
- `--list_checkpoints` List available checkpoints (both legacy and stage-based) and exit.
- `--stages int` Number of passes over the dataset. Stage > 1 re-runs unsolved tasks.

Per-stage run checkpoints are always saved to `tmp/checkpoints/run-<timestamp>/stage-<n>.json` and include summary fields such as `processed_count`, `solved_stage`, `solved_cumulative`, `unsolved_remaining`, and `results_*`.

Notes:
- Provider is fixed to OpenRouter in this script; ensure `OPENROUTER_API_KEY` is set.
- Tokens are counted with `tiktoken` when available and reported at the end.

### benchmark_openrouter.py
Prompt-only baseline using OpenRouter’s Chat Completions. It asks the model to replace `sorry` with a proof, extracts the Lean proof body, and verifies with Lake.

Flags:
- `--subset_size int` Use 0 or -1 for all tasks.
- `--json_file Path` Path to tasks (default `valid.json`).
- `--model str` OpenRouter model ID.
- `--log_level {DEBUG,INFO,...}` Logging verbosity.
- `--concurrency int` Number of tasks verified in parallel.
- `--stages int` Multi-pass runs; later stages retry only unsolved tasks.

Requires `OPENROUTER_API_KEY`.

### benchmark_openai.py
Prompt-only baseline using the OpenAI Responses API. Similar flow: prompt → extract proof body → verify with Lake.

Flags:
- `--subset_size int` Use 0 or -1 for all tasks.
- `--json_file Path`
- `--model str` OpenAI model name.
- `--log_level {DEBUG,INFO,...}`
- `--concurrency int`
- `--effort {low,medium,high}` Reasoning effort used in the Responses API.
- `--max_output_tokens int` Maximum tokens requested for generation.
- `--stages int`

Requires `OPENAI_API_KEY`.

### process_lean.py
Parses the Lean sources and emits `valid.json` with normalized theorems where any existing proofs are replaced by `sorry` so every item is solvable by the LLM/agent. The script:
- Reads `miniF2F-lean4/MiniF2F/Valid.lean` (path defined in `config.py`).
- Extracts declarations (`theorem`, `lemma`, `def`, `example`, `instance`, `abbrev`).
- Detects whether the original declaration had a proof and records `is_solved`.
- Rewrites proof blocks to a placeholder `sorry`.
- Writes a list of objects to `valid.json` with fields: `name`, `statement`, `is_solved`.

No CLI flags; just run:

```bash
python process_lean.py
```

### verify_task.py
Minimal helper to compile ad‑hoc Lean code with Lake inside `miniF2F-lean4`.

Usage (mutually exclusive):
- `--file <path>`: path to a Lean file whose contents are compiled.
- `--code "<lean code>"`: pass Lean code directly as a string.
- If neither is provided, the script reads Lean code from STDIN.

Example:

```bash
python verify_task.py --code "theorem t1 : 2 + 2 = 4 := by norm_num"
```

The script ensures `import MiniF2F.Minif2fImport` is present, writes a temp file under `miniF2F-lean4`, runs `lake env lean` and forwards compiler output, returning Lean’s exit code.

## Configuration
All defaults live in `config.py`:
- Paths: `MINIF2F_DIR`, `LEAN_SOURCE_FILE`, `LEAN_OUTPUT_FILE`, `LOG_DIR`, `TMP_DIR`.
- Providers and models: `AVAILABLE_PROVIDERS`, `DEFAULT_MODEL`, `AVAILABLE_MODELS`.
- Agent tuning: `DEFAULT_MAX_STEPS`, `DEFAULT_PLANNING_INTERVAL`, `DEFAULT_CONCURRENCY`, `DEFAULT_SUBSET_SIZE`.
- Timeouts and logging: `LLM_REQUEST_TIMEOUT`, `LEAN_TIMEOUT`, `LOG_FORMAT`.
- Validation helpers: `validate_config()` and `validate_provider_credentials()` (scripts call these and will error early if prerequisites are missing).

## Checkpoints, logs, and outputs
- `valid.json`: generated by `process_lean.py`; consumed by benchmarks and agents.
- **Stage-based checkpoints** (recommended): `tmp/checkpoints/run-<YYYYMMDD-HHMMSS>/stage-<n>.json` - always written per stage; includes cumulative and per-stage results and the list of unsolved items. Use `--resume_run <run_id>` to resume.
- **Legacy checkpoint files**: `tmp/checkpoints/<name>.json` (if you use `--save_checkpoint` / `--checkpoint` in `agents/math_prover_agent.py`).
- Logs:
  - `log/llm_requests.log`: provider requests (benchmarks).
  - `log/tools.log`: output from Lean verifier and search tools.
  - `log/agent_benchmark.log`: multi-agent run logs.

## Project structure

```
agents/
  math_prover_agent.py      # Multi-agent benchmark (OpenRouter). CLI flags documented above.
  tools.py                  # Tools for agents: Lean verifier (via Lake) and moogle.ai semantic search.
benchmark_openrouter.py     # Prompt baseline using OpenRouter Chat Completions.
benchmark_openai.py         # Prompt baseline using OpenAI Responses API.
config.py                   # Central configuration (paths, models, timeouts, validation helpers).
process_lean.py             # Produces valid.json from Lean sources, replacing proofs with sorry.
verify_task.py              # Compile/verify ad-hoc Lean code inside miniF2F project with Lake.
miniF2F-lean4/              # Vendored MiniF2F Lean project; see its README for build notes.
requirements.txt            # Python dependencies.
tmp/                        # Checkpoints and auxiliary outputs created at runtime.
log/                        # Runtime logs.
LICENSE                     # License for this repository.
```

### About `agents/tools.py`
- `LeanVerifier`: lightweight wrapper that writes a temp `.lean` file under `miniF2F-lean4/`, calls `lake env lean <file>`, cleans diagnostics, and returns `{success, exit_code, output}`.
  - Looks for `LAKE_BINARY` or falls back to `$HOME/.elan/bin/lake`.
  - Ensures `import MiniF2F.Minif2fImport` is injected if missing.
- `VerifyLeanProof` tool (`verify_lean_proof`): validates a complete Lean theorem string.
- `MoogleSemanticSearch` tool (`moogle_semantic_search`): calls `https://www.moogle.ai/api/search`, decodes Brotli if needed, and returns filtered fields (`declarationName`, `declarationCode`, `declarationDocstring`, `declarationType`, `sourceCodeUrl`, `mathlibPath`).

## Troubleshooting
- Lean tooling not found: ensure `elan` installed and `lake` on PATH. You can set `LAKE_BINARY` to point to your `lake` binary.
- miniF2F not built: run `lake exe cache get && lake build` inside `miniF2F-lean4/`.
- Insufficient credits (OpenRouter): the agent script will detect provider error code 402 and stop; add credits and re-run. Stage checkpoints allow resuming.
- Long compile times: `LEAN_TIMEOUT` is set high in `config.py`. Adjust if needed.
- Empty `valid.json`: ensure `miniF2F-lean4/MiniF2F/Valid.lean` exists and `process_lean.py` can read it (the script validates configuration and paths).

## Attribution and licenses
- This repository vendors and uses the MiniF2F Lean project under `miniF2F-lean4/`. Please consult and respect its documentation and license. See `miniF2F-lean4/README.md` for build instructions and helpful links.
- We also rely on `smolagents`, Lean 4, Lake, and (optionally) moogle.ai’s public search API.
- See `LICENSE` in this repository for the project’s license.

If you use this repo in academic work, please also cite MiniF2F:
- MiniF2F: “MiniF2F: a cross-system benchmark for formal Olympiad-level mathematics” (`https://arxiv.org/abs/2109.00110`).

## FAQ
- Which provider should I use? The multi-agent pipeline currently uses OpenRouter; prompt baselines are provided for both OpenRouter and OpenAI.
- Where do results go? Logs in `log/`, checkpoints in `tmp/checkpoints/...`. The source tasks are in `valid.json`.
- Can I resume from a previous run? Yes. Use `--stages` for automatic re-tries across stages. For the multi-agent script, you can resume from stage-based checkpoints with `--resume_run <run_id>` (recommended) or use legacy `--checkpoint`/`--save_checkpoint` options.


