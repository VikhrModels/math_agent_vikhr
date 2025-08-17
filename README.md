# Math Agent Vikhr

Multi-agent system for automated theorem proving using Lean 4, powered by `smolagents`.

## Quick Start

1) Install dependencies:

```bash
pip install -r requirements.txt
```

2) Run the multi-agent system:

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
Prompt-only baseline using OpenRouter's Chat Completions. It asks the model to replace `sorry` with a proof, extracts the Lean proof body, and verifies with Lake.

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

The script ensures `import MiniF2F.Minif2fImport` is present, writes a temp file under `miniF2F-lean4`, runs `lake env lean` and forwards compiler output, returning Lean's exit code.

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
- `agents/`: Multi-agent system implementation
- `miniF2F-lean4/`: Lean 4 theorem library
- `config.py`: Configuration and defaults
- `requirements.txt`: Python dependencies
- `tmp/`: Checkpoints and temporary files
- `log/`: Log files

## FAQ
- Which provider should I use? The multi-agent pipeline currently uses OpenRouter; prompt baselines are provided for both OpenRouter and OpenAI.
- Where do results go? Logs in `log/`, checkpoints in `tmp/checkpoints/...`. The source tasks are in `valid.json`.
- Can I resume from a previous run? Yes. Use `--stages` for automatic re-tries across stages. For the multi-agent script, you can resume from stage-based checkpoints with `--resume_run <run_id>` (recommended) or use legacy `--checkpoint`/`--save_checkpoint` options.


