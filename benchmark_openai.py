import os
import re
import json
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import concurrent.futures

from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, before_sleep_log

from agents.tools import lean_verifier as verifier

# --- Helpers ---------------------------------------------------------------
def _extract_output_text(response) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    outputs = getattr(response, "output", None)
    chunks: list[str] = []
    if isinstance(outputs, list):
        for msg in outputs:
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                        txt = part.get("text")
                        if isinstance(txt, str):
                            chunks.append(txt)
    return "\n".join(chunks).strip()

# Import configuration
from configs.config_loader import (
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    DEFAULT_MODEL,
    MINIF2F_DIR,
    LOG_DIR,
    TMP_DIR,
    LEAN_OUTPUT_FILE,
    DEFAULT_SUBSET_SIZE,
    DEFAULT_LOG_LEVEL,
    LOG_FORMAT,
    LLM_REQUEST_TIMEOUT,
    validate_config,
)

# --- Logging setup ---
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, DEFAULT_LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_DIR / "llm_requests.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def make_client() -> OpenAI:
    validate_config()
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")
    return OpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY)


def read_json_file(filepath: Path) -> list[dict]:
    logger.info(f"Reading JSON file: {filepath}")
    with filepath.open('r', encoding='utf-8') as f:
        return json.load(f)


def select_micro_subset_stratified(all_theorems: list[dict], subset_size: int) -> list[dict]:
    if subset_size <= 0:
        logger.info(f"Special value for subset_size={subset_size}: using the full dataset ({len(all_theorems)} tasks).")
        return all_theorems
    solved = [t for t in all_theorems if t['is_solved']]
    unsolved = [t for t in all_theorems if not t['is_solved']]
    total = len(solved) + len(unsolved)
    if total == 0:
        logger.warning("No theorems found to select from.")
        return []
    n_solved = round(subset_size * (len(solved) / total)) if len(solved) > 0 else 0
    n_unsolved = subset_size - n_solved
    selected = random.sample(solved, min(n_solved, len(solved))) + \
               random.sample(unsolved, min(n_unsolved, len(unsolved)))
    random.shuffle(selected)
    return selected


def _ensure_run_dir(base_dir: Path, run_id: str) -> Path:
    run_dir = base_dir / "checkpoints" / f"run-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def save_stage_checkpoint(
    run_dir: Path,
    run_id: str,
    stage_index: int,
    stages_total: int,
    stage_results: dict,
    cumulative_results: dict,
    processed_count: int,
    total_count: int,
    model_id: str,
    provider: str,
) -> None:
    now_iso = datetime.now().isoformat(timespec="seconds")
    unsolved_remaining = [name for name, ok in cumulative_results.items() if ok is not True]
    checkpoint_data = {
        "run_id": run_id,
        "timestamp": now_iso,
        "stage_index": stage_index,
        "stages_total": stages_total,
        "model": model_id,
        "provider": provider,
        "processed_count": processed_count,
        "total_count": total_count,
        "solved_stage": sum(1 for v in stage_results.values() if v is True),
        "solved_cumulative": sum(1 for v in cumulative_results.values() if v is True),
        "pass_rate_stage": (sum(1 for v in stage_results.values() if v is True) / total_count * 100) if total_count > 0 else 0.0,
        "unsolved_remaining": unsolved_remaining,
        "results_stage": stage_results,
        "results_cumulative": cumulative_results,
    }
    stage_file = run_dir / f"stage-{stage_index}.json"
    try:
        with stage_file.open('w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Stage checkpoint saved: {stage_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save stage checkpoint: {e}")


def verify_lean_proof(theorem_name: str, theorem_statement: str, generated_proof_body: str) -> bool:
    logger.info("Attempting Lean 4 verification via Lake.")
    if not generated_proof_body.strip():
        logger.warning("  ❌ Empty proof body provided.")
        return False
    if ':=' not in theorem_statement:
        logger.error("  ❌ Expected ':=' in theorem statement but not found.")
        logger.debug(f"    Full statement: {theorem_statement}")
        return False
    base = theorem_statement.split(':=')[0].strip()
    lean_code = "import MiniF2F.Minif2fImport\n\n" + base + " :=\n" + generated_proof_body
    try:
        result = verifier.verify_lean_code(lean_code, theorem_name)
        if result.get("success"):
            logger.info(f"  ✅ Verification successful for {theorem_name}.")
            return True
        logger.warning(f"  ❌ Verification failed for {theorem_name}.")
        return False
    except Exception as e:
        logger.error(f"  ❌ Verification error for {theorem_name}: {e}")
        return False


def extract_proof_body(llm_response_content: str) -> str | None:
    md = re.search(r'```(?:lean)?\s*(.*?)\s*```', llm_response_content, re.DOTALL | re.IGNORECASE)
    content = md.group(1).strip() if md else llm_response_content.strip()
    begin_match = re.search(r'\b(begin|by\b)', content, re.IGNORECASE | re.DOTALL)
    if not begin_match:
        logger.debug(f"No 'begin'/'by' in content:\n{llm_response_content}")
        return None
    start = begin_match.start()
    area = content[start:]
    if begin_match.group(1).lower() == 'by':
        lines = [ln.strip() for ln in area.split('\n') if ln.strip()]
        proof = '\n  '.join(lines)
        return proof if proof and not proof.endswith('...') else None
    # begin...end matching
    open_blocks = 0
    for m in re.finditer(r'\b(begin|end)\b', area, re.IGNORECASE):
        if m.group(1).lower() == 'begin':
            open_blocks += 1
        else:
            open_blocks -= 1
            if open_blocks == 0:
                return area[:m.end()]
    return None


client: Optional[OpenAI] = None
SELECTED_MODEL: str = DEFAULT_MODEL
REASONING_EFFORT: str = "low"  # one of: low, medium, high
MAX_OUTPUT_TOKENS: int = 4096

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5), before_sleep=before_sleep_log(logger, logging.WARNING))
def _call_llm_with_retry(input_messages: list[dict], model_name: str, extra_headers: dict, max_tokens: int):
    return client.responses.create(
        extra_headers=extra_headers,
        model=model_name,
        input=input_messages,
        reasoning={"effort": REASONING_EFFORT},
        timeout=LLM_REQUEST_TIMEOUT,
        extra_body={"max_output_tokens": max_tokens},
    )


def generate_and_verify_proof(theorem: dict) -> bool:
    logger.info(f"\n--- Processing Theorem: {theorem['name']} ---")
    logger.info(f"Original statement (first 200 chars):\n{theorem['statement'][:200]}...")
    try:
        system_prompt = (
            "You are a Lean 4 theorem prover. Your task is to replace `sorry` with a complete proof. "
            "Generate ONLY the proof code, no explanations or comments. "
            "Use simple tactics: `simp`, `norm_num`, `ring`, `linarith`, `tauto`, `exact`, `apply`, `rw`. "
            "For simple calculations use `norm_num` or `ring`. "
            "For linear equations use `linarith`. "
            "For rewriting use `rw [h₀, h₁]` then `ring` or `linarith`. "
            "Response must be ONLY the proof code in ```lean``` block. "
            "Keep proofs short and direct."
        )
        user_prompt = (
            f"Replace `sorry` with a simple Lean 4 proof using only basic tactics (simp, norm_num, ring, linarith, tauto, exact, apply, rw, cases, induction). Return ONLY the proof code in ```lean``` block.\n\n"
            f"Now prove this theorem:\n\n```lean\n{theorem['statement']}\n```"
        )
        input_messages: list[dict] = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        logger.info(f"Sending request to LLM for {theorem['name']}. Prompt (first 500 chars):\n{json.dumps(input_messages, ensure_ascii=False, indent=2)[:500]}...")
        resp = _call_llm_with_retry(
            input_messages=input_messages,
            model_name=SELECTED_MODEL,
            extra_headers={
                "HTTP-Referer": "https://github.com/umbra2728/math_agent_vikhr",
                "X-Title": "Math Agent Vikhr",
            },
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        generated_content = _extract_output_text(resp)
        if not generated_content:
            logger.error(f"  ❌ Empty response content from LLM for {theorem['name']}")
            logger.debug(f"  Raw response object: {resp}")
            return False
        logger.info(f"Received LLM response for {theorem['name']}. Content:\n{generated_content[:200]}...")
        proof_body = extract_proof_body(generated_content)
        if not proof_body:
            logger.warning(f"  ❌ Failed to extract Lean proof body from LLM response for {theorem['name']}.")
            return False
        logger.info(f"  Attempting to verify proof for {theorem['name']}...")
        return verify_lean_proof(theorem['name'], theorem['statement'], proof_body)
    except Exception as e:
        logger.error(f"  An error occurred during LLM interaction for {theorem['name']}: {e}")
        return False


def process_theorem_task(theorem: dict) -> tuple[bool, str]:
    success = generate_and_verify_proof(theorem)
    return success, theorem['name']


def main():
    parser = argparse.ArgumentParser(description="Automated Lean 4 theorem proving with LLM (OpenAI)")
    parser.add_argument("--subset_size", type=int, default=DEFAULT_SUBSET_SIZE,
                        help="Total number of tasks for the micro-subset. Use 0 or -1 to use the full dataset.")
    parser.add_argument("--json_file", type=Path, default=LEAN_OUTPUT_FILE,
                        help="Path to the JSON file containing theorems.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Model name to use for proof generation.")
    parser.add_argument("--log_level", type=str, default=DEFAULT_LOG_LEVEL,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level.")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Number of theorems to process in parallel (default: 4 for OpenAI).")
    parser.add_argument("--effort", type=str, choices=["low", "medium", "high"], default="low",
                        help="Reasoning effort level for OpenAI Responses API (default: low).")
    parser.add_argument("--max_output_tokens", type=int, default=4096,
                        help="Max output tokens for the model response (default: 4096).")
    # Stages (multi-pass)
    parser.add_argument("--stages", type=int, default=1,
                        help="Number of passes over the dataset. 1 = single pass; 2 = rerun on unsolved, etc.")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    logger.info("Starting MiniF2F Lean Prover Benchmark (OpenAI)...")
    logger.info(f"Configuration: Subset Size={args.subset_size}, JSON File='{args.json_file}', Model='{args.model}', Log Level='{args.log_level}', Concurrency={args.concurrency}")

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        all_theorems = read_json_file(args.json_file)
    except Exception as e:
        logger.critical(f"Failed to load JSON file: {e}. Exiting.")
        return
    if not all_theorems:
        logger.warning("No theorems found in the JSON file. Exiting.")
        return
    micro_subset = select_micro_subset_stratified(all_theorems, args.subset_size) if args.subset_size > 0 else all_theorems
    if not micro_subset:
        logger.warning("No tasks selected for the micro-subset. Exiting.")
        return

    global client
    client = make_client()
    global SELECTED_MODEL
    SELECTED_MODEL = args.model
    global REASONING_EFFORT, MAX_OUTPUT_TOKENS
    REASONING_EFFORT = args.effort
    MAX_OUTPUT_TOKENS = args.max_output_tokens
    # Prepare staged processing
    name_to_theorem = {t['name']: t for t in micro_subset}
    stages_total = max(1, int(getattr(args, 'stages', 1)))
    logger.info(f"Stages requested: {stages_total}")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = _ensure_run_dir(TMP_DIR, run_id)

    cumulative_results: dict[str, bool] = {}
    for stage_index in range(1, stages_total + 1):
        if stage_index == 1:
            tasks_this_stage = micro_subset
        else:
            tasks_this_stage = [name_to_theorem[name] for name in name_to_theorem.keys() if cumulative_results.get(name) is not True]
        if not tasks_this_stage:
            logger.info(f"Stage {stage_index}/{stages_total}: nothing to process (all solved).")
            save_stage_checkpoint(run_dir, run_id, stage_index, stages_total, {}, cumulative_results, 0, 0, args.model, "openai")
            continue

        logger.info(f"\n--- Stage {stage_index}/{stages_total}: Processing {len(tasks_this_stage)} tasks ---")
        stage_results: dict[str, bool] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_to_theorem = {executor.submit(process_theorem_task, th): th['name'] for th in tasks_this_stage}
            processed_in_stage = 0
            for future in concurrent.futures.as_completed(future_to_theorem):
                name = future_to_theorem[future]
                try:
                    success, _ = future.result(timeout=360)
                    stage_results[name] = success
                    cumulative_results[name] = success
                except concurrent.futures.TimeoutError:
                    logger.error(f"❌ Timeout processing theorem {name}")
                    stage_results[name] = False
                    cumulative_results[name] = False
                except Exception as e:
                    logger.error(f"❌ Error processing theorem {name}: {e}")
                    stage_results[name] = False
                    cumulative_results[name] = False
                processed_in_stage += 1
                logger.info(f"Stage {stage_index}: progress {processed_in_stage}/{len(tasks_this_stage)}")
                save_stage_checkpoint(run_dir, run_id, stage_index, stages_total, stage_results, cumulative_results, processed_in_stage, len(tasks_this_stage), args.model, "openai")

        # end-of-stage snapshot
        save_stage_checkpoint(run_dir, run_id, stage_index, stages_total, stage_results, cumulative_results, len(stage_results), len(tasks_this_stage), args.model, "openai")

    logger.info("\n--- Summary of Results (Cumulative) ---")
    solved = sum(1 for ok in cumulative_results.values() if ok)
    total = len(name_to_theorem)
    for name in sorted(name_to_theorem.keys()):
        ok = cumulative_results.get(name, False)
        logger.info(f"{name}: {'PASSED' if ok else 'FAILED'}")
    logger.info(f"\nTotal tasks processed: {len(cumulative_results)}")
    logger.info(f"Successfully proven: {solved}")
    logger.info(f"Pass rate: {solved / total * 100:.2f}%" if total > 0 else "No tasks processed.")


if __name__ == "__main__":
    main()


