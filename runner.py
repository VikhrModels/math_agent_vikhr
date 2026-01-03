import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

from configs.config_loader import (
    BASE_DIR,
    TMP_DIR,
)


def _load_run_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Invalid run config structure: expected mapping at top-level")
    if "model_list" not in data or not isinstance(data["model_list"], list) or not data["model_list"]:
        raise ValueError("run.yaml must include non-empty 'model_list'")
    return data


def _resolve_dataset_path(dataset: str | None) -> Path:
    # Minif2F default
    if dataset in (None, "minif2f"):
        return BASE_DIR / "valid.json"
    # Fallback to treat as path
    p = Path(dataset)
    return p if p.is_absolute() else (BASE_DIR / dataset)


def _ensure_dirs():
    (BASE_DIR / "results").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "results" / "details").mkdir(parents=True, exist_ok=True)


def _write_model_results(model_key: str, summary: Dict[str, Any], results: Dict[str, Any]) -> None:
    model_dir = BASE_DIR / "results" / "details" / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (model_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _read_model_summary_if_exists(model_key: str) -> Dict[str, Any] | None:
    path = BASE_DIR / "results" / "details" / model_key / "summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _generate_leaderboard(model_summaries: Dict[str, Dict[str, Any]]) -> None:
    # Sort by score desc; if missing score, treat as 0
    items = sorted(
        model_summaries.items(),
        key=lambda kv: (kv[1].get("score") or 0.0),
        reverse=True,
    )
    md_lines: List[str] = []
    md_lines.append("## Leaderboard")
    md_lines.append("")
    md_lines.append("| Model | Score | Evaluation Time (s) | Notes |")
    md_lines.append("|---|---:|---:|---|")
    for model_key, s in items:
        model_name = s.get("model_name", model_key)
        score = s.get("score")
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "-"
        t = s.get("evaluation_time")
        t_str = f"{t:.1f}" if isinstance(t, (int, float)) else "-"
        notes = s.get("notes", "")
        md_lines.append(f"| {model_name} | {score_str} | {t_str} | {notes} |")
    (BASE_DIR / "results" / "leaderboard.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def _run_backend(api_type: str, model_cfg: Dict[str, Any], dataset_path: Path, max_workers: int | None, num_examples: int | None) -> Dict[str, Any]:
    # Import lazily to keep startup light
    if api_type == "openrouter":
        import benchmark_openrouter as backend
        return backend.run_benchmark({
            "dataset_path": str(dataset_path),
            "model": model_cfg.get("model_name"),
            "concurrency": model_cfg.get("parallel", max_workers or 1),
            "num_examples": num_examples,
            "max_tokens": model_cfg.get("max_tokens"),
        })
    if api_type == "openai":
        import benchmark_openai as backend
        return backend.run_benchmark({
            "dataset_path": str(dataset_path),
            "model": model_cfg.get("model_name"),
            "concurrency": model_cfg.get("parallel", max_workers or 1),
            "num_examples": num_examples,
            "max_output_tokens": model_cfg.get("max_tokens"),
        })
    if api_type == "agent":
        from agents import math_prover_agent as backend
        return backend.run_benchmark({
            "dataset_path": str(dataset_path),
            "model": model_cfg.get("model_name"),
            "concurrency": model_cfg.get("parallel", max_workers or 1),
            "num_examples": num_examples,
            "max_steps": model_cfg.get("max_steps"),
            "planning_interval": model_cfg.get("planning_interval"),
        })
    raise ValueError(f"Unsupported api_type: {api_type}")


def main():
    parser = argparse.ArgumentParser(description="Unified runner (DOoM-style)")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "configs" / "run.yaml")
    parser.add_argument("--dataset", type=str, default="minif2f")
    parser.add_argument("--max-workers", dest="max_workers", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    _ensure_dirs()

    run_cfg = _load_run_config(args.config)
    dataset_path = _resolve_dataset_path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    model_summaries: Dict[str, Dict[str, Any]] = {}
    global_num_examples = run_cfg.get("num_examples")
    for model_key in run_cfg["model_list"]:
        model_cfg = run_cfg.get(model_key)
        if not isinstance(model_cfg, dict):
            raise ValueError(f"Model config block not found for '{model_key}' in run.yaml")
        api_type = model_cfg.get("api_type", "openrouter")

        if not args.no_cache:
            cached = _read_model_summary_if_exists(model_key)
            if cached is not None:
                model_summaries[model_key] = cached
                continue

        per_model_num_examples = model_cfg.get("num_examples", global_num_examples)
        start = time.time()
        result = _run_backend(
            api_type=api_type,
            model_cfg=model_cfg,
            dataset_path=dataset_path,
            max_workers=args.max_workers,
            num_examples=per_model_num_examples,
        )
        duration = time.time() - start

        # Expected result contains: score (float), results (dict), optional fields
        score = result.get("score")
        results = result.get("results", {})
        summary = {
            "model_key": model_key,
            "model_name": model_cfg.get("model_name", model_key),
            "api_type": api_type,
            "score": score,
            "evaluation_time": result.get("evaluation_time", duration),
            "notes": result.get("notes", ""),
        }
        _write_model_results(model_key, summary, results)
        model_summaries[model_key] = summary

    _generate_leaderboard(model_summaries)


if __name__ == "__main__":
    main()


