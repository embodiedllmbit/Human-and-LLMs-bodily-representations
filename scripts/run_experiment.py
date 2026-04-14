#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the full video understanding experiment across models and prompt types."""

import argparse
import json
import logging
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from answer_parser import extract_final_answer
from model_config import EXPERIMENT_ROOT, VIDEO_DIR, get_experiment_configs
from prompt_factory import (
    build_prompt,
    is_detail_prompt_type,
    load_video_descriptions_from_csv,
    normalize_prompt_type,
    to_script_prompt_type,
)

DEFAULT_VIDEO_DESCRIPTIONS_CSV = str(Path(EXPERIMENT_ROOT) / "detailprompt.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("experiment.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


_OPTION_ONLY_PATTERN = re.compile(r"^\s*([A-Da-d])\s*[\.\)]?\s*$")


def normalize_option_letter(value: Any) -> Optional[str]:
    """Return A/B/C/D when value is a standalone option letter."""
    if value is None:
        return None

    match = _OPTION_ONLY_PATTERN.match(str(value))
    if match:
        return match.group(1).upper()
    return None


def resolve_model_answer(
    model_json: Optional[Dict[str, Any]],
    stdout_text: str = "",
) -> Tuple[Optional[str], str, str]:
    """
    Resolve final answer robustly from structured fields and text fallback.

    Returns:
        (answer, method, status)
    """
    if model_json is not None:
        final_answer = normalize_option_letter(model_json.get("final_answer"))
        if final_answer is not None:
            return final_answer, "final_answer_field", "ok"

        response_field = normalize_option_letter(model_json.get("response"))
        if response_field is not None:
            return response_field, "response_field", "ok"

        response_text = str(model_json.get("response") or "")
        if response_text:
            answer, method = extract_final_answer(response_text)
            if answer is not None:
                return answer, f"response_text_{method}", "ok"

        for key in ("original_output", "raw_response"):
            text = str(model_json.get(key) or "")
            if not text:
                continue
            answer, method = extract_final_answer(text)
            if answer is not None:
                return answer, f"{key}_{method}", "ok"

    stdout = str(stdout_text or "")
    if stdout:
        answer, method = extract_final_answer(stdout)
        if answer is not None:
            return answer, f"stdout_{method}", "ok"

    return None, "not_found", "not_found"


def parse_json_from_stdout(stdout: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from stdout, allowing logs around the payload."""
    text = (stdout or "").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = text[first : last + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None

    return None


def get_inference_time_from_metadata(metadata_path: Optional[str]) -> Optional[float]:
    """Read inference time from metadata file."""
    if not metadata_path or not os.path.exists(metadata_path):
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        record = data[-1] if isinstance(data, list) and data else data
        if not isinstance(record, dict):
            return None

        for key in ["processing_duration_seconds", "inference_time_seconds"]:
            value = record.get(key)
            if value is not None:
                return float(value)
    except Exception:
        return None

    return None


def build_inference_command(
    mode: str,
    config: Dict[str, Any],
    video_path: str,
    prompt_text: str,
    script_prompt_type: str,
    video_id: str,
    output_dir: str,
) -> List[str]:
    """Build CLI command for the model chat script."""
    cmd = [
        "python3",
        config["script_path"],
        "--video",
        video_path,
        "--prompt",
        prompt_text,
        "--prompt-type",
        script_prompt_type,
        "--video-id",
        video_id,
        "--output",
        output_dir,
    ]

    mode_arg_name = config.get("mode_arg_name")
    mode_arg_values = config.get("mode_arg_values", {})
    if mode_arg_name:
        mapped_mode = mode_arg_values.get(mode, mode)
        cmd.extend([f"--{mode_arg_name}", mapped_mode])
    elif config.get("supports_think_tag", False):
        cmd.extend(["--mode", "thinking" if mode == "think" else "base"])

    return cmd


def run_single_inference(
    model_name: str,
    mode: str,
    config: Dict[str, Any],
    question_data: Dict[str, Any],
    prompt_type: str,
    script_prompt_type: str,
    prompt_text: str,
    output_dir: str,
    timeout: int = 300,
) -> Dict[str, Any]:
    """Run one inference call and return structured result."""
    script_dir = os.path.dirname(config["script_path"])

    output_dir = os.path.abspath(output_dir)

    cmd = build_inference_command(
        mode=mode,
        config=config,
        video_path=question_data["video_path"],
        prompt_text=prompt_text,
        script_prompt_type=script_prompt_type,
        video_id=question_data["video_id"],
        output_dir=output_dir,
    )

    result = {
        "model": model_name,
        "mode": mode,
        "prompt_type": prompt_type,
        "question_id": question_data["id"],
        "video_id": question_data["video_id"],
        "question_type": question_data["question_type"],
        "correct_answer": question_data["correct_answer"],
        "model_answer": None,
        "correct": None,
        "inference_time_seconds": None,
        "raw_response": None,
        "original_output": None,
        "thinking_content": None,
        "answer_parse_method": None,
        "answer_parse_status": None,
        "metadata_path": None,
        "artifacts_output_dir": output_dir,
        "success": False,
        "error": None,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=script_dir,
        )

        if process.returncode != 0:
            stderr = (process.stderr or "").strip()
            stdout = (process.stdout or "").strip()
            message = stderr if stderr else stdout
            result["error"] = f"Return code {process.returncode}: {message[:500]}"
            return result

        stdout = (process.stdout or "").strip()
        model_json = parse_json_from_stdout(stdout)

        if model_json is not None:
            if not bool(model_json.get("success", True)):
                result["error"] = str(model_json.get("error", "model returned success=false"))
                return result

            raw_output_value = model_json.get("original_output")
            if raw_output_value is None:
                raw_output_value = model_json.get("response")

            model_answer, parse_method, parse_status = resolve_model_answer(
                model_json=model_json,
                stdout_text=stdout,
            )
            result["model_answer"] = model_answer
            result["answer_parse_method"] = parse_method
            result["answer_parse_status"] = parse_status
            result["correct"] = result["model_answer"] == result["correct_answer"]
            result["raw_response"] = None if raw_output_value is None else str(raw_output_value)
            result["original_output"] = None if raw_output_value is None else str(raw_output_value)
            result["thinking_content"] = model_json.get("thinking_content")
            metadata_path = model_json.get("metadata_path")
            if metadata_path:
                metadata_path = str(metadata_path)
                if not os.path.isabs(metadata_path):
                    metadata_path = os.path.abspath(os.path.join(script_dir, metadata_path))
            result["metadata_path"] = metadata_path

            inference_time = model_json.get("inference_time_seconds")
            if inference_time is None:
                inference_time = get_inference_time_from_metadata(metadata_path)
            result["inference_time_seconds"] = inference_time
            result["success"] = True
            return result

        model_answer, parse_method, parse_status = resolve_model_answer(
            model_json=None,
            stdout_text=stdout,
        )
        result["model_answer"] = model_answer
        result["answer_parse_method"] = parse_method
        result["answer_parse_status"] = parse_status
        result["correct"] = result["model_answer"] == result["correct_answer"]
        result["raw_response"] = stdout
        result["original_output"] = stdout
        think_match = re.search(r"<think>(.*?)</think>", stdout, flags=re.DOTALL)
        if think_match:
            result["thinking_content"] = think_match.group(1).strip()
        result["success"] = True
        return result

    except subprocess.TimeoutExpired:
        result["error"] = f"Timeout after {timeout}s"
    except Exception as exc:
        result["error"] = str(exc)

    return result


def save_results(output_file: str, results: List[Dict[str, Any]], start_time: datetime, total_questions: int) -> None:
    """Persist results and aggregate statistics."""
    stats: Dict[str, Dict[str, Any]] = {}

    for row in results:
        key = f"{row['model']}_{row['mode']}_{row.get('prompt_type', 'simple')}"
        if key not in stats:
            stats[key] = {"total": 0, "correct": 0, "inference_times": []}

        if row["success"]:
            stats[key]["total"] += 1
            if row["correct"]:
                stats[key]["correct"] += 1
            if row["inference_time_seconds"] is not None:
                stats[key]["inference_times"].append(row["inference_time_seconds"])

    model_stats = {}
    for key, value in stats.items():
        total = value["total"]
        accuracy = (value["correct"] / total * 100) if total > 0 else 0.0
        avg_time = (
            sum(value["inference_times"]) / len(value["inference_times"])
            if value["inference_times"]
            else None
        )
        model_stats[key] = {
            "total": total,
            "correct": value["correct"],
            "accuracy_percent": round(accuracy, 2),
            "avg_inference_time_seconds": round(avg_time, 3) if avg_time is not None else None,
        }

    output_data = {
        "experiment_info": {
            "start_time": start_time.isoformat(),
            "last_update": datetime.now().isoformat(),
            "total_questions": total_questions,
            "total_results": len(results),
            "model_statistics": model_stats,
        },
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2, ensure_ascii=False)


def normalize_prompt_type_list(prompt_types: Sequence[str]) -> List[str]:
    """Normalize and deduplicate prompt types while preserving order."""
    normalized: List[str] = []
    seen = set()
    for item in prompt_types:
        value = normalize_prompt_type(item)
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


def result_key(question_id: str, model_name: str, mode: str, prompt_type: str) -> str:
    return f"{question_id}_{model_name}_{mode}_{prompt_type}"


def sanitize_name(value: str) -> str:
    """Sanitize value for use in path components."""
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return sanitized or "unknown"


def build_artifacts_output_dir(
    artifacts_root: str,
    model_name: str,
    mode: str,
    prompt_type: str,
    question_id: str,
) -> str:
    """Build a unique artifacts output directory for one inference task."""
    return os.path.join(
        artifacts_root,
        sanitize_name(model_name),
        sanitize_name(mode),
        sanitize_name(prompt_type),
        sanitize_name(question_id),
    )


def resolve_video_path(question: Dict[str, Any]) -> str:
    """Resolve video path for compatibility across different project roots."""
    raw_path = str(question.get("video_path") or "").strip()
    video_id = str(question.get("video_id") or "").strip()

    if raw_path and os.path.exists(raw_path):
        return raw_path

    if raw_path:
        basename = os.path.basename(raw_path)
        if basename:
            candidate = os.path.join(VIDEO_DIR, basename)
            if os.path.exists(candidate):
                return candidate

    if video_id:
        candidate = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        if os.path.exists(candidate):
            return candidate

    return raw_path


def run_experiment(
    questions_file: str,
    output_file: str,
    models_to_run: Optional[Sequence[str]] = None,
    resume: bool = False,
    timeout: int = 300,
    max_questions: Optional[int] = None,
    video_id: Optional[str] = None,
    prompt_types: Optional[Sequence[str]] = None,
    video_descriptions_csv: str = DEFAULT_VIDEO_DESCRIPTIONS_CSV,
    save_every: int = 1,
    artifacts_root: Optional[str] = None,
) -> None:
    """Run the complete experiment."""
    prompt_types = normalize_prompt_type_list(prompt_types or ["simple"])
    logger.info("Prompt types: %s", ", ".join(prompt_types))

    with open(questions_file, "r", encoding="utf-8") as file:
        questions_data = json.load(file)

    questions = questions_data["questions"]
    for question in questions:
        question["video_path"] = resolve_video_path(question)

    if video_id:
        questions = [item for item in questions if item["video_id"] == video_id]
        logger.info("Questions after video filter (%s): %d", video_id, len(questions))

    if max_questions is not None:
        questions = questions[:max_questions]
        logger.info("Questions after max limit (%d): %d", max_questions, len(questions))

    if not questions:
        raise ValueError("No questions left after filters")

    logger.info("Loaded %d questions", len(questions))

    video_descriptions: Dict[str, str] = {}
    if any(is_detail_prompt_type(prompt_type) for prompt_type in prompt_types):
        video_descriptions = load_video_descriptions_from_csv(video_descriptions_csv)
        question_video_ids = {item["video_id"] for item in questions}
        missing = sorted([video for video in question_video_ids if video not in video_descriptions])
        if missing:
            logger.warning(
                "Missing video descriptions for %d/%d videos; fallback text will be used",
                len(missing),
                len(question_video_ids),
            )
        logger.info("Loaded %d video descriptions", len(video_descriptions))

    configs = get_experiment_configs()
    if models_to_run:
        selected_models = set(models_to_run)
        configs = [cfg for cfg in configs if cfg[0] in selected_models]

    if not configs:
        raise ValueError("No model configurations selected")

    logger.info("Selected %d model configurations", len(configs))
    for model_name, mode, _ in configs:
        logger.info("  - %s (%s)", model_name, mode)

    existing_results: Dict[str, Dict[str, Any]] = {}
    if resume and os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as file:
            existing_data = json.load(file)
        for row in existing_data.get("results", []):
            stored_prompt_type = normalize_prompt_type(row.get("prompt_type", "simple"))
            key = result_key(row["question_id"], row["model"], row["mode"], stored_prompt_type)
            row["prompt_type"] = stored_prompt_type
            existing_results[key] = row
        logger.info("Loaded %d existing results from resume file", len(existing_results))

    results: List[Dict[str, Any]] = list(existing_results.values())

    total_tasks = len(questions) * len(configs) * len(prompt_types)
    completed = len(existing_results)
    logger.info("Total tasks: %d | Already completed: %d", total_tasks, completed)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)
    if artifacts_root is None:
        output_stem = Path(output_file).with_suffix("").name
        artifacts_root = str(Path(output_file).resolve().parent / f"{output_stem}_artifacts")
    artifacts_root = os.path.abspath(artifacts_root)
    os.makedirs(artifacts_root, exist_ok=True)
    logger.info("Artifacts root: %s", artifacts_root)

    experiment_start = datetime.now()

    for prompt_type in prompt_types:
        script_prompt_type = to_script_prompt_type(prompt_type)
        logger.info("\n%s", "=" * 72)
        logger.info("Prompt type: %s (script value: %s)", prompt_type, script_prompt_type)
        logger.info("%s", "=" * 72)

        for model_name, mode, config in configs:
            logger.info("\n%s", "=" * 72)
            logger.info("Model: %s (%s) | Prompt: %s", model_name, mode, prompt_type)
            logger.info("%s", "=" * 72)

            model_correct = 0
            model_total = 0

            for question in questions:
                key = result_key(question["id"], model_name, mode, prompt_type)
                if key in existing_results:
                    continue

                completed += 1
                logger.info(
                    "[%d/%d] %s - %s",
                    completed,
                    total_tasks,
                    question["question_type"],
                    question["id"][:25] + "...",
                )

                prompt_text = build_prompt(
                    question_data=question,
                    prompt_type=prompt_type,
                    video_descriptions=video_descriptions,
                )
                output_dir = build_artifacts_output_dir(
                    artifacts_root=artifacts_root,
                    model_name=model_name,
                    mode=mode,
                    prompt_type=prompt_type,
                    question_id=question["id"],
                )
                os.makedirs(output_dir, exist_ok=True)

                result = run_single_inference(
                    model_name=model_name,
                    mode=mode,
                    config=config,
                    question_data=question,
                    prompt_type=prompt_type,
                    script_prompt_type=script_prompt_type,
                    prompt_text=prompt_text,
                    output_dir=output_dir,
                    timeout=timeout,
                )

                results.append(result)
                existing_results[key] = result

                if result["success"]:
                    model_total += 1
                    if result["correct"]:
                        model_correct += 1

                    status = "PASS" if result["correct"] else "FAIL"
                    infer_time = (
                        f"{result['inference_time_seconds']:.2f}s"
                        if result["inference_time_seconds"] is not None
                        else "N/A"
                    )
                    logger.info(
                        "  %s answer=%s expected=%s time=%s",
                        status,
                        result["model_answer"],
                        result["correct_answer"],
                        infer_time,
                    )
                else:
                    logger.warning("  ERROR: %s", str(result["error"])[:200])

                if save_every <= 1 or completed % save_every == 0:
                    save_results(
                        output_file=output_file,
                        results=results,
                        start_time=experiment_start,
                        total_questions=questions_data.get("total_questions", len(questions)),
                    )

            if model_total > 0:
                accuracy = model_correct / model_total * 100.0
                logger.info(
                    "%s(%s,%s): %d/%d (%.1f%%)",
                    model_name,
                    mode,
                    prompt_type,
                    model_correct,
                    model_total,
                    accuracy,
                )

    save_results(
        output_file=output_file,
        results=results,
        start_time=experiment_start,
        total_questions=questions_data.get("total_questions", len(questions)),
    )

    elapsed = (datetime.now() - experiment_start).total_seconds()
    logger.info("\nExperiment finished in %.2f hours", elapsed / 3600.0)
    logger.info("Results written to: %s", output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run video understanding experiments")
    parser.add_argument("--questions", required=True, help="Path to question JSON")
    parser.add_argument("--output", default="./experiment_results.json", help="Output JSON path")
    parser.add_argument("--models", nargs="+", help="Run only selected model names")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per inference call (seconds)")
    parser.add_argument("--max-questions", type=int, default=None, help="Maximum number of questions to run")
    parser.add_argument("--video-id", type=str, default=None, help="Run only a specific video_id")
    parser.add_argument(
        "--prompt-types",
        nargs="+",
        default=["simple"],
        help="Prompt types to run (supported: simple, detail, embodied_simple, embodied_detail)",
    )
    parser.add_argument(
        "--video-descriptions-csv",
        default=DEFAULT_VIDEO_DESCRIPTIONS_CSV,
        help="Path to CSV containing per-video detail descriptions",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save output every N completed tasks (default: 1)",
    )
    parser.add_argument(
        "--artifacts-root",
        default=None,
        help="Root directory for per-question artifacts (defaults to sibling folder of output file)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        questions_file=args.questions,
        output_file=args.output,
        models_to_run=args.models,
        resume=args.resume,
        timeout=args.timeout,
        max_questions=args.max_questions,
        video_id=args.video_id,
        prompt_types=args.prompt_types,
        video_descriptions_csv=args.video_descriptions_csv,
        save_every=args.save_every,
        artifacts_root=args.artifacts_root,
    )


if __name__ == "__main__":
    main()
