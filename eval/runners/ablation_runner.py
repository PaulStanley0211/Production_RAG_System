"""Ablation runner — measures each pipeline component's contribution.

Runs the retrieval and generation eval suites under each ablation
configuration, captures the numbers, and produces a comparison table.

Configurations:
    baseline                    everything on (production behavior)
    no_reranker                 disable cross-encoder reranking
    no_crag                     disable CRAG self-correction loop
    no_content_filter           disable content filter on chunks
    dense_only                  retrieve with dense embeddings only
    sparse_only                 retrieve with BM25 only

For each:
  1. Update .env with the right flags
  2. Restart the app container so it picks up the new env
  3. Wait for /ready
  4. Flush Redis cache
  5. Run retrieval_runner → capture metrics
  6. Run e2e_runner → capture metrics
  7. Save under the configuration name

Output: eval/reports/ablation_<timestamp>.md

Run from repo root:
    python -m eval.runners.ablation_runner

Wall time ~60-90 minutes total. Watch the logs to monitor progress.
"""

import asyncio
import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

from eval.runners.e2e_runner import run_e2e_eval
from eval.runners.retrieval_runner import run_retrieval_eval

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

ENV_PATH = Path(".env")
REPORTS_DIR = Path("eval/reports")

API_BASE = "http://localhost:8000"
READY_TIMEOUT_SECONDS = 90
READY_POLL_INTERVAL = 2.0


# ============================================================
# Ablation configurations
# ============================================================

@dataclass
class AblationConfig:
    """One configuration to test — flag values + a human-readable name."""

    name: str
    description: str
    env_overrides: dict[str, str]


CONFIGURATIONS: list[AblationConfig] = [
    AblationConfig(
        name="baseline",
        description="Full system: hybrid retrieval + reranker + CRAG + content filter",
        env_overrides={
            "DISABLE_RERANKER": "false",
            "DISABLE_CRAG": "false",
            "DISABLE_CONTENT_FILTER": "false",
            "RETRIEVAL_MODE": "hybrid",
        },
    ),
    AblationConfig(
        name="no_reranker",
        description="Hybrid retrieval, but no cross-encoder rerank — RRF order only",
        env_overrides={
            "DISABLE_RERANKER": "true",
            "DISABLE_CRAG": "false",
            "DISABLE_CONTENT_FILTER": "false",
            "RETRIEVAL_MODE": "hybrid",
        },
    ),
    AblationConfig(
        name="no_crag",
        description="Linear retrieve+grade+generate; no decompose retry",
        env_overrides={
            "DISABLE_RERANKER": "false",
            "DISABLE_CRAG": "true",
            "DISABLE_CONTENT_FILTER": "false",
            "RETRIEVAL_MODE": "hybrid",
        },
    ),
    AblationConfig(
        name="no_content_filter",
        description="Skip content filter on retrieved chunks",
        env_overrides={
            "DISABLE_RERANKER": "false",
            "DISABLE_CRAG": "false",
            "DISABLE_CONTENT_FILTER": "true",
            "RETRIEVAL_MODE": "hybrid",
        },
    ),
    AblationConfig(
        name="dense_only",
        description="Only dense embeddings, no BM25",
        env_overrides={
            "DISABLE_RERANKER": "false",
            "DISABLE_CRAG": "false",
            "DISABLE_CONTENT_FILTER": "false",
            "RETRIEVAL_MODE": "dense_only",
        },
    ),
    AblationConfig(
        name="sparse_only",
        description="Only BM25, no dense embeddings",
        env_overrides={
            "DISABLE_RERANKER": "false",
            "DISABLE_CRAG": "false",
            "DISABLE_CONTENT_FILTER": "false",
            "RETRIEVAL_MODE": "sparse_only",
        },
    ),
]


# ============================================================
# Env file management
# ============================================================

# Keys this runner manages. Other keys in .env are preserved untouched.
MANAGED_KEYS = {
    "DISABLE_RERANKER",
    "DISABLE_CRAG",
    "DISABLE_CONTENT_FILTER",
    "RETRIEVAL_MODE",
}


def update_env(overrides: dict[str, str]) -> None:
    """Update .env in-place with the given overrides. Preserves other keys."""
    if not ENV_PATH.exists():
        log.warning(".env not found; creating new file")
        ENV_PATH.write_text("", encoding="utf-8")

    lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    seen_keys: set[str] = set()
    new_lines: list[str] = []

    # Replace existing keys
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue
        if "=" not in stripped:
            new_lines.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in overrides:
            new_lines.append(f"{key}={overrides[key]}")
            seen_keys.add(key)
        else:
            new_lines.append(line)

    # Append any keys that weren't already in the file
    missing = set(overrides.keys()) - seen_keys
    if missing:
        new_lines.append("")
        new_lines.append("# Ablation flags (managed by ablation_runner)")
        for key in sorted(missing):
            new_lines.append(f"{key}={overrides[key]}")

    ENV_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


# ============================================================
# Container lifecycle
# ============================================================

def restart_app() -> None:
    """Recreate the app container so new .env values take effect.

    `docker compose restart` only restarts the existing container with its
    baked-in env — it does NOT re-read env_file. Recreate is required.
    """
    log.info("Recreating app container to pick up new env...")
    result = subprocess.run(
        ["docker", "compose", "up", "-d", "--force-recreate", "app"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"docker compose up -d --force-recreate app failed: {result.stderr}"
        )


def flush_redis() -> None:
    """Clear the semantic cache so each ablation runs fresh."""
    log.info("Flushing Redis cache...")
    result = subprocess.run(
        ["docker", "exec", "rag-redis", "redis-cli", "FLUSHDB"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.warning("Redis flush failed (non-fatal): %s", result.stderr)


async def wait_for_ready() -> None:
    """Poll /ready until 200 or timeout."""
    deadline = time.time() + READY_TIMEOUT_SECONDS
    log.info("Waiting for app /ready...")
    async with httpx.AsyncClient() as client:
        while time.time() < deadline:
            try:
                resp = await client.get(f"{API_BASE}/ready", timeout=5.0)
                if resp.status_code == 200:
                    log.info("App ready")
                    return
            except Exception:
                pass
            await asyncio.sleep(READY_POLL_INTERVAL)
    raise RuntimeError(f"App did not become ready in {READY_TIMEOUT_SECONDS}s")


# ============================================================
# Running one ablation
# ============================================================

async def run_one_ablation(config: AblationConfig) -> dict:
    """Run both eval suites under one configuration. Returns combined results."""
    log.info("=" * 70)
    log.info("ABLATION: %s — %s", config.name, config.description)
    log.info("Env: %s", config.env_overrides)
    log.info("=" * 70)

    # 1. Update .env
    update_env(config.env_overrides)

    # 2. Restart app to pick up env
    restart_app()

    # 3. Wait for ready
    await wait_for_ready()

    # 4. Flush cache
    flush_redis()

    # 5. Retrieval eval
    log.info("Running retrieval eval for %s...", config.name)
    retrieval_result = await run_retrieval_eval()

    # 6. E2E eval
    log.info("Running e2e eval for %s...", config.name)
    e2e_result = await run_e2e_eval()

    return {
        "config": {
            "name": config.name,
            "description": config.description,
            "env_overrides": config.env_overrides,
        },
        "retrieval": retrieval_result,
        "e2e": e2e_result,
    }


# ============================================================
# Reporting
# ============================================================

def _format_report(all_results: list[dict]) -> str:
    """Build the final ablation comparison report."""
    lines: list[str] = []
    lines.append("# Ablation Study")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_\n")

    lines.append(
        "Each row is a different system configuration evaluated against the "
        "same retrieval and generation test sets. The baseline row is the "
        "production system; subsequent rows turn off one component at a time. "
        "Differences vs. baseline are the contribution of that component.\n"
    )

    # Comparison table
    lines.append("## Summary\n")
    lines.append(
        "| Configuration | HR@5 | MRR | nDCG@5 | Faithfulness | Relevance | Behavior |"
    )
    lines.append("|---|---|---|---|---|---|---|")

    for r in all_results:
        name = r["config"]["name"]
        retr = r["retrieval"]["overall"]
        e2e = r["e2e"]["overall"]

        lines.append(
            f"| {name} | "
            f"{retr['hit_rate_at_k']:.3f} | "
            f"{retr['mrr']:.3f} | "
            f"{retr['ndcg_at_k']:.3f} | "
            f"{e2e['faithfulness']:.3f} | "
            f"{e2e['answer_relevance']:.3f} | "
            f"{e2e['behavior_accuracy']:.3f} |"
        )
    lines.append("")

    # Per-config detail
    lines.append("## Per-configuration detail\n")
    for r in all_results:
        cfg = r["config"]
        lines.append(f"### {cfg['name']}")
        lines.append(f"_{cfg['description']}_\n")
        lines.append("Env overrides:")
        for k, v in cfg["env_overrides"].items():
            lines.append(f"- `{k}={v}`")
        lines.append("")

        retr = r["retrieval"]["overall"]
        e2e = r["e2e"]["overall"]
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Hit Rate@5 | {retr['hit_rate_at_k']:.3f} |")
        lines.append(f"| MRR | {retr['mrr']:.3f} |")
        lines.append(f"| nDCG@5 | {retr['ndcg_at_k']:.3f} |")
        lines.append(
            f"| Faithfulness | {e2e['faithfulness']:.3f} "
            f"(n={e2e['faithfulness_n']}) |"
        )
        lines.append(f"| Answer relevance | {e2e['answer_relevance']:.3f} |")
        lines.append(f"| Behavior accuracy | {e2e['behavior_accuracy']:.3f} |")
        lines.append("")

    lines.append("## How to read this\n")
    lines.append(
        "- **Drops vs. baseline** are the value of the disabled component.\n"
        "- **Identical numbers vs. baseline** mean the component made no "
        "measurable difference on this corpus + test set. Either the test "
        "doesn't exercise that component, or the corpus is too small to "
        "stress it.\n"
        "- **Improvements vs. baseline** would suggest the component is "
        "actively hurting on this data — worth investigating.\n"
    )

    return "\n".join(lines)


def write_report(all_results: list[dict]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"ablation_{timestamp}.md"
    path.write_text(_format_report(all_results), encoding="utf-8")
    return path


# ============================================================
# CLI entrypoint
# ============================================================

async def main() -> None:
    log.info("Ablation study starting — %d configurations", len(CONFIGURATIONS))
    start = time.time()

    all_results: list[dict] = []
    for cfg in CONFIGURATIONS:
        try:
            result = await run_one_ablation(cfg)
            all_results.append(result)
        except Exception as e:
            log.exception("Configuration %s FAILED: %s", cfg.name, e)
            # Continue to next configuration rather than aborting the whole run
            all_results.append({
                "config": {
                    "name": cfg.name,
                    "description": cfg.description,
                    "env_overrides": cfg.env_overrides,
                },
                "retrieval": {"overall": {
                    "hit_rate_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0,
                }},
                "e2e": {"overall": {
                    "faithfulness": 0.0, "faithfulness_n": 0,
                    "answer_relevance": 0.0, "behavior_accuracy": 0.0,
                }},
                "error": str(e),
            })

    # ---- Restore production defaults at the end ----
    log.info("Restoring baseline configuration in .env")
    update_env(CONFIGURATIONS[0].env_overrides)
    restart_app()
    try:
        await wait_for_ready()
    except Exception as e:
        log.warning("Baseline restoration check failed (you may need to restart manually): %s", e)

    # ---- Write the comparison report ----
    report_path = write_report(all_results)

    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print("Ablation study complete")
    print(f"  {len(all_results)} configurations evaluated")
    print(f"  Wall time: {elapsed/60:.1f} minutes")
    print(f"  Report: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())