from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "toy_nc_cli"))

from src.toy_terminal import TerminalConfig, generate_episodes
from src.mamba_model import MambaBaseline, MambaConfig, action_kind_breakdown_mamba, evaluate_mamba


def run_experiment(
    name: str,
    condition_level: str,
    train_variant_indices_by_family=None,
    test_variant_indices_by_family=None,
    train_n: int = 90,
    test_n: int = 45,
    seed: int = 0,
) -> dict:
    print(f"[run_experiment] start name={name} condition={condition_level} seed={seed}", flush=True)
    config = TerminalConfig(rows=10, cols=40, context_width=32, patch_radius=1)
    families = ["pwd", "whoami", "echo_home", "date", "env_head", "python_arith"]
    train_eps = generate_episodes(
        train_n,
        config,
        noisy=False,
        seed=seed,
        context_mode="command",
        families=families,
        variant_indices_by_family=train_variant_indices_by_family,
    )
    test_eps = generate_episodes(
        test_n,
        config,
        noisy=False,
        seed=seed + 1,
        context_mode="command",
        families=families,
        variant_indices_by_family=test_variant_indices_by_family,
    )

    model = MambaBaseline(
        rows=config.rows,
        cols=config.cols,
        families=families,
        config=MambaConfig(epochs=40, hidden_dim=192, num_layers=3, context_width=32, seed=seed),
        condition_level=condition_level,
    )
    losses = model.fit(train_eps)
    metrics = evaluate_mamba(model, test_eps)
    breakdown = action_kind_breakdown_mamba(model, test_eps)

    result = {
        "name": name,
        "condition_level": condition_level,
        "train_n": train_n,
        "test_n": test_n,
        "seed": seed,
        "losses": losses,
        "metrics": metrics,
        "action_breakdown": breakdown,
    }
    print(f"[run_experiment] done name={name} metrics={metrics}", flush=True)
    return result


def main() -> int:
    out_dir = ROOT / "experiments" / "toy_nc_cli" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    paraphrase_families = ["pwd", "whoami", "echo_home", "date", "env_head"]
    train_variants = {family: (0,) for family in paraphrase_families}
    test_variants = {family: (1, 2) for family in paraphrase_families}

    results = {
        "standard_family": run_experiment("standard_family", "family", seed=301),
        "standard_command": run_experiment("standard_command", "command", seed=302),
        "paraphrase_family": run_experiment(
            "paraphrase_family",
            "family",
            train_variant_indices_by_family=train_variants,
            test_variant_indices_by_family=test_variants,
            seed=303,
        ),
        "paraphrase_command": run_experiment(
            "paraphrase_command",
            "command",
            train_variant_indices_by_family=train_variants,
            test_variant_indices_by_family=test_variants,
            seed=304,
        ),
    }

    out_path = out_dir / "mamba_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
