from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .toy_terminal import TerminalConfig, generate_episodes
from .cell_model import CellUpdateModel, ModelConfig, evaluate_model, arithmetic_exact_match, action_kind_breakdown


@lru_cache(maxsize=64)
def fit_bundle(
    train_n: int,
    test_n: int,
    noisy_train: bool,
    noisy_test: bool,
    condition_level: str,
    context_mode: str,
    hidden_size: int,
    max_iter: int,
    negative_ratio: int,
    seed: int,
    rows: int = 10,
    cols: int = 40,
    context_width: int = 20,
    families: tuple[str, ...] | None = None,
    train_variant_indices_by_family: tuple[tuple[str, tuple[int, ...]], ...] | None = None,
    test_variant_indices_by_family: tuple[tuple[str, tuple[int, ...]], ...] | None = None,
) -> dict:
    config = TerminalConfig(rows=rows, cols=cols, context_width=context_width, patch_radius=1)
    family_list = None if families is None else list(families)
    train_variant_map = None if train_variant_indices_by_family is None else dict(train_variant_indices_by_family)
    test_variant_map = None if test_variant_indices_by_family is None else dict(test_variant_indices_by_family)
    train_eps = generate_episodes(
        train_n,
        config,
        noisy=noisy_train,
        seed=seed,
        context_mode=context_mode,
        families=family_list,
        variant_indices_by_family=train_variant_map,
    )
    test_eps = generate_episodes(
        test_n,
        config,
        noisy=noisy_test,
        seed=seed + 1,
        context_mode=context_mode,
        families=family_list,
        variant_indices_by_family=test_variant_map,
    )
    model = CellUpdateModel(
        config,
        ModelConfig(
            hidden_size=hidden_size,
            max_iter=max_iter,
            negative_ratio=negative_ratio,
            random_state=seed,
        ),
        condition_level,
    ).fit(train_eps)
    metrics = evaluate_model(model, test_eps)
    metrics["arithmetic_exact_match"] = arithmetic_exact_match(model, test_eps)
    action_breakdown = action_kind_breakdown(model, test_eps)
    return {
        "config": config,
        "train_eps": train_eps,
        "test_eps": test_eps,
        "model": model,
        "metrics": metrics,
        "action_breakdown": action_breakdown,
    }


@lru_cache(maxsize=16)
def conditioning_study(
    train_n: int,
    test_n: int,
    hidden_size: int,
    max_iter: int,
    negative_ratio: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for condition_level in ["none", "family", "command"]:
        bundle = fit_bundle(
            train_n=train_n,
            test_n=test_n,
            noisy_train=False,
            noisy_test=False,
            condition_level=condition_level,
            context_mode="command",
            hidden_size=hidden_size,
            max_iter=max_iter,
            negative_ratio=negative_ratio,
            seed=seed,
        )
        rows.append({"conditioning": condition_level, **bundle["metrics"]})
    return pd.DataFrame(rows)


@lru_cache(maxsize=16)
def conditioning_study_multiseed(
    train_n: int,
    test_n: int,
    hidden_size: int,
    max_iter: int,
    negative_ratio: int,
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        frame = conditioning_study(train_n, test_n, hidden_size, max_iter, negative_ratio, seed)
        frame = frame.copy()
        frame["seed"] = seed
        rows.append(frame)
    raw = pd.concat(rows, ignore_index=True)
    summary = (
        raw.groupby("conditioning", as_index=False)
        .agg(
            char_acc_mean=("char_acc", "mean"),
            char_acc_std=("char_acc", "std"),
            changed_acc_mean=("changed_acc", "mean"),
            changed_acc_std=("changed_acc", "std"),
            exact_line_acc_mean=("exact_line_acc", "mean"),
            exact_line_acc_std=("exact_line_acc", "std"),
            arith_exact_mean=("arithmetic_exact_match", "mean"),
            runs=("seed", "count"),
        )
        .fillna(0.0)
    )
    return summary


@lru_cache(maxsize=16)
def noise_study(
    train_n: int,
    test_n: int,
    hidden_size: int,
    max_iter: int,
    negative_ratio: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for noisy_train, noisy_test in [(False, False), (True, False), (False, True), (True, True)]:
        bundle = fit_bundle(
            train_n=train_n,
            test_n=test_n,
            noisy_train=noisy_train,
            noisy_test=noisy_test,
            condition_level="command",
            context_mode="command",
            hidden_size=hidden_size,
            max_iter=max_iter,
            negative_ratio=negative_ratio,
            seed=seed,
        )
        rows.append(
            {
                "train_data": "noisy" if noisy_train else "clean",
                "test_data": "noisy" if noisy_test else "clean",
                "setting": f"{'noisy' if noisy_train else 'clean'}→{'noisy' if noisy_test else 'clean'}",
                **bundle["metrics"],
            }
        )
    return pd.DataFrame(rows)


@lru_cache(maxsize=16)
def noise_study_multiseed(
    train_n: int,
    test_n: int,
    hidden_size: int,
    max_iter: int,
    negative_ratio: int,
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        frame = noise_study(train_n, test_n, hidden_size, max_iter, negative_ratio, seed)
        frame = frame.copy()
        frame["seed"] = seed
        rows.append(frame)
    raw = pd.concat(rows, ignore_index=True)
    summary = (
        raw.groupby(["setting", "train_data", "test_data"], as_index=False)
        .agg(
            char_acc_mean=("char_acc", "mean"),
            char_acc_std=("char_acc", "std"),
            changed_acc_mean=("changed_acc", "mean"),
            changed_acc_std=("changed_acc", "std"),
            exact_line_acc_mean=("exact_line_acc", "mean"),
            exact_line_acc_std=("exact_line_acc", "std"),
            arith_exact_mean=("arithmetic_exact_match", "mean"),
            runs=("seed", "count"),
        )
        .fillna(0.0)
    )
    return summary


@lru_cache(maxsize=16)
def paraphrase_generalization_study(
    train_n: int,
    test_n: int,
    hidden_size: int,
    max_iter: int,
    negative_ratio: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    families = ("pwd", "whoami", "echo_home", "date", "env_head")
    train_variants = tuple((family, (0,)) for family in families)
    test_variants = tuple((family, (1, 2)) for family in families)
    for condition_level in ["none", "family", "command"]:
        bundle = fit_bundle(
            train_n=train_n,
            test_n=test_n,
            noisy_train=False,
            noisy_test=False,
            condition_level=condition_level,
            context_mode="command",
            hidden_size=hidden_size,
            max_iter=max_iter,
            negative_ratio=negative_ratio,
            seed=seed,
            families=families,
            train_variant_indices_by_family=train_variants,
            test_variant_indices_by_family=test_variants,
        )
        breakdown = bundle["action_breakdown"]
        rows.append(
            {
                "conditioning": condition_level,
                "overall_changed_acc": bundle["metrics"]["changed_acc"],
                "typing_changed_acc": breakdown.get("type_char", {}).get("changed_acc", float("nan")),
                "enter_changed_acc": breakdown.get("enter", {}).get("changed_acc", float("nan")),
                "exact_line_acc": bundle["metrics"]["exact_line_acc"],
            }
        )
    return pd.DataFrame(rows)


@lru_cache(maxsize=16)
def paraphrase_generalization_multiseed(
    train_n: int,
    test_n: int,
    hidden_size: int,
    max_iter: int,
    negative_ratio: int,
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        frame = paraphrase_generalization_study(train_n, test_n, hidden_size, max_iter, negative_ratio, seed)
        frame = frame.copy()
        frame["seed"] = seed
        rows.append(frame)
    raw = pd.concat(rows, ignore_index=True)
    summary = (
        raw.groupby("conditioning", as_index=False)
        .agg(
            overall_changed_acc_mean=("overall_changed_acc", "mean"),
            overall_changed_acc_std=("overall_changed_acc", "std"),
            typing_changed_acc_mean=("typing_changed_acc", "mean"),
            typing_changed_acc_std=("typing_changed_acc", "std"),
            enter_changed_acc_mean=("enter_changed_acc", "mean"),
            enter_changed_acc_std=("enter_changed_acc", "std"),
            exact_line_acc_mean=("exact_line_acc", "mean"),
            exact_line_acc_std=("exact_line_acc", "std"),
            runs=("seed", "count"),
        )
        .fillna(0.0)
    )
    return summary
