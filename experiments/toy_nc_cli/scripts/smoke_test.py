from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "experiments" / "toy_nc_cli" / "src"
sys.path.insert(0, str(SRC.parent))

from src.toy_terminal import TerminalConfig, generate_episodes, encode_screen, decode_screen
from src.cell_model import CellUpdateModel, ModelConfig, evaluate_model, arithmetic_exact_match


def main() -> int:
    config = TerminalConfig(rows=10, cols=40, context_width=20, patch_radius=1)

    eps_a = generate_episodes(n=4, config=config, noisy=False, seed=123, context_mode="command")
    eps_b = generate_episodes(n=4, config=config, noisy=False, seed=123, context_mode="command")
    determinism_ok = all(
        (a.family == b.family)
        and (a.command_text == b.command_text)
        and all((fa == fb).all() for fa, fb in zip(a.frames, b.frames))
        for a, b in zip(eps_a, eps_b)
    )

    sample = eps_a[0].frames[0]
    roundtrip_ok = (decode_screen(encode_screen(sample)) == sample).all()

    train = generate_episodes(n=10, config=config, noisy=False, seed=7, context_mode="command")
    test = generate_episodes(n=12, config=config, noisy=False, seed=17, context_mode="command")

    model = CellUpdateModel(config, ModelConfig(hidden_size=128, max_iter=60, negative_ratio=12, random_state=0), "command")
    model.fit(train[:8])

    copy_overfit = evaluate_model(None, train[:8], baseline=True)
    learned_overfit = evaluate_model(model, train[:8])
    heldout = evaluate_model(model, test)
    heldout_copy = evaluate_model(None, test, baseline=True)
    arith = arithmetic_exact_match(model, test)

    result = {
        "determinism_ok": bool(determinism_ok),
        "roundtrip_ok": bool(roundtrip_ok),
        "copy_overfit": copy_overfit,
        "learned_overfit": learned_overfit,
        "heldout_copy": heldout_copy,
        "heldout": heldout,
        "heldout_arithmetic_exact_match": arith,
    }

    out_dir = ROOT / "experiments" / "toy_nc_cli" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoke_test.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
