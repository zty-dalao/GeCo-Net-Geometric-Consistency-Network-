"""Evaluate a deep-prior checkpoint with the original fixed-range metrics."""

import runpy
import sys
from pathlib import Path

import submodel.decoder.model as original_model_module

from submodel.deep_encoder.model import DecoderPretrainer


def _argument_value(name: str) -> str | None:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


def main() -> None:
    original_model_module.DecoderPretrainer = DecoderPretrainer
    if "--output-dir" not in sys.argv:
        checkpoint = _argument_value("--checkpoint")
        checkpoint_stem = Path(checkpoint).stem if checkpoint else "evaluation"
        output_dir = Path(__file__).resolve().parent / "metrics" / checkpoint_stem
        sys.argv.extend(["--output-dir", str(output_dir)])
    runpy.run_module("submodel.decoder.evaluate_metrics", run_name="__main__")


if __name__ == "__main__":
    main()
