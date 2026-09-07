"""Run the proven decoder-pretraining loop with the deep prior encoder.

The original loop is executed unchanged after replacing only its model class.
This keeps loss definitions, AMP, validation, TensorBoard, and checkpoint
semantics identical while avoiding two diverging copies of a long trainer.
"""

import runpy
import sys
from pathlib import Path

import submodel.decoder.model as original_model_module

from submodel.deep_encoder.model import DecoderPretrainer


def main() -> None:
    original_model_module.DecoderPretrainer = DecoderPretrainer
    if "--output-root" not in sys.argv:
        sys.argv.extend(["--output-root", str(Path(__file__).resolve().parent)])
    runpy.run_module("submodel.decoder.train", run_name="__main__")


if __name__ == "__main__":
    main()
