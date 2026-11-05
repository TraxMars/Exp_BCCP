"""Re-test a run's saved Student without retraining (the counterpart of DuCiSC's test_3d.py).

  CUDA_VISIBLE_DEVICES=0 python code/test.py --run_dir Results_baseline_pl/seed_42/LA_8l [--nms]
Writes <run_dir>/retest_metrics.json: metrics without post-processing, plus an "nms1" entry
(largest connected component) with --nms.
"""
import argparse
import json
from pathlib import Path

import torch

from data import SPECS
from evaluate import ducisc_line, evaluate
from networks import VNetDS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--nms", action="store_true", help="Also report metrics after getLargestCC")
    parser.add_argument("--checkpoint", default="best_model.pth")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads((args.run_dir / "config.json").read_text())
    config = config.get("args", config)   # runs from before the merged code stored the arguments flat
    spec = SPECS[config["dataset"]]
    # Same convolution algorithms as during training, so a re-test reproduces test_metrics.json exactly.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = VNetDS(n_channels=1, n_classes=2, has_dropout=True).to(args.device)
    model.load_state_dict(torch.load(args.run_dir / args.checkpoint, map_location=args.device, weights_only=True))
    result = evaluate(model, Path(config["data_root"]) / spec["directory"], config["dataset"],
                      config.get("patch_size") or spec["patch"], args.device, split=args.split, nms1=args.nms)
    for name, entry in [("nms0", result)] + ([("nms1", result["nms1"])] if args.nms else []):
        print(f"{name}: {ducisc_line(entry)}; empty predictions {entry['empty_prediction_cases']}/{entry['cases']}")
    output = args.output or args.run_dir / "retest_metrics.json"
    output.write_text(json.dumps(dict(result, checkpoint=args.checkpoint), indent=2) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
