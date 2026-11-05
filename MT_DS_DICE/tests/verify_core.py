"""Check backbone equivalence, initialization order, deep supervision gradients and Mean Teacher."""
import argparse
import copy
import importlib.util
from pathlib import Path
import sys

import torch
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))
from networks import VNetDS
from objectives import pseudo_label_consistency, supervised_loss, update_ema
from evaluate import predict_volume


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, default=(
        PROJECT.parents[1] / "DuCiSC-main/DuCiSC-main/code/networks/VNet.py"))
    # The pre-simplification VNetDS (with normalization options), at tag archive/step1-step2.
    parser.add_argument("--legacy", type=Path, default=PROJECT.parent / "MT_DS/code/networks/vnet_ds.py")
    args = parser.parse_args()
    torch.set_num_threads(2)
    # Construction order decides which random numbers each layer's initialization consumes.
    for seed in (42, 500, 1337):
        torch.manual_seed(seed)
        legacy = load_module("legacy_vnet", args.legacy).VNetDS(
            n_channels=1, n_classes=2, normalization="batchnorm", has_dropout=True).state_dict()
        torch.manual_seed(seed)
        current = VNetDS(n_channels=1, n_classes=2, has_dropout=True).state_dict()
        assert list(legacy) == list(current)
        for key in legacy:
            torch.testing.assert_close(current[key], legacy[key], rtol=0, atol=0)
    print("PASS: identical parameter names, order and seeded initial weights as the legacy VNetDS")

    torch.manual_seed(42)
    original_module = load_module("original_vnet", args.original)
    original = original_module.DuCiSC(n_channels=1, n_classes=2, normalization="batchnorm").eval()
    model = VNetDS(n_channels=1, n_classes=2).eval()
    model.load_state_dict(original.state_dict(), strict=True)
    inputs = torch.randn(2, 1, 32, 32, 32)
    with torch.no_grad():
        expected, _ = original(inputs, deepsup=True)
        actual = model(inputs, deepsup=True)
        assert [tuple(x.shape[2:]) for x in actual] == [(4, 4, 4), (8, 8, 8), (16, 16, 16), (32, 32, 32)]
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(model(inputs), expected[-1], rtol=0, atol=0)
    print("PASS: exact original predictions and parameter keys at all four scales")
    del original, expected, actual

    teacher = copy.deepcopy(model).requires_grad_(False)
    # A teacher target passed with requires_grad must still be isolated by the loss.
    student_logits = [torch.randn(2, 2, 4, 4, 4, requires_grad=True)]
    teacher_logits = [torch.randn(2, 2, 4, 4, 4, requires_grad=True)]
    pseudo_label_consistency(student_logits, teacher_logits)[0].backward()
    assert student_logits[0].grad is not None and teacher_logits[0].grad is None
    print("PASS: the pseudo-label loss sends gradients only to the student")

    model.train()
    teacher.train()
    labels = torch.randint(0, 2, (2, 32, 32, 32))
    outputs = model(inputs, deepsup=True)
    with torch.no_grad():
        targets = teacher(inputs + 0.1, deepsup=True)
    supervised, _, _ = supervised_loss(outputs, labels)
    objective = supervised + 0.3 * pseudo_label_consistency(outputs, targets)[0]
    objective.backward()
    for head in (model.out_conv, model.out_conv_2, model.out_conv_3, model.out_conv_4):
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in teacher.parameters())
    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    update_ema(model, teacher, 0.99, 0)
    for left, right in zip(model.parameters(), teacher.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    print("PASS: supervised and pseudo-label gradients, all four heads, and first EMA update")

    class SignModel(torch.nn.Module):
        def forward(self, image):
            return torch.cat([-image, image], dim=1)

    for shape in ((9, 12, 7), (22, 25, 19)):
        image = np.random.default_rng(42).normal(size=shape).astype(np.float32)
        prediction = predict_volume(SignModel(), image, (16, 16, 16), (8, 8, 8), "cpu")
        np.testing.assert_array_equal(prediction, image > 0)
    print("PASS: sliding-window overlap, padding and restored volume dimensions")
    print("All core verification checks passed.")


if __name__ == "__main__":
    main()
