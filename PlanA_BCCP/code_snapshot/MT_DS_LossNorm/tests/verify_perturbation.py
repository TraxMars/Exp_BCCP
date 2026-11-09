"""Check the step-2 decoder-tap perturbation branch."""
import copy
from pathlib import Path
import sys

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))
from networks import VNetDS
from objectives import consistency_loss
from perturbation import TAP_FIRST_HEAD, channel_dropout, frozen_bn_statistics


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = VNetDS(n_channels=1, n_classes=2, normalization="batchnorm", has_dropout=True)
    images = torch.randn(2, 1, 32, 32, 32)

    # Re-running the tail on the clean tap reproduces the remaining heads exactly.
    model.eval()
    with torch.no_grad():
        taps = {}
        outputs = model(images, deepsup=True, taps=taps)
        for tap, first in TAP_FIRST_HEAD.items():
            tail = model.decode_from(tap, taps[tap], taps)
            assert len(tail) == len(outputs) - first
            for left, right in zip(tail, outputs[first:]):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
    print("PASS: decode_from reproduces the D3/D2 tails exactly")

    # Whole channels are dropped per sample, survivors rescaled to keep C active on average.
    feature = torch.rand(64, 32, 4, 4, 4) + 0.5
    torch.manual_seed(1)
    perturbed = channel_dropout(feature, 0.5)
    ratio = perturbed / feature
    per_channel = ratio.flatten(2)
    torch.testing.assert_close(per_channel.min(-1).values, per_channel.max(-1).values)
    kept = per_channel[..., 0] > 0
    torch.testing.assert_close(per_channel[..., 0].sum(1), torch.full((64,), 32.0))
    assert 0.4 < kept.float().mean() < 0.6
    assert torch.equal(channel_dropout(feature, 0.0), feature)
    guided = channel_dropout(feature, torch.linspace(0, 0.9, 32).expand(64, 32))
    assert (guided.flatten(2)[..., 0] > 0).float().mean(0)[0] == 1
    print("PASS: channel dropout masks whole channels with CHAP rescaling")

    # The perturbed branch uses batch statistics but leaves running statistics untouched.
    model.train()
    taps = {}
    outputs = model(images, deepsup=True, taps=taps)
    buffers = [buffer.clone() for buffer in model.buffers()]
    momenta = [layer.momentum for layer in model.modules() if isinstance(layer, torch.nn.BatchNorm3d)]
    with frozen_bn_statistics(model):
        tail = model.decode_from("d2", channel_dropout(taps["d2"], 0.5), taps)
    for left, right in zip(buffers, model.buffers()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert momenta == [layer.momentum for layer in model.modules() if isinstance(layer, torch.nn.BatchNorm3d)]
    loss = consistency_loss(tail, [output.detach() + 0.1 for output in outputs[2:]])
    loss.backward()
    assert model.block_one.conv[0].weight.grad.abs().sum() > 0
    assert model.block_eight.conv[0].weight.grad.abs().sum() > 0
    print("PASS: frozen BN running statistics; perturbation loss reaches encoder and tail")
    print("All perturbation verification checks passed.")


if __name__ == "__main__":
    main()
