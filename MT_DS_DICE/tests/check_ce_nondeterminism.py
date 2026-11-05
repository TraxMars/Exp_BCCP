"""Show that CUDA F.cross_entropy (mean reduction, 3D) is not bitwise deterministic.

Its forward uses atomic accumulation, so the logged CE can differ by about one ULP between
identical runs. The CE gradient does not depend on that value, so MT_DS trajectories stay
bitwise identical; under LossNorm (gradient / L) the ULP noise enters the update and runs diverge.
"""
import torch
import torch.nn.functional as F


def main():
    torch.manual_seed(0)
    logits = torch.randn(2, 2, 112, 112, 80, device="cuda")
    target = (torch.rand(2, 112, 112, 80, device="cuda") > 0.7).long()
    fused = {F.cross_entropy(logits, target).item() for _ in range(200)}
    elementwise = {F.cross_entropy(logits, target, reduction="none").mean().item() for _ in range(200)}
    print(f"F.cross_entropy (mean), 200 repeats: {len(fused)} distinct values {sorted(fused)}")
    print(f"F.cross_entropy(reduction='none').mean(), 200 repeats: {len(elementwise)} distinct values")
    torch.use_deterministic_algorithms(True)
    try:
        F.cross_entropy(logits, target)
    except RuntimeError as error:
        print("torch deterministic mode:", str(error).splitlines()[0][:120])


if __name__ == "__main__":
    main()
