"""Check the pseudo-label consistency against DuCiSC's get_loss, and that diagnostics never touch training.

  1. pseudo_label_consistency == DuCiSC get_loss(..., deepsup=True, if_pseudo=True,
     uncertain_area_all=None) as (ce + dice) / 2: values and gradients w.r.t. the student outputs,
     with VNetDS output shapes and one scale whose pseudo-label is entirely background.
  2. gradient_diagnostics equals separately back-propagated objectives and leaves every .grad None.
  3. The hidden GT cannot change parameter gradients; the boundary band ignores patch borders.
"""
import copy
from pathlib import Path
import sys

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))
sys.path.insert(0, str(PROJECT / "tests"))
from diagnostics import boundary_band, boundary_diagnostics, gradient_diagnostics
from ducisc_reference import get_loss
from networks import VNetDS
from objectives import pseudo_label_consistency, supervised_loss

TOLERANCE = 1e-6
LA_SHAPES = [(14, 14, 10), (28, 28, 20), (56, 56, 40), (112, 112, 80)]   # VNetDS heads, 112x112x80 patch


def random_outputs(device, generator):
    student = [torch.randn(2, 2, *shape, generator=generator).to(device).requires_grad_() for shape in LA_SHAPES]
    teacher = [torch.randn(2, 2, *shape, generator=generator).to(device) for shape in LA_SHAPES]
    teacher[0][:, 0] += 100   # coarsest scale: Teacher argmax is all background
    return student, teacher


def check_ducisc(device):
    generator = torch.Generator().manual_seed(0)
    for trial in range(3):
        student, teacher = random_outputs(device, generator)
        pseudo_labels = [torch.argmax(output, dim=1) for output in teacher]   # as in train_DuCiSC.py, line 284
        assert pseudo_labels[0].sum() == 0 and all(label.sum() > 0 for label in pseudo_labels[1:])
        ours, ce, dice = pseudo_label_consistency(student, teacher)
        grads_ours = torch.autograd.grad(ours, student)
        ce_ref, dice_ref = get_loss(student, pseudo_labels, deepsup=True, if_pseudo=True, uncertain_area_all=None)
        reference = (ce_ref + dice_ref) / 2
        grads_ref = torch.autograd.grad(reference, student)
        value_gap = max(abs(ours - reference).item(), abs(ce - ce_ref).item(), abs(dice - dice_ref).item())
        grad_gap = max((left - right).abs().max().item() for left, right in zip(grads_ours, grads_ref))
        print(f"  {device} trial {trial}: L_pl={ours.item():.8f} reference={reference.item():.8f} "
              f"max|value diff|={value_gap:.3e} max|grad diff|={grad_gap:.3e}")
        assert value_gap <= TOLERANCE and grad_gap <= TOLERANCE
    teacher_leaf = [output.clone().requires_grad_() for output in teacher]
    pseudo_label_consistency(student, teacher_leaf)[0].backward()
    assert all(output.grad is None for output in teacher_leaf)


def small_batch():
    torch.manual_seed(0)
    images = torch.randn(4, 1, 32, 32, 32)
    labels = (torch.rand(4, 32, 32, 32) > 0.6).long()
    return images, labels


def check_gradient_diagnostics():
    images, labels = small_batch()
    student = VNetDS(has_dropout=False).train()
    teacher = copy.deepcopy(student).requires_grad_(False)
    supervised = student(images[:2], deepsup=True)
    unlabeled = student(images[2:], deepsup=True)
    with torch.no_grad():
        targets = teacher(images[2:] + 0.1, deepsup=True)
    loss_sup = supervised_loss(supervised, labels[:2])[0]
    loss_cons = 0.3 * pseudo_label_consistency(unlabeled, targets)[0]
    parameters = {"b7": student.block_seven.conv[-3].weight, "b3": student.block_three.conv[-3].weight}
    row = gradient_diagnostics(loss_sup, loss_cons, parameters)
    assert all(parameter.grad is None for parameter in student.parameters())
    for name, parameter in parameters.items():
        loss_sup.backward(retain_graph=True)
        sup = parameter.grad.clone()
        student.zero_grad(set_to_none=True)
        loss_cons.backward(retain_graph=True)
        cons = parameter.grad.clone()
        student.zero_grad(set_to_none=True)
        torch.testing.assert_close(row[f"gnorm_sup_{name}"], sup.norm().item())
        torch.testing.assert_close(row[f"gnorm_cons_{name}"], cons.norm().item())
        torch.testing.assert_close(row[f"cos_{name}"],
                                   torch.nn.functional.cosine_similarity(sup.flatten(), cons.flatten(), dim=0).item())
    print(f"  {row}")


def check_hidden_gt():
    images, labels = small_batch()
    model = VNetDS(has_dropout=False).train()
    gradients = []
    for hidden in (labels[2:], 1 - labels[2:]):
        state = copy.deepcopy(model.state_dict())
        supervised = model(images[:2], deepsup=True)
        unlabeled = model(images[2:], deepsup=True)
        with torch.no_grad():
            targets = model(images[2:] * 0.9, deepsup=True)
        loss = supervised_loss(supervised, labels[:2])[0] + 0.3 * pseudo_label_consistency(unlabeled, targets)[0]
        row = boundary_diagnostics(unlabeled[-1], targets[-1], hidden)
        assert all(isinstance(value, float) for value in row.values())
        model.zero_grad(set_to_none=True)
        loss.backward()
        gradients.append([parameter.grad.clone() for parameter in model.parameters()])
        model.load_state_dict(state)
    for left, right in zip(*gradients):
        torch.testing.assert_close(left, right, rtol=0, atol=0)

    mask = torch.zeros(1, 20, 20, 20, dtype=torch.long)
    mask[:, 5:15, 5:15, 5:15] = 1
    band = boundary_band(mask, 2)
    assert band[0, 10, 10, 10] == 0 and band[0, 10, 10, 5] and band[0, 10, 10, 3] and not band[0, 10, 10, 2]
    full = torch.ones(1, 20, 20, 20, dtype=torch.long)
    assert not boundary_band(full, 2).any()


def main():
    torch.set_num_threads(4)
    print("1. pseudo_label_consistency vs DuCiSC get_loss(if_pseudo=True, uncertain_area_all=None)")
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        check_ducisc(device)
    print(f"PASS: values and student-output gradients agree within {TOLERANCE:g}; Teacher gets no gradient")
    print("2. gradient_diagnostics")
    check_gradient_diagnostics()
    print("PASS: probe norms / cosine equal separate backward passes; all parameter .grad stay None")
    check_hidden_gt()
    print("PASS: hidden GT leaves parameter gradients bitwise unchanged; band = dilation - erosion, borders ignored")


if __name__ == "__main__":
    main()
