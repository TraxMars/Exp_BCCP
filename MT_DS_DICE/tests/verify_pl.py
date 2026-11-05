"""Check the pseudo-label consistency against DuCiSC's get_loss, and that diagnostics never touch training.

  1. pseudo_label_consistency == DuCiSC get_loss(..., deepsup=True, if_pseudo=True,
     uncertain_area_all=None) as (ce + dice) / 2: values and gradients w.r.t. the student outputs,
     with VNetDS output shapes and one scale whose pseudo-label is entirely background.
  2. gradient_diagnostics equals separately back-propagated objectives and leaves every .grad None.
  3. The hidden GT cannot change parameter gradients; the boundary band ignores patch borders.
"""
import copy
import math
from pathlib import Path
import sys

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))
sys.path.insert(0, str(PROJECT / "tests"))
from diagnostics import boundary_band, boundary_diagnostics, gradient_diagnostics, labeled_band_ce
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
    loss_pl = 0.3 * pseudo_label_consistency(unlabeled, targets)[0]
    parameters = {"b7": student.block_seven.conv[-3].weight, "b3": student.block_three.conv[-3].weight}
    row = gradient_diagnostics(loss_sup, loss_pl, parameters)
    assert all(parameter.grad is None for parameter in student.parameters())
    for name, parameter in parameters.items():
        loss_sup.backward(retain_graph=True)
        sup = parameter.grad.clone()
        student.zero_grad(set_to_none=True)
        loss_pl.backward(retain_graph=True)
        pl = parameter.grad.clone()
        student.zero_grad(set_to_none=True)
        torch.testing.assert_close(row[f"gnorm_sup_{name}"], sup.norm().item())
        torch.testing.assert_close(row[f"gnorm_pl_{name}"], pl.norm().item())
        torch.testing.assert_close(row[f"cos_{name}"],
                                   torch.nn.functional.cosine_similarity(sup.flatten(), pl.flatten(), dim=0).item())
    print(f"  {row}")


def check_hidden_gt():
    images, labels = small_batch()
    model = VNetDS(has_dropout=False).train()
    gradients, rows = [], []
    for hidden in (labels[2:], 1 - labels[2:]):   # the hidden GT and its inverse
        state = copy.deepcopy(model.state_dict())
        supervised = model(images[:2], deepsup=True)
        unlabeled = model(images[2:], deepsup=True)
        with torch.no_grad():
            targets = model(images[2:] * 0.9, deepsup=True)
        loss_sup = supervised_loss(supervised, labels[:2])[0]
        loss_pl = pseudo_label_consistency(unlabeled, targets)[0]
        loss = loss_sup + 0.3 * loss_pl
        model.zero_grad(set_to_none=True)
        # As in train.py: oracle L_true, gradient probes, then the no_grad boundary statistics.
        row = gradient_diagnostics(loss_sup, 0.3 * loss_pl, dict(
            b7=model.block_seven.conv[-3].weight, b3=model.block_three.conv[-3].weight),
            supervised_loss(unlabeled, hidden)[0])
        assert all(parameter.grad is None for parameter in model.parameters())
        row.update(boundary_diagnostics(unlabeled[-1], targets[-1], hidden), **labeled_band_ce(supervised[-1], labels[:2]))
        assert all(isinstance(value, float) for value in row.values())
        loss.backward()
        gradients.append([parameter.grad.clone() for parameter in model.parameters()])
        rows.append(row)
        model.load_state_dict(state)
    for left, right in zip(*gradients):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    # The oracle columns did see the GT (they change), the training gradient did not.
    assert rows[0]["cos_pl_true_b7"] != rows[1]["cos_pl_true_b7"] and rows[0]["gnorm_pl_b7"] == rows[1]["gnorm_pl_b7"]

    mask = torch.zeros(1, 20, 20, 20, dtype=torch.long)
    mask[:, 5:15, 5:15, 5:15] = 1
    band = boundary_band(mask, 2)
    assert band[0, 10, 10, 10] == 0 and band[0, 10, 10, 5] and band[0, 10, 10, 3] and not band[0, 10, 10, 2]
    full = torch.ones(1, 20, 20, 20, dtype=torch.long)
    assert not boundary_band(full, 2).any()


def check_oracle_gradient_columns():
    """Gradient-space oracle columns against an independent per-channel computation (scipy Spearman)."""
    from scipy.stats import spearmanr
    generator = torch.Generator().manual_seed(1)
    weight = torch.randn(16, 8, 3, 3, 3, generator=generator, requires_grad=True)
    g_sup, g_pl, g_true = (torch.randn(16, 8, 3, 3, 3, generator=generator) for _ in range(3))
    row = gradient_diagnostics((weight * g_sup).sum(), (weight * g_pl).sum(), dict(w=weight), (weight * g_true).sum())
    cos = torch.nn.functional.cosine_similarity
    pl_sup = [cos(g_pl[c].flatten(), g_sup[c].flatten(), dim=0).item() for c in range(16)]
    pl_true = [cos(g_pl[c].flatten(), g_true[c].flatten(), dim=0).item() for c in range(16)]
    flagged = [b for a, b in zip(pl_sup, pl_true) if a < 0]
    expected = dict(cos_pl_true_w=cos(g_pl.flatten(), g_true.flatten(), dim=0).item(),
                    cos_sup_true_w=cos(g_sup.flatten(), g_true.flatten(), dim=0).item(),
                    frac_ch_pl_true_neg_w=sum(v < 0 for v in pl_true) / 16, frac_ch_pl_sup_neg_w=sum(v < 0 for v in pl_sup) / 16,
                    ch_rank_corr_w=spearmanr(pl_sup, pl_true).statistic,
                    hit_rate_w=sum(v < 0 for v in flagged) / len(flagged))
    for key, value in expected.items():
        assert abs(row[key] - value) < 1e-5, (key, row[key], value)
    assert weight.grad is None
    same = gradient_diagnostics((weight * g_sup).sum(), (weight * g_sup).sum(), dict(w=weight), (weight * g_true).sum())
    assert math.isnan(same["hit_rate_w"]) and same["frac_ch_pl_sup_neg_w"] == 0   # no channel with cos(pl, sup) < 0


def check_new_columns():
    """Hand-computed values of the error / confidence-split and labeled-band columns."""
    hidden = torch.zeros(1, 12, 12, 12, dtype=torch.long)
    hidden[:, 3:9, 3:9, 3:9] = 1
    teacher = torch.zeros(1, 2, 12, 12, 12)
    teacher[:, 1] = 10.0 * (2 * hidden - 1)              # correct, confidence ~1
    teacher[:, 1, 3, 3:9, 3:9] = -0.1                     # one wrong face, confidence ~0.52
    teacher[:, 1, 8, 3:9, 3:9] = -10.0                    # one wrong face, confidence ~1
    band = boundary_band(hidden, 2)
    pseudo = teacher.argmax(dim=1)
    wrong = pseudo != hidden
    row = boundary_diagnostics(torch.randn(1, 2, 12, 12, 12), teacher, hidden)
    assert abs(row["teacher_err_gt_band"] - (wrong & band).sum().item() / band.sum().item()) < 1e-7
    assert wrong.sum() == 72 and (wrong & band).sum() == 72
    assert abs(row["conf_wrong_095"] - 0.5) < 1e-7 and row["conf_right_095"] == 1.0
    logits = torch.zeros(1, 2, 12, 12, 12)
    lab = labeled_band_ce(logits, hidden)
    assert abs(lab["ce_lab_in"] - math.log(2)) < 1e-6 and abs(lab["ce_lab_out"] - math.log(2)) < 1e-6
    empty = boundary_diagnostics(torch.randn(1, 2, 12, 12, 12), torch.zeros(1, 2, 12, 12, 12),
                                 torch.zeros(1, 12, 12, 12, dtype=torch.long))
    assert all(math.isnan(empty[key]) for key in ("ce_pl_in", "teacher_err_gt_band", "conf_wrong_095", "band_iou"))


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
    print("PASS: inverting the hidden GT leaves all parameter gradients bitwise unchanged (it does change the oracle "
          "columns); probes leave .grad None; band = dilation - erosion, borders ignored")
    check_new_columns()
    check_oracle_gradient_columns()
    print("PASS: cos_pl_true / cos_sup_true / frac_ch_* / ch_rank_corr / hit_rate match an independent per-channel "
          "computation (scipy Spearman); hit_rate is NaN without negative pl-sup channels")
    print("PASS: teacher_err_gt_band / conf_wrong_095 / conf_right_095 / ce_lab_in / ce_lab_out match hand-computed "
          "values; NaN when the patch has no foreground")


if __name__ == "__main__":
    main()
