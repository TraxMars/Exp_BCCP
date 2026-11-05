"""Deep-supervised CE + Dice for the labeled GT and for the EMA Teacher's pseudo-labels."""
import torch
import torch.nn.functional as F


def deep_ce_dice(outputs, targets):
    """Mean over scales of CE and foreground Dice; targets holds one label map per scale."""
    ce_terms, dice_terms = [], []
    for output, target in zip(outputs, targets, strict=True):
        ce_terms.append(F.cross_entropy(output, target))
        foreground = output.softmax(dim=1)[:, 1]
        intersection = 2 * (foreground * target).sum() + 1e-8
        denominator = foreground.sum() + target.sum() + 1e-8
        dice_terms.append(1 - intersection / denominator)
    ce, dice = torch.stack(ce_terms).mean(), torch.stack(dice_terms).mean()
    return (ce + dice) / 2, ce, dice


def supervised_loss(outputs, labels):
    return deep_ce_dice(outputs, [
        F.interpolate(labels[:, None].float(), size=output.shape[2:], mode="nearest")[:, 0].long()
        for output in outputs])


def pseudo_label_consistency(student_outputs, teacher_outputs):
    """DuCiSC's pseudo-label loss without its dynamic-threshold mask: each scale's own Teacher argmax."""
    return deep_ce_dice(student_outputs, [teacher.detach().argmax(dim=1) for teacher in teacher_outputs])


@torch.no_grad()
def update_ema(student, teacher, decay, global_step):
    decay = min(1 - 1 / (global_step + 1), decay)
    for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
        teacher_parameter.mul_(decay).add_(student_parameter, alpha=1 - decay)
    # Teacher BN buffers follow its own train-mode forward (DuCiSC policy); only parameters use EMA.
