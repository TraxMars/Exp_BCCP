"""Verbatim copies from the original DuCiSC repository, used only as the test reference.

  Binary_dice_loss  DuCiSC-main/code/utils/losses.py, lines 4-8
                    (sha256 404792a5fec093617958f732967fcbdc95d60f444accfae318474f54c96f4725)
  convert_to_soft_label, custom_cross_entropy, get_loss
                    DuCiSC-main/code/train_DuCiSC.py, lines 100-158
                    (sha256 eb4aae2b123bedb9edd60d30d12f647c6c5baa9910911b7168f846d1881c4838)

The original binds the module-level name `dice_loss = losses.Binary_dice_loss` in its __main__ block
(line 260); the same binding is made below. Function bodies are unmodified.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- utils/losses.py, lines 4-8 ----
def Binary_dice_loss(predictive, target, ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss


# ---- train_DuCiSC.py, lines 100-158 ----
def convert_to_soft_label(deep_label):
    foreground_prob = deep_label.unsqueeze(1) 
    background_prob = 1 - foreground_prob  #[batch_size, 1, W, H, D]
    soft_label = torch.cat([background_prob, foreground_prob], dim=1)  # [batch_size, 2, W, H, D]
    return soft_label

def custom_cross_entropy(output, target, certain_index=None):
    target = convert_to_soft_label(target)
    log_output = F.log_softmax(output, dim=1)
    if certain_index is None:
        loss = F.kl_div(log_output, target, reduction="mean")
    else:
        # certain_index: [B, W, H, D], target: [B, 2, W, H, D]
        certain_index = certain_index.unsqueeze(1).repeat(1, 2, 1, 1, 1)
        loss = F.kl_div(log_output, target, reduction="none")[certain_index].mean()
    return loss

def get_loss(outputs, label_batch, deepsup, if_pseudo=False, if_soft=False, uncertain_area_all=None):
    if deepsup:
        loss_seg, loss_seg_dice = 0, 0
        for i in range(len(outputs)):
            output = outputs[i] # B C W H D
            if if_pseudo: # pseudo label
                deep_label = label_batch[i].long()
            elif if_soft: # soft label
                deep_label = nn.functional.interpolate(label_batch.float().unsqueeze(1), size=output.shape[2:], mode='trilinear').float()[:,0,:,:,:]
            else: # gt label
                deep_label = nn.functional.interpolate(label_batch.float().unsqueeze(1), size=output.shape[2:], mode='nearest').long()[:,0,:,:,:]
            # ce loss
            if if_pseudo: # pseudo label
                if uncertain_area_all is None:
                    loss_seg = loss_seg + F.cross_entropy(output, deep_label)
                else:
                    certain_index = (uncertain_area_all[i]==0)
                    loss_seg = loss_seg + F.cross_entropy(output, deep_label, reduction='none')[certain_index].mean()
            elif if_soft: # soft label
                if uncertain_area_all is None:
                    loss_seg = loss_seg + custom_cross_entropy(output, deep_label)
                else:
                    certain_index = (uncertain_area_all[i]==0)
                    loss_seg = loss_seg + custom_cross_entropy(output, deep_label, certain_index)
            else: # gt label
                loss_seg = loss_seg + F.cross_entropy(output, deep_label)
            
            # dice loss
            output_soft = F.softmax(output, dim=1)
            if uncertain_area_all is not None:
                certain_index = (uncertain_area_all[i]==0)
                loss_seg_dice = loss_seg_dice + dice_loss(output_soft[:, 1, :, :, :][certain_index], deep_label[certain_index])
            else:
                loss_seg_dice = loss_seg_dice + dice_loss(output_soft[:, 1, :, :, :], deep_label)
        loss_seg_dice = loss_seg_dice / len(outputs)
        loss_seg = loss_seg / len(outputs)
        return loss_seg, loss_seg_dice
    else:
        loss_seg = F.cross_entropy(outputs, label_batch)
        outputs_soft = F.softmax(outputs, dim=1)
        loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)
        return loss_seg, loss_seg_dice


# ---- train_DuCiSC.py, line 260 (module scope there, inside the __main__ block) ----
dice_loss = Binary_dice_loss
