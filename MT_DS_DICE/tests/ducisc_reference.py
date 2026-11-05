"""Verbatim copies from the original DuCiSC repository, used only as the test reference.

  Binary_dice_loss  DuCiSC-main/code/utils/losses.py, lines 4-8
                    (sha256 404792a5fec093617958f732967fcbdc95d60f444accfae318474f54c96f4725)
  convert_to_soft_label, custom_cross_entropy, get_loss
                    DuCiSC-main/code/train_DuCiSC.py, lines 100-158
                    (sha256 eb4aae2b123bedb9edd60d30d12f647c6c5baa9910911b7168f846d1881c4838)
  getLargestCC, var_all_case, test_all_case, test_single_case_first_output, calculate_metric_percase
                    DuCiSC-main/code/utils/test_patch.py, lines 12-16, 18-58, 60-105, 107-168, 236-242
                    (sha256 45e7807e078be9f3aabc1aae637a5601eab72f5bd8b16c36d701640ccc400436);
                    the only edit: `.cuda()` -> `.to(device)`, with the module-level `device` below.

The original binds the module-level name `dice_loss = losses.Binary_dice_loss` in its __main__ block
(line 260); the same binding is made below. Function bodies are unmodified.
"""
import h5py
import math
import nibabel as nib
import numpy as np
import os
from medpy import metric
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# ---- utils/test_patch.py, lines 12-16, 18-58, 60-105, 107-168, 236-242 ----
def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def var_all_case(model, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, dataset_name="LA"):
    if dataset_name == "LA":
        with open('./data/LA/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = ["./data/LA/2018LA_Seg_TrainingSet/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
    elif dataset_name == "Pancreas":
        with open('./data/Pancreas/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = ["./data/Pancreas/Pancreas_h5/" + item.replace('\n', '') + ".h5" for item in image_list]
    elif dataset_name == "Nurves":
        with open('./data/Nurves/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = ["./data/Nurves/Nurves_h5/" + item.replace('\n', '') + ".h5" for item in image_list]
    loader = tqdm(image_list)
    total_dice = 0.0
    total_jc = 0.0
    total_hd = 0.0
    total_asd = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        label[label<0] = 0
        prediction, score_map = test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if np.sum(prediction)==0:
            dice, jc, hd, asd = 0, 0, 0, 0
        else:
            dice = metric.binary.dc(prediction, label) # metric.binary.dc(prediction, label)
            jc = metric.binary.jc(prediction, label)
            hd = metric.binary.hd95(prediction, label)
            asd = metric.binary.asd(prediction, label)
        total_dice += dice
        total_jc += jc
        total_hd += hd
        total_asd += asd
    avg_dice = total_dice / len(image_list)
    avg_jc = total_jc / len(image_list)
    avg_hd = total_hd / len(image_list)
    avg_asd = total_asd / len(image_list)
    print(f'average dice: {avg_dice*100:.2f}, average jc: {avg_jc*100:.2f}, average hd: {avg_hd:.2f}, average asd: {avg_asd:.2f}')
    return avg_dice, avg_jc, avg_hd, avg_asd

def test_all_case(model_name, num_outputs, model, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None, metric_detail=1, nms=0):

    loader = tqdm(image_list) if not metric_detail else image_list
    ith = 0
    total_metric = {'dice' : [], 'jc': [], 'hd': [], 'asd': []}
    infer_info = []
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        name = os.path.basename(image_path.split(".")[0])
        image = h5f['image'][:]
        label = h5f['label'][:]
        label[label<0] = 0
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if nms:
            prediction = getLargestCC(prediction)
            
        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
        total_metric['dice'].append(single_metric[0])
        total_metric['jc'].append(single_metric[1])
        total_metric['hd'].append(single_metric[2])
        total_metric['asd'].append(single_metric[3])
        if metric_detail:
            info = f'case {ith}, dice: {single_metric[0]*100:.2f}, jc: {single_metric[1]*100:.2f}, hd: {single_metric[2]:.2f}, asd: {single_metric[3]:.2f}'
            print(info)
            infer_info.append(info)
        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path +  f"{ith}_pred_{single_metric[0]:.2f}.nii.gz")
            nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)), test_save_path +  f"{ith}_scores.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path +  f"{ith}_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + f"{ith}_gt.nii.gz")
        ith += 1

    final_metric = f'dice: {100*np.mean(total_metric["dice"]):.2f}({np.std(total_metric["dice"]):.2f}), jc: {100*np.mean(total_metric["jc"]):.2f}({np.std(total_metric["jc"]):.2f}), hd: {np.mean(total_metric["hd"]):.2f}({np.std(total_metric["hd"]):.2f}), asd: {np.mean(total_metric["asd"]):.2f}({np.std(total_metric["asd"]):.2f})'
    print(final_metric)
    infer_info.append(final_metric)
    
    with open(test_save_path+f'../{model_name}_performance_{nms}.txt', 'w') as f:
        for i in range(len(infer_info)):
            f.writelines(f'{infer_info[i]} \n')
    return final_metric


def test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).to(device)

                with torch.no_grad():
                    y = model(test_patch)
                    if len(y) > 1:
                        y = y[0]
                    y = F.softmax(y, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,1,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1

    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = (score_map[0]>0.5).astype(np.int32)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map

def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd
