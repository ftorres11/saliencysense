# -*- coding: utf-8 -*-

# Torch Imports
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# CASME imports
from casme.stats import AverageMeter, StatisticsContainer
from casme.model_basics import (get_masks_and_check_predictions,
                                BoxCoords, classification_accuracy,
                                binarize_mask, get_rectangular_mask)
from casme.utils.torch_utils import ImagePathDataset
import casme.tasks.imagenet.utils as imagenet_utils

# Package imports
import cv2
import numpy as np
import pandas as pd

import os
import time
import pdb
#import pyutils.io as io

AXIS_RANGE = np.arange(224)

# ========================================================================
def mask_to_bbox(mask, axis_range=AXIS_RANGE):
    x_range = mask.any(axis=0)
    y_range = mask.any(axis=1)
    img_x_range = axis_range[x_range]
    img_y_range = axis_range[y_range]
    if len(img_x_range) and len(img_y_range):
        bbox = BoxCoords(
            xmin=int(img_x_range[0]),
            xmax=int(img_x_range[-1] + 1),
            ymin=int(img_y_range[0]),
            ymax=int(img_y_range[-1] + 1),
        )
    else:
        bbox = BoxCoords(
            xmin=0,
            xmax=224,
            ymin=0,
            ymax=224,
        )
    return bbox


def masks_to_bboxes(masks, axis_range=AXIS_RANGE):
    bbox_ls = []
    for i in range(masks.shape[0]):
        bbox_ls.append(mask_to_bbox(
            mask=masks[i],
            axis_range=axis_range,
        ))
    return bbox_ls


def score(args, model, data_loader, bboxes, original_classifier,
          record_bboxes=False):
    if 'special' in model.keys():
        print("=> Special mode evaluation: {}.".format(model['special']))

    # setup meters
    batch_time = 0
    data_time = 0
    f1_meter = AverageMeter()
    f1a_meter = AverageMeter()
    le_meter = AverageMeter()
    om_meter = AverageMeter()
    sm_meter = AverageMeter()
    sm1_meter = AverageMeter()
    sm2_meter = AverageMeter()
    sm_acc_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    binarized_meter = AverageMeter()
    statistics = StatisticsContainer()

    end = time.time()
    candidate_bbox_ls = []

    # data loop
    for i, ((input_, target), paths) in enumerate(data_loader):
        if i > len(data_loader)*args.pot:
            break

        data_time += time.time() - end

        # compute continuous mask, rectangular mask and compare class predictions with targets
        if 'special' in model.keys():
            is_correct = target.ge(0).numpy()
            if model['special'] == 'max':
                continuous = binarized = np.ones((args.batch_size, 224, 224))
                rectangular = continuous
                bbox_coords = [BoxCoords(0, 224, 0, 224)] * len(target)
            elif model['special'] == 'min':
                continuous = binarized = np.zeros((args.batch_size, 224, 224))
                rectangular = continuous
                bbox_coords = [BoxCoords(0, 0, 0, 0)] * len(target)
            elif model['special'] == 'center':
                continuous = binarized = np.zeros((args.batch_size, 224, 224))
                continuous[:, 33:-33, 33:-33] = 1
                rectangular = continuous
                bbox_coords = [BoxCoords(33, 224-33, 33, 224-33)] * len(target)
            elif model['special'] == 'none':
                continuous = binarized = np.zeros((args.batch_size, 224, 224))
                rectangular = continuous
                bbox_coords = [BoxCoords(0, 0, 0, 0)] * len(target)
            elif model['special'] == 'ground_truth':
                # Special handling later
                rectangular = continuous = binarized = np.zeros((args.batch_size, 224, 224))
                bbox_coords = [None] * len(target)
            elif model['special'] == 'external':
                # Special handling later
                rectangular = continuous = binarized = np.zeros((args.batch_size, 224, 224))
                bbox_coords = [BoxCoords.from_dict(model["bboxes"][get_path_stub(path)]) for path in paths]
                for j, bbox_coord in enumerate(bbox_coords):
                    rectangular[j, bbox_coord.yslice, bbox_coord.xslice] = 1
            elif model['special'] == 'grad_cam':
                continuous, binarized, rectangular, is_correct, bbox_coords = get_torchray_saliency(
                    original_classifier=original_classifier,
                    input_=input_, target=target,
                    method="grad_cam",
                )
            elif model['special'] == 'guided_backprop':
                continuous, binarized, rectangular, is_correct, bbox_coords = get_torchray_saliency(
                    original_classifier=original_classifier,
                    input_=input_, target=target,
                    method="guided_backprop",
                )
            elif model['special'] == 'optimal_cam':
                cam = model['cam']
                continuous, binarized, rectangular, is_correct, bbox_coords = get_optimal_saliency(
                    original_classifier=original_classifier,
                    input_=input_, target=target,
                    cam = cam,
                )
            elif model['special'] == 'grad_cam_plusplus':
                cam = model['cam']
                continuous, binarized, rectangular, is_correct, bbox_coords = get_pytorch_saliency(
                    original_classifier=original_classifier,
                    input_=input_, target=target,
                    cam=cam,
                )
            elif model['special'] == 'score_cam':
                cam = model['cam']
                continuous, binarized, rectangular, is_correct, bbox_coords = get_pytorch_saliency(
                    original_classifier=original_classifier,
                    input_=input_, target=target,
                    cam=cam,
                )
            else:
                raise KeyError(model["special"])
        else:
            continuous, binarized, rectangular, is_correct, bbox_coords, classifier_outputs = \
                get_masks_and_check_predictions(
                    input_=input_, target=target, model=model,
                    use_p=args.use_p,
                )
            acc1, acc5 = classification_accuracy(classifier_outputs, target.to(device), topk=(1, 5))
            top1_meter.update(acc1.item(), n=target.shape[0])
            top5_meter.update(acc5.item(), n=target.shape[0])
            if record_bboxes:
                candidate_bbox_ls += masks_to_bboxes(rectangular)

        # update statistics
        statistics.update(torch.tensor(continuous).unsqueeze(1))
        binarized_meter.update(binarized.reshape(target.shape[0], -1).mean(-1).mean(), n=target.shape[0])

        # image loop
        for idx, path in enumerate(paths):
            gt_boxes = get_image_bboxes(bboxes_dict=bboxes, path=path)

            # compute localization metrics
            f1s_for_image = []
            ious_for_image = []
            for gt_box in gt_boxes:
                if model.get('special') == 'ground_truth':
                    gt_mask = np.zeros([224, 224])
                    truncated_gt_box = gt_box.clamp(0, 224)
                    gt_mask[truncated_gt_box.yslice, truncated_gt_box.xslice] = 1
                    f1_for_box, iou_for_box = get_loc_scores(gt_box, gt_mask, gt_mask)
                else:
                    f1_for_box, iou_for_box = get_loc_scores(gt_box, continuous[idx], rectangular[idx])

                f1s_for_image.append(f1_for_box)
                ious_for_image.append(iou_for_box)

            f1_meter.update(np.array(f1s_for_image).max())
            f1a_meter.update(np.array(f1s_for_image).mean())
            le_meter.update(1 - np.array(ious_for_image).max())
            om_meter.update(1 - (np.array(ious_for_image).max() * is_correct[idx]))

        if model.get('special') == 'ground_truth':
            saliency_metric, sm1_ls, sm2_ls = compute_saliency_metric_ground_truth(
                input_=input_,
                target=target,
                bboxes=bboxes,
                paths=paths,
                classifier=original_classifier,
            )
            sm_acc = None
        else:
            saliency_metric, sm1_ls, sm2_ls, sm_acc = compute_saliency_metric(
                input_=input_,
                target=target,
                bbox_coords=bbox_coords,
                classifier=original_classifier,
            )

        for sm, sm1, sm2 in zip(saliency_metric, sm1_ls, sm2_ls):
            sm_meter.update(sm)
            sm1_meter.update(sm1)
            sm2_meter.update(sm2)
        if sm_acc is not None:
            sm_acc_meter.update(sm_acc, n=len(saliency_metric))

        # measure elapsed time
        batch_time += time.time() - end
        end = time.time()

        del saliency_metric, continuous, binarized, rectangular, is_correct, bbox_coords, f1s_for_image, ious_for_image, sm1_ls, sm2_ls, gt_boxes

        # print log
        if i % args.print_freq == 0 and i > 0:
            print('[{0}/{1}]\t'
                  'Time {batch_time:.3f}\t'
                  'Data {data_time:.3f}\n'
                  'F1 {F1.avg:.3f} ({F1.val:.3f})\t'
                  'F1a {F1a.avg:.3f} ({F1a.val:.3f})\t'
                  'OM {OM.avg:.3f} ({OM.val:.3f})\t'
                  'LE {LE.avg:.3f} ({LE.val:.3f})\n'
                  'SM {SM.avg:.3f} ({SM.val:.3f})\t'
                  'SM1 {SM1.avg:.3f} ({SM1.val:.3f})\t'
                  'SM2 {SM2.avg:.3f} ({SM2.val:.3f})\t'
                  ''.format(
                      i, len(data_loader), batch_time=batch_time, data_time=data_time,
                      F1=f1_meter, F1a=f1a_meter, OM=om_meter, LE=le_meter, SM=sm_meter,
                      SM1=sm1_meter, SM2=sm2_meter), flush=True)
            statistics.print_out()

    print('Final:\t'
          'Time {batch_time:.3f}\t'
          'Data {data_time:.3f}\n'
          'F1 {F1.avg:.3f} ({F1.val:.3f})\t'
          'F1a {F1a.avg:.3f} ({F1a.val:.3f})\t'
          'OM {OM.avg:.3f} ({OM.val:.3f})\t'
          'LE {LE.avg:.3f} ({LE.val:.3f})\n'
          'SM {SM.avg:.3f} ({SM.val:.3f})\t'
          'SM1 {SM1.avg:.3f} ({SM1.val:.3f})\t'
          'SM2 {SM2.avg:.3f} ({SM2.val:.3f})\t'
          ''.format(
                batch_time=batch_time, data_time=data_time, F1=f1_meter, F1a=f1a_meter, OM=om_meter,
                LE=le_meter, SM=sm_meter, SM1=sm1_meter, SM2=sm2_meter), flush=True)
    statistics.print_out()

    results = {
        'F1': f1_meter.avg,
        'F1a': f1a_meter.avg,
        'OM': om_meter.avg,
        'LE': le_meter.avg,
        'SM': sm_meter.avg,
        'SM1': sm1_meter.avg,
        'SM2': sm2_meter.avg,
        'top1': top1_meter.avg,
        'top5': top5_meter.avg,
        'sm_acc': sm_acc_meter.avg,
        'binarized': binarized_meter.avg,
        **statistics.get_dictionary(),
    }
    return results, candidate_bbox_ls


def get_loc_scores(bbox, continuous_mask, rectangular_mask):
    if bbox.area == 0:
        return 0, 0

    truncated_bbox = bbox.clamp(0, 224)

    gt_box = np.zeros((224, 224))
    gt_box[truncated_bbox.yslice, truncated_bbox.xslice] = 1

    f1 = compute_f1(continuous_mask, gt_box, bbox.area)
    iou = compute_iou(rectangular_mask, gt_box, bbox.area)

    return f1, 1*(iou > 0.5)


def clip(x, a, b):
    if x < a:
        return a
    if x > b:
        return b

    return x


def compute_f1(m, gt_box, gt_box_size):
    with torch.no_grad():
        inside = (m*gt_box).sum()
        precision = inside / (m.sum() + 1e-6)
        recall = inside / gt_box_size

        return (2 * precision * recall)/(precision + recall + 1e-6)


def compute_iou(m, gt_box, gt_box_size):
    with torch.no_grad():
        intersection = (m*gt_box).sum()
        return intersection / (m.sum() + gt_box_size - intersection)


def compute_saliency_metric(input_, target, bbox_coords, classifier):
    resized_sliced_input_ls = []
    area_ls = []
    for i, bbox in enumerate(bbox_coords):
        sliced_input_single = imagenet_utils.tensor_to_image_arr(input_[i, :, bbox.yslice, bbox.xslice])
        if bbox.area > 0 and sliced_input_single.shape[0]*sliced_input_single.shape[1] > 0:
            resized_sliced_input_single = cv2.resize(
                sliced_input_single,
                (224, 224),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            resized_sliced_input_single = np.zeros([224, 224, 3])
        area_ls.append(bbox.area)
        resized_sliced_input_ls.append(resized_sliced_input_single)
    resized_input = torch.tensor(np.moveaxis(np.array(resized_sliced_input_ls), 3, 1)).float()
    with torch.no_grad():
        cropped_upscaled_yhat = classifier(resized_input.to(device))
    term_1 = (torch.tensor(area_ls).float() / (224 * 224)).clamp(0.05, 1).log().numpy()
    term_2 = torch.softmax(cropped_upscaled_yhat, dim=-1).detach().cpu()[
        torch.arange(cropped_upscaled_yhat.size(0)), target].log().numpy()
    saliency_metric = term_1 - term_2
    # Note: not reduced

    # Slightly redundant, but doing to validate we're computing things correctly
    acc1, = classification_accuracy(cropped_upscaled_yhat, target.to(device), topk=(1,))

    return saliency_metric, term_1, term_2, acc1.item()


def compute_saliency_metric_ground_truth(input_, target, bboxes, paths, classifier):
    # We need this because there are multiple ground truth bounding boxes per image
    resized_sliced_input_ls = []
    area_ls = []
    img_idx_ls = []
    target_ls = []
    for i, path in enumerate(paths):
        for bbox in get_image_bboxes(bboxes, paths[i]):
            bbox = bbox.clamp(0, 224)
            sliced_input_single = imagenet_utils.tensor_to_image_arr(input_[i, :, bbox.yslice, bbox.xslice])
            # if bbox.area > 1:  # minimum is set to 1 pixel because of inclusive boundaries
            if bbox.area > 0:
                resized_sliced_input_single = cv2.resize(
                    sliced_input_single,
                    (224, 224),
                    interpolation=cv2.INTER_CUBIC,
                )
            else:
                resized_sliced_input_single = np.zeros([224, 224, 3])
            area_ls.append(bbox.area)
            resized_sliced_input_ls.append(resized_sliced_input_single)
            img_idx_ls.append(i)
            target_ls.append(target[i])
    resized_input = torch.tensor(np.moveaxis(np.array(resized_sliced_input_ls), 3, 1)).float()
    with torch.no_grad():
        cropped_upscaled_yhat = classifier(resized_input.to(device), return_intermediate=False)
    term_1 = (torch.tensor(area_ls).float() / (224 * 224)).clamp(0.05, 1).log()
    term_2 = torch.softmax(cropped_upscaled_yhat, dim=-1).detach().cpu()[
        torch.arange(cropped_upscaled_yhat.size(0)), torch.tensor(target_ls)].log()
    df = pd.DataFrame({
        "term_1": term_1.numpy(),
        "term_2": term_2.numpy(),
        "img_idx": img_idx_ls,
    })
    averaged_within_img_df = df.groupby("img_idx").mean().sort_index()
    saliency_metric = (averaged_within_img_df["term_1"] - averaged_within_img_df["term_2"]).values
    return saliency_metric, averaged_within_img_df["term_1"].values, averaged_within_img_df["term_2"].values


def get_image_bboxes(bboxes_dict, path):
    ls = []
    for bbox in bboxes_dict[get_path_stub(path)]:
        ls.append(BoxCoords.from_dict(bbox))
    return ls


def get_path_stub(path):
    return os.path.basename(path).split('.')[0]
