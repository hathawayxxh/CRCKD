import os
import cv2
import sys
# from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import pandas as pd

import torch
from torch.nn import functional as F

from utils.metrics import compute_isic_metrics
from utils.metric_logger import MetricLogger
from utils.util import get_timestamp, get_att_maps, save_img, get_att_masked_feature, compute_test_relation_matrix


def epochVal_metrics(model, dataLoader):
    training = model.training
    model.eval()

    meters = MetricLogger()
    false_index = []

    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    att_pred = torch.FloatTensor().cuda()
    final_pred = torch.FloatTensor().cuda()
    
    gt_study, pred_study, att_pred_study, final_pred_study, studies = {}, {},  {}, {}, []

    input_path = "./visualization/split2/images/"
    saliency_path = "./visualization/split2/att_maps/"

    all_feature4, all_labels = None, None

    with torch.no_grad():
        batch_num = 0
        for i, (study, _, image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            feature3, feature4, SE_att4, _, output_b3, output, att_output, fmaps_b3, fmaps_b4 = model(image)
            output = F.softmax(output, dim=1) #[bs, n_classes]
            att_output = F.softmax(att_output, dim=1)
            final_output = (output + att_output)/2

            # if i == 0:
            # compute_test_relation_matrix(i, label, feature3, feature4, output)

            att4, _ = get_att_maps(fmaps_b4, norm="l2")
            get_att_masked_feature(att4, fmaps_b4, iter=i, mode='test')
            # print("features:", feature3.shape, feature4.shape)
            # print(label.shape)

            feature4 = feature4.cpu().numpy()
            all_feature4 = feature4 if all_feature4 is None else np.concatenate((all_feature4, feature4), axis=0)
            batch_label = np.reshape(np.argmax(label.cpu().numpy(), axis=1), (-1, 1))
            all_labels = batch_label if all_labels is None else np.concatenate((all_labels, batch_label), axis=0)

            att_b3, s_map3 = get_att_maps(fmaps_b3)
            att_b4, s_map4 = get_att_maps(fmaps_b4)
            SE_att4, SE_smap4 = get_att_maps(SE_att4)

            saliency_b3 = s_map3.repeat(1, 3, 1, 1)
            saliency_b4 = s_map4.repeat(1, 3, 1, 1)
            saliency_SE = SE_smap4.repeat(1, 3, 1, 1)

            for i in range(len(study)):
                # print(study[i])
                if study[i] in pred_study:
                    assert torch.equal(gt_study[study[i]], label[i])
                    pred_study[study[i]] = torch.max(pred_study[study[i]], output[i])
                    att_pred_study[study[i]] = torch.max(att_pred_study[study[i]], att_output[i])
                    final_pred_study[study[i]] = torch.max(final_pred_study[study[i]], final_output[i])
                else:
                    gt_study[study[i]] = label[i]
                    pred_study[study[i]] = output[i]
                    att_pred_study[study[i]] = att_output[i]
                    final_pred_study[study[i]] = final_output[i]
                    studies.append(study[i])

            for index in range(label.shape[0]):
                if output[index].argmax() != label[index].argmax():
                    false_index.append(batch_num * label.shape[0] + index)
                    # print(label[index].argmax(), output[index].argmax())
                # img = image[index].cpu().numpy().transpose((1, 2, 0))
                # att_map_b3 = cv2.resize(saliency_b3[index].cpu().numpy().transpose((1, 2, 0)), (224, 224))
                # att_map_b4 = cv2.resize(saliency_b4[index].cpu().numpy().transpose((1, 2, 0)), (224, 224))
                # att_map_SE = cv2.resize(saliency_SE[index].cpu().numpy().transpose((1, 2, 0)), (224, 224))
                # img_index = batch_num * 64 + index
                # # save_img(img, img_index, input_path, img_name='.jpg', mode="image")
                # # save_img(att_map_b3, img_index, saliency_path, img_name='_att3.jpg', mode="heatmap")
                # # save_img(att_map_b4, img_index, saliency_path, img_name='_att4.jpg', mode="heatmap")
                # save_img(att_map_SE, img_index, saliency_path, img_name='_SE_att4.jpg', mode="heatmap")

            batch_num += 1

        for study in studies:
            gt = torch.cat((gt, gt_study[study].view(1, -1)), 0)
            pred = torch.cat((pred, pred_study[study].view(1, -1)), 0)
            att_pred = torch.cat((att_pred, att_pred_study[study].view(1, -1)), 0)
            final_pred = torch.cat((final_pred, final_pred_study[study].view(1, -1)), 0)

        ACC, BACC, Prec, Rec, F1, AUC_ovo, AUC_ovr, SPEC, kappa = compute_isic_metrics(gt, pred)
        ACC2, BACC2, Prec2, Rec2, F1_2, AUC_ovo2, AUC_ovr2, SPEC2, kappa2 = compute_isic_metrics(gt, att_pred)
        ACC3, BACC3, Prec3, Rec3, F1_3, AUC_ovo3, AUC_ovr3, SPEC3, kappa3 = compute_isic_metrics(gt, final_pred)

        # print("false index:", len(false_index), false_index)

    all_labels = np.array(all_labels)
    # print("all features and labels:", all_feature4.shape, all_labels.shape)
    feature4_labels = np.concatenate((all_feature4, all_labels), axis=1)

    model.train(training)

    return ACC, BACC, Prec, Rec, F1, AUC_ovo, SPEC, kappa, feature4_labels, \
           ACC2, BACC2, Prec2, Rec2, F1_2, AUC_ovo2, SPEC2, kappa2, \
           ACC3, BACC3, Prec3, Rec3, F1_3, AUC_ovo3, SPEC3, kappa3

