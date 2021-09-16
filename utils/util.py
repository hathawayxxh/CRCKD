# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import cv2
import pickle
import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data.sampler import Sampler
import torch.nn as nn
import torch.nn.functional as F

import networks

np.set_printoptions(threshold=np.inf)


##### Metrics #####
def metric_all_epochs(auc_all, acc_all, bacc_all, prec_all, rec_all, F1_all, ACC, BACC, Prec, Rec, F1, AUC):
    auc_all.append(AUC)
    acc_all.append(ACC)
    bacc_all.append(BACC)
    prec_all.append(Prec)
    rec_all.append(Rec)
    F1_all.append(F1)

    return auc_all, acc_all, bacc_all, prec_all, rec_all, F1_all

def save_img(img, img_index, root_path, img_name, mode="image"):
    img = np.uint8(255 * img)
    if mode == "image":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif mode == "heatmap":
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    img_path = os.path.join(root_path, str(img_index) + img_name)
    cv2.imwrite(img_path, img)


def load_model(path):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

        # size of the top layer
        N = checkpoint['state_dict']['top_layer.bias'].size()

        # build skeleton of the model
        sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
        model = models.__dict__[checkpoint['arch']](sobel=sob, out=int(N[0]))

        # deal with a dataparallel table
        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}

        # load weights
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        size_per_pseudolabel = int(self.N / len(self.images_lists)) + 1
        res = np.zeros(size_per_pseudolabel * len(self.images_lists))

        for i in range(len(self.images_lists)):
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res[i * size_per_pseudolabel: (i + 1) * size_per_pseudolabel] = indexes

        np.random.shuffle(res)
        return res[:self.N].astype('int')

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.N


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger():
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)

def get_timestamp():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    return timestampDate + '-' + timestampTime

def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.extend([ param_group['lr'] ])
    return lr


###############################
####### Attention Maps ########
###############################

def get_att_maps(f_maps, mode="sum_square", norm="l2"):
    """
    given feature maps: [bs, c, h, w], compute the normalized attention map.
    """
    batch_size, c, h, w = f_maps.shape
    if mode == "avg":
        att_map = torch.mean(F.relu(f_maps), dim=1, keepdims=True)

    elif mode == "sum_square":  # same as the paper: lane detection with self attention distillation.
        att_map = torch.mean(torch.square(F.relu(f_maps)), dim=1, keepdims=True)

    att_map_min = torch.min(att_map.view(batch_size, -1), dim=1)[0].view(batch_size, 1, 1, 1)
    att_map_max = torch.max(att_map.view(batch_size, -1), dim=1)[0].view(batch_size, 1, 1, 1)
    s_map = torch.div(att_map - att_map_min + 1e-8, att_map_max - att_map_min + 1e-8)
    # print("max:", torch.max(att_map), "min:", torch.min(att_map))

    if norm == "l2":
        att_map = F.normalize(att_map.view(batch_size, -1), dim=1).view(batch_size, 1, h, w)
    elif norm == "softmax":
        att_map = F.softmax(att_map.view(batch_size, -1), dim=1).view(batch_size, 1, h, w)
    elif norm == "log_softmax":
        att_map = F.log_softmax(att_map.view(batch_size, -1), dim=1).view(batch_size, 1, h, w)

    return att_map, s_map


def get_att_mask(att_map, iter, mode):
    """
    threshold the attention maps to binary masks
    """
    att_save_path = '/apdcephfs/private_xiaohanxing/STAC_CRD_MT_codes/visualization/att_masks/'
    bs, _, h, w = att_map.shape
    # thresh = 0.6 * torch.max(att_map.view(bs, -1), axis=-1)[0].view(bs, 1, 1, 1).repeat(1, 1, h, w)
    thresh = torch.mean(att_map.view(bs, -1), axis=-1).view(bs, 1, 1, 1).repeat(1, 1, h, w)
    att_mask = torch.where(att_map >= thresh, torch.ones_like(att_map), torch.zeros_like(att_map))
    # print(att_mask)

    if mode == 'test':
        att_mask_save = np.uint8(255 * torch.squeeze(att_mask).cpu().detach().numpy()) #[bs, h, w]
        # print(att_mask_save.shape)

        for i in range(bs):
            img = cv2.resize(att_mask_save[i], (224, 224)) #grayimage
            cv2.imwrite(att_save_path + str(bs * iter + i) + '.jpg', img)

    return att_mask


def get_att_masked_feature(att_map, feature, iter, mode):
    """
    threshold the attention maps to binary masks, and then use the mask to select the feature maps,
    and generate the average local masked feature
    """
    bs, c, h, w = feature.shape
    att_mask = get_att_mask(att_map, iter, mode) #[bs, 1, h, w]
    num_selected = torch.sum(att_mask.view(bs, -1), axis=1).view(bs, 1)
    masked_avg_feature = torch.sum((att_mask * feature).view(bs, c, -1), axis=-1)/num_selected #[bs, c]

    return masked_avg_feature




##########################
##### Relation graph #####
##########################

def save_relation(GT_matrix, feature, batch, img_name):
    root_path = "/apdcephfs/private_xiaohanxing/STAC_CRD_MT_codes/visualization/relation_matrix/"

    norm = torch.norm(feature, 2, 1, keepdim=True)
    relation_matrix = feature.mm(feature.t())/norm.mm(norm.t())
    # print(relation_matrix)
    # for i in range(8, 15):
    #     print(relation_matrix[i])

    if batch == 9:
        ## compute average relation of samples from the same class
        pos_relation = torch.sum(torch.mul(relation_matrix, GT_matrix))/torch.sum(GT_matrix)
        # print(pos_relation)

        ## compute average relation of samples from different classes.
        neg_relation = torch.sum(torch.mul(relation_matrix, (1-GT_matrix)))/torch.sum(1-GT_matrix)
        # print(neg_relation)

    # print(torch.min(relation_matrix, axis=1))
    relation_img = Image.fromarray((255*relation_matrix).cpu().detach().numpy().astype(np.uint8))
    # relation_img = relation_img.resize((128, 128), Image.NEAREST)
    # plt.imsave(os.path.join(root_path, str(batch) + img_name), relation_img, cmap='Blues')
    plt.imsave(os.path.join(root_path, str(batch) + img_name), relation_img, cmap='winter')



def compute_relation_matrix(batch, labels, feature3, feature4, ema_feature4, ema_output):
    label_class = torch.argmax(labels, axis=1)
    pred = torch.argmax(ema_output, axis=1)

    sorted_index = torch.argsort(label_class)
    # print("original labels:", label_class)
    # print("sorted_index:", sorted_index)
    # print("feature3:", feature3.shape)

    sorted_labels = torch.zeros_like(label_class)
    sorted_pred = torch.zeros_like(pred)
    sorted_feature3 = torch.zeros_like(feature3)
    sorted_feature4 = torch.zeros_like(feature4)
    sorted_ema_feature4 = torch.zeros_like(ema_feature4)

    for ind in range(sorted_index.shape[0]):
        sorted_labels[ind] = label_class[sorted_index[ind]]
        sorted_pred[ind] = pred[sorted_index[ind]]
        sorted_feature3[ind, :] = feature3[sorted_index[ind], :]
        sorted_feature4[ind, :] = feature4[sorted_index[ind], :]
        sorted_ema_feature4[ind, :] = ema_feature4[sorted_index[ind], :]

    # print("sorted_labels:", sorted_labels)
    # print("sorted predictions:", sorted_pred)

    # relation_b3 = sorted_feature3.mm(sorted_feature3.t())
    # relation_b4 = sorted_feature4.mm(sorted_feature4.t())
    # relation_ema_b4 = sorted_ema_feature4.mm(sorted_ema_feature4.t())

    # print("sample norm:", torch.norm(sorted_ema_feature4, 2, 1))
    # print("ema_feature4:", sorted_ema_feature4.cpu().detach().numpy())
    # print("relation matrix of block4:", relation_ema_b4.cpu().detach().numpy())

    # save_relation(sorted_feature3, batch, img_name='_relation_b3.jpg')
    save_relation(sorted_feature4, batch, img_name='_relation_b4.jpg')
    save_relation(sorted_ema_feature4, batch, img_name='_relation_ema_b4.jpg')


def compute_test_relation_matrix(batch, labels, feature3, feature4, output):
    label_class = torch.argmax(labels, axis=1)
    pred = torch.argmax(output, axis=1)

    sorted_index = torch.argsort(label_class)
    # print("original labels:", label_class)
    # print("sorted_index:", sorted_index)
    # print("feature3:", feature3.shape)

    sorted_labels = torch.zeros_like(label_class)
    sorted_pred = torch.zeros_like(pred)
    sorted_feature3 = torch.zeros_like(feature3)
    sorted_feature4 = torch.zeros_like(feature4)

    for ind in range(sorted_index.shape[0]):
        sorted_labels[ind] = label_class[sorted_index[ind]]
        sorted_pred[ind] = pred[sorted_index[ind]]
        sorted_feature3[ind, :] = feature3[sorted_index[ind], :]
        sorted_feature4[ind, :] = feature4[sorted_index[ind], :]

    # print("batch", str(batch), sorted_labels)


    GT_matrix = torch.zeros(pred.shape[0], pred.shape[0])
    for i in range(7):
        class_samples = torch.zeros_like(sorted_labels)
        class_samples[sorted_labels == i] = 1
        class_matrix = torch.mm(torch.unsqueeze(class_samples, 1).float(), torch.unsqueeze(class_samples, 0).float())
        GT_matrix = GT_matrix + class_matrix.cpu()
    # print(GT_matrix)

    # save_relation(sorted_feature3, batch, img_name='_relation_b3.jpg')
    save_relation(GT_matrix.cuda(), sorted_feature4, batch, img_name='_relation_b4.jpg')

    GT_matrix_img = Image.fromarray((255*GT_matrix).detach().numpy().astype(np.uint8))
    plt.imsave(os.path.join("/apdcephfs/private_xiaohanxing/STAC_CRD_MT_codes/visualization/GT_relation/",
                            str(batch) + '_GT_matrix.jpg'), GT_matrix_img, cmap='Blues')


