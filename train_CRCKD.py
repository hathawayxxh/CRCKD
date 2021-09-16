import os
import sys
import shutil
import argparse
import logging
import time
import random
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.data.sampler import WeightedRandomSampler

from networks.models import DenseNet121
from utils import losses, ramps
from utils.CCD_CRP_loss import CCD_CRP_Loss
from utils.metric_logger import MetricLogger
from dataloaders import dataset
from utils.util import get_timestamp, get_att_maps, compute_relation_matrix, get_att_masked_feature, metric_all_epochs
from utils.transform import transforms_for_rot, transforms_back_rot
from validation import epochVal_metrics

print(torch.cuda.is_available())

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./APTOS_images/')
parser.add_argument('--csv_file_path', type=str, default='./5cv_split_dataset/')
parser.add_argument('--feature_save_path', type=str, default='./feature_visualization/')
parser.add_argument('--which_split', type=str,  default='split1', help='model_name')

parser.add_argument('--exp', type=str,  default='sup4_base', help='model_name')
parser.add_argument('--train', type=str,  default='True', help='model_name')
parser.add_argument('--ema_consistency', type=int, default=0, help='whether train baseline model')
parser.add_argument('--begin_ema', type=int, default=20, help='start to use the EMA loss')

parser.add_argument('--consistency_feature_weight', type=int, default=0, help='consistency feature weight')

parser.add_argument('--deep_sup3', type=str, default="False", help='add deep supervision on block3')

### for CCD loss
parser.add_argument('--CCD_distill', type=int, default=0, help='whether use the CCD loss')
parser.add_argument('--mask_type', type=str, default="None", choices=['dual', 'semi', 'None'])
parser.add_argument('--CCD_b4_weight', type=float, default=0, help='whether use the CCD loss')
parser.add_argument('--CCD_b3_weight', type=float, default=0, help='whether use the CCD loss')

parser.add_argument('--CCD_mode', type=str, default="sup", choices=['sup', 'unsup'])

parser.add_argument('--s_dim', type=int, default=128, help='feature dim of the student model')
parser.add_argument('--t_dim', type=int, default=128, help='feature dim of the EMA teacher')
parser.add_argument('--feat_dim', type=int, default=128, help='reduced feature dimension')
parser.add_argument('--n_data', type=int, default=2929, help='total number of training samples.')
parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax', 'multi_pos'])
parser.add_argument('--nce_p', default=1, type=int, help='number of positive samples for NCE')
parser.add_argument('--nce_k', default=4096, type=int, help='number of negative samples for NCE')
parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

### for similarity-preserving (SP) loss
parser.add_argument('--consistency_relation_weight', type=int, default=0, help='consistency relation weight')
parser.add_argument('--b44_SRC_weight', type=float, default=0, help='SRC loss between block4 and ema_block4')

### for Categorical Relation Preserving (CRP) loss
parser.add_argument('--CRP_distill', type=float, default=0, help='the weight of AR loss')
parser.add_argument('--anchor_type', type=str,  default="center", choices=['center', 'class'])
parser.add_argument('--class_anchor', default=30, type=int, help='number of anchors in each class')

### whether use the attention branch to get attention masks.
parser.add_argument('--att_branch', type=str,  default="False", help='whether to train auxiliary attention branch')
parser.add_argument('--att_pred_MT', type=float,  default=0, help='weight of the pred MT loss of the attention branch')
parser.add_argument('--att_loss_weight', type=float,  default=0, help='weight of the CE loss from the attention branch')

parser.add_argument('--semi', type=str,  default="False", help='semi or full supervision')

parser.add_argument('--epochs', type=int,  default=80, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=32, help='number of labeled data per batch')
parser.add_argument('--drop_rate', type=int, default=0, help='dropout rate')
parser.add_argument('--labeled_num', type=int, default=2929, help='number of labeled')
parser.add_argument('--base_lr', type=float,  default=1e-4, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=2020, help='random seed')
parser.add_argument('--gpu', type=str,  default='0, 1', help='GPU to use')
### tune
parser.add_argument('--resume', type=str,  default=None, help='model to resume')
parser.add_argument('--start_epoch', type=int,  default=0, help='start_epoch')
parser.add_argument('--global_step', type=int,  default=0, help='global_step')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="kl", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=30, help='consistency_rampup')

### by beemo
parser.add_argument('--optim_str', type=str,  default='Adam', help='optim')
parser.add_argument('--sch_str', type=str,  default='OneCycleLR', help='sch_str')
args = parser.parse_args()

train_data_path = args.root_path
save_path = "./5cv_results/"
snapshot_path = save_path + args.which_split + '/' + args.exp + "/"
log_path = save_path + args.which_split + '/logs/'

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
base_lr = args.base_lr
labeled_bs = args.labeled_bs * len(args.gpu.split(','))

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


if __name__ == "__main__":
    ## make logging file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        os.makedirs(snapshot_path + './checkpoint')
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logging.basicConfig(filename=log_path + args.exp +".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        net = DenseNet121(hidden_units=args.feat_dim, out_size=dataset.N_CLASSES, drop_rate=args.drop_rate)

        if len(args.gpu.split(',')) > 1:
            net = torch.nn.DataParallel(net)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, ':', param.size())

    ema_model = create_model(ema=True)

    module_list = nn.ModuleList([])
    module_list.append(model)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model)

    CCD_criterion = CCD_CRP_Loss(args).cuda()
    if args.CCD_distill == 1:
        module_list.append(CCD_criterion.embed_s)
        module_list.append(CCD_criterion.embed_t)
        trainable_list.append(CCD_criterion.embed_s)
        trainable_list.append(CCD_criterion.embed_t)

    optimizer = torch.optim.Adam(trainable_list.parameters(), lr=args.base_lr,
                                 betas=(0.9, 0.999), weight_decay=5e-4)

    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        logging.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        args.global_step = checkpoint['global_step']
        # best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        ema_model.load_state_dict(checkpoint['ema_state_dict'], strict=False)
        # print(checkpoint['optimizer']['param_groups']['params'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    # dataset
    train_dataset, test_dataset, n_data = dataset.isic_dataloaders_sample(args, p=args.nce_p, mode=args.mode)
    class_index = train_dataset.class_index ### the image index of every class.
    train_targets = np.argmax(train_dataset.labels, axis=1) #labels of the 7012 train samples.
    test_targets = np.argmax(test_dataset.labels, axis=1)
    # print(class_index.shape)

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=12, pin_memory=True, worker_init_fn=worker_init_fn)

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=12, pin_memory=True, worker_init_fn=worker_init_fn)

    sch_str = args.sch_str
    epoch_cnt = args.epochs
    if sch_str == 'StepLR':
        sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoch_cnt // 3, gamma=0.1)
    elif sch_str == 'CosineAnnealingLR':
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_cnt)
    elif sch_str == 'OneCycleLR':
        sch = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, epochs=epoch_cnt,
                                                  steps_per_epoch=len(train_dataloader))
    elif sch_str == 'CosineAnnealingWarmRestarts':
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)


    module_list.train()

    loss_fn = losses.cross_entropy_loss()

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')

    iter_num = args.global_step
    lr_ = base_lr
    module_list.train()

    auc_all, acc_all, bacc_all, prec_all, rec_all, F1_all = [], [], [], [], [], []
    att_auc_all, att_acc_all, att_bacc_all, att_prec_all, att_rec_all, att_F1_all = [], [], [], [], [], []
    final_auc_all, final_acc_all, final_bacc_all, final_prec_all, final_rec_all, final_F1_all = [], [], [], [], [], []

    for epoch in range(args.start_epoch, args.epochs):
        if args.train == 'True':
            aux_weight = 0.2*(1 - epoch/args.epochs)
            # print("aux_weight:", aux_weight)
            start_time = time.time()
            meters_loss = MetricLogger(delimiter="  ")
            meters_loss_classification = MetricLogger(delimiter="  ")
            meters_loss_CCD = MetricLogger(delimiter="  ")
            meters_loss_consistency = MetricLogger(delimiter="  ")
            meters_loss_consistency_relation = MetricLogger(delimiter="  ")
            meters_loss_center_relation = MetricLogger(delimiter="  ")
            meters_loss_consistency_feature = MetricLogger(delimiter="  ")
            meters_loss_consistency_attention = MetricLogger(delimiter="  ")

            time1 = time.time()
            iter_max = len(train_dataloader)

            all_feature4, all_masked_feature4, all_embed4, all_labels = None, None, None, None

            for i, ((image_batch, ema_image_batch), label_batch, index, sample_idx) in enumerate(train_dataloader):
                time2 = time.time()
                # print("items:", item)
                # print("indexs:", index)
                # print("labels:", label_batch.shape, torch.argmax(label_batch, axis=1))
                # print("sample_idx:", sample_idx.shape)
                image_batch, ema_image_batch, label_batch = image_batch.cuda(), ema_image_batch.cuda(), label_batch.cuda()
                inputs = image_batch
                ema_inputs = ema_image_batch
                ema_inputs_rot, rot_mask, flip_mask = transforms_for_rot(ema_inputs)


                feature3, feature4, SE_att, att_feature4, logit_b3, logit_b4, att_logit4, \
                    fmaps_b3, fmaps_b4 = model(inputs)

                att3, smap3 = get_att_maps(fmaps_b3, norm="l2")
                att4, smap4 = get_att_maps(fmaps_b4, norm="l2")
                # print("pred and labels:", logit_b4.shape, label_batch.shape)

                batch_label = np.reshape(np.argmax(label_batch.cpu().numpy(), axis=1), (-1, 1))
                all_labels = batch_label if all_labels is None else np.concatenate((all_labels, batch_label), axis=0)
                feature4_np = feature4.cpu().detach().numpy()
                all_feature4 = feature4_np if all_feature4 is None else np.concatenate((all_feature4, feature4_np), axis=0)

                masked_feature4 = get_att_masked_feature(att4, fmaps_b4, iter=i, mode='train')
                masked_feature4_np = masked_feature4.cpu().detach().numpy()
                all_masked_feature4 = masked_feature4_np if all_masked_feature4 is None else np.concatenate((
                    all_masked_feature4, masked_feature4_np), axis=0)


                with torch.no_grad():
                    ema_feature3, ema_feature4, ema_SE_att, ema_att_feature4, ema_logit_b3, ema_logit_b4, ema_att_logit4, \
                        ema_fmaps_b3, ema_fmaps_b4 = ema_model(ema_inputs_rot)
                    ema_att3, ema_smap3 = get_att_maps(ema_fmaps_b3, norm="l2") #[bs, 1, 7, 7]
                    ema_att4, ema_smap4 = get_att_maps(ema_fmaps_b4, norm="l2")
                    ema_masked_feature4 = get_att_masked_feature(ema_att4, ema_fmaps_b4, iter=i, mode='train')


                ## calculate the loss
                # print(logit_b4[:labeled_bs].shape, label_batch.shape)
                loss_classification = loss_fn(logit_b4[:labeled_bs], label_batch[:labeled_bs])

                if args.att_branch == "True":
                    loss_classification += args.att_loss_weight * loss_fn(att_logit4[:labeled_bs], label_batch[:labeled_bs])
                if args.deep_sup3 == "True":
                    loss_classification += aux_weight * loss_fn(logit_b3[:labeled_bs], label_batch[:labeled_bs])

                ## MT loss (have no effect in the beginneing)
                if args.ema_consistency == 1:
                    consistency_weight = get_current_consistency_weight(epoch)
                    consistency_dist = torch.sum(consistency_criterion(logit_b4, ema_logit_b4))/batch_size
                    consistency_loss = consistency_weight * consistency_dist

                    consistency_relation_dist = args.b44_SRC_weight * losses.SP_loss(feature4, ema_feature4)
                    consistency_relation_loss = consistency_weight * consistency_relation_dist * args.consistency_relation_weight

                    consistency_feature_dist = 0.1 * torch.sum(losses.feature_mse_loss(feature4, ema_feature4))/batch_size
                    consistency_feature_loss = consistency_weight * consistency_feature_dist * args.consistency_feature_weight

                    # print(CCD_loss)


                else:
                    consistency_loss = 0.0
                    consistency_relation_loss = 0.0
                    consistency_feature_loss = 0.0
                    consistency_weight = 0.0
                    consistency_dist = 0.0

                if args.mask_type == "dual":
                    CCD_loss, embed_s, embed_t = CCD_criterion(masked_feature4, ema_masked_feature4, index.cuda(),
                                                               args.nce_p, sample_idx.cuda())
                elif args.mask_type == "semi":
                    CCD_loss, embed_s, embed_t = CCD_criterion(feature4, ema_masked_feature4, index.cuda(),
                                                               args.nce_p, sample_idx.cuda())
                elif args.mask_type == "None":
                    CCD_loss, relation_loss, embed_s, embed_t = CCD_criterion(feature4, ema_feature4,
                                index.cuda(), label_batch.cuda(), class_index, args.nce_p, sample_idx.cuda())

                relation_loss = args.CRP_distill * relation_loss # * consistency_weight

                CCD_loss = args.CCD_b4_weight * CCD_loss

                loss = loss_classification + CCD_loss

                if (epoch >= args.begin_ema) and (args.ema_consistency == 1):
                    loss = loss_classification + consistency_loss + consistency_relation_loss + consistency_feature_loss \
                            + args.CCD_b4_weight * CCD_loss + relation_loss

                elif epoch >= args.begin_ema:
                    loss = loss_classification + args.CCD_b4_weight * CCD_loss + relation_loss
                    # loss = loss_classification

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                update_ema_variables(model, ema_model, args.ema_decay, iter_num)

                meters_loss.update(loss=loss)
                meters_loss_classification.update(loss=loss_classification)
                meters_loss_CCD.update(loss=CCD_loss)
                meters_loss_consistency.update(loss=consistency_loss)
                meters_loss_consistency_relation.update(loss=consistency_relation_loss)
                meters_loss_center_relation.update(loss=relation_loss)
                meters_loss_consistency_feature.update(loss=consistency_feature_loss)

                iter_num = iter_num + 1

                if sch_str == 'OneCycleLR':
                    sch.step()

                # write tensorboard
                if i % 40 == 0:
                    writer.add_scalar('lr', lr_, iter_num)
                    writer.add_scalar('loss/loss', loss, iter_num)
                    writer.add_scalar('loss/loss_classification', loss_classification, iter_num)
                    writer.add_scalar('train/CCD_loss', CCD_loss, iter_num)
                    writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
                    writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
                    writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)
                    writer.add_scalar('train/consistency_relation_loss', consistency_relation_loss, iter_num)
                    writer.add_scalar('train/consistency_feature_loss', consistency_feature_loss, iter_num)

                    logging.info("\nEpoch: {}, iteration: {}/{}, ==> train <===, loss: {:.6f}, classification loss: {:.6f}, "
                                 "CCD loss: {:.6f}, consistency loss: {:.6f}, consistency relation loss: {:.6f}, "
                                 "consistency feature loss: {:.6f}, center_relation_loss: {:.6f}, "
                                 "consistency weight: {:.6f}, lr: {}"
                                .format(epoch, i, iter_max, meters_loss.loss.avg, meters_loss_classification.loss.avg,
                                        meters_loss_CCD.loss.avg, meters_loss_consistency.loss.avg,
                                        meters_loss_consistency_relation.loss.avg, meters_loss_consistency_feature.loss.avg,
                                        meters_loss_center_relation.loss.avg,
                                        consistency_weight, optimizer.param_groups[0]['lr']))

                    image = inputs[-1, :, :]
                    grid_image = make_grid(image, 5, normalize=True)
                    writer.add_image('raw/Image', grid_image, iter_num)

                    image = ema_inputs[-1, :, :]
                    grid_image = make_grid(image, 5, normalize=True)
                    writer.add_image('noise/Image', grid_image, iter_num)

            timestamp = get_timestamp()

        # test student
        ACC, BACC, Prec, Rec, F1, AUC, SPEC, kappa, feature4_labels, ACC2, BACC2, Prec2, Rec2, F1_2, AUC2, SPEC2, kappa2, \
            ACC3, BACC3, Prec3, Rec3, F1_3, AUC3, SPEC3, kappa3 = epochVal_metrics(
            model, test_dataloader)
        logging.info("\nTEST: Epoch: {}".format(epoch))
        logging.info("\nTEST Accuracy: {:6f}, Precision: {:6f}, Recall: {:6f}, F1 score: {:6f}, AUC: {:6f}, "
                     "Specificity: {:6f}, Kappa: {:6f}".format(ACC, Prec, Rec, F1, AUC, SPEC, kappa))

        if (epoch+1) >= args.epochs-10:
            auc_all, acc_all, bacc_all, prec_all, rec_all, F1_all = metric_all_epochs(
                auc_all, acc_all, bacc_all, prec_all, rec_all, F1_all, ACC, BACC, Prec, Rec, F1, AUC)
            att_auc_all, att_acc_all, att_bacc_all, att_prec_all, att_rec_all, att_F1_all = metric_all_epochs(
                att_auc_all, att_acc_all, att_bacc_all, att_prec_all, att_rec_all, att_F1_all, ACC2, BACC2, Prec2, Rec2, F1_2, AUC2)
            final_auc_all, final_acc_all, final_bacc_all, final_prec_all, final_rec_all, final_F1_all = metric_all_epochs(
                final_auc_all, final_acc_all, final_bacc_all, final_prec_all, final_rec_all, final_F1_all, ACC3, BACC3, Prec3, Rec3, F1_3, AUC3)


        # save model
        if ((epoch+1) >= args.epochs-20) and ((epoch+1) % 5 == 0):
            save_mode_path = os.path.join(snapshot_path + 'checkpoint/', 'epoch_' + str(epoch+1) + '.pth')
            torch.save({    'epoch': epoch + 1,
                            'global_step': iter_num,
                            'state_dict': model.state_dict(),
                            'ema_state_dict': ema_model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'epochs'    : epoch,
                       }
                       , save_mode_path
            )
            logging.info("save model to {}".format(save_mode_path))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        epoch_time = round(time.time() - start_time, 2)
        print("Time cost for epoch: {} is {}s".format(epoch, epoch_time))

    logging.info("\nAverage performance of the last 10 epochs:")
    logging.info("\nGlobal branch TEST AUC: {:6f}, Accuracy: {:6f}, Precision: {:6f}, Balanced Accuracy: {:6f}, F1: {:6f}"
                 .format(np.mean(auc_all), np.mean(acc_all), np.mean(prec_all), np.mean(rec_all), np.mean(F1_all)))

    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(iter_num+1)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
