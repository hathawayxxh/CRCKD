import torch
import torch.nn
from torch.nn import functional as F
import numpy as np

np.set_printoptions(threshold=np.inf)

CLASS_NUM = [1805, 370, 999, 193, 295]
CLASS_WEIGHT = torch.Tensor([3662/i for i in CLASS_NUM]).cuda()


class Loss_Zeros(object):
    """
    map all uncertainty values to 0
    """
    
    def __init__(self):
        self.base_loss = torch.nn.BCELoss(reduction='mean')
    
    def __call__(self, output, target):
        target[target == -1] = 0
        return self.base_loss(output, target)

class Loss_Ones(object):
    """
    map all uncertainty values to 1
    """
    
    def __init__(self):
        self.base_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
    
    def __call__(self, output, target):
        target[target == -1] = 1
        return self.base_loss(output, target)

class cross_entropy_loss(object):
    """
    map all uncertainty values to a unique value "2"
    """
    
    def __init__(self):
        self.base_loss = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHT, reduction='mean')
    
    def __call__(self, output, target):
        # target[target == -1] = 2
        # output_softmax = F.softmax(output, dim=1)
        target = torch.argmax(target, dim=1)
        return self.base_loss(output, target.long())


class focal_loss(object):

    def __init__(self):
        self.base_loss = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHT, reduction='none')

    def __call__(self, output, target, gamma=1):
        # target[target == -1] = 2
        output_softmax = F.softmax(output, dim=1).detach()
        label = torch.argmax(target, dim=1)
        weighted_ce_loss = self.base_loss(output, label.long())
        focal_weight = (1-torch.sum(torch.mul(target, output_softmax), 1))
        # print(weighted_ce_loss.shape, focal_weight.shape)
        weighted_focal_loss = torch.mul(focal_weight, weighted_ce_loss).mean()
        # print(weighted_focal_loss)
        return weighted_focal_loss


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent

def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2 * CLASS_WEIGHT
    return mse_loss

def cam_attention_map(activations, channel_weight):
    # activations 48*49*1024
    # channel_weight 48*1024
    attention = activations.permute(1,0,2).mul(channel_weight)
    attention = attention.permute(1,0,2)
    attention = torch.sum(attention, -1)
    attention = torch.reshape(attention, (48, 7, 7))

    return attention



def relation_mse_loss(activations, ema_activations):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    activations = torch.reshape(activations, (activations.shape[0], -1))
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

    similarity = activations.mm(activations.t())
    norm = torch.reshape(torch.norm(similarity, 2, 1), (-1, 1))
    norm_similarity = similarity / norm

    ema_similarity = ema_activations.mm(ema_activations.t())
    ema_norm = torch.reshape(torch.norm(ema_similarity, 2, 1), (-1, 1))
    ema_norm_similarity = ema_similarity / ema_norm

    assert norm_similarity.size() == ema_norm_similarity.size()

    # print("block3 similarity:", norm_similarity.cpu().detach().numpy())
    # print("ema_block4 similarity:", ema_norm_similarity.cpu().detach().numpy())

    similarity_mse_loss = (norm_similarity-ema_norm_similarity)**2
    return similarity_mse_loss


def SP_loss(f_s, f_t):
    """similarity preserving loss: constraint the similarity matrix of the teacher and student."""
    bsz = f_s.shape[0]
    f_s = f_s.view(bsz, -1)
    f_t = f_t.view(bsz, -1)

    G_s = torch.mm(f_s, torch.t(f_s))
    # G_s = G_s / G_s.norm(2)
    G_s = F.normalize(G_s)
    G_t = torch.mm(f_t, torch.t(f_t))
    # G_t = G_t / G_t.norm(2)
    G_t = F.normalize(G_t)

    G_diff = G_t - G_s
    loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
    return loss


def cos_relation_mse_loss(activations, ema_activations):
    """Takes softmax on both sides and returns MSE loss"""

    activations = torch.reshape(activations, (activations.shape[0], -1))
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

    act_norm = torch.norm(activations, 2, 1, keepdim=True)
    similarity = activations.mm(activations.t())/act_norm.mm(act_norm.t())

    ema_act_norm = torch.norm(ema_activations, 2, 1, keepdim=True)
    ema_similarity = ema_activations.mm(ema_activations.t())/ema_act_norm.mm(ema_act_norm.t())

    assert similarity.size() == ema_similarity.size()

    # print("block3 similarity:", similarity.cpu().detach().numpy())
    # print("ema_block4 similarity:", ema_similarity.cpu().detach().numpy())

    similarity_mse_loss = (similarity-ema_similarity)**2
    return similarity_mse_loss


def attention_mse_loss(attention, ema_attention):
    if attention.size() != ema_attention.size():
        avg_pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        attention = avg_pool(attention)
    assert attention.size() == ema_attention.size()
    att_mse_loss = (attention - ema_attention)**2
    return att_mse_loss



def attention_kl_loss(attention, ema_attention):
    assert attention.size() == ema_attention.size()
    att_kl_loss = F.kl_div(attention, ema_attention, reduction='none')
    return att_kl_loss


def feature_mse_loss(activations, ema_activations):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """

    assert activations.size() == ema_activations.size()

    activations = torch.reshape(activations, (activations.shape[0], -1))
    ema_activations = torch.reshape(ema_activations, (ema_activations.shape[0], -1))

    similarity_mse_loss = (activations-ema_activations)**2
    return similarity_mse_loss



def sigmoid_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = torch.sigmoid(input_logits)
    target_softmax = torch.sigmoid(target_logits)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    mse_loss = loss_fn(input_softmax, target_softmax)
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)
