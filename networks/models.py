# encoding: utf-8

"""
The main CheXpert models implementation.
Including:
    DenseNet-121
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from . import densenet


class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, hidden_units, out_size, drop_rate=0):
        super(DenseNet121, self).__init__()
        self.densenet121 = densenet.densenet121(pretrained=True, drop_rate=drop_rate)
        num_ftrs_b3 = self.densenet121.classifier_b3.in_features
        num_ftrs = self.densenet121.classifier.in_features

        self.densenet121.fc_layer = nn.Linear(num_ftrs, hidden_units)
        self.densenet121.classifier = nn.Linear(hidden_units, out_size)
        self.densenet121.classifier_b3 = nn.Linear(num_ftrs_b3, out_size)
        self.densenet121.att_classifier = nn.Linear(num_ftrs, out_size)

        # Official init from torch repo.
        for m in self.densenet121.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, x):
        # print("number of modules in the network:", len(self.densenet121.features))
        # print("module9:", self.densenet121.features[8])

        fmaps_b3 = F.relu(self.densenet121.norm_after_b3(
            self.densenet121.features[0:9](x)), inplace=True) #before transition layer

        feature3 = F.adaptive_avg_pool2d(fmaps_b3, (1, 1)).view(fmaps_b3.size(0), -1)
        if self.drop_rate > 0:
            feature3 = self.drop_layer(feature3)

        # print("fmaps_b3:", fmaps_b3.shape)
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)
        # print("output shape:", out.shape) #[bs, 1024, 7, 7]
        fmaps_b4 = out

        out = F.adaptive_avg_pool2d(fmaps_b4, (1, 1)).view(fmaps_b4.size(0), -1)

        att4, attended_feature = self.densenet121.attention_gate(features)
        # print("attended feature:", attended_feature.shape)
        attended_feature = F.adaptive_avg_pool2d(
            F.relu(attended_feature, inplace=True), (1, 1)).view(attended_feature.size(0), -1)

        # print("feature before FC classifier:", out.shape)

        # print("feature3:", feature3)
        # print("feature4:", out)

        if self.drop_rate > 0:
            out = self.drop_layer(out)

        feature4 = self.densenet121.fc_layer(out)

        logit_b3 = self.densenet121.classifier_b3(feature3)
        logit_b4 = self.densenet121.classifier(feature4)
        attended_logits = self.densenet121.att_classifier(attended_feature)

        return feature3, feature4, att4, attended_feature, \
               logit_b3, logit_b4, attended_logits, fmaps_b3, fmaps_b4

