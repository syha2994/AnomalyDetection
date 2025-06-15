import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainRelated_Feature_Selection(nn.Module):
    def __init__(self, num_channels=256):
        super(DomainRelated_Feature_Selection, self).__init__()
        self.num_channels = num_channels    # the first layer
        #  initialize to 0
        self.theta1 = nn.Parameter(torch.zeros(1, num_channels, 1, 1)).to("cuda")
        self.theta2 = nn.Parameter(torch.zeros(1, num_channels * 2, 1, 1)).to("cuda")
        self.theta3 = nn.Parameter(torch.zeros(1, num_channels * 4, 1, 1)).to("cuda")

    def forward(self, xs, priors, learnable=True, conv=False, max=True):
        features = []
        for idx, (x, prior) in enumerate(zip(xs, priors)):
            theta = 1
            if learnable:
                #  to avoid losing local weight, theta should be as non-zero value as possible
                if idx < 3:
                    #
                    theta = torch.clamp(torch.sigmoid(eval("self.theta{}".format((idx + 1)))) * 1.0 + 0.5, max=1)
                else:
                    theta = torch.clamp(torch.sigmoid(eval("self.theta{}".format(idx - 2))) * 1.0 + 0.5, max=1)

            b, c, h, w = x.shape
            if not conv:
                prior_flat = prior.view(b, c, -1)
                if max:
                    prior_flat_ = prior_flat.max(dim=-1, keepdim=True)[0]
                    prior_flat = prior_flat - prior_flat_
                weights = F.softmax(prior_flat, dim=-1)
                weights = weights.view(b, c, h, w)

                global_inf = prior.mean(dim=(-2, -1), keepdim=True)

                inter_weights = weights * (theta + global_inf)

                x_ = x * inter_weights
                features.append(x_)
            else:
                #  generated in a convolutional way
                pass

        return features


def domain_related_feature_selection(student_features, teacher_features, use_max_normalization=True):
    features_list = []
    theta = 1
    for (student_feature, teacher_feature) in zip(student_features, teacher_features):
        b, c, h, w = student_feature.shape

        teacher_flat = teacher_feature.view(b, c, -1)
        if use_max_normalization:
            prior_flat_ = teacher_flat.max(dim=-1, keepdim=True)[0]
            teacher_flat = teacher_flat - prior_flat_
        weights = F.softmax(teacher_flat, dim=-1)
        weights = weights.view(b, c, h, w)

        global_inf = teacher_feature.mean(dim=(-2, -1), keepdim=True)

        inter_weights = weights * (theta + global_inf)

        weighted_student_feature = student_feature * inter_weights
        features_list.append(weighted_student_feature)

    return features_list
