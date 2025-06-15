import torch
import torch.nn as nn
from .Loss import losses
import numpy as np


class EarlyStopping:
    def __init__(self, patience=3, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        self.counter = 0
        self.best_pro = None
        self.early_stop = False
        self.pro_min = 0.0

    def __call__(self, pro):
        current_pro = pro

        if self.best_pro is None:
            self.best_pro = current_pro
        elif current_pro < self.best_pro + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_pro = current_pro
            self.save_checkpoint(current_pro)
            self.counter = 0

    def save_checkpoint(self, pro):
        if self.verbose:
            print(f'PRO increases ({self.pro_min:.1f} --> {pro:.1f}). Saving model...')
        self.pro_min = pro


class UniNet(nn.Module):
    def __init__(self, c, source_teacher, target_teacher, bottleneck, student, DFS=None):
        super().__init__()
        self._class_ = c._class_
        self.T = c.T # 코사인 유사도 결과 조절 하는 스케일링 파라미터
        self.n = 1 if target_teacher is None else 2
        self.teacher = Teachers(source_teacher=source_teacher, target_teacher=target_teacher)
        self.bottleneck = BottleNeck(bottleneck)
        self.student = Student(student=student)
        self.dfs = DFS

    def train_or_eval(self, type='train'):
        self.type = type
        self.teacher.train_eval(type)
        self.bottleneck.train_eval(type)
        self.student.train_eval(type)

        return self

    def feature_selection(self, teacher_features, student_features, max):
        if self._class_ in ['transistor']:
            return student_features

        if self.dfs is not None:
            selected_features = self.dfs(student_features, teacher_features, learnable=True, conv=False, max=max)
        else:
            from .DFS import domain_related_feature_selection
            selected_features = domain_related_feature_selection(student_features, teacher_features, use_max_normalization=max)
        return selected_features

    def loss_computation(self, teacher_feature, student_feature, margin=1, mask=None, stop_gradient=False):
        T = 0.1 if self._class_ in ['transistor', 'pill', 'cable', 'bottle', "grid", 'foam'] else self.T
        loss = losses(teacher_feature, student_feature, T, margin, mask=mask, stop_gradient=stop_gradient)

        return loss

    def forward(self, x, max=True, mask=None, stop_gradient=False):
        source_target_features, bottleneck_inputs = self.teacher(x)
        bottleneck_out = self.bottleneck(bottleneck_inputs)
        student_features = self.student(bottleneck_out)

        student_features = [d.chunk(dim=0, chunks=2) for d in student_features]
        student_features = [student_features[0][0], student_features[1][0], student_features[2][0],
                            student_features[0][1], student_features[1][1], student_features[2][1]]

        if self.type == 'train':
            student_features_ = self.feature_selection(source_target_features, student_features, max)
            loss = self.loss_computation(source_target_features, student_features_, mask=mask, stop_gradient=stop_gradient)

            return loss
        else:
            return source_target_features, student_features


class Teachers(nn.Module):
    def __init__(self, source_teacher, target_teacher):
        super().__init__()
        self.source_teacher = source_teacher
        self.target_teacher = target_teacher

    def train_eval(self, type='train'):
        self.type = type
        self.source_teacher.eval()
        if self.target_teacher is not None:
            if type == "train":
                self.target_teacher.train()
            else:
                self.target_teacher.eval()

        return self

    def forward(self, x):
        with torch.no_grad():
            source_teacher_feature = self.source_teacher(x)

        if self.target_teacher is None:
            return source_teacher_feature
        else:
            target_teacher_feature = self.target_teacher(x)
            bottleneck_input = [torch.cat([a, b], dim=0) for a, b in zip(target_teacher_feature, source_teacher_feature)]  # 512, 1024, 2048

            return source_teacher_feature + target_teacher_feature, bottleneck_input


class BottleNeck(nn.Module):
    def __init__(self, bottleneck):
        super().__init__()
        self.bottleneck = bottleneck

    def train_eval(self, type='train'):
        if type == 'train':
            self.bottleneck.train()
        else:
            self.bottleneck.eval()

        return self

    def forward(self, x):
        bottleneck_output = self.bottleneck(x)
        return bottleneck_output


class Student(nn.Module):
    def __init__(self, student):
        super().__init__()
        self.student_decoder = student

    def train_eval(self, type='train'):
        if type == 'train':
            self.student_decoder.train()
        else:
            self.student_decoder.eval()

        return self

    def forward(self, bottleneck_out, skips=None):
        student_decoded_features = self.student_decoder(bottleneck_out)
        return student_decoded_features
