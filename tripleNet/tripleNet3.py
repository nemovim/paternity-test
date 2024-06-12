import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights

class TripleSiameseNetwork(nn.Module):
    def __init__(self, layer='conn'):
        super(TripleSiameseNetwork, self).__init__()
        # ResNet18을 사용하여 feature extractor 정의
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor1 = nn.Sequential(*list(resnet.children())[:-1])  # 마지막 fc layer 제거
        self.feature_extractor2 = nn.Sequential(*list(resnet.children())[:-1])  # 마지막 fc layer 제거
        self.feature_extractor3 = nn.Sequential(*list(resnet.children())[:-1])  # 마지막 fc layer 제거
        self.layer = layer
        
        # Fully connected layer
        self.fc = nn.Linear(3 * 512, 2)  # ResNet18의 output은 512차원
        # Fully convolutional layer
        self.conv = nn.Conv1d(in_channels=3, out_channels=2, kernel_size=512, stride=1)  # 1D convolutional layer

        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),

            nn.Linear(64, 2),
            nn.Sigmoid(),
        )
        
    def forward1(self, x):
        x = self.feature_extractor1(x)
        x = x.view(x.size(0), -1)  # flatten
        return x

    def forward2(self, x):
        x = self.feature_extractor2(x)
        x = x.view(x.size(0), -1)  # flatten
        return x

    def forward3(self, x):
        x = self.feature_extractor3(x)
        x = x.view(x.size(0), -1)  # flatten
        return x
    
    def forward(self, img1, img2, img3):
        # 세 개의 이미지를 각각 feature extractor에 통과시킴
        v1 = self.forward1(img1)
        v2 = self.forward2(img2)
        v3 = self.forward3(img3)
        
        # Distance vector 계산 (L1 distance)
        # d1 = torch.abs(v1 - v2)
        # d2 = torch.abs(v2 - v3)
        # d3 = torch.abs(v3 - v1)

        d = v1 * v2 * v3
        
        # if self.layer == 'conn':
        #     # Fully connected layer에 통과시켜 최종 출력 계산
        #     concat = torch.cat((d1, d2, d3), 1)
        #     output = self.fc(concat)
        # elif self.layer == 'conv':
        #     # Fully convolutional layer
        #     concat = torch.stack((d1, d2, d3), dim=1)
        #     output = self.conv(concat).squeeze(-1)  # (batch_size, 2)

        # output = self.cls_head(torch.cat((d1, d2, d3), 1))
        output = self.cls_head(d)

        return output

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.bce_loss = nn.BCELoss()

    def forward(self, output, label1, label2):
        # label1: 자식이 엄마의 친자식일 확률 라벨 (0 또는 1)
        # label2: 자식이 아빠의 친자식일 확률 라벨 (0 또는 1)
        prob1, prob2 = output[:, 0], output[:, 1]
        
        # loss1 = (1 - label1) * torch.pow(prob1, 2) + label1 * torch.pow(torch.clamp(self.margin - prob1, min=0.0), 2)
        # loss2 = (1 - label2) * torch.pow(prob2, 2) + label2 * torch.pow(torch.clamp(self.margin - prob2, min=0.0), 2)

        # loss1 = (1 - label1) * torch.pow(prob1, 2) + label1 * torch.pow(1 - prob1, 2)
        # loss2 = (1 - label2) * torch.pow(prob2, 2) + label2 * torch.pow(1 - prob2, 2)

        loss1 = self.bce_loss(prob1, label1.reshape(prob1.size(0)))
        loss2 = self.bce_loss(prob2, label2.reshape(prob2.size(0)))
        
        loss = (loss1 + loss2)
        return loss



