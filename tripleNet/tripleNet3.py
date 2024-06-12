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
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # 마지막 fc layer 제거
        self.layer = layer
        
        # Fully connected layer
        self.fc = nn.Linear(3 * 512, 2)  # ResNet18의 output은 512차원
        # Fully convolutional layer
        self.conv = nn.Conv1d(in_channels=3, out_channels=2, kernel_size=512, stride=1)  # 1D convolutional layer

        self.out = nn.Sigmoid()
        
    def forward_one(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # flatten
        return x
    
    def forward(self, img1, img2, img3):
        # 세 개의 이미지를 각각 feature extractor에 통과시킴
        v1 = self.forward_one(img1)
        v2 = self.forward_one(img2)
        v3 = self.forward_one(img3)
        
        # Distance vector 계산 (L1 distance)
        d1 = torch.abs(v1 - v2)
        d2 = torch.abs(v2 - v3)
        d3 = torch.abs(v3 - v1)
        
        if self.layer == 'conn':
            # Fully connected layer에 통과시켜 최종 출력 계산
            concat = torch.cat((d1, d2, d3), 1)
            output = self.fc(concat)
        elif self.layer == 'conv':
            # Fully convolutional layer
            concat = torch.stack((d1, d2, d3), dim=1)
            output = self.conv(concat).squeeze(-1)  # (batch_size, 2)

        output = self.out(output)

        return output

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, label1, label2):
        # label1: 자식이 엄마의 친자식일 확률 라벨 (0 또는 1)
        # label2: 자식이 아빠의 친자식일 확률 라벨 (0 또는 1)
        prob1, prob2 = output[:, 0], output[:, 1]
        
        # loss1 = (1 - label1) * torch.pow(prob1, 2) + label1 * torch.pow(torch.clamp(self.margin - prob1, min=0.0), 2)
        # loss2 = (1 - label2) * torch.pow(prob2, 2) + label2 * torch.pow(torch.clamp(self.margin - prob2, min=0.0), 2)

        loss1 = (1 - label1) * torch.pow(prob1, 2) + label1 * torch.pow(1 - prob1, 2)
        loss2 = (1 - label2) * torch.pow(prob2, 2) + label2 * torch.pow(1 - prob2, 2)
        
        loss = torch.mean(loss1 + loss2)
        return loss



