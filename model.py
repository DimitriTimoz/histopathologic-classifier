import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader

class CNN(nn.Module): 
    def __init__(self, num_classes: int):
        super(CNN, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs
    
    def evaluate(self, dataloader: DataLoader, device: torch.device):
        self.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                logits = self.forward(x)
                _, predicted = torch.max(logits, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        accuracy = correct / total if total > 0 else 0.0
        return accuracy

class Attention_Module(nn.Module):
    def __init__(self):
        super(Attention_Module, self).__init__()
        
    def forward(self, x, fc_weights, gama):
        cams = F.conv2d(x, fc_weights)
        cams = F.relu(cams)
        N, C, H, W = cams.size()
        cam_mean = torch.mean(cams, dim=1)  # N 28 28
        
        zero = torch.zeros_like(cam_mean)
        one = torch.ones_like(cam_mean)
        mean_drop_cam = zero
        for i in range(C):
            sub_cam = cams[:, i, :, :]
            sub_cam_max = torch.max(sub_cam.view(N, -1), dim=-1)[0].view(N, 1, 1)
            thr = (sub_cam_max * gama)
            thr = thr.expand(sub_cam.shape)
            sub_cam_with_drop = torch.where(sub_cam > thr, zero, sub_cam)
            mean_drop_cam = mean_drop_cam + sub_cam_with_drop
        mean_drop_cam = mean_drop_cam / C  # Diviser par C au lieu de 4
        mean_drop_cam = torch.unsqueeze(mean_drop_cam, dim=1)

        x = x * mean_drop_cam
        return x


class CNN_PDA(nn.Module): 
    def __init__(self, num_classes: int, gamma: float = 0.5):
        super(CNN_PDA, self).__init__()

        self.model = models.resnet18(pretrained=True)

        self.num_features = self.model.fc.in_features
        
        self.model.fc = nn.Linear(self.num_features, num_classes)
        
        self.attention_module = Attention_Module()
        
        self.gamma = gamma
        
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = self.model.fc

    def forward(self, x, apply_attention=None):

        features = self.feature_extractor(x)
        
        if apply_attention and self.training:
            fc_weights = self.fc.weight.unsqueeze(-1).unsqueeze(-1)
            features = self.attention_module(features, fc_weights, self.gamma)

        features = self.avgpool(features)
        features = torch.flatten(features, 1)

        return self.fc(features)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, apply_attention=False)
            probs = F.softmax(logits, dim=1)
        return probs
    
    def evaluate(self, dataloader: DataLoader, device: torch.device):
        self.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                logits = self.forward(x, apply_attention=False)
                _, predicted = torch.max(logits, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def get_attention_maps(self, x):
        self.eval()
        with torch.no_grad():
            features = self.feature_extractor(x)
            fc_weights = self.fc.weight.unsqueeze(-1).unsqueeze(-1)
            
            cams = F.conv2d(features, fc_weights)
            cams = F.relu(cams)
            
        return cams
