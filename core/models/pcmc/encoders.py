import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from lightly.models.modules import SimCLRProjectionHead, SwaVProjectionHead, SwaVPrototypes
from lightly.loss import SwaVLoss

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead

from lightly.loss import BarlowTwinsLoss
from lightly.models.modules import BarlowTwinsProjectionHead

class SimCLR(nn.Module):
    def __init__(self, arch, temperature, encoder_dim, pretrained=False):
        super().__init__()

        self.backbone = get_backbone(arch, pretrained)
        self.projection_head = SimCLRProjectionHead(encoder_dim, encoder_dim, 128)

        self.temperature = temperature
        self.cross_ent = torch.nn.CrossEntropyLoss().to('cuda')

    def info_nce_loss(self, h0, h1):

        features = torch.cat((h0, h1))

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.exp(torch.matmul(features, features.T) / self.temperature)

        mask = torch.eye(similarity_matrix.shape[0]).to('cuda')
        mask2 = torch.zeros_like(mask).to('cuda')
        mask2[:len(mask) // 2, :] = mask[len(mask) // 2:, :]
        mask2[len(mask) // 2:, :] = mask[:len(mask) // 2, :]
        positives = similarity_matrix[mask2.to(torch.bool)]

        mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool).to('cuda')
        mask2 = mask2[~mask].reshape(mask2.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].reshape(similarity_matrix.shape[0], -1)

        negatives = similarity_matrix[~mask2.to(torch.bool)].reshape(similarity_matrix.shape[0], -1)

        denom = torch.sum(negatives, dim=1)

        loss = torch.mean(-torch.log(positives / denom))

        return loss

    def forward(self, x):
        z = self.backbone(x).flatten(start_dim=1)
        h = self.projection_head(z)
        return h, z

    def training_step(self, batch):
        x0, x1 = batch
        h0, z0 = self.forward(x0.to('cuda'))
        h1, z1 = self.forward(x1.to('cuda'))
        
        loss = self.info_nce_loss(h0, h1)

        return loss 
    
    def embed(self, x):
        with torch.no_grad():
            z = self.backbone(x).flatten(start_dim=1)
            return z

    def configure_optimizers(self, lr, wd, epochs, num_iters):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=
            lambda step: self.get_lr(  # pylint: disable=g-long-lambda
                step,
                epochs * num_iters,
                lr,  # lr_lambda computes multiplicative factor
                1e-4
            )
        )
        
        return optimizer, scheduler
        
    def get_lr(self, step, total_steps, lr_max, lr_min):
        """Compute learning rate according to cosine annealing schedule."""
        if total_steps == 0:
            total_steps = 1
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


class SwaV(nn.Module):
    def __init__(self, arch, temperature, encoder_dim, n_prototypes, pretrained=False):
        super().__init__()


        self.backbone = get_backbone(arch, pretrained)
        self.projection_head = SwaVProjectionHead(encoder_dim, encoder_dim, 128)
        self.prototypes = SwaVPrototypes(128, n_prototypes=n_prototypes)
        self.criterion = SwaVLoss(temperature)

    def forward(self, x):
        z = self.backbone(x).flatten(start_dim=1)
        h = self.projection_head(z)
        h = nn.functional.normalize(h, dim=1, p=2)
        p = self.prototypes(h)
        return p, z
    
    def embed(self, x):
        with torch.no_grad():
            z = self.backbone(x).flatten(start_dim=1)
            return z

    def training_step(self, batch):
        self.prototypes.normalize()
        x0, x1 = batch
        p0, z0 = self.forward(x0.to('cuda'))
        p1, z1 = self.forward(x1.to('cuda'))

        loss = self.criterion([p0, p1], [])
        return loss

    def configure_optimizers(self, lr, wd, epochs, num_iters):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=
            lambda step: self.get_lr(  # pylint: disable=g-long-lambda
                step,
                epochs * num_iters,
                lr,  # lr_lambda computes multiplicative factor
                1e-4
            )
        )
        
        return optimizer, scheduler
        
    def get_lr(self, step, total_steps, lr_max, lr_min):
        """Compute learning rate according to cosine annealing schedule."""
        if total_steps == 0:
            total_steps = 1
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


class SimSiam(nn.Module):
    def __init__(self, arch, encoder_dim, pretrained=False):
        super().__init__()


        self.backbone = get_backbone(arch, pretrained)
        self.projection_head = SimSiamProjectionHead(encoder_dim, encoder_dim, 128)
        self.prediction_head = SimSiamPredictionHead(128, 64, 128)
        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch):
        x0, x1 = batch
        z0, p0 = self.forward(x0.to('cuda'))
        z1, p1 = self.forward(x1.to('cuda'))
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        return loss

    def embed(self, x):
        with torch.no_grad():
            z = self.backbone(x).flatten(start_dim=1)
            return z

    def configure_optimizers(self, lr, wd, epochs, num_iters):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=
            lambda step: self.get_lr(  # pylint: disable=g-long-lambda
                step,
                epochs * num_iters,
                lr,  # lr_lambda computes multiplicative factor
                1e-4
            )
        )
        
        return optimizer, scheduler
        
    def get_lr(self, step, total_steps, lr_max, lr_min):
        """Compute learning rate according to cosine annealing schedule."""
        if total_steps == 0:
            total_steps = 1
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


class BarlowTwins(nn.Module):
    def __init__(self, arch, encoder_dim, pretrained=False):
        super().__init__()

        self.backbone = get_backbone(arch, pretrained)
        self.projection_head = BarlowTwinsProjectionHead(encoder_dim, 2048, 2048)
        self.criterion = BarlowTwinsLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch):
        x0, x1 = batch
        z0 = self.forward(x0.to('cuda'))
        z1 = self.forward(x1.to('cuda'))
        loss = self.criterion(z0, z1)
        return loss

    def embed(self, x):
        with torch.no_grad():
            z = self.backbone(x).flatten(start_dim=1)
            return z

    def configure_optimizers(self, lr, wd, epochs, num_iters):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda=
            lambda step: self.get_lr(  # pylint: disable=g-long-lambda
                step,
                epochs * num_iters,
                lr,  # lr_lambda computes multiplicative factor
                1e-4
            )
        )
        
        return optimizer, scheduler
        
    def get_lr(self, step, total_steps, lr_max, lr_min):
        """Compute learning rate according to cosine annealing schedule."""
        if total_steps == 0:
            total_steps = 1
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
    


def get_backbone(arch, pretrained):

    if arch == 'resnet18':
        weights = torchvision.models.ResNet18_Weights if pretrained else None
        backbone = torchvision.models.resnet18(weights)
        backbone.fc = torch.nn.Identity()
        return backbone
    elif arch == 'resnet34':
        weights = torchvision.models.ResNet34_Weights if pretrained else None
        backbone = torchvision.models.resnet34(weights)
        backbone.fc = torch.nn.Identity()
        return backbone
    elif arch == 'resnet50':
        weights = torchvision.models.ResNet50_Weights if pretrained else None
        backbone = torchvision.models.resnet50(weights)
        backbone.fc = torch.nn.Identity()
        return backbone