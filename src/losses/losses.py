import numpy as np

from torch import Tensor
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from src.utils import concat_all_gather


class KLDivergence(nn.Module):
    "KL divergence between the estimated normal distribution and a prior distribution"
    def __init__(self):
        super(KLDivergence, self).__init__()
        """
        N :  the index N spans all dimensions of input 
        N = H x W x D
        """
        self.N = 80*96*80
    def forward(self, z_mean, z_log_sigma):
        z_log_var = z_log_sigma * 2
        return 0.5 * ((z_mean**2 + z_log_var.exp() - z_log_var - 1).sum())

class L2Loss(nn.Module): 
    "Measuring the `Euclidian distance` between prediction and ground truh using `L2 Norm`"
    def __init__(self):
        super(L2Loss, self).__init__()
        
    def forward(self, x, y): 
        N = y.shape[0]*y.shape[1]*y.shape[2]*y.shape[3]*y.shape[4]
        return  ( (x - y)**2 ).sum() / N

class L1Loss(nn.Module): 
    "Measuring the `Euclidian distance` between prediction and ground truh using `L1 Norm`"
    def __init__(self):
        super(L1Loss, self).__init__()
        
    def forward(self, x, y): 
        N = y.shape[0]*y.shape[1]*y.shape[2]*y.shape[3]*y.shape[4]
        return  ( (x - y).abs()).sum() / N


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
                
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + \
            batch_center * (1 - self.center_momentum)
