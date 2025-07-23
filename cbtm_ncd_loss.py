import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from vdpmm_pytorch import VDPMMExpectation, normwish
from utils import MarginLoss, entropy
from itertools import info_nce_logits

class AverageMeter:
    def __init__(self, name, fmt=':.4f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class CBTM_NCD_Loss(nn.Module):
    def __init__(self, n_clusters, feature_dim, device, batch_size, instance_temperature, cluster_temperature, entropy_q, temperature, threshold_known, threshold_novel):
        super(CBTM_NCD_Loss, self).__init__()
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim
        self.device = device
        self.batch_size = batch_size
        self.entropy_q = entropy_q
        self.temperature = temperature
        self.threshold_known = threshold_known
        self.threshold_novel = threshold_novel
        self.bce = nn.BCELoss()
        self.ce = MarginLoss(m=-0.5)
        self.vdpmm_expectation = VDPMMExpectation().to(device)
        self.criterion_instance = InstanceLoss(batch_size, instance_temperature, device).to(device)
        self.criterion_cluster = ClusterLoss(n_clusters, cluster_temperature, device).to(device)
        self.elbo_losses = AverageMeter('elbo_loss')
        self.ce_sup_losses = AverageMeter('ce_sup_loss')
        self.con_unsup_losses = AverageMeter('con_unsup_loss')
        self.reg_losses = AverageMeter('reg_loss')
        self.ce_unsup_losses = AverageMeter('ce_unsup_loss')
        self.proto_losses = AverageMeter('proto_loss')
        self.instance_losses = AverageMeter('instance_loss')
        self.cluster_losses = AverageMeter('cluster_loss')
        self.ce_margin_losses = AverageMeter('ce_margin_loss')

    def compute_elbo(self, data, pseudo_data, labels, gammas, pseudo_gammas, params):
        log_p_theta = 0
        log_q_theta = 0
        log_p_z = torch.sum(gammas * torch.log(gammas + 1e-10))
        log_q_z = log_p_z
        log_p_x = 0
        for k in range(self.n_clusters):
            cluster_data = data[gammas.argmax(dim=1) == k]
            if len(cluster_data) > 0:
                log_p_x += torch.sum(normwish(cluster_data, params['mean'][k], params['beta'][k], params['a'][k], params['B'][:, :, k]))
        log_p_t = 0
        for k in range(self.n_clusters):
            cluster_pseudo = pseudo_data[pseudo_gammas.argmax(dim=1) == k]
            if len(cluster_pseudo) > 0:
                log_p_t += torch.sum(normwish(cluster_pseudo, params['mean'][k], params['beta'][k], params['a'][k], params['B'][:, :, k]))
        log_q_t = torch.sum(pseudo_gammas * torch.log(pseudo_gammas + 1e-10))
        log_p_y = 0
        for k in range(self.n_clusters):
            cluster_indices = (gammas.argmax(dim=1) == k).nonzero(as_tuple=True)[0]
            if len(cluster_indices) > 0:
                cluster_labels = labels[cluster_indices]
                log_p_y += -torch.sum((cluster_labels.float() - k) ** 2)
        elbo = log_p_theta - log_q_theta + log_p_z - log_q_z + log_p_x + log_p_t - log_q_t + log_p_y
        return elbo

    def compute_warmup_losses(self, output, output2, feat, feat2, instance_out1, instance_out2, cluster_out1, cluster_out2, target, labeled_len, proto_matrix):
        feat_l_w, feat_u_w = feat[:labeled_len], feat[labeled_len:]
        feat_l_s, feat_u_s = feat2[:labeled_len], feat2[labeled_len:]
        output_u_s = output2[labeled_len:]

        loss_instance = self.criterion_instance(instance_out1, instance_out2)
        loss_cluster = self.criterion_cluster(cluster_out1, cluster_out2)

        feat_unsupcon = torch.cat([feat_l_w, feat_u_w, feat_l_s, feat_u_s], dim=0)
        unsupcon_logits, unsup_labels = info_nce_logits(feat_unsupcon, self.temperature)
        loss_con_unsup = F.cross_entropy(unsupcon_logits, unsup_labels)

        prob_reg = F.softmax(output / self.entropy_q, dim=1)
        loss_reg = -entropy(torch.mean(prob_reg, 0))

        loss_ce_sup = F.cross_entropy(output[:labeled_len] / self.temperature, target)

        prob_u_s = F.softmax(output_u_s, dim=1)
        max_probs, targets_u_pl = torch.max(prob_u_s, dim=1)
        mask_novel = max_probs > self.threshold_novel
        mask_known = max_probs > self.threshold_known
        index_chosen_novel = mask_novel.nonzero().squeeze(1)
        index_chosen_known = mask_known.nonzero().squeeze(1)

        loss_ce_pseudo_novel = (F.cross_entropy(output_u_s / self.temperature, targets_u_pl, reduction='none')[
            index_chosen_novel]).mean() if index_chosen_novel.numel() > 0 else 0.0
        loss_ce_pseudo_known = (F.cross_entropy(output_u_s / self.temperature, targets_u_pl, reduction='none')[
            index_chosen_known]).mean() if index_chosen_known.numel() > 0 else 0.0
        loss_ce_unsup = loss_ce_pseudo_novel + loss_ce_pseudo_known

        feat_norm_u_w = F.normalize(feat_u_w, dim=1)
        logits_u_proto = torch.matmul(feat_norm_u_w, proto_matrix.T.detach())
        loss_proto = (F.cross_entropy(logits_u_proto / self.temperature, targets_u_pl, reduction='none')[
            index_chosen_novel]).mean() if index_chosen_novel.numel() > 0 else 0.0

        return loss_ce_sup, loss_con_unsup, loss_reg, loss_ce_unsup, loss_proto, loss_instance, loss_cluster

    def compute_ncd_losses(self, output, output2, instance_out1, instance_out2, cluster_out1, cluster_out2, target, labeled_len):
        loss_instance = self.criterion_instance(instance_out1, instance_out2)
        loss_cluster = self.criterion_cluster(cluster_out1, cluster_out2)
        loss_ce_margin = self.ce(output[:labeled_len], target)
        return loss_ce_margin, loss_instance, loss_cluster

    def forward(self, features, pseudo_features, labels, params, output, output2, feat, feat2, instance_out1, instance_out2, cluster_out1, cluster_out2, target, labeled_len, proto_matrix, epoch, warmup_epochs):
        gammas = self.vdpmm_expectation(features, params)
        pseudo_gammas = self.vdpmm_expectation(pseudo_features, params) if pseudo_features.size(0) > 0 else torch.zeros_like(gammas[:1])
        elbo = self.compute_elbo(features, pseudo_features, labels, gammas, pseudo_gammas, params)
        self.elbo_losses.update(-elbo.item(), self.batch_size)

        if epoch < warmup_epochs:
            loss_ce_sup, loss_con_unsup, loss_reg, loss_ce_unsup, loss_proto, loss_instance, loss_cluster = self.compute_warmup_losses(
                output, output2, feat, feat2, instance_out1, instance_out2, cluster_out1, cluster_out2, target, labeled_len, proto_matrix
            )
            self.ce_sup_losses.update(loss_ce_sup.item(), self.batch_size)
            self.con_unsup_losses.update(loss_con_unsup.item(), self.batch_size)
            self.reg_losses.update(loss_reg.item(), self.batch_size)
            self.ce_unsup_losses.update(loss_ce_unsup.item(), self.batch_size)
            self.proto_losses.update(loss_proto.item(), self.batch_size)
            self.instance_losses.update(loss_instance.item(), self.batch_size)
            self.cluster_losses.update(loss_cluster.item(), self.batch_size)
            total_loss = -elbo + loss_ce_sup + loss_con_unsup + 5 * loss_reg + loss_ce_unsup + loss_proto + loss_instance + loss_cluster
        else:
            loss_ce_margin, loss_instance, loss_cluster = self.compute_ncd_losses(
                output, output2, instance_out1, instance_out2, cluster_out1, cluster_out2, target, labeled_len
            )
            self.ce_margin_losses.update(loss_ce_margin.item(), self.batch_size)
            self.instance_losses.update(loss_instance.item(), self.batch_size)
            self.cluster_losses.update(loss_cluster.item(), self.batch_size)
            total_loss = loss_ce_margin + loss_instance + loss_cluster

        return total_loss