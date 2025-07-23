import torch
import torch.nn as nn
import torch.nn.functional as F
from models import resnet18, resnet50
from cbtm_ncd_loss import CBTM_NCD_Loss
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle

class CBTM_NCD(nn.Module):
    def __init__(self, args, num_classes, device):
        super(CBTM_NCD, self).__init__()
        self.args = args
        self.device = device
        self.num_classes = num_classes
        self.feature_dim = 512 if 'cifar' in args.dataset else 2048
        self.model = resnet18(num_classes=num_classes) if 'cifar' in args.dataset else resnet50(num_classes=num_classes)
        self.model.to(device)
        self.loss_fn = CBTM_NCD_Loss(
            n_clusters=num_classes, feature_dim=self.feature_dim, device=device,
            batch_size=args.batch_size, instance_temperature=args.instance_temperature,
            cluster_temperature=args.cluster_temperature, entropy_q=args.entropy_q,
            temperature=args.temperature, threshold_known=args.threshold_known,
            threshold_novel=args.threshold_novel
        )

        if args.dataset == 'cifar10-lt':
            state_dict = torch.load('./pretrained/simclr_cifar_10.pth.tar')
        elif args.dataset == 'cifar100-lt':
            state_dict = torch.load('./pretrained/simclr_cifar_100.pth.tar')
        elif args.dataset == 'imagenet':
            state_dict = torch.load('./pretrained/simclr_imagenet_100.pth.tar')
        self.model.load_state_dict(state_dict, strict=False)

        for name, param in self.model.named_parameters():
            if 'linear' not in name and 'layer4' not in name:
                param.requires_grad = False

        with torch.no_grad():
            dummy_input = torch.randn(2, 3, 32 if 'cifar' in args.dataset else 224, 32 if 'cifar' in args.dataset else 224).to(device)
            _, feat, _, _ = self.model(dummy_input)
            feat_dim = feat.shape[1]
        self.model.register_buffer('prototype', torch.zeros(num_classes, feat_dim).to(device))

    def forward(self, x):
        output, feat, instance_out, cluster_out = self.model(x)
        return output, feat, instance_out, cluster_out

    def update_prototype_safe(self, features, targets, momentum=0.9):
        with torch.no_grad():
            proto_copy = self.model.prototype.clone()
            valid_targets = targets[targets != -1]
            unique_classes = torch.unique(valid_targets)
            for cls_id in unique_classes:
                cls_mask = (valid_targets == cls_id)
                if cls_mask.any():
                    cls_features = features[cls_mask].mean(dim=0)
                    proto_copy[cls_id] = proto_copy[cls_id] * momentum + cls_features * (1 - momentum)
            self.model.prototype.copy_(proto_copy)

    def train_epoch(self, train_label_loader, train_unlabel_loader, optimizer, epoch, tf_writer, scaler):
        self.train()
        params = {
            'a': torch.ones(self.num_classes, device=self.device),
            'g': torch.ones(self.num_classes, 2, device=self.device),
            'mean': torch.randn(self.num_classes, self.feature_dim, device=self.device),
            'beta': torch.ones(self.num_classes, device=self.device),
            'B': torch.stack([torch.eye(self.feature_dim, device=self.device)] * self.num_classes, dim=2),
            'eq_alpha': 1.0
        }
        total_loss = 0
        unlabel_loader_iter = cycle(train_unlabel_loader)
        progress_bar = tqdm(train_label_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, ((x, x2), target) in enumerate(progress_bar):
            ((ux, ux2), _) = next(unlabel_loader_iter)
            x = torch.cat([x, ux], dim=0).to(self.device)
            x2 = torch.cat([x2, ux2], dim=0).to(self.device)
            target = map_categories_to_target(self.args, target).to(self.device)  # 假设映射函数存在
            labeled_len = len(target)

            optimizer.zero_grad()
            output, feat, instance_out1, cluster_out1 = self(x)
            output2, feat2, instance_out2, cluster_out2 = self(x2)

            features = feat[:labeled_len].detach()
            pseudo_features = feat[labeled_len:].detach()
            labels = target

            features_np = features.cpu().numpy()
            pseudo_features_np = pseudo_features.cpu().numpy()
            features_scaled = scaler.fit_transform(features_np) if features_np.size > 0 else features_np
            pseudo_features_scaled = scaler.transform(pseudo_features_np) if pseudo_features_np.size > 0 else pseudo_features_np
            features_tensor = torch.tensor(features_scaled, device=self.device)
            pseudo_features_tensor = torch.tensor(pseudo_features_scaled, device=self.device)

            loss = self.loss_fn(
                features_tensor, pseudo_features_tensor, labels, params,
                output, output2, feat, feat2, instance_out1, instance_out2,
                cluster_out1, cluster_out2, target, labeled_len,
                self.model.prototype, epoch, self.args.warmup_epochs
            )

            try:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0))
                optimizer.step()
            except RuntimeError as e:
                print(f"Gradient error handled: {e}")
                optimizer.zero_grad()
                continue

            self.update_prototype_safe(features, labels)
            optimizer.zero_grad()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / (batch_idx + 1))

        tf_writer.add_scalar('Loss/Total', total_loss / (batch_idx + 1), epoch)
        tf_writer.add_scalar('Loss/ELBO', self.loss_fn.elbo_losses.avg, epoch)
        tf_writer.add_scalar('Loss/CE_Sup', self.loss_fn.ce_sup_losses.avg, epoch)
        tf_writer.add_scalar('Loss/Con_Unsup', self.loss_fn.con_unsup_losses.avg, epoch)
        tf_writer.add_scalar('Loss/Reg', self.loss_fn.reg_losses.avg, epoch)
        tf_writer.add_scalarskyLoss/CE_Unsup', self.loss_fn.ce_unsup_losses.avg, epoch)
        tf_writer.add_scalar('Loss/Proto', self.loss_fn.proto_losses.avg, epoch)
        tf_writer.add_scalar('Loss/Instance', self.loss_fn.instance_losses.avg, epoch)
        tf_writer.add_scalar('Loss/Cluster', self.loss_fn.cluster_losses.avg, epoch)
        tf_writer.add_scalar('Loss/CE_Margin', self.loss_fn.ce_margin_losses.avg, epoch)
        return total_loss / (batch_idx + 1)