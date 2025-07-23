import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
from cbtm_ncd_data import load_data
from cbtm_ncd_model import CBTM_NCD
from utils import cluster_acc

def main():
    parser = argparse.ArgumentParser(description='CBTM-NCD Training')
    parser.add_argument('--dataset', type=str, default='cifar10-lt', choices=['cifar10-lt', 'cifar100-lt', 'imagenet'])
    parser.add_argument('--dataset_root', type=str, default='./data', help='Dataset root directory')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--labeled_num', type=int, default=5, help='Number of labeled classes')
    parser.add_argument('--labeled_ratio', type=float, default=0.0, help='Ratio of labeled data')
    parser.add_argument('--num_classes', type=int, default=10, help='Total number of classes')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warm-up epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--instance_temperature', type=float, default=0.5, help='Temperature for instance loss')
    parser.add_argument('--cluster_temperature', type=float, default=5.0, help='Temperature for cluster loss')
    parser.add_argument('--entropy_q', type=float, default=1.0, help='Entropy temperature')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for pseudo losses')
    parser.add_argument('--threshold_known', type=float, default=0.9, help='Threshold for known classes')
    parser.add_argument('--threshold_novel', type=float, default=0.7, help='Threshold for novel classes')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_label_loader, train_unlabel_loader, test_loader, num_classes = load_data(args)
    model = CBTM_NCD(args, num_classes, device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scaler = StandardScaler()
    tf_writer = SummaryWriter()

    for epoch in range(args.epochs):
        train_loss = model.train_epoch(train_label_loader, train_unlabel_loader, optimizer, epoch, tf_writer, scaler)
        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}")

    torch.save(model.state_dict(), f'model_{args.dataset}.pth')
    tf_writer.close()

if __name__ == "__main__":
    main()