import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import clip
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from collections import Counter
import matplotlib
matplotlib.use("Agg")
from models.resnet import resnet50

# Assuming vdpmm_pytorch.py is available
from vdpmm_pytorch import VDPMMExpectation, VDPMMMaximize, normwish

# Data transformations
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Long-Tailed Dataset
class LongTailDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        for label, class_dir in enumerate(os.listdir(root_dir)):
            class_dir_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_dir_path):
                for img_file in os.listdir(class_dir_path):
                    img_path = os.path.join(class_dir_path, img_file)
                    if img_file.endswith(('.png', '.jpg', '.JPEG')):
                        self.data.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Pseudo-Sample Dataset
class PseudoSampleDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Load class list
def load_class_list(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# Calculate clustering accuracy
def calculate_acc(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-w)
    acc = w[row_ind, col_ind].sum() / y_pred.size
    return acc

# Unified ELBO-based CBTM-NCD Framework
class CBTM_NCD_Unified(nn.Module):
    def __init__(self, n_clusters=20, alpha=1.0, feature_dim=512, device='cuda'):
        super(CBTM_NCD_Unified, self).__init__()
        self.device = device
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.feature_dim = feature_dim
        self.feature_extractor = resnet50(num_classes=10).to(device)
        self.vdpmm_expectation = VDPMMExpectation().to(device)
        self.vdpmm_maximize = VDPMMMaximize(a0=10, beta0=1.0).to(device)
        self.transform = transform
        self.clip_model, self.clip_preprocess = clip.load("pretrained/ViT-B-32.pt", device=device)
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            "./stable-diffusion-2-base/models--stabilityai--stable-diffusion-2-base/snapshots/fa386bb446685d8ad8a8f06e732a66ad10be6f47/scheduler"
        )
        self.diffusion = StableDiffusionPipeline.from_pretrained(
            "./stable-diffusion-2-base/models--stabilityai--stable-diffusion-2-base/snapshots/fa386bb446685d8ad8a8f06e732a66ad10be6f47/",
            scheduler=self.scheduler,
            torch_dtype=torch.float16
        ).to(device)

    def forward(self, x):
        _, features = self.feature_extractor(x)
        features = features.squeeze(-1).squeeze(-1)
        return features

    def generate_pseudo_samples(self, cluster_labels, cluster_centers, text_list, dataset, n_samples=5):
        pseudo_images = []
        pseudo_labels = []
        cluster_counts = Counter(cluster_labels)
        for cluster_idx in range(self.n_clusters):
            if cluster_counts[cluster_idx] < 100:  # Target tail classes
                cluster_indices = np.where(cluster_labels == cluster_idx)[0]
                if len(cluster_indices) == 0:
                    continue
                cluster_images = [dataset[i][0] for i in cluster_indices]
                processed_images = [
                    self.clip_preprocess(Image.fromarray(np.uint8(img.permute(1, 2, 0).cpu().numpy() * 255))).unsqueeze(0)
                    for img in cluster_images
                ]
                image_batch = torch.cat(processed_images).to(self.device)
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_batch).cpu().numpy()
                    text_features = self.clip_model.encode_text(clip.tokenize(text_list).to(self.device)).cpu().numpy()
                similarity_matrix = image_features @ text_features.T
                best_word_idx = np.argmax(similarity_matrix.mean(axis=0))
                best_word = text_list[best_word_idx]
                for i in range(n_samples):
                    prompt = f"a picture of {best_word} {i + 1}"
                    image = self.diffusion(prompt).images[0]
                    image = image.resize((256, 256))
                    pseudo_images.append(transform(image).to(self.device))
                    pseudo_labels.append(cluster_idx)
        return pseudo_images, pseudo_labels

    def compute_elbo(self, data, pseudo_data, labels, params, gammas, pseudo_gammas):
        # ELBO terms
        log_p_theta = 0  # Prior on parameters (simplified)
        log_q_theta = 0  # Variational posterior on parameters (simplified)
        log_p_z = torch.sum(gammas * torch.log(gammas + 1e-10))  # p(Z | theta_k)
        log_q_z = log_p_z  # q(Z)
        log_p_x = 0  # p(X | Z, theta_k)
        for k in range(self.n_clusters):
            cluster_data = data[gammas.argmax(dim=1) == k]
            if len(cluster_data) > 0:
                log_p_x += torch.sum(normwish(cluster_data, params['mean'][k], params['beta'][k], params['a'][k], params['B'][:, :, k]))
        log_p_t = 0  # p(T | theta_k)
        for k in range(self.n_clusters):
            cluster_pseudo = pseudo_data[pseudo_gammas.argmax(dim=1) == k]
            if len(cluster_pseudo) > 0:
                log_p_t += torch.sum(normwish(cluster_pseudo, params['mean'][k], params['beta'][k], params['a'][k], params['B'][:, :, k]))
        log_q_t = torch.sum(pseudo_gammas * torch.log(pseudo_gammas + 1e-10))  # q(T)
        log_p_y = 0  # p(Y | T, theta_k) (simplified classification term)
        for k in range(self.n_clusters):
            cluster_indices = (gammas.argmax(dim=1) == k).nonzero(as_tuple=True)[0]
            if len(cluster_indices) > 0:
                cluster_labels = labels[cluster_indices]
                # Assume a simple Gaussian likelihood for labels
                log_p_y += -torch.sum((cluster_labels.float() - k) ** 2)  # Simplified
        elbo = log_p_theta - log_q_theta + log_p_z - log_q_z + log_p_x + log_p_t - log_q_t + log_p_y
        return elbo

    def train(self, dataloader, text_list, dataset, epochs=200, lr=0.01, max_iter=20):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scaler = StandardScaler()
        params = {
            'a': torch.ones(self.n_clusters, device=self.device),
            'g': torch.ones(self.n_clusters, 2, device=self.device),
            'mean': torch.randn(self.n_clusters, self.feature_dim, device=self.device),
            'beta': torch.ones(self.n_clusters, device=self.device),
            'B': torch.stack([torch.eye(self.feature_dim, device=self.device)] * self.n_clusters, dim=2),
            'eq_alpha': self.alpha
        }

        for epoch in range(epochs):
            self.train()
            total_elbo = 0
            features = []
            labels_list = []
            with torch.no_grad():
                for images, labels in dataloader:
                    images = images.to(self.device)
                    features.append(self.forward(images).cpu())
                    labels_list.append(labels)
            features = torch.cat(features, dim=0).numpy()
            labels = torch.cat(labels_list, dim=0).to(self.device)
            features_scaled = scaler.fit_transform(features)
            features_tensor = torch.tensor(features_scaled, device=self.device)

            # VDPMM clustering
            for _ in range(max_iter):
                gammas = self.vdpmm_expectation(features_tensor, params)
                cluster_labels = gammas.argmax(dim=1).cpu().numpy()
                params = self.vdpmm_maximize(features_tensor, params, gammas)

            # Generate pseudo-samples
            pseudo_images, pseudo_labels = self.generate_pseudo_samples(cluster_labels, params['mean'].cpu().numpy(), text_list, dataset)
            if pseudo_images:
                pseudo_dataset = PseudoSampleDataset(pseudo_images, pseudo_labels, transform=self.transform)
                pseudo_dataloader = DataLoader(pseudo_dataset, batch_size=64, shuffle=True)
                pseudo_features = []
                with torch.no_grad():
                    for images, _ in pseudo_dataloader:
                        images = images.to(self.device)
                        pseudo_features.append(self.forward(images).cpu())
                pseudo_features = torch.cat(pseudo_features, dim=0).numpy()
                pseudo_features_scaled = scaler.transform(pseudo_features)
                pseudo_features_tensor = torch.tensor(pseudo_features_scaled, device=self.device)
                pseudo_gammas = self.vdpmm_expectation(pseudo_features_tensor, params)
            else:
                pseudo_features_tensor = torch.zeros_like(features_tensor[:1])
                pseudo_gammas = torch.zeros_like(gammas[:1])

            # Optimize ELBO
            optimizer.zero_grad()
            elbo = self.compute_elbo(features_tensor, pseudo_features_tensor, labels, params, gammas, pseudo_gammas)
            loss = -elbo  # Maximize ELBO by minimizing -ELBO
            loss.backward()
            optimizer.step()
            total_elbo += elbo.item()

            print(f"Epoch {epoch+1}, ELBO: {total_elbo / len(dataloader)}")
            if epoch in [140, 180]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1

        # Final clustering and evaluation
        final_features = []
        final_labels = []
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                final_features.append(self.forward(images).cpu())
                final_labels.append(labels)
        final_features = torch.cat(final_features, dim=0).numpy()
        final_labels = torch.cat(final_labels, dim=0).numpy()
        final_features_scaled = scaler.fit_transform(final_features)
        final_features_tensor = torch.tensor(final_features_scaled, device=self.device)

        for _ in range(max_iter):
            gammas = self.vdpmm_expectation(final_features_tensor, params)
            cluster_labels = gammas.argmax(dim=1).cpu().numpy()
            params = self.vdpmm_maximize(final_features_tensor, params, gammas)

        acc = calculate_acc(final_labels, cluster_labels)
        print(f"Final Clustering Accuracy: {acc * 100:.2f}%")
        return cluster_labels

    def fit(self, data_dir, text_list_file, epochs=200):
        dataset = LongTailDataset(data_dir, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=1)
        text_list = load_class_list(text_list_file)
        cluster_labels = self.train(dataloader, text_list, dataset, epochs=200)
        return cluster_labels

# Example usage
if __name__ == "__main__":
    data_dir = './imagenet10_lt'
    text_list_file = 'imagenet10-lt.txt'
    model = CBTM_NCD_Unified(n_clusters=20, device='cuda' if torch.cuda.is_available() else 'cpu')
    model.fit(data_dir, text_list_file)