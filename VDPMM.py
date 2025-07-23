import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import psi  # For digamma function, used in expectation
import numpy as np

def normwish(data, mean, beta, a, B):
    """
    Placeholder for normal-Wishart log-probability.
    Args:
        data: torch.Tensor of shape (N, D), input data
        mean: torch.Tensor of shape (D,), component mean
        beta: float, precision scaling
        a: float, degrees of freedom for Wishart
        B: torch.Tensor of shape (D, D), scale matrix
    Returns:
        log_prob: torch.Tensor of shape (N,), log-probability
    """
    N, D = data.shape
    # Assuming a multivariate normal with Wishart prior
    # This is a simplified version; replace with actual normwish if available
    diff = data - mean
    inv_B = torch.inverse(B)
    log_det_B = torch.logdet(B)
    log_prob = -0.5 * D * torch.log(torch.tensor(2 * np.pi)) \
               -0.5 * log_det_B \
               -0.5 * torch.sum(diff @ inv_B * diff, dim=1) \
               + 0.5 * a * log_det_B
    return log_prob

class VDPMMExpectation(nn.Module):
    def __init__(self):
        super(VDPMMExpectation, self).__init__()

    def forward(self, data, params):
        """
        Computes the expectation step for VDP-MM with Gaussian components.
        Args:
            data: torch.Tensor of shape (N, D), input data
            params: dict with keys:
                - 'a': torch.Tensor of shape (K,), degrees of freedom
                - 'g': torch.Tensor of shape (K, 2), variational parameters
                - 'mean': torch.Tensor of shape (K, D), component means
                - 'beta': torch.Tensor of shape (K,), precision scaling
                - 'B': torch.Tensor of shape (D, D, K), scale matrices
        Returns:
            gammas: torch.Tensor of shape (N, K), responsibility matrix
        """
        K = params['a'].shape[0]
        N, D = data.shape
        device = data.device

        eq_log_Vs = torch.zeros(K, 1, device=device)
        eq_log_1_Vs = torch.zeros(K, 1, device=device)
        log_V_prob = torch.zeros(K, 1, device=device)
        pob = torch.zeros(N, K, device=device)
        log_gamma_tilde = torch.zeros(N, K, device=device)

        for i in range(K):
            # Compute E[log V_s] and E[log (1-V_s)] using digamma function
            eq_log_Vs[i] = torch.tensor(psi(params['g'][i, 0].item()) - psi(params['g'][i, 0].item() + params['g'][i, 1].item()))
            eq_log_1_Vs[i] = torch.tensor(psi(params['g'][i, 1].item()) - psi(params['g'][i, 0].item() + params['g'][i, 1].item()))
            # Compute log V_prob
            log_V_prob[i] = eq_log_Vs[i] + torch.sum(eq_log_1_Vs[:i], dim=0)
            # Compute normal-Wishart probability
            pob[:, i] = normwish(data, params['mean'][i, :], params['beta'][i], params['a'][i], params['B'][:, :, :, i])
            # Compute log responsibilities
            log_gamma_tilde[:, i] = log_V_prob[i] + pob[:, i]

        gammas = log_gamma_tilde
        # Normalize responsibilities
        gammas = gammas / torch.sum(gammas, dim=1, keepdim=True)

        return gammas

class VDPMMMaximize(nn.Module):
    def __init__(self, a0=10, beta0=1.0):
        super(VDPMMMaximize, self).__init__()
        self.a0 = a0
        self.beta0 = beta0

    def forward(self, data, params, gammas):
        """
        Maximization step for VDP-MM with Gaussian components.
        Args:
            data: torch.Tensor of shape (N, D), input data
            params: dict with keys:
                - 'a': torch.Tensor of shape (K,), degrees of freedom
                - 'g': torch.Tensor of shape (K, 2), variational parameters
                - 'beta': torch.Tensor of shape (K,), precision scaling
                - 'mean': torch.Tensor of shape (K, D), component means
                - 'B': torch.Tensor of shape (D, D, K), scale matrices
                - 'eq_alpha': float, expected concentration parameter
            gammas: torch.Tensor of shape (N, K), responsibility matrix
        Returns:
            params: dict with updated variational parameters
        """
        D = data.shape[1]
        N = data.shape[0]
        K = params['a'].shape[0]
        device = data.device

        gammas = torch.where(torch.isnan(gammas), torch.tensor(0.0, device=device), gammas)
        # Compute convenience variables
        Ns = torch.sum(gammas, dim=0) + 1e-10
        mus = torch.zeros(K, K, D, device=device)
        sigs = torch.zeros(D, D, K, device=device)
        mean0 = torch.mean(data, dim=0)
        B0 = 0.1 * D * torch.cov(data.T)

        # Compute mus
        mus = torch.matmul(gammas.T, data) / Ns.unsqueeze(-1)
        
        # Compute sigs
        for i in range(K):
            diff0 = data - mus[i, :].unsqueeze(0).repeat(N, 1)
            diff1 = diff0 * gammas[:, i].sqrt().unsqueeze(1)
            sigs[:, :, i] = torch.matmul(diff1.T, diff1)

        # Update variational parameters
        params['g'][:, 0] = 1.0 + Ns
        temp1 = params['eq_alpha'] + torch.flip(torch.cumsum(torch.flip(Ns, (0,)), dim=0), (0,)) - Ns
        params['g'][:, 1] = temp1
        params['beta'] = Ns + self.beta0
        params['a'] = Ns + self.a0
        params['mean'] = (Ns.unsqueeze(-1) * mus + self.beta0 * mean0.unsqueeze(0)) / ((Ns + self.beta0).unsqueeze(-1))

        for i in range(K):
            diff = mus[i, :] - mean0
            params['B'][:, :, i] = sigs[:, :, i] + Ns[i] * self.beta0[i] * torch.outer(diff, diff) / (Ns[i] + self.beta0) + B0

        return params

class Autoencoder(nn.Module):
    def __init__(self,
         dims, activation='relu', init='he_normal'):
            """
            PyTorch autoencoder model, symmetric.
            Args:
                dims: list of number of units in each layer
                activation: str, activation function
                init: str, str kernel initializer
            """
            super(Autoencoder, self).__init__()
            self.dims = dims
            n_stacks = len(dims) - 1

            # Encoder layers
            encoder_layers = nn.ModuleList()
            for i in range(n_stacks):
                layer = nn.Linear(dims[i], dims[i+1])
                if init == 'glorot_uniform':
                    nn.init.uniform_(layer.weight, -0.1, 0.1)
                elif init == 'he_normal':
                    nn.init.kaiming_normal_(layer.weight)
                self.encoder_layers.append(layer)
                if i < n_stacks - 1 and activation:
                    if activation == 'relu':
                        self.encoder_layers.append(nn.ReLU())
                    elif activation == 'elu':
                        self.encoder_layers.append(nn.ELU())

            # Decoder layers
            self.decoder_layers = nn.ModuleList()
            for i in range(n_stacks-1, -1, -1):
                layer = nn.Linear(dims[i+1], dims[i])
                if init == 'glorot_uniform':
                    nn.init.uniform_(layer.weight, -0.1, 0.1)
                elif init == 'he_normal':
                    nn.init.kaiming_normal_(layer.weight)
                self.decoder_layers.append(layer)
                if i > 0 and activation:
                    if activation == 'relu':
                        self.decoder_layers.append(nn.ReLU())
                    elif activation == 'elu':
                        self.decoder_layers.append(nn.ELU())

    def forward(self, x):
            h = x
            # Encoder pass
            for layer in self.encoder_layers:
                h = layer(h)
            encoded = h
            # Decoder pass
            y = encoded
            for layer in self.decoder_layers:
                y = layer(y)
            return y, encoded

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, alpha=1.0):
        """
        Clustering layer using Student's t-distribution.
        Args:
            n_clusters: int, number of clusters
            alpha: float, degrees of freedom for t-distribution
        """
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha

    def forward(self, inputs, clusters):
        """
        Compute soft labels.
        Args:
            inputs: torch.Tensor of shape (n_samples, n_features)
            clusters: torch.Tensor of shape (n_clusters, n_features)
        Returns:
            q: torch.Tensor of shape (n_samples, n_clusters), soft labels
        """
        # Compute q_ij using Student's t-distribution
        diff = inputs.unsqueeze(1) - clusters.unsqueeze(0)
        q = 1.0 / (1.0 + torch.sum(diff ** 2, dim=2) / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

class DEC(nn.Module):
    def __init__(self, dims, n_clusters=10, alpha=1.0, init='he_normal'):
        """
        Deep Embedded Clustering model in PyTorch.
        Args:
            dims: list of number of units in each layer
            n_clusters: int, number of clusters
            alpha: float, degrees of freedom for clustering
            init: str, kernel initializer
        """
        super(DEC, self).__init__()
        self.autoencoder = Autoencoder(dims, init=init)
        self.clustering = nn.Parameter(torch.randn(n_clusters, dims[-1]))
        self.alpha = alpha

    def forward(self, x):
        _, encoded = self.autoencoder(x)
        q = ClusteringLayer(self.clustering.shape[0], self.alpha)(encoded, self.clustering)
        return q

    def pretrain(self, x, y=None, optimizer='adam', epochs=50, batch_size=256, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Pretrain the autoencoder.
        """
        self.autoencoder.to(device)
        x = torch.tensor(x, dtype=torch.float32, device=device)
        dataset = torch.utils.data.TensorDataset(x, x)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if optimizer == 'adam':
            optimizer = torch.optim.Adam(self.autoencoder.parameters())
        else:
            optimizer = torch.optim.SGD(self.autoencoder.parameters(), lr=1.0, momentum=0.9)
        
        criterion = nn.MSELoss()
        self.autoencoder.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                output, _ = self.autoencoder(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')

    def fit(self, x, y=None, maxiter=20000, batch_size=256, lr=0.01, tol=1e-3, update_interval=140, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Train the DEC model.
        """
        self.to(device)
        x = torch.tensor(x, dtype=torch.float32, device=device)
        dataset = torch.utils.data.TensorDataset(x)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        criterion = nn.KLDivLoss(reduction='sum')

        # Initialize cluster centers using k-means
        from sklearn.cluster import KMeans
        with torch.no_grad():
            _, encoded = self.autoencoder(x)
            kmeans = KMeans(n_clusters=self.clustering.shape[0], n_init=20)
            y_pred = kmeans.fit_predict(encoded.cpu().numpy())
            self.clustering.data.copy_(torch.tensor(kmeans.cluster_centers_, device=device))

        y_pred_last = y_pred
        for ite in range(maxiter):
            if ite % update_interval == 0:
                with torch.no_grad():
                    q = self(x)
                    p = self.target_distribution(q)
                    y_pred = q.argmax(1).cpu().numpy()
                    delta_label = np.sum(y_pred != y_pred_last) / y_pred.shape[0]
                    y_pred_last = np.copy(y_pred)
                    if ite > 0 and delta_label < tol:
                        print(f"Iteration {ite}, delta_label {delta_label} < tol {tol}, stopping training")
                        break

            self.train()
            total_loss = 0
            for batch_x, in dataloader:
                optimizer.zero_grad()
                q = self(batch_x[0])
                p = self.target_distribution(q)
                loss = criterion(F.log_softmax(q, dim=1), p)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if ite % update_interval == 0:
                print(f"Iteration {ite}, Loss: {total_loss}")

        return y_pred

    @staticmethod
    def target_distribution(q):
        """
        Compute target distribution for DEC training.
        """
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T