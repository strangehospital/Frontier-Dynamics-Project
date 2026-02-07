"""
Set Theoretic Learning Environment (STLE) - Core Implementation
Proof of Concept Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List


class SimpleNormalizingFlow(nn.Module):
    """
    Simplified Normalizing Flow for density estimation
    Uses RealNVP-style affine coupling layers
    """
    def __init__(self, dim: int, num_layers: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        # Coupling layers
        self.scale_nets = nn.ModuleList()
        self.translate_nets = nn.ModuleList()
        
        for _ in range(num_layers):
            self.scale_nets.append(nn.Sequential(
                nn.Linear(dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim // 2),
                nn.Tanh()  # Bounded for stability
            ))
            
            self.translate_nets.append(nn.Sequential(
                nn.Linear(dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim // 2)
            ))
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform from base to data distribution"""
        log_det_sum = torch.zeros(z.shape[0], device=z.device)
        x = z
        
        for i in range(self.num_layers):
            # Split
            x1, x2 = x.chunk(2, dim=-1)
            
            # Affine coupling
            s = self.scale_nets[i](x1)
            t = self.translate_nets[i](x1)
            
            x2_new = x2 * torch.exp(s) + t
            x = torch.cat([x1, x2_new], dim=-1)
            
            log_det_sum += s.sum(dim=-1)
            
            # Permute for next layer (simple swap)
            if i < self.num_layers - 1:
                x = torch.cat([x[:, self.dim//2:], x[:, :self.dim//2]], dim=-1)
        
        return x, log_det_sum
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform from data to base distribution"""
        log_det_sum = torch.zeros(x.shape[0], device=x.device)
        z = x
        
        for i in reversed(range(self.num_layers)):
            # Unpermute
            if i < self.num_layers - 1:
                z = torch.cat([z[:, self.dim//2:], z[:, :self.dim//2]], dim=-1)
            
            # Split
            z1, z2 = z.chunk(2, dim=-1)
            
            # Inverse affine coupling
            s = self.scale_nets[i](z1)
            t = self.translate_nets[i](z1)
            
            z2_new = (z2 - t) * torch.exp(-s)
            z = torch.cat([z1, z2_new], dim=-1)
            
            log_det_sum -= s.sum(dim=-1)
        
        return z, log_det_sum
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability density"""
        z, log_det = self.inverse(x)
        
        # Standard normal base distribution
        log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * self.dim * np.log(2 * np.pi)
        
        return log_pz + log_det


class STLEModel(nn.Module):
    """
    Complete STLE Implementation
    """
    def __init__(self, input_dim: int, latent_dim: int = 32, num_classes: int = 10):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder: map data to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, latent_dim)
        )
        
        # Normalizing Flow per class
        self.flows = nn.ModuleList([
            SimpleNormalizingFlow(latent_dim, num_layers=4, hidden_dim=32)
            for _ in range(num_classes)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        # Certainty budget (learned from data)
        self.register_buffer('class_counts', torch.ones(num_classes))
        self.register_buffer('total_samples', torch.tensor(1.0))
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space"""
        return self.encoder(x)
    
    def compute_densities(self, z: torch.Tensor) -> torch.Tensor:
        """Compute density for each class"""
        batch_size = z.shape[0]
        log_densities = torch.zeros(batch_size, self.num_classes, device=z.device)
        
        for c in range(self.num_classes):
            log_densities[:, c] = self.flows[c].log_prob(z)
        
        # Convert to densities (with numerical stability)
        densities = torch.exp(log_densities - log_densities.max(dim=-1, keepdim=True)[0])
        
        return densities
    
    def compute_mu_x(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute accessibility μ_x(x)
        Returns: (mu_x, alpha) where alpha are Dirichlet parameters
        """
        # Encode
        z = self.encode(x)
        
        # Compute densities
        densities = self.compute_densities(z)
        
        # Pseudo-counts (certainty budget allocation)
        # beta = N_c * P(z | class_c)
        beta = self.class_counts.unsqueeze(0) * densities
        
        # Dirichlet concentration (alpha = beta_prior + beta_data)
        beta_prior = 1.0  # Flat prior
        alpha = beta_prior + beta
        alpha_0 = alpha.sum(dim=-1, keepdim=True)
        
        # Accessibility = max concentration / total concentration
        # High alpha_0 means high certainty
        mu_x = alpha.max(dim=-1)[0] / alpha_0.squeeze(-1)
        
        return mu_x, alpha
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass with all uncertainty estimates
        """
        # Encode
        z = self.encode(x)
        
        # Classification logits
        logits = self.classifier(z)
        
        # Compute accessibility and Dirichlet parameters
        mu_x, alpha = self.compute_mu_x(x)
        mu_y = 1.0 - mu_x
        
        # Predictive distribution (mean of Dirichlet)
        alpha_0 = alpha.sum(dim=-1, keepdim=True)
        p_pred = alpha / alpha_0
        
        # Epistemic uncertainty (inverse of total concentration)
        epistemic = 1.0 / alpha_0.squeeze(-1)
        
        # Aleatoric uncertainty (entropy of predictive distribution)
        aleatoric = -(p_pred * torch.log(p_pred + 1e-10)).sum(dim=-1)
        
        return {
            'logits': logits,
            'prediction': p_pred,
            'mu_x': mu_x,
            'mu_y': mu_y,
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'alpha': alpha,
            'latent': z
        }
    
    def update_class_counts(self, y_train: torch.Tensor):
        """Update certainty budget based on training data"""
        self.class_counts = torch.bincount(
            y_train, minlength=self.num_classes
        ).float()
        self.total_samples = torch.tensor(float(len(y_train)))


class STLELoss(nn.Module):
    """
    PAC-Bayes regularized STLE training loss
    """
    def __init__(self, num_classes: int, entropy_weight: float = 1e-5, 
                 kl_weight: float = 0.01):
        super().__init__()
        self.num_classes = num_classes
        self.entropy_weight = entropy_weight
        self.kl_weight = kl_weight
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss components
        """
        alpha = outputs['alpha']
        logits = outputs['logits']
        
        # Standard cross-entropy
        ce_loss = F.cross_entropy(logits, targets)
        
        # Uncertain cross-entropy (UCE) - encourage high concentration on correct class
        alpha_0 = alpha.sum(dim=-1)
        target_alpha = alpha[torch.arange(len(targets)), targets]
        
        # Digamma trick for differentiability
        uce_loss = -torch.mean(
            torch.digamma(target_alpha + 1e-10) - torch.digamma(alpha_0 + 1e-10)
        )
        
        # Entropy regularizer (encourage uncertainty far from data)
        entropy_reg = self.dirichlet_entropy(alpha).mean()
        
        # KL complexity penalty (prevent overfitting)
        kl_penalty = self.kl_to_uniform_prior(alpha)
        
        # Combined loss
        total_loss = (
            0.5 * ce_loss + 
            0.5 * uce_loss - 
            self.entropy_weight * entropy_reg + 
            self.kl_weight * kl_penalty
        )
        
        return {
            'total': total_loss,
            'ce': ce_loss,
            'uce': uce_loss,
            'entropy': entropy_reg,
            'kl': kl_penalty
        }
    
    def dirichlet_entropy(self, alpha: torch.Tensor) -> torch.Tensor:
        """Differential entropy of Dirichlet distribution"""
        alpha_0 = alpha.sum(dim=-1)
        K = alpha.shape[-1]
        
        entropy = (
            torch.lgamma(alpha_0) - torch.lgamma(alpha).sum(dim=-1) +
            (alpha_0 - K) * torch.digamma(alpha_0) -
            ((alpha - 1.0) * torch.digamma(alpha)).sum(dim=-1)
        )
        
        return entropy
    
    def kl_to_uniform_prior(self, alpha: torch.Tensor) -> torch.Tensor:
        """KL divergence to uniform Dirichlet prior"""
        prior_alpha = torch.ones_like(alpha)
        
        alpha_0 = alpha.sum(dim=-1)
        prior_alpha_0 = prior_alpha.sum(dim=-1)
        
        kl = (
            torch.lgamma(alpha_0) - torch.lgamma(prior_alpha_0) -
            torch.lgamma(alpha).sum(dim=-1) + torch.lgamma(prior_alpha).sum(dim=-1) +
            ((alpha - prior_alpha) * (torch.digamma(alpha) - 
             torch.digamma(alpha_0.unsqueeze(1)))).sum(dim=-1)
        )
        
        return kl.mean()


class STLETrainer:
    """
    Training pipeline for STLE
    """
    def __init__(self, model: STLEModel, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = STLELoss(model.num_classes)
        
    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, 
              X_val: torch.Tensor = None, y_val: torch.Tensor = None,
              epochs: int = 50, batch_size: int = 128, lr: float = 1e-3,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train STLE model
        """
        # Update class counts
        self.model.update_class_counts(y_train)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Move data to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        
        if X_val is not None:
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'train_mu_x': []
        }
        
        for epoch in range(epochs):
            self.model.train()
            
            # Shuffle training data
            perm = torch.randperm(len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]
            
            epoch_losses = []
            epoch_correct = 0
            epoch_total = 0
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_shuffled[i:i+batch_size]
                batch_y = y_train_shuffled[i:i+batch_size]
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                
                # Compute loss
                loss_dict = self.loss_fn(outputs, batch_y)
                loss = loss_dict['total']
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track metrics
                epoch_losses.append(loss.item())
                preds = outputs['logits'].argmax(dim=-1)
                epoch_correct += (preds == batch_y).sum().item()
                epoch_total += len(batch_y)
            
            # Epoch metrics
            train_loss = np.mean(epoch_losses)
            train_acc = epoch_correct / epoch_total
            
            # Compute mean mu_x on training data
            with torch.no_grad():
                train_outputs = self.model(X_train[:1000])  # Sample for speed
                train_mu_x = train_outputs['mu_x'].mean().item()
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_mu_x'].append(train_mu_x)
            
            # Validation
            if X_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val, batch_size)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | "
                          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                          f"Train μ_x: {train_mu_x:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | "
                          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                          f"Train μ_x: {train_mu_x:.4f}")
        
        return history
    
    def evaluate(self, X: torch.Tensor, y: torch.Tensor, 
                 batch_size: int = 128) -> Tuple[float, float]:
        """Evaluate model"""
        self.model.eval()
        
        X = X.to(self.device)
        y = y.to(self.device)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                outputs = self.model(batch_X)
                loss_dict = self.loss_fn(outputs, batch_y)
                
                total_loss += loss_dict['total'].item() * len(batch_X)
                preds = outputs['logits'].argmax(dim=-1)
                correct += (preds == batch_y).sum().item()
                total += len(batch_X)
        
        return total_loss / total, correct / total
    
    def predict(self, X: torch.Tensor, batch_size: int = 128) -> Dict[str, np.ndarray]:
        """Make predictions with uncertainty"""
        self.model.eval()
        
        X = X.to(self.device)
        
        all_outputs = {
            'predictions': [], 'mu_x': [], 'mu_y': [],
            'epistemic': [], 'aleatoric': [], 'probs': []
        }
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                outputs = self.model(batch_X)
                
                all_outputs['predictions'].append(outputs['logits'].argmax(dim=-1).cpu().numpy())
                all_outputs['mu_x'].append(outputs['mu_x'].cpu().numpy())
                all_outputs['mu_y'].append(outputs['mu_y'].cpu().numpy())
                all_outputs['epistemic'].append(outputs['epistemic'].cpu().numpy())
                all_outputs['aleatoric'].append(outputs['aleatoric'].cpu().numpy())
                all_outputs['probs'].append(outputs['prediction'].cpu().numpy())
        
        # Concatenate batches
        return {k: np.concatenate(v) for k, v in all_outputs.items()}


def compute_ood_metrics(mu_x_id: np.ndarray, mu_x_ood: np.ndarray) -> Dict[str, float]:
    """
    Compute OOD detection metrics
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    # Labels: 1 = ID, 0 = OOD
    labels = np.concatenate([np.ones(len(mu_x_id)), np.zeros(len(mu_x_ood))])
    scores = np.concatenate([mu_x_id, mu_x_ood])
    
    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)
    
    # FPR at 95% TPR
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]
    
    n_id = len(mu_x_id)
    tpr_threshold = int(0.95 * n_id)
    
    fpr_at_95_tpr = (sorted_labels[tpr_threshold:] == 0).sum() / len(mu_x_ood)
    
    return {
        'auroc': auroc,
        'aupr': aupr,
        'fpr_at_95_tpr': fpr_at_95_tpr
    }


if __name__ == "__main__":
    print("STLE Core Implementation Loaded Successfully")
    print("=" * 60)
    print("Available components:")
    print("  - STLEModel: Main model with accessibility computation")
    print("  - STLETrainer: Training pipeline")
    print("  - STLELoss: PAC-Bayes regularized loss")
    print("  - compute_ood_metrics: OOD detection evaluation")
