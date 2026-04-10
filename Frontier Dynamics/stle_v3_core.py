"""
Set Theoretic Learning Environment (STLE) v3 - Core Implementation
Evidence-Scaled Posterior Networks with Multi-Domain Dirichlet Formulation

Key differences from v2:
  - Evidence-scaled formula: α_c = β + λ·N_c·p(z|domain_c), μ_x = (α_0 - K) / α_0
  - Logsumexp stabilization for numerical safety at large N
  - Auto-calibrated evidence scale λ (prevents saturation)
  - Optional projection layer (encoder) for dimensionality reduction
  - Multi-domain native (K domains, not binary accessible/inaccessible)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional


class SimpleNormalizingFlow(nn.Module):
    """
    RealNVP-style normalizing flow for per-domain density estimation.
    """
    def __init__(self, dim: int, num_layers: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        self.scale_nets = nn.ModuleList()
        self.translate_nets = nn.ModuleList()
        
        for _ in range(num_layers):
            self.scale_nets.append(nn.Sequential(
                nn.Linear(dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim // 2),
                nn.Tanh()
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
            x1, x2 = x.chunk(2, dim=-1)
            s = self.scale_nets[i](x1)
            t = self.translate_nets[i](x1)
            x2_new = x2 * torch.exp(s) + t
            x = torch.cat([x1, x2_new], dim=-1)
            log_det_sum += s.sum(dim=-1)
            
            if i < self.num_layers - 1:
                x = torch.cat([x[:, self.dim // 2:], x[:, :self.dim // 2]], dim=-1)
        
        return x, log_det_sum
    
    def inverse(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform from data to base distribution"""
        log_det_sum = torch.zeros(x.shape[0], device=x.device)
        z = x
        
        for i in reversed(range(self.num_layers)):
            if i < self.num_layers - 1:
                z = torch.cat([z[:, self.dim // 2:], z[:, :self.dim // 2]], dim=-1)
            
            z1, z2 = z.chunk(2, dim=-1)
            s = self.scale_nets[i](z1)
            t = self.translate_nets[i](z1)
            z2_new = (z2 - t) * torch.exp(-s)
            z = torch.cat([z1, z2_new], dim=-1)
            log_det_sum -= s.sum(dim=-1)
        
        return z, log_det_sum
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability density"""
        z, log_det = self.inverse(x)
        log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * self.dim * np.log(2 * np.pi)
        return log_pz + log_det


class STLEv3Model(nn.Module):
    """
    STLE v3: Evidence-Scaled Posterior Networks
    
    Computes accessibility via multi-domain Dirichlet formulation:
        α_c = β + λ · N_c · p(z | domain_c)
        α_0 = Σ_c α_c
        μ_x = (α_0 - K) / α_0
    
    Components:
        - Optional encoder/projection (for high-dimensional inputs)
        - Per-domain normalizing flows (density estimation)
        - Evidence-scaled Dirichlet concentration
        - Classification head (for supervised training)
    
    Args:
        input_dim: dimension of input features
        latent_dim: dimension of latent space (if using projection)
        num_classes: number of domains/classes (K)
        use_projection: whether to project input to latent space
        flow_layers: number of coupling layers per flow
        flow_hidden: hidden dimension in flow coupling networks
        beta_prior: Dirichlet prior parameter (must be > 0)
        evidence_scale: initial λ (auto-calibrated after training)
    """
    def __init__(self, input_dim: int, latent_dim: int = 32, num_classes: int = 2,
                 use_projection: bool = True, flow_layers: int = 4,
                 flow_hidden: int = 64, beta_prior: float = 1.0,
                 evidence_scale: float = 0.01):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim if use_projection else input_dim
        self.num_classes = num_classes
        self.use_projection = use_projection
        self.beta_prior = beta_prior
        self.evidence_scale = evidence_scale
        
        # Optional encoder/projection layer
        if use_projection:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, latent_dim),
            )
        else:
            self.encoder = nn.Identity()
        
        # Per-domain normalizing flows
        self.flows = nn.ModuleList([
            SimpleNormalizingFlow(self.latent_dim, num_layers=flow_layers, hidden_dim=flow_hidden)
            for _ in range(num_classes)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        # Domain counts (certainty budget, updated from training data)
        self.register_buffer('class_counts', torch.ones(num_classes))
        self.register_buffer('total_samples', torch.tensor(1.0))
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Project input to latent space (or identity if no projection)."""
        return self.encoder(x)
    
    def compute_log_densities(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log-density under each domain's flow.
        
        Returns: [batch, K] tensor of log-densities.
        
        IMPORTANT: Uses torch.stack comprehension, NOT loop-based list.append().
        Loop accumulation causes O(epochs × batch × K) memory leak.
        """
        log_probs = torch.stack(
            [self.flows[c].log_prob(z) for c in range(self.num_classes)],
            dim=-1
        )
        return log_probs
    
    def compute_mu_x(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute accessibility μ_x using evidence-scaled Posterior Networks.
        
        Formula:
            α_c = β + λ · N_c · p(z | domain_c)
            α_0 = Σ_c α_c
            μ_x = (α_0 - K) / α_0
        
        Returns:
            mu_x: accessibility scores [batch]
            alpha: Dirichlet concentrations [batch, K]
        """
        z = self.encode(x)
        
        # Log-densities from per-domain flows
        log_probs = self.compute_log_densities(z)
        
        # Logsumexp stabilization (prevents underflow at log-densities ≈ -100)
        log_prob_max = log_probs.max(dim=-1, keepdim=True)[0]
        densities = torch.exp(log_probs - log_prob_max)
        
        # Evidence-scaled Dirichlet concentration
        # α_c = β + λ · N_c · p(z | domain_c)
        N_c = self.class_counts.unsqueeze(0)  # [1, K]
        evidence = self.evidence_scale * N_c * densities
        alpha = self.beta_prior + evidence
        alpha_0 = alpha.sum(dim=-1)  # [batch]
        
        # Accessibility: μ_x = (α_0 - K) / α_0
        K = self.num_classes
        mu_x = (alpha_0 - K) / alpha_0
        mu_x = torch.clamp(mu_x, min=0.0, max=1.0)
        
        return mu_x, alpha
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass with all uncertainty estimates.
        
        Returns dict with:
            logits: classification logits [batch, K]
            prediction: predictive distribution [batch, K]
            mu_x: accessibility [batch]
            mu_y: inaccessibility [batch]
            epistemic: epistemic uncertainty [batch]
            aleatoric: aleatoric uncertainty [batch]
            alpha: Dirichlet concentrations [batch, K]
            latent: latent representations [batch, latent_dim]
        """
        z = self.encode(x)
        logits = self.classifier(z)
        
        mu_x, alpha = self.compute_mu_x(x)
        mu_y = 1.0 - mu_x
        
        # Predictive distribution (mean of Dirichlet)
        alpha_0 = alpha.sum(dim=-1, keepdim=True)
        p_pred = alpha / alpha_0
        
        # Epistemic uncertainty (inverse of total concentration)
        epistemic = 1.0 / alpha_0.squeeze(-1)
        
        # Aleatoric uncertainty (entropy of predictive)
        aleatoric = -(p_pred * torch.log(p_pred + 1e-10)).sum(dim=-1)
        
        return {
            'logits': logits,
            'prediction': p_pred,
            'mu_x': mu_x,
            'mu_y': mu_y,
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'alpha': alpha,
            'latent': z,
        }
    
    def update_class_counts(self, y_train: torch.Tensor):
        """Update certainty budget from training labels."""
        self.class_counts = torch.bincount(
            y_train, minlength=self.num_classes
        ).float()
        self.total_samples = torch.tensor(float(len(y_train)))
    
    def calibrate_evidence_scale(self, X: torch.Tensor, target_mu_x: float = 0.9):
        """
        Auto-calibrate λ so training data achieves median μ_x ≈ target.
        Call after training the flows and projection.
        """
        self.eval()
        candidates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        best_lambda = self.evidence_scale
        best_error = float('inf')
        
        # Sample subset for efficiency
        n_sample = min(400, len(X))
        indices = torch.randperm(len(X))[:n_sample]
        X_sample = X[indices]
        
        with torch.no_grad():
            for lam in candidates:
                old_scale = self.evidence_scale
                self.evidence_scale = lam
                mu_x, _ = self.compute_mu_x(X_sample)
                median_mu_x = float(mu_x.median())
                error = abs(median_mu_x - target_mu_x)
                
                if error < best_error:
                    best_error = error
                    best_lambda = lam
                
                self.evidence_scale = old_scale
        
        self.evidence_scale = best_lambda
        return best_lambda


class STLEv3Loss(nn.Module):
    """
    Training loss for STLE v3.
    
    Combines:
        - Cross-entropy loss (classification)
        - Uncertain Cross-Entropy (UCE, Dirichlet-aware)
        - Entropy regularizer (encourage uncertainty far from data)
        - KL complexity penalty (prevent overfitting)
    """
    def __init__(self, num_classes: int, entropy_weight: float = 1e-5,
                 kl_weight: float = 0.01):
        super().__init__()
        self.num_classes = num_classes
        self.entropy_weight = entropy_weight
        self.kl_weight = kl_weight
    
    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute all loss components."""
        alpha = outputs['alpha']
        logits = outputs['logits']
        
        # Standard cross-entropy
        ce_loss = F.cross_entropy(logits, targets)
        
        # Uncertain Cross-Entropy (UCE)
        alpha_0 = alpha.sum(dim=-1)
        target_alpha = alpha[torch.arange(len(targets)), targets]
        
        # NaN guard: clamp alpha values before digamma
        alpha_0_safe = torch.clamp(alpha_0, min=1e-6)
        target_alpha_safe = torch.clamp(target_alpha, min=1e-6)
        
        uce_loss = -torch.mean(
            torch.digamma(target_alpha_safe) - torch.digamma(alpha_0_safe)
        )
        
        # Entropy regularizer
        entropy_reg = self.dirichlet_entropy(alpha).mean()
        
        # KL complexity penalty
        kl_penalty = self.kl_to_uniform_prior(alpha)
        
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
            'kl': kl_penalty,
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
        """KL divergence from posterior Dirichlet to uniform prior"""
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


class STLEv3Trainer:
    """
    Training pipeline for STLE v3.
    
    Handles:
        - Mini-batch training with loss computation
        - Class count updates (certainty budget)
        - Evidence scale calibration after training
        - Evaluation and prediction with full uncertainty
    """
    def __init__(self, model: STLEv3Model, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = STLEv3Loss(model.num_classes)
    
    def train(self, X_train: torch.Tensor, y_train: torch.Tensor,
              X_val: torch.Tensor = None, y_val: torch.Tensor = None,
              epochs: int = 50, batch_size: int = 128, lr: float = 1e-3,
              calibrate: bool = True, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train STLE v3 model.
        
        After training, auto-calibrates evidence scale λ if calibrate=True.
        """
        self.model.update_class_counts(y_train)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        
        if X_val is not None:
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'train_mu_x': [], 'evidence_scale': [],
        }
        
        for epoch in range(epochs):
            self.model.train()
            
            perm = torch.randperm(len(X_train))
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]
            
            epoch_losses = []
            epoch_correct = 0
            epoch_total = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss_dict = self.loss_fn(outputs, batch_y)
                loss = loss_dict['total']
                
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                
                epoch_losses.append(loss.item())
                preds = outputs['logits'].argmax(dim=-1)
                epoch_correct += (preds == batch_y).sum().item()
                epoch_total += len(batch_y)
            
            train_loss = np.mean(epoch_losses)
            train_acc = epoch_correct / epoch_total
            
            with torch.no_grad():
                sample_size = min(500, len(X_train))
                train_outputs = self.model(X_train[:sample_size])
                train_mu_x = train_outputs['mu_x'].mean().item()
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_mu_x'].append(train_mu_x)
            history['evidence_scale'].append(self.model.evidence_scale)
            
            if X_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val, batch_size)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | "
                          f"Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
                          f"Val Acc: {val_acc:.4f} | μ_x: {train_mu_x:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | "
                          f"Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
                          f"μ_x: {train_mu_x:.4f}")
        
        # Auto-calibrate evidence scale
        if calibrate:
            best_lambda = self.model.calibrate_evidence_scale(X_train)
            if verbose:
                print(f"Evidence scale λ calibrated: {best_lambda}")
        
        return history
    
    def evaluate(self, X: torch.Tensor, y: torch.Tensor,
                 batch_size: int = 128) -> Tuple[float, float]:
        """Evaluate model on data."""
        self.model.eval()
        X = X.to(self.device)
        y = y.to(self.device)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i + batch_size]
                batch_y = y[i:i + batch_size]
                
                outputs = self.model(batch_X)
                loss_dict = self.loss_fn(outputs, batch_y)
                
                total_loss += loss_dict['total'].item() * len(batch_X)
                preds = outputs['logits'].argmax(dim=-1)
                correct += (preds == batch_y).sum().item()
                total += len(batch_X)
        
        return total_loss / total, correct / total
    
    def predict(self, X: torch.Tensor, batch_size: int = 128) -> Dict[str, np.ndarray]:
        """Make predictions with full uncertainty quantification."""
        self.model.eval()
        X = X.to(self.device)
        
        all_outputs = {
            'predictions': [], 'mu_x': [], 'mu_y': [],
            'epistemic': [], 'aleatoric': [], 'probs': [],
        }
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i + batch_size]
                outputs = self.model(batch_X)
                
                all_outputs['predictions'].append(outputs['logits'].argmax(dim=-1).cpu().numpy())
                all_outputs['mu_x'].append(outputs['mu_x'].cpu().numpy())
                all_outputs['mu_y'].append(outputs['mu_y'].cpu().numpy())
                all_outputs['epistemic'].append(outputs['epistemic'].cpu().numpy())
                all_outputs['aleatoric'].append(outputs['aleatoric'].cpu().numpy())
                all_outputs['probs'].append(outputs['prediction'].cpu().numpy())
        
        return {k: np.concatenate(v) for k, v in all_outputs.items()}


def compute_ood_metrics(mu_x_id: np.ndarray, mu_x_ood: np.ndarray) -> Dict[str, float]:
    """Compute OOD detection metrics: AUROC, AUPR, FPR@95%TPR."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    labels = np.concatenate([np.ones(len(mu_x_id)), np.zeros(len(mu_x_ood))])
    scores = np.concatenate([mu_x_id, mu_x_ood])
    
    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)
    
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]
    n_id = len(mu_x_id)
    tpr_threshold = int(0.95 * n_id)
    fpr_at_95_tpr = (sorted_labels[tpr_threshold:] == 0).sum() / len(mu_x_ood)
    
    return {
        'auroc': auroc,
        'aupr': aupr,
        'fpr_at_95_tpr': fpr_at_95_tpr,
    }


if __name__ == "__main__":
    print("STLE v3 Core Implementation Loaded Successfully")
    print("=" * 60)
    print("Available components:")
    print("  - STLEv3Model: Evidence-scaled Posterior Networks")
    print("  - STLEv3Trainer: Training pipeline with λ calibration")
    print("  - STLEv3Loss: UCE + entropy + KL regularized loss")
    print("  - compute_ood_metrics: OOD detection evaluation")
    print()
    print("v3 formula: α_c = β + λ·N_c·p(z|c), μ_x = (α_0 - K) / α_0")
