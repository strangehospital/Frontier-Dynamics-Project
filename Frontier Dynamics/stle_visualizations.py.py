"""
STLE Visualization Generator
Creates visual demonstrations of STLE concepts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from stle_minimal_demo import MinimalSTLE, generate_moons_data, generate_circles_data


def create_decision_boundary_plot(model, X_train, y_train, X_test, save_path):
    """
    Visualize decision boundary with accessibility coloring
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Create meshgrid
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predict on grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_pred = model.predict(grid_points)
    
    # Plot 1: Classification Decision Boundary
    Z_class = grid_pred['predictions'].reshape(xx.shape)
    axes[0].contourf(xx, yy, Z_class, alpha=0.4, cmap='RdYlBu', levels=1)
    axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', 
                   edgecolors='k', s=50, linewidths=1.5, label='Training')
    axes[0].set_title('Classification Decision Boundary', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].legend()
    
    # Plot 2: Accessibility μ_x
    Z_mu_x = grid_pred['mu_x'].reshape(xx.shape)
    im = axes[1].contourf(xx, yy, Z_mu_x, levels=20, cmap='viridis', alpha=0.8)
    axes[1].scatter(X_train[:, 0], X_train[:, 1], c='red', marker='x', 
                   s=30, linewidths=1, label='Training data')
    axes[1].set_title('Accessibility μ_x (Knowledge Map)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    plt.colorbar(im, ax=axes[1], label='μ_x')
    axes[1].legend()
    
    # Plot 3: Learning Frontier
    # Identify frontier regions
    frontier_mask = (grid_pred['mu_x'] > 0.2) & (grid_pred['mu_x'] < 0.8)
    accessible_mask = grid_pred['mu_x'] >= 0.8
    inaccessible_mask = grid_pred['mu_x'] <= 0.2
    
    Z_frontier = np.zeros_like(Z_mu_x)
    Z_frontier[accessible_mask.reshape(xx.shape)] = 2  # Accessible
    Z_frontier[frontier_mask.reshape(xx.shape)] = 1  # Frontier
    Z_frontier[inaccessible_mask.reshape(xx.shape)] = 0  # Inaccessible
    
    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']  # Red, Yellow, Green
    axes[2].contourf(xx, yy, Z_frontier, levels=[0, 0.5, 1.5, 2.5], 
                    colors=colors, alpha=0.6)
    axes[2].scatter(X_train[:, 0], X_train[:, 1], c='black', marker='x', 
                   s=30, linewidths=1.5, label='Training data')
    
    # Legend
    red_patch = mpatches.Patch(color='#ff6b6b', label='Inaccessible (μ_x < 0.2)')
    yellow_patch = mpatches.Patch(color='#ffd93d', label='Frontier (0.2 ≤ μ_x < 0.8)')
    green_patch = mpatches.Patch(color='#6bcf7f', label='Accessible (μ_x ≥ 0.8)')
    axes[2].legend(handles=[green_patch, yellow_patch, red_patch], loc='upper right')
    
    axes[2].set_title('Learning Frontier (x ∩ y)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Feature 1')
    axes[2].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    return fig


def create_ood_comparison_plot(model, X_id, X_ood, save_path):
    """
    Visualize in-distribution vs out-of-distribution accessibility
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Predict
    pred_id = model.predict(X_id)
    pred_ood = model.predict(X_ood)
    
    # Plot 1: Scatter plot with μ_x coloring
    scatter_id = axes[0].scatter(X_id[:, 0], X_id[:, 1], c=pred_id['mu_x'], 
                                cmap='viridis', s=50, alpha=0.7, 
                                edgecolors='k', linewidths=0.5,
                                vmin=0, vmax=1, label='In-Distribution')
    scatter_ood = axes[0].scatter(X_ood[:, 0], X_ood[:, 1], c=pred_ood['mu_x'], 
                                 cmap='viridis', s=50, alpha=0.7, 
                                 marker='^', edgecolors='k', linewidths=0.5,
                                 vmin=0, vmax=1, label='Out-of-Distribution')
    
    axes[0].set_title('Accessibility: ID vs OOD', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].legend()
    plt.colorbar(scatter_id, ax=axes[0], label='μ_x')
    
    # Plot 2: Histogram comparison
    axes[1].hist(pred_id['mu_x'], bins=30, alpha=0.6, label='In-Distribution (Moons)', 
                color='blue', edgecolor='black')
    axes[1].hist(pred_ood['mu_x'], bins=30, alpha=0.6, label='Out-of-Distribution (Circles)', 
                color='red', edgecolor='black')
    axes[1].axvline(pred_id['mu_x'].mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'ID mean: {pred_id["mu_x"].mean():.3f}')
    axes[1].axvline(pred_ood['mu_x'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'OOD mean: {pred_ood["mu_x"].mean():.3f}')
    
    axes[1].set_title('Accessibility Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('μ_x (Accessibility)')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    return fig


def create_uncertainty_plot(model, X_test, y_test, save_path):
    """
    Visualize epistemic vs aleatoric uncertainty
    """
    pred = model.predict(X_test)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Epistemic uncertainty
    scatter1 = axes[0].scatter(X_test[:, 0], X_test[:, 1], c=pred['epistemic'], 
                              cmap='Reds', s=100, alpha=0.7, edgecolors='k', linewidths=0.5)
    axes[0].set_title('Epistemic Uncertainty\n(Reducible with more data)', 
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    plt.colorbar(scatter1, ax=axes[0], label='Epistemic')
    
    # Plot 2: Aleatoric uncertainty
    scatter2 = axes[1].scatter(X_test[:, 0], X_test[:, 1], c=pred['aleatoric'], 
                              cmap='Blues', s=100, alpha=0.7, edgecolors='k', linewidths=0.5)
    axes[1].set_title('Aleatoric Uncertainty\n(Irreducible randomness)', 
                     fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    plt.colorbar(scatter2, ax=axes[1], label='Aleatoric')
    
    # Plot 3: Scatter plot epistemic vs aleatoric
    correct = pred['predictions'] == y_test
    axes[2].scatter(pred['epistemic'][correct], pred['aleatoric'][correct], 
                   alpha=0.6, s=50, label='Correct', color='green')
    axes[2].scatter(pred['epistemic'][~correct], pred['aleatoric'][~correct], 
                   alpha=0.6, s=50, label='Incorrect', color='red')
    axes[2].set_title('Uncertainty Decomposition', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Epistemic Uncertainty')
    axes[2].set_ylabel('Aleatoric Uncertainty')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    return fig


def create_complementarity_plot(model, X_test, save_path):
    """
    Visualize complementarity μ_x + μ_y = 1
    """
    pred = model.predict(X_test)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: μ_x vs μ_y scatter
    axes[0].scatter(pred['mu_x'], pred['mu_y'], alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
    axes[0].plot([0, 1], [1, 0], 'r--', linewidth=2, label='Perfect complementarity')
    axes[0].set_title('Complementarity: μ_x + μ_y = 1', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('μ_x (Accessibility)')
    axes[0].set_ylabel('μ_y (Inaccessibility)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    
    # Plot 2: Error distribution
    error = np.abs(pred['mu_x'] + pred['mu_y'] - 1.0)
    axes[1].hist(error, bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[1].axvline(error.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean error: {error.mean():.2e}')
    axes[1].axvline(error.max(), color='orange', linestyle='--', linewidth=2,
                   label=f'Max error: {error.max():.2e}')
    axes[1].set_title('Complementarity Error Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('|μ_x + μ_y - 1|')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    return fig


def generate_all_visualizations():
    """
    Generate complete set of STLE visualizations
    """
    print("\n" + "="*60)
    print("Generating STLE Visualizations")
    print("="*60 + "\n")
    
    np.random.seed(42)
    
    # Generate data
    print("Generating datasets...")
    X_train, y_train = generate_moons_data(n_samples=400)
    X_test, y_test = generate_moons_data(n_samples=200)
    X_ood, _ = generate_circles_data(n_samples=300)
    
    # Train model
    print("Training STLE model...")
    model = MinimalSTLE(input_dim=2, num_classes=2)
    model.fit(X_train, y_train, epochs=100, lr=0.05)
    print()
    
    # Create output directory
    import os
    os.makedirs('/mnt/user-data/outputs', exist_ok=True)
    
    # Generate visualizations
    print("Creating visualizations...\n")
    
    create_decision_boundary_plot(
        model, X_train, y_train, X_test,
        '/mnt/user-data/outputs/stle_decision_boundary.png'
    )
    
    create_ood_comparison_plot(
        model, X_test, X_ood,
        '/mnt/user-data/outputs/stle_ood_comparison.png'
    )
    
    create_uncertainty_plot(
        model, X_test, y_test,
        '/mnt/user-data/outputs/stle_uncertainty_decomposition.png'
    )
    
    create_complementarity_plot(
        model, X_test,
        '/mnt/user-data/outputs/stle_complementarity.png'
    )
    
    print("\n" + "="*60)
    print("All visualizations generated successfully!")
    print("="*60)


if __name__ == "__main__":
    generate_all_visualizations()
