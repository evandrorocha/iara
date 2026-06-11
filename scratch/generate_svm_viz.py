import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic dataset (overlapping moons)
X, y = make_moons(n_samples=100, noise=0.25, random_state=42)

# Setup plot
fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), dpi=300)
plt.subplots_adjust(hspace=0.35, wspace=0.25)

# Define grid for plotting decision boundaries
h = .02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Colors
color_class0 = '#e76f51' # Muted Coral
color_class1 = '#017ba5' # PEE Blue (UFRJ)
cmap_contour = ListedColormap(['#fff1ec', '#e9f4f8'])

# Configurations for the 4 subplots: (Title, C, gamma, description)
# Row 1: Impact of C (gamma = 1.0)
# Row 2: Impact of Gamma (C = 1.0)
configs = [
    # (Title, C, gamma, row, col, explanation)
    ("Baixo C (C = 0.1, γ = 1.0)\nMargem Larga, Mais Erros Tolerados", 0.1, 1.0, 0, 0),
    ("Alto C (C = 100.0, γ = 1.0)\nMargem Estreita, Pouco Tolerante a Erros", 100.0, 1.0, 0, 1),
    ("Baixo Gamma (C = 1.0, γ = 0.1)\nRaio de Influência Amplo (Fronteira Suave)", 1.0, 0.1, 1, 0),
    ("Alto Gamma (C = 1.0, γ = 10.0)\nRaio de Influência Estrito (Ajuste Local)", 1.0, 10.0, 1, 1)
]

for title, C, gamma, r, c in configs:
    ax = axes[r, c]
    
    # Fit model
    clf = SVC(C=C, gamma=gamma, kernel='rbf')
    clf.fit(X, y)
    
    # Plot decision boundary regions
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cmap_contour, alpha=0.9)
    
    # Plot margin lines and decision boundary
    Z_dec = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z_dec = Z_dec.reshape(xx.shape)
    # levels: -1 (margin), 0 (boundary), 1 (margin)
    ax.contour(xx, yy, Z_dec, levels=[-1, 0, 1], colors=['#b0b0b0', '#017ba5', '#b0b0b0'],
               linestyles=['--', '-', '--'], linewidths=[0.9, 1.6, 0.9])
    
    # Highlight support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=70,
               facecolors='none', edgecolors='#555555', linewidths=0.8, 
               label='Vetores de Suporte' if (r == 0 and c == 0) else "")
    
    # Plot data points
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c=color_class0, edgecolors='#444444', 
               linewidths=0.6, s=35, label='Classe A' if (r == 0 and c == 0) else "", alpha=0.95)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c=color_class1, edgecolors='#444444', 
               linewidths=0.6, s=35, label='Classe B' if (r == 0 and c == 0) else "", alpha=0.95)
    
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10, color='#2c3e50')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add a thin gray border around the plot
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(0.8)

# Add single legend to the first plot
axes[0, 0].legend(loc="upper left", fontsize=8, framealpha=0.9, facecolor='#ffffff', edgecolor='#e0e0e0')

# General label styling or layout improvements
fig.suptitle("Impacto de C e Gamma no SVM com Kernel RBF", fontsize=15, fontweight='bold', color='#1a252f', y=0.97)

# Save to destination folder
output_path = r"c:\Users\ev_ro\git\IARA\presentation\imgs\svm_c_gamma.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Visualization saved to: {output_path}")
