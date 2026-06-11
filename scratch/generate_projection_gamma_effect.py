import os
import matplotlib.pyplot as plt
import numpy as np

# Coordinates for Class A (8 blue circles - clustered in the center)
class_a = np.array([
    [3.5, 3.5], [3.0, 3.2], [3.8, 3.0], [3.2, 3.8],
    [3.7, 3.7], [2.8, 3.6], [4.2, 3.3], [3.5, 2.7]
])

# Coordinates for Class B (9 red triangles - surrounding Class A in a ring)
class_b = np.array([
    [1.0, 3.5], [3.5, 1.0], [6.0, 3.5], [3.5, 6.0],
    [1.7, 1.7], [5.3, 5.3], [1.7, 5.3], [5.3, 1.7], 
    [2.3, 4.8]
])

# Define grid for plotting contour decision boundaries
grid_size = 200
x_range = np.linspace(0.2, 6.8, grid_size)
y_range = np.linspace(0.2, 6.8, grid_size)
X, Y = np.meshgrid(x_range, y_range)

# RBF Landmark positions (same as generate_hyperplane_flow.py)
landmarks = np.array([class_a[0], class_b[0], class_b[2]])

# RBF similarity calculation function
def rbf_feature_map(X, Y, landmarks, gamma=0.4):
    phi = []
    for l in landmarks:
        dist_sq = (X - l[0])**2 + (Y - l[1])**2
        phi.append(np.exp(-gamma * dist_sq))
    return np.array(phi)

# Define visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig = plt.figure(figsize=(19, 4.8), dpi=150)
ax1 = fig.add_subplot(1, 4, 1, projection='3d')
ax2 = fig.add_subplot(1, 4, 2)
ax3 = fig.add_subplot(1, 4, 3)
ax4 = fig.add_subplot(1, 4, 4)

blue_color = '#3b82f6'
red_color = '#ef4444'
star_color = '#eab308'
line_color = '#1f2937'

# Weights and bias learned in the projected space
w = np.array([2.2, -1.6, -1.6])
b = -0.45

# ----------------- PANEL 1: PROJETADO (3D) -----------------
ax = ax1

# Compute 3D RBF features for all points
def rbf_point_map(x, y, landmarks, gamma=0.4):
    dist_sq = (x - landmarks[:, 0])**2 + (y - landmarks[:, 1])**2
    return np.exp(-gamma * dist_sq)

phi_a = np.array([rbf_point_map(p[0], p[1], landmarks) for p in class_a])
phi_b = np.array([rbf_point_map(p[0], p[1], landmarks) for p in class_b])

# Plot points in 3D feature space
ax.scatter(phi_a[:, 0], phi_a[:, 1], phi_a[:, 2], c=blue_color, marker='o', s=60, edgecolors='black', linewidths=0.8, label='Classe A (\u03a6)', zorder=4)
ax.scatter(phi_b[:, 0], phi_b[:, 1], phi_b[:, 2], c=red_color, marker='^', s=60, edgecolors='black', linewidths=0.8, label='Classe B (\u03a6)', zorder=4)

# Plot the flat decision boundary plane: w[0]*p1 + w[1]*p2 + w[2]*p3 + b = 0
p1_vals = np.linspace(0.0, 1.0, 15)
p2_vals = np.linspace(0.0, 1.0, 15)
P1_grid, P2_grid = np.meshgrid(p1_vals, p2_vals)
P3_grid = -(w[0]*P1_grid + w[1]*P2_grid + b) / w[2]
P3_grid = np.clip(P3_grid, 0.0, 1.0)

ax.plot_surface(P1_grid, P2_grid, P3_grid, color='#9ca3af', alpha=0.3, zorder=2)

ax.set_title("1. Espaço \u03a6(x) (3D)\nHiperplano Reto: wᵀz + b = 0", fontsize=11, fontweight='bold', pad=10)
ax.set_xlabel("\u03a6_1", fontsize=8)
ax.set_ylabel("\u03a6_2", fontsize=8)
ax.set_zlabel("\u03a6_3", fontsize=8)
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)
ax.set_zlim(0, 1.0)
ax.view_init(elev=20, azim=45)
ax.grid(True, linestyle=':', alpha=0.4)

# ----------------- PANEL 2: MAPEAMENTO DE VOLTA (2D) -----------------
ax = ax2
ax.scatter(class_a[:, 0], class_a[:, 1], c=blue_color, marker='o', s=80, edgecolors='black', linewidths=0.8, zorder=3)
ax.scatter(class_b[:, 0], class_b[:, 1], c=red_color, marker='^', s=80, edgecolors='black', linewidths=0.8, zorder=3)

# Calculate decision function contour in 2D with standard gamma = 0.4
PHI_std = rbf_feature_map(X, Y, landmarks, gamma=0.4)
Z_std = w[0]*PHI_std[0] + w[1]*PHI_std[1] + w[2]*PHI_std[2] + b
ax.contour(X, Y, Z_std, levels=[0], colors=line_color, linestyles='-', linewidths=2.5, zorder=2)

ax.set_title("2. Mapeamento de Volta (2D)\nFronteira Curva: wᵀ\u03a6(x) + b = 0", fontsize=11, fontweight='bold', pad=10)
ax.set_xlim(0.5, 6.5)
ax.set_ylim(0.5, 6.5)
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.6)

# ----------------- PANEL 3: GAMMA PEQUENO (2D) -----------------
ax = ax3
ax.scatter(class_a[:, 0], class_a[:, 1], c=blue_color, marker='o', s=80, edgecolors='black', linewidths=0.8, zorder=3)
ax.scatter(class_b[:, 0], class_b[:, 1], c=red_color, marker='^', s=80, edgecolors='black', linewidths=0.8, zorder=3)

# Small gamma = 0.1 -> Smooth boundary (low curvature)
PHI_small = rbf_feature_map(X, Y, landmarks, gamma=0.1)
# Adjust weights slightly for visual clarity with low gamma
w_small = np.array([2.5, -1.2, -1.2])
b_small = -0.5
Z_small = w_small[0]*PHI_small[0] + w_small[1]*PHI_small[1] + w_small[2]*PHI_small[2] + b_small
ax.contour(X, Y, Z_small, levels=[0], colors='#059669', linestyles='-', linewidths=2.5, zorder=2)

ax.set_title("3. \u03b3 Pequeno (Suave)\nBaixa Curvatura (Generalista)", fontsize=11, fontweight='bold', pad=10)
ax.set_xlim(0.5, 6.5)
ax.set_ylim(0.5, 6.5)
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.6)

# ----------------- PANEL 4: GAMMA GRANDE (2D) -----------------
ax = ax4
ax.scatter(class_a[:, 0], class_a[:, 1], c=blue_color, marker='o', s=80, edgecolors='black', linewidths=0.8, zorder=3)
ax.scatter(class_b[:, 0], class_b[:, 1], c=red_color, marker='^', s=80, edgecolors='black', linewidths=0.8, zorder=3)

# Large gamma = 1.8 -> Tight boundary (high curvature / overfitting-prone)
PHI_large = rbf_feature_map(X, Y, landmarks, gamma=1.8)
w_large = np.array([3.2, -1.8, -1.8])
b_large = -0.3
Z_large = w_large[0]*PHI_large[0] + w_large[1]*PHI_large[1] + w_large[2]*PHI_large[2] + b_large
ax.contour(X, Y, Z_large, levels=[0], colors='#dc2626', linestyles='-', linewidths=2.5, zorder=2)

ax.set_title("4. \u03b3 Grande (Apertado)\nAlta Curvatura (Especialista)", fontsize=11, fontweight='bold', pad=10)
ax.set_xlim(0.5, 6.5)
ax.set_ylim(0.5, 6.5)
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.6)

# Adjust layouts
plt.tight_layout()

# Save paths
brain_dest = r"C:\Users\ev_ro\.gemini\antigravity-ide\brain\b9e437c3-f2d4-4681-96ac-ed18adc3bd7b\svm_projection_and_gamma_effect.png"
presentation_dest = r"c:\Users\ev_ro\git\IARA\presentation\imgs\svm_projection_and_gamma_effect.png"

# Save figure
plt.savefig(brain_dest, bbox_inches='tight', dpi=150)
plt.savefig(presentation_dest, bbox_inches='tight', dpi=150)
plt.close()

print("Mathematical SVM Projection & Gamma Infographic generated successfully!")
