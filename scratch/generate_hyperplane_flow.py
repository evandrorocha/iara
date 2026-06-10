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
    [2.3, 4.8]  # (2.3, 4.8) is closer to the center, acting as a potential violator
])

# Define grid for plotting contour decision boundaries
grid_size = 200
x_range = np.linspace(0.2, 6.8, grid_size)
y_range = np.linspace(0.2, 6.8, grid_size)
X, Y = np.meshgrid(x_range, y_range)

# RBF Landmark positions (1 from Class A center, 2 from Class B outer ring)
landmarks = np.array([class_a[0], class_b[0], class_b[2]])

# RBF similarity calculation function
def rbf_feature_map(X, Y, landmarks, gamma=0.5):
    phi = []
    for l in landmarks:
        dist_sq = (X - l[0])**2 + (Y - l[1])**2
        phi.append(np.exp(-gamma * dist_sq))
    return np.array(phi)

# Generate similarity map for the grid
PHI = rbf_feature_map(X, Y, landmarks, gamma=0.4)

# Define visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig = plt.figure(figsize=(18, 5.2), dpi=150)
ax1 = fig.add_subplot(1, 4, 1)
ax2 = fig.add_subplot(1, 4, 2)
ax3 = fig.add_subplot(1, 4, 3)
ax4 = fig.add_subplot(1, 4, 4, projection='3d')

blue_color = '#3b82f6'
red_color = '#ef4444'
star_color = '#eab308'
line_color = '#1f2937'

# ----------------- PANEL 1: INICIALIZAÇÃO -----------------
ax = ax1
ax.scatter(class_a[:, 0], class_a[:, 1], c=blue_color, marker='o', s=100, label='Classe A', edgecolors='black', linewidths=0.8, zorder=3)
ax.scatter(class_b[:, 0], class_b[:, 1], c=red_color, marker='^', s=100, label='Classe B', edgecolors='black', linewidths=0.8, zorder=3)

# Initial random weights where outer landmarks have positive weight (bad boundary)
w_init = np.array([-0.2, 1.2, 1.2])
b_init = -0.5
Z_init = w_init[0]*PHI[0] + w_init[1]*PHI[1] + w_init[2]*PHI[2] + b_init
ax.contour(X, Y, Z_init, levels=[0], colors='#9ca3af', linestyles='--', linewidths=2.5, zorder=2)

ax.set_title("1. Inicialização\n(Fronteira Linear em \u03a6(x))", fontsize=13, fontweight='bold', pad=10)
ax.set_xlim(0.5, 6.5)
ax.set_ylim(0.5, 6.5)
ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none', fontsize=9)
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.6)

# ----------------- PANEL 2: SELEÇÃO DE MARCOS -----------------
ax = ax2
ax.scatter(class_a[:, 0], class_a[:, 1], c=blue_color, marker='o', s=100, edgecolors='black', linewidths=0.8, zorder=3)
ax.scatter(class_b[:, 0], class_b[:, 1], c=red_color, marker='^', s=100, edgecolors='black', linewidths=0.8, zorder=3)

# Plot landmarks (stars) exactly on top of selected training points
ax.scatter(landmarks[:, 0], landmarks[:, 1], c=star_color, marker='*', s=250, edgecolors='black', linewidths=1.0, label='Marcos Selecionados', zorder=4)

ax.set_title("2. Seleção de Marcos\n(Amostras do Próprio Treino)", fontsize=13, fontweight='bold', pad=10)
ax.set_xlim(0.5, 6.5)
ax.set_ylim(0.5, 6.5)
ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none', fontsize=9)
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.6)

# ----------------- PANEL 3: AJUSTE POR GRADIENTE -----------------
ax = ax3
ax.scatter(class_a[:, 0], class_a[:, 1], c=blue_color, marker='o', s=100, edgecolors='black', linewidths=0.8, zorder=3)
ax.scatter(class_b[:, 0], class_b[:, 1], c=red_color, marker='^', s=100, edgecolors='black', linewidths=0.8, zorder=3)

# Show old random boundary (dashed)
ax.contour(X, Y, Z_init, levels=[0], colors='#9ca3af', linestyles='--', linewidths=2.0, zorder=2)

# Intermediate weights (approaching enclosing the center)
w_inter = np.array([1.2, -0.6, -0.6])
b_inter = -0.3
Z_inter = w_inter[0]*PHI[0] + w_inter[1]*PHI[1] + w_inter[2]*PHI[2] + b_inter
ax.contour(X, Y, Z_inter, levels=[0], colors=line_color, linestyles='-', linewidths=2.5, zorder=2)

# Highlight violator point class_b[8] = (2.3, 4.8) - which is inside Class A's zone for the intermediate boundary
violator = class_b[8]
ax.scatter(violator[0], violator[1], facecolors='none', edgecolors='#ef4444', s=250, linewidths=2.0, zorder=4)

# Force arrow pointing from the violating point to the boundary showing adjustment
ax.annotate(
    "Força de Correção\n(Gradiente)",
    xy=(violator[0], violator[1]), xytext=(0.8, 5.2),
    arrowprops=dict(facecolor='#ef4444', shrink=0.08, width=2, headwidth=8, headlength=8),
    fontsize=9, fontweight='bold', color='#ef4444', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ef4444", lw=0.5, alpha=0.9),
    zorder=5
)

ax.set_title("3. Ajuste por Gradiente (SGD)\n(Giro/Deslocamento da Curva)", fontsize=13, fontweight='bold', pad=10)
ax.set_xlim(0.5, 6.5)
ax.set_ylim(0.5, 6.5)
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.6)

# ----------------- PANEL 4: FRONTEIRA FINAL (3D) -----------------
ax = ax4

# Compute 3D RBF features for all points
def rbf_point_map(x, y, landmarks, gamma=0.4):
    dist_sq = (x - landmarks[:, 0])**2 + (y - landmarks[:, 1])**2
    return np.exp(-gamma * dist_sq)

phi_a = np.array([rbf_point_map(p[0], p[1], landmarks) for p in class_a])
phi_b = np.array([rbf_point_map(p[0], p[1], landmarks) for p in class_b])

# Plot points in 3D feature space
ax.scatter(phi_a[:, 0], phi_a[:, 1], phi_a[:, 2], c=blue_color, marker='o', s=80, edgecolors='black', linewidths=0.8, label='Classe A (\u03a6)', zorder=4)
ax.scatter(phi_b[:, 0], phi_b[:, 1], phi_b[:, 2], c=red_color, marker='^', s=80, edgecolors='black', linewidths=0.8, label='Classe B (\u03a6)', zorder=4)

# Plot landmarks in 3D
phi_landmarks = np.array([rbf_point_map(l[0], l[1], landmarks) for l in landmarks])
ax.scatter(phi_landmarks[:, 0], phi_landmarks[:, 1], phi_landmarks[:, 2], c=star_color, marker='*', s=200, edgecolors='black', linewidths=1.0, label='Marcos (\u03a6)', zorder=5)

# Plot the flat decision boundary plane: w_final[0]*p1 + w_final[1]*p2 + w_final[2]*p3 + b_final = 0
w_final = np.array([2.2, -1.6, -1.6])
b_final = -0.45
p1_vals = np.linspace(0.0, 1.0, 10)
p2_vals = np.linspace(0.0, 1.0, 10)
P1_grid, P2_grid = np.meshgrid(p1_vals, p2_vals)
P3_grid = -(w_final[0]*P1_grid + w_final[1]*P2_grid + b_final) / w_final[2]
P3_grid = np.clip(P3_grid, 0.0, 1.0)

ax.plot_surface(P1_grid, P2_grid, P3_grid, color='#9ca3af', alpha=0.3, edgecolors='none', zorder=2)

ax.set_title("4. Espaço \u03a6(x) (3D)\n(Hiperplano Reto Separador)", fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel("\u03a6_1 (Centro)", fontsize=8)
ax.set_ylabel("\u03a6_2 (Esquerda)", fontsize=8)
ax.set_zlabel("\u03a6_3 (Direita)", fontsize=8)
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)
ax.set_zlim(0, 1.0)
ax.view_init(elev=20, azim=45)
ax.grid(True, linestyle=':', alpha=0.4)

ax.set_title("4. Fronteira Final\n(Classes Separadas por RBF)", fontsize=13, fontweight='bold', pad=10)
ax.set_xlim(0.5, 6.5)
ax.set_ylim(0.5, 6.5)
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.6)

# Adjust layouts
plt.tight_layout()

# Save paths
brain_dest = r"C:\Users\ev_ro\.gemini\antigravity-ide\brain\b9e437c3-f2d4-4681-96ac-ed18adc3bd7b\svm_hyperplane_generation_flow_v3.png"
presentation_dest = r"c:\Users\ev_ro\git\IARA\presentation\imgs\svm_hyperplane_generation.png"

# Save figure
plt.savefig(brain_dest, bbox_inches='tight', dpi=150)
plt.savefig(presentation_dest, bbox_inches='tight', dpi=150)
plt.close()

print("Non-linear RBF-SVM Infographics generated successfully!")
