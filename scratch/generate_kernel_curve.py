"""
Figura: hiperplano plano no espaço Nyström → fronteira curva no espaço original.

(a) Espaço Nyström 3D: plano plano separa as classes
(b) Espaço original 2D: a mesma fronteira aparece como curva
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
N = 150
C0, C1 = "#e74c3c", "#2980b9"
GAMMA = 2.5

# Círculos concêntricos: NAVIO (centro) vs FUNDO (anel externo)
r0  = rng.uniform(0, 0.55, N);  th0 = rng.uniform(0, 2*np.pi, N)
r1  = rng.uniform(0.75, 1.2, N); th1 = rng.uniform(0, 2*np.pi, N)
X0  = np.c_[r0*np.cos(th0), r0*np.sin(th0)]
X1  = np.c_[r1*np.cos(th1), r1*np.sin(th1)]
X   = np.vstack([X0, X1])
y   = np.array([0]*N + [1]*N)

# Nyström m=3 → espaço 3D visualizável
sc    = StandardScaler().fit(X)
nys   = Nystroem(kernel='rbf', gamma=GAMMA, n_components=3, random_state=7)
X_phi = nys.fit_transform(sc.transform(X))

clf = LinearSVC(C=10.0, max_iter=5000, random_state=0)
clf.fit(X_phi, y)
preds   = clf.predict(X_phi)
n_err   = int((preds != y).sum())

# ---------------------------------------------------------------
# Figura: 2 painéis lado a lado (3D esquerda, 2D direita)
# ---------------------------------------------------------------
fig = plt.figure(figsize=(13, 5.5), facecolor='white')

# ── Painel (a): espaço Nyström 3D ──────────────────────────────
ax3d = fig.add_subplot(1, 2, 1, projection='3d')

# Pontos
for cls, color, label in [(0, C0, "NAVIO"), (1, C1, "FUNDO")]:
    mask  = y == cls
    right = mask & (preds == cls)
    wrong = mask & (preds != cls)
    ax3d.scatter(X_phi[right,0], X_phi[right,1], X_phi[right,2],
                 c=color, s=34, alpha=0.85, edgecolors='white',
                 linewidths=0.3, depthshade=True, label=label)
    if wrong.sum() > 0:
        ax3d.scatter(X_phi[wrong,0], X_phi[wrong,1], X_phi[wrong,2],
                     c=color, s=60, alpha=0.95, edgecolors='black',
                     linewidths=1.5, depthshade=True)

# Plano de decisão
w, b = clf.coef_[0], clf.intercept_[0]
lo   = X_phi.min(axis=0) - 0.05
hi   = X_phi.max(axis=0) + 0.05
k    = int(np.argmax(np.abs(w)))
oth  = [i for i in range(3) if i != k]
g0, g1 = np.meshgrid(np.linspace(lo[oth[0]], hi[oth[0]], 30),
                      np.linspace(lo[oth[1]], hi[oth[1]], 30))
gk  = -(w[oth[0]]*g0 + w[oth[1]]*g1 + b) / w[k]
gk  = np.clip(gk, lo[k], hi[k])
coords = [None]*3
coords[oth[0]] = g0; coords[oth[1]] = g1; coords[k] = gk
ax3d.plot_surface(coords[0], coords[1], coords[2],
                  alpha=0.30, color='#555', linewidth=0, antialiased=True)

ax3d.set_xlabel(r"$\phi_1$", fontsize=9, labelpad=1)
ax3d.set_ylabel(r"$\phi_2$", fontsize=9, labelpad=1)
ax3d.set_zlabel(r"$\phi_3$", fontsize=9, labelpad=1)
ax3d.tick_params(labelsize=0, length=0)
ax3d.set_title(f"(a)  Espaço Nyström — 3D\n"
               r"Hiperplano plano: $w^T\phi(x)+b=0$",
               fontsize=10.5, pad=5)
ax3d.view_init(25, -50)
ax3d.legend(fontsize=8.5, loc='upper left', framealpha=0.9)

# ── Seta central ───────────────────────────────────────────────
fig.text(0.497, 0.56, "→", fontsize=24, ha='center', va='center', color='#333')
fig.text(0.497, 0.43,
         "projeção de\nvolta ao espaço\noriginal",
         fontsize=7.8, ha='center', va='center', color='#333', style='italic')

# ── Painel (b): espaço original 2D com fronteira curva ─────────
ax2d = fig.add_subplot(1, 2, 2)
ax2d.set_facecolor('#f9f9f9')

# Regiões de decisão via grade densa
xx, yy = np.meshgrid(np.linspace(-1.45, 1.45, 400),
                      np.linspace(-1.45, 1.45, 400))
grid   = np.c_[xx.ravel(), yy.ravel()]
Z      = clf.decision_function(nys.transform(sc.transform(grid))).reshape(xx.shape)

ax2d.contourf(xx, yy, Z, levels=[-200, 0], colors=[C1], alpha=0.13)
ax2d.contourf(xx, yy, Z, levels=[0, 200],  colors=[C0], alpha=0.13)
ax2d.contour( xx, yy, Z, levels=[0], colors='black', linewidths=2.2)

# Pontos
ax2d.scatter(X0[:,0], X0[:,1], c=C0, s=36, alpha=0.85,
             edgecolors='white', linewidths=0.4, label='NAVIO', zorder=3)
ax2d.scatter(X1[:,0], X1[:,1], c=C1, s=36, alpha=0.85,
             edgecolors='white', linewidths=0.4, label='FUNDO', zorder=3)

# Anotação na fronteira
ax2d.annotate("fronteira curva\n(hiperplano projetado)",
              xy=(0.65, 0.02), xytext=(0.60, -0.80),
              fontsize=7.8, color='black', ha='center',
              arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

ax2d.set_xlim(-1.45, 1.45); ax2d.set_ylim(-1.45, 1.45)
ax2d.set_aspect('equal')
ax2d.set_xlabel(r"$x_1$", fontsize=11)
ax2d.set_ylabel(r"$x_2$", fontsize=11)
ax2d.set_title("(b)  Espaço Original — 2D\n"
               "Fronteira curva: projeção do hiperplano",
               fontsize=10.5, pad=8)
ax2d.tick_params(labelbottom=False, labelleft=False, length=0)
ax2d.legend(fontsize=8.5, loc='upper right', framealpha=0.9)
ax2d.spines[['top','right']].set_visible(False)

plt.tight_layout(pad=2.2)
out = "/workspace/presentation/imgs/kernel_projection_curve.png"
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
print(f"Saved: {out}  |  erros no treino: {n_err}")
plt.close()
