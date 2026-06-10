"""
Figura: kernel trick em 3D — caso separável vs caso com sobreposição.

(a) 2D original: círculos concêntricos (não separável linearmente)
(b) 3D Nyström: plano plano separa PERFEITAMENTE (caso ideal)
(c) 3D Nyström: plano plano com SOBREPOSIÇÃO — margem suave (soft margin)
    Alguns pontos ficam no lado errado; o SGD minimiza o erro total.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)

C0, C1 = "#e74c3c", "#2980b9"
GAMMA = 2.5
N = 120

def make_circles(n, r_inner=(0, 0.55), r_outer=(0.75, 1.2), noise_outer=0.0):
    """Círculos concêntricos com ruído opcional no anel externo."""
    r0  = rng.uniform(*r_inner, n)
    th0 = rng.uniform(0, 2*np.pi, n)
    X0  = np.c_[r0*np.cos(th0), r0*np.sin(th0)]

    r1  = rng.uniform(*r_outer, n)
    th1 = rng.uniform(0, 2*np.pi, n)
    X1  = np.c_[r1*np.cos(th1), r1*np.sin(th1)]

    if noise_outer > 0:
        # Injeta pontos de fundo no centro e vice-versa
        n_noise = int(n * noise_outer)
        idx0 = rng.choice(n, n_noise, replace=False)
        idx1 = rng.choice(n, n_noise, replace=False)
        X0[idx0] = X1[idx1].copy()   # pontos de navio vão para o anel externo
        X1[idx1] = rng.uniform(-0.45, 0.45, (n_noise, 2))  # fundo no centro

    X = np.vstack([X0, X1])
    y = np.array([0]*n + [1]*n)
    return X, y, X0, X1

def nystroem_3d(X, y, gamma=GAMMA, C_svm=10.0):
    sc    = StandardScaler().fit(X)
    nys   = Nystroem(kernel='rbf', gamma=gamma, n_components=3, random_state=7)
    X_phi = nys.fit_transform(sc.transform(X))
    clf   = LinearSVC(C=C_svm, max_iter=5000, random_state=0)
    clf.fit(X_phi, y)
    return X_phi, clf

def decision_plane(clf, X_phi, n=35):
    w, b = clf.coef_[0], clf.intercept_[0]
    lo = X_phi.min(axis=0) - 0.05
    hi = X_phi.max(axis=0) + 0.05
    # Resolve a componente de maior peso
    k = int(np.argmax(np.abs(w)))
    others = [i for i in range(3) if i != k]
    g0, g1 = np.meshgrid(np.linspace(lo[others[0]], hi[others[0]], n),
                          np.linspace(lo[others[1]], hi[others[1]], n))
    gk = -(w[others[0]]*g0 + w[others[1]]*g1 + b) / w[k]
    gk = np.clip(gk, lo[k], hi[k])
    coords = [None]*3
    coords[others[0]] = g0
    coords[others[1]] = g1
    coords[k]         = gk
    return coords[0], coords[1], coords[2]

def plot_3d(ax, X_phi, y, clf, title, view=(25, -50)):
    preds = clf.predict(X_phi)
    n_err = int((preds != y).sum())

    # Plano de decisão
    Gx, Gy, Gz = decision_plane(clf, X_phi)
    ax.plot_surface(Gx, Gy, Gz, alpha=0.28, color="#777",
                    linewidth=0, antialiased=True)

    for cls, color, label in [(0, C0, "NAVIO"), (1, C1, "FUNDO")]:
        mask  = y == cls
        right = mask & (preds == y)
        wrong = mask & (preds != y)
        ax.scatter(X_phi[right,0], X_phi[right,1], X_phi[right,2],
                   c=color, s=34, alpha=0.85, edgecolors="white",
                   linewidths=0.3, depthshade=True, label=label)
        ax.scatter(X_phi[wrong,0], X_phi[wrong,1], X_phi[wrong,2],
                   c=color, s=60, alpha=0.95, edgecolors="black",
                   linewidths=1.5, depthshade=True)

    ax.set_xlabel(r"$\phi_1$", fontsize=9, labelpad=1)
    ax.set_ylabel(r"$\phi_2$", fontsize=9, labelpad=1)
    ax.set_zlabel(r"$\phi_3$", fontsize=9, labelpad=1)
    ax.tick_params(labelsize=0, length=0)
    ax.set_title(title + f"\n({n_err} erro(s) — borda preta)", fontsize=9.8, pad=5)
    ax.view_init(*view)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

# ---------------------------------------------------------------
# Dados
# ---------------------------------------------------------------
X_sep,  y_sep,  X0_sep,  X1_sep  = make_circles(N, noise_outer=0.0)
X_over, y_over, X0_over, X1_over = make_circles(N, noise_outer=0.22)

X_phi_sep,  clf_sep  = nystroem_3d(X_sep,  y_sep,  C_svm=10.0)
X_phi_over, clf_over = nystroem_3d(X_over, y_over, C_svm=1.0)

# ---------------------------------------------------------------
# Plot: 3 painéis
# ---------------------------------------------------------------
fig = plt.figure(figsize=(16, 5.2), facecolor="white")

# --- (a) 2D original ---
ax2 = fig.add_subplot(1, 3, 1)
ax2.set_facecolor("#f9f9f9")
ax2.scatter(X0_sep[:,0], X0_sep[:,1], c=C0, s=36, alpha=0.80,
            edgecolors="white", linewidths=0.4, label="NAVIO")
ax2.scatter(X1_sep[:,0], X1_sep[:,1], c=C1, s=36, alpha=0.80,
            edgecolors="white", linewidths=0.4, label="FUNDO")
th = np.linspace(0, 2*np.pi, 200)
ax2.plot(0.65*np.cos(th), 0.65*np.sin(th), 'k--', lw=1.1, alpha=0.35,
         label="fronteira ideal (curva)")
ax2.set_xlim(-1.4, 1.4); ax2.set_ylim(-1.4, 1.4); ax2.set_aspect("equal")
ax2.set_xlabel(r"$x_1$", fontsize=10); ax2.set_ylabel(r"$x_2$", fontsize=10)
ax2.set_title("(a)  Espaço Original 2D\n"
              "Não separável por linha reta", fontsize=10, pad=6)
ax2.tick_params(labelbottom=False, labelleft=False, length=0)
ax2.legend(fontsize=8, loc="upper right", framealpha=0.9)
ax2.spines[["top","right"]].set_visible(False)

# Setas
for xpos, label in [(0.365, "caso\nseparável"), (0.635, "caso com\nsobreposição")]:
    fig.text(xpos, 0.54, "→", fontsize=20, ha="center", va="center", color="#555")
    fig.text(xpos, 0.42, "Nyström\n" + r"$\phi(x) \in \mathbb{R}^3$" + f"\n({label})",
             fontsize=7.5, ha="center", va="center", color="#444", style="italic")

# --- (b) 3D separável ---
ax3a = fig.add_subplot(1, 3, 2, projection='3d')
plot_3d(ax3a, X_phi_sep, y_sep, clf_sep,
        "(b)  Nyström 3D — separável\nPlano plano separa perfeitamente")

# --- (c) 3D com sobreposição ---
ax3b = fig.add_subplot(1, 3, 3, projection='3d')
plot_3d(ax3b, X_phi_over, y_over, clf_over,
        "(c)  Nyström 3D — com sobreposição\nMargem suave: plano minimiza erros totais",
        view=(28, -45))

# Nota margem suave
fig.text(0.83, 0.10,
         "Margem suave (soft margin):\nSGD tolera violações para encontrar\n"
         "o plano que minimiza o erro total.\nControle via parâmetro C.",
         fontsize=7.8, ha="center", va="center", color="#336", style="italic",
         bbox=dict(boxstyle="round,pad=0.4", fc="#eef", ec="#99b", alpha=0.88))

plt.tight_layout(pad=1.8)
out_path = "presentation/imgs/sgd_linear_classifier.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}  |  erros sep={int((clf_sep.predict(X_phi_sep)!=y_sep).sum())}  "
      f"erros overlap={int((clf_over.predict(X_phi_over)!=y_over).sum())}")
plt.close()
