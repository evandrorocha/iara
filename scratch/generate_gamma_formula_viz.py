import matplotlib.pyplot as plt
import numpy as np

# Set figure facecolor and create figure
fig = plt.figure(figsize=(10, 6.2), facecolor='#fbfbfb', dpi=300)

# Layout: Grid with 2 columns at the top, and a formula block at the bottom
ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=2) # Variance panel
ax2 = plt.subplot2grid((3, 2), (0, 1), rowspan=2) # Dimension panel
ax3 = plt.subplot2grid((3, 2), (2, 0), colspan=2) # Formula panel

# Set facecolors and hide spines
for ax in [ax1, ax2]:
    ax.set_facecolor('#ffffff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dddddd')
    ax.spines['bottom'].set_color('#dddddd')

# ----------------- PANEL 1: VARIANCE ADJUSTMENT -----------------
ax1.set_title("1. Ajuste de Escala (Variância " + r"$\sigma^2$" + ")", fontsize=11, fontweight='bold', pad=12, color='#1a252f')
np.random.seed(42)
concentrated = np.random.normal(0, 0.2, (40, 2))
spread = np.random.normal(0, 0.9, (40, 2))

# Plot concentrated
ax1.scatter(concentrated[:, 0] - 1.5, concentrated[:, 1], color='#e76f51', s=15, alpha=0.8, label='Dados Concentrados (Var Baixa)')
circle1 = plt.Circle((-1.5, 0), 0.45, color='#e76f51', fill=False, linestyle='--', linewidth=0.8)
ax1.add_patch(circle1)

# Plot spread
ax1.scatter(spread[:, 0] + 1.5, spread[:, 1], color='#017ba5', s=15, alpha=0.8, label='Dados Espalhados (Var Alta)')
circle2 = plt.Circle((1.5, 0), 1.9, color='#017ba5', fill=False, linestyle='--', linewidth=0.8)
ax1.add_patch(circle2)

ax1.set_xlim(-3.0, 4.0)
ax1.set_ylim(-2.5, 2.5)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.legend(loc='lower center', fontsize=7.5, framealpha=0.9, facecolor='#ffffff', edgecolor='#eeeeee')

# Text explaining Variance adjustment
ax1.text(0.5, -2.1, 
         "A variância " + r"$\sigma^2$" + " normaliza a dispersão:\n"
         "• Se os dados são espalhados: gamma diminui\n"
         "• Se são muito concentrados: gamma aumenta",
         fontsize=8.5, ha='center', color='#2c3e50', 
         bbox=dict(facecolor='#fdfdfd', edgecolor='#e9ecef', boxstyle='round,pad=0.5'))

# ----------------- PANEL 2: DIMENSION ADJUSTMENT -----------------
ax2.set_title("2. Ajuste de Alta Dimensão (" + r"$d$" + ")", fontsize=11, fontweight='bold', pad=12, color='#1a252f')

# Draw a 2D vector vs high-dimensional vector representation
ax2.text(0.1, 0.82, "Espaço 2D (" + r"$d=2$" + ")", fontsize=9.5, fontweight='bold', color='#2c3e50')
ax2.arrow(0.15, 0.64, 0.12, 0.12, head_width=0.02, color='#e76f51', linewidth=1.2)
ax2.arrow(0.15, 0.64, 0.18, 0.0, head_width=0.02, color='#e76f51', linewidth=1.2)
ax2.text(0.15, 0.50, "Distância ao quadrado " + r"$D^2 = \Delta x^2 + \Delta y^2$" + "\n(Soma de apenas 2 termos)", fontsize=8, color='#555555')

ax2.text(0.1, 0.32, "Espaço de Características (" + r"$d=256$" + " ou " + r"$1024$" + ")", fontsize=9.5, fontweight='bold', color='#2c3e50')
# Draw a long matrix vector block
rect = plt.Rectangle((0.15, 0.17), 0.7, 0.07, facecolor='#017ba5', alpha=0.15, edgecolor='#017ba5', linewidth=0.8)
ax2.add_patch(rect)
ax2.text(0.5, 0.205, "[ " + r"$x_1, x_2, x_3, \dots, x_d$" + " ]", ha='center', va='center', fontsize=8.5, color='#017ba5', fontweight='bold')
ax2.text(0.15, 0.04, "Distância ao quadrado " + r"$D^2 = \sum_{i=1}^d \Delta x_i^2$" + "\n(A soma de centenas de termos infla a distância!)", fontsize=8, color='#555555')

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xticks([])
ax2.set_yticks([])

# ----------------- PANEL 3: THE FORMULA -----------------
ax3.axis('off')
# Display the math formula centrally
formula_text = r"$\gamma_{\mathrm{scale}} = \frac{1}{d \times \sigma^2}$"
ax3.text(0.5, 0.60, formula_text, fontsize=22, ha='center', va='center', color='#017ba5')

desc_text = (
    r"$\mathbf{d}$" + " = número de features (evita a inflação de distância em alta dimensão)\n" +
    r"$\mathbf{\sigma^2}$" + " = variância dos dados (normaliza a dispersão física das variáveis)"
)
ax3.text(0.5, 0.18, desc_text, fontsize=9, ha='center', va='center', color='#2c3e50',
         bbox=dict(facecolor='#f1f3f5', edgecolor='#e9ecef', boxstyle='round,pad=0.6'))

plt.tight_layout()
output_path = r"c:\Users\ev_ro\git\IARA\presentation\imgs\gamma_scale_explanation.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Formula explanation saved to: {output_path}")
