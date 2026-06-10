import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Create figure and axis in dark navy background
fig = plt.figure(figsize=(15, 6.2), dpi=150)
fig.patch.set_facecolor('#0d1b2a')
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor('#0d1b2a')
ax.axis('off')

# Set coordinate bounds
ax.set_xlim(0, 10.5)
ax.set_ylim(0, 6.2)

# Colors
navy_bg = '#0d1b2a'
text_color = '#ffffff'
sub_text_color = '#94a3b8'
teal_box = '#0891b2'
teal_border = '#22d3ee'
orange_circle = '#ea580c'
orange_border = '#f97316'
purple_box = '#6d28d9'
purple_border = '#a78bfa'
gold_star = '#fbbf24'
vector_bg = '#1e1b4b'
vector_border = '#818cf8'

# Title
ax.text(5.25, 5.8, "Como Landmarks se Tornam Dimensões (Projeção Nyström)", 
        color=text_color, fontsize=15, fontweight='bold', ha='center', va='center')

# ----------------- STAGE 1: JANELA DE ENTRADA -----------------
# Label
ax.text(1.25, 5.2, "1. Janela de Entrada", color=teal_border, fontsize=11, fontweight='bold', ha='center')
ax.text(1.25, 4.95, "Áudio original em 2D", color=sub_text_color, fontsize=9, ha='center')

# Box
rect_in = patches.FancyBboxPatch((0.4, 1.2), 1.7, 3.4, boxstyle="round,pad=0.1", 
                                 facecolor='#0f172a', edgecolor=teal_border, lw=2.0, zorder=2)
ax.add_patch(rect_in)

ax.text(1.25, 4.2, "Espectrograma Mel\n(Filtros de Banda Larga)", color=text_color, fontsize=10, ha='center', va='center')
ax.text(1.25, 1.6, "Vetor x ∈ ℝ²⁵⁶\n(256 Features)", color=teal_border, fontsize=11, fontweight='bold', ha='center', va='center')

# Mock bar chart inside the input box
bar_x = np.linspace(0.6, 1.9, 12)
np.random.seed(42)
bar_y = np.random.uniform(2.0, 3.2, 12)
for bx, by in zip(bar_x, bar_y):
    ax.plot([bx, bx], [2.0, by], color=teal_border, lw=3, alpha=0.8, zorder=3)

# ----------------- STAGE 2: RBF COMPARATORS -----------------
ax.text(4.75, 5.2, "2. Banco de Comparadores de Similaridade (RBF)", color=orange_border, fontsize=11, fontweight='bold', ha='center')
ax.text(4.75, 4.95, "Comparação em paralelo contra os 4000 Marcos", color=sub_text_color, fontsize=9, ha='center')

rows_y = [4.1, 2.9, 1.4]
labels = ["Landmark L₁", "Landmark L₂", "Landmark L₄₀₀₀"]
similarities = ["s₁ = 0.85", "s₂ = 0.12", "s₄₀₀₀ = 0.43"]
subs = ["(Similar ao Centro)", "(Distante do Marco 2)", "(Similaridade Média)"]

for i, y in enumerate(rows_y):
    # RBF Kernel Circle
    circle = patches.Circle((4.75, y), radius=0.38, facecolor='#1e293b', edgecolor=orange_border, lw=2.0, zorder=3)
    ax.add_patch(circle)
    ax.text(4.75, y, "RBF\nKernel", color=text_color, fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Input arrow from x to kernel
    ax.annotate("", xy=(4.32, y), xytext=(2.2, y),
                arrowprops=dict(arrowstyle="-|>", color=teal_border, lw=1.5, mutation_scale=12))
    
    # Landmark Star representator
    ax.scatter(3.3, y + 0.35, c=gold_star, marker='*', s=150, edgecolors='black', linewidths=0.5, zorder=4)
    ax.text(3.3, y - 0.2, labels[i], color=gold_star, fontsize=9.5, fontweight='bold', ha='center')
    
    # Arrow from Landmark to Kernel
    ax.annotate("", xy=(4.4, y + 0.15), xytext=(3.3, y + 0.15),
                arrowprops=dict(arrowstyle="-|>", color=gold_star, lw=1.2, linestyle=':', mutation_scale=10))
    
    # Output arrow from kernel to vector
    ax.annotate("", xy=(6.85, y), xytext=(5.18, y),
                arrowprops=dict(arrowstyle="-|>", color=orange_border, lw=1.5, mutation_scale=12))
    
    # Text on output arrow
    ax.text(6.0, y + 0.22, similarities[i], color=orange_border, fontsize=9.5, fontweight='bold', ha='center')
    ax.text(6.0, y - 0.22, subs[i], color=sub_text_color, fontsize=8, ha='center')

# Ellipsis dots in Stage 2
ax.text(4.75, 2.15, "•\n•\n•", color=sub_text_color, fontsize=12, ha='center', va='center')

# ----------------- STAGE 3: PROJECTED FEATURE VECTOR -----------------
ax.text(7.7, 5.2, "3. Vetor Projetado", color=vector_border, fontsize=11, fontweight='bold', ha='center')
ax.text(7.7, 4.95, "Dimensão m=4000", color=sub_text_color, fontsize=9, ha='center')

# Column Vector Box
rect_vec = patches.FancyBboxPatch((7.3, 0.8), 0.8, 3.8, boxstyle="round,pad=0.08", 
                                  facecolor=vector_bg, edgecolor=vector_border, lw=2.0, zorder=2)
ax.add_patch(rect_vec)

# Stacking elements in vector
ax.text(7.7, 4.25, "s₁", color=text_color, fontsize=11, fontweight='bold', ha='center')
ax.text(7.7, 3.65, "s₂", color=text_color, fontsize=11, fontweight='bold', ha='center')
ax.text(7.7, 2.9, "•\n•\n•", color=sub_text_color, fontsize=11, ha='center', va='center')
ax.text(7.7, 1.85, "s₄₀₀₀", color=text_color, fontsize=11, fontweight='bold', ha='center')

# Brackets and description
ax.text(8.25, 2.7, "}", color=vector_border, fontsize=70, ha='center', va='center', weight='light')
ax.text(8.45, 2.8, "Vetor z ∈ ℝ⁴⁰⁰⁰\n(Espaço Projetado)", color=text_color, fontsize=9.5, fontweight='bold', ha='left', va='center')

# ----------------- STAGE 4: CLASSIFICATION -----------------
ax.text(9.7, 5.2, "4. Classificação", color=purple_border, fontsize=11, fontweight='bold', ha='center')
ax.text(9.7, 4.95, "Fronteira Linear Rápida", color=sub_text_color, fontsize=9, ha='center')

# Arrow to SVM
ax.annotate("", xy=(9.2, 2.1), xytext=(8.0, 2.1),
            arrowprops=dict(arrowstyle="-|>", color=vector_border, lw=1.8, mutation_scale=15))

# SVM Box
rect_svm = patches.FancyBboxPatch((9.2, 1.1), 1.1, 2.0, boxstyle="round,pad=0.08", 
                                  facecolor='#3b0764', edgecolor=purple_border, lw=2.0, zorder=2)
ax.add_patch(rect_svm)

ax.text(9.75, 2.5, "SVM Linear", color=text_color, fontsize=10.5, fontweight='bold', ha='center')
ax.text(9.75, 1.9, "f(z) = w^T z + b", color=purple_border, fontsize=10, fontweight='bold', ha='center')
ax.text(9.75, 1.4, "Decisão: sign(f)", color=text_color, fontsize=9, ha='center')

# Tight margins
plt.tight_layout()

# Save paths
brain_dest = r"C:\Users\ev_ro\.gemini\antigravity-ide\brain\b9e437c3-f2d4-4681-96ac-ed18adc3bd7b\nystroem_projection_mechanism_flow.png"
presentation_dest = r"c:\Users\ev_ro\git\IARA\presentation\imgs\nystroem_projection.png"

# Save figure
plt.savefig(brain_dest, bbox_inches='tight', dpi=150, facecolor=navy_bg)
plt.savefig(presentation_dest, bbox_inches='tight', dpi=150, facecolor=navy_bg)
plt.close()

print("Matplotlib Nystroem Projection Diagram generated successfully!")
