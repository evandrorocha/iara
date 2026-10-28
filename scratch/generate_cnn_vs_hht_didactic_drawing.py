import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Configuração visual premium
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.titlesize': 14,
    'figure.dpi': 300
})

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1.0], hspace=0.35, wspace=0.28)

# -------------------------------------------------------------------------
# PAINEL 1: O MUNDO IDEAL DA CNN (STFT / MEL / LOFAR)
# -------------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("(A) O Mundo Ideal da CNN:\nEspectrograma STFT / MEL (Contínuo)", fontweight='bold', color='#1d3557', pad=10)

# Simular STFT com bandas suaves e harmônicos contínuos
time = np.linspace(0, 10, 120)
freq = np.linspace(0, 100, 100)
T, F = np.meshgrid(time, freq)
# Harmônico tonal contínuo
stft = 0.8 * np.exp(-((F - 30)**2) / 35.0) + 0.6 * np.exp(-((F - 65)**2) / 45.0)
# Banda larga difusa
stft += 0.3 * np.exp(-((F - 15)**2) / 120.0) + 0.15 * np.random.randn(*stft.shape)
stft = np.clip(stft, 0, 1)

im1 = ax1.imshow(stft, aspect='auto', origin='lower', cmap='viridis', extent=[0, 10, 0, 100])
ax1.set_xlabel("Tempo (s)", fontweight='bold')
ax1.set_ylabel("Frequência (Hz / Mel)", fontweight='bold')

# Desenhar Kernel Convolucional 3x3 deslizando sobre linha contínua
rect1 = patches.Rectangle((4.0, 22), 2.0, 16, linewidth=2.5, edgecolor='#e63946', facecolor='#e63946', alpha=0.3)
ax1.add_patch(rect1)
ax1.text(5.0, 42, "Kernel 3x3 deslizando\n(detecta borda contínua)", color='#e63946', fontweight='bold', fontsize=9, ha='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#e63946', alpha=0.9))

ax1.text(5.0, 5, "✔ Alta Coerência Espacial 2D\n✔ Pixels vizinhos correlacionados", 
         color='white', fontweight='bold', fontsize=9.5, ha='center',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#1d3557', alpha=0.85))

# -------------------------------------------------------------------------
# PAINEL 2: O CHOQUE CONCEITUAL COM O HHT (ESPECTRO DE HILBERT)
# -------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title("(B) O Choque Conceitual com o HHT:\nFrequência Instantânea Dispersa", fontweight='bold', color='#c1121f', pad=10)

# Simular HHT (trajetórias instantâneas finas, dispersas, com saltos e descontinuidades)
hht_grid = np.zeros((100, 120))
# Trajetória não-estacionária oscilante e pontilhada
for t_idx in range(120):
    if t_idx % 12 in [8, 9, 10]: continue # Gaps temporais / quebras de modo
    f_inst_1 = int(30 + 15 * np.sin(t_idx * 0.25) + 3 * np.random.randn())
    f_inst_2 = int(70 + 8 * np.cos(t_idx * 0.15) + 2 * np.random.randn())
    if 0 <= f_inst_1 < 100:
        hht_grid[f_inst_1, t_idx] = 1.0
        if np.random.rand() > 0.6 and f_inst_1 + 1 < 100: hht_grid[f_inst_1+1, t_idx] = 0.6
    if 0 <= f_inst_2 < 100:
        hht_grid[f_inst_2, t_idx] = 0.85

# Ruído esparso pontual
sparse_noise = (np.random.rand(100, 120) > 0.985).astype(float) * 0.4
hht_grid = np.clip(hht_grid + sparse_noise, 0, 1)

im2 = ax2.imshow(hht_grid, aspect='auto', origin='lower', cmap='magma', extent=[0, 10, 0, 100])
ax2.set_xlabel("Tempo (s)", fontweight='bold')
ax2.set_ylabel("Frequência Instantânea $\\omega(t)$", fontweight='bold')

# Desenhar Kernel Convolucional 3x3 no HHT
rect2 = patches.Rectangle((4.0, 22), 2.0, 16, linewidth=2.5, edgecolor='#00b4d8', facecolor='#00b4d8', alpha=0.3)
ax2.add_patch(rect2)
ax2.text(5.0, 42, "Kernel 3x3 faz MÉDIA:\nBorra as raias finas!", color='#0077b6', fontweight='bold', fontsize=9, ha='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#0077b6', alpha=0.9))

ax2.text(5.0, 5, "✖ Esparsidade extrema (>95% zeros)\n✖ Sem continuidade 2D / Saltos", 
         color='white', fontweight='bold', fontsize=9.5, ha='center',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#780000', alpha=0.85))

# -------------------------------------------------------------------------
# PAINEL 3: A AÇÃO MATEMÁTICA DO FILTRO CONVOLUCIONAL
# -------------------------------------------------------------------------
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_title("(C) Mecanismo do Erro:\nBorramento Destrutivo da Convolução", fontweight='bold', color='#333333', pad=10)
ax3.axis('off')

# Desenhar esquema conceitual
# Entrada HHT pontual
ax3.text(0.18, 0.88, "Entrada HHT (Fina):", fontweight='bold', fontsize=10, ha='center')
grid_in = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
for r in range(3):
    for c in range(3):
        color = '#ffb703' if grid_in[r, c] == 1 else '#f1f1f1'
        txt = "1" if grid_in[r, c] == 1 else "0"
        rect = patches.Rectangle((0.06 + c*0.08, 0.62 + (2-r)*0.08), 0.075, 0.075, facecolor=color, edgecolor='black', linewidth=1)
        ax3.add_patch(rect)
        ax3.text(0.06 + c*0.08 + 0.037, 0.62 + (2-r)*0.08 + 0.037, txt, ha='center', va='center', fontweight='bold')

# Operador Convolução
ax3.text(0.38, 0.74, "$\\circledast$\nKernel", fontweight='bold', fontsize=14, ha='center', va='center', color='#e63946')

# Kernel 3x3
grid_k = np.array([[0.1, 0.1, 0.1], [0.1, 0.2, 0.1], [0.1, 0.1, 0.1]])
for r in range(3):
    for c in range(3):
        rect = patches.Rectangle((0.46 + c*0.08, 0.62 + (2-r)*0.08), 0.075, 0.075, facecolor='#e0fbfc', edgecolor='black', linewidth=1)
        ax3.add_patch(rect)
        ax3.text(0.46 + c*0.08 + 0.037, 0.62 + (2-r)*0.08 + 0.037, f"{grid_k[r, c]:.1f}", ha='center', va='center', fontsize=7.5)

# Seta de igualdade
ax3.text(0.75, 0.74, "$=$", fontweight='bold', fontsize=16, ha='center', va='center')

# Saída borrada
ax3.text(0.88, 0.88, "Saída Diluída:", fontweight='bold', fontsize=10, ha='center')
rect_out = patches.Rectangle((0.82, 0.68), 0.12, 0.12, facecolor='#ffccd5', edgecolor='red', linewidth=1.5)
ax3.add_patch(rect_out)
ax3.text(0.88, 0.74, "0.2\n(Fraco)", ha='center', va='center', fontweight='bold', color='red', fontsize=9)

# Texto explicativo embaixo
box_text = (
    "Por que a CNN colapsa no HHT:\n"
    "1. Convolução age como filtro passa-baixas:\n"
    "   Ao somar ponderado os vizinhos, ela dilui a\n"
    "   frequência instantânea e amplifica o ruído;\n"
    "2. Pooling 2D destrói a esparsidade:\n"
    "   Max-pool perde a posição exata de tempo;\n"
    "3. Falta de Textura Espacial:\n"
    "   CNN depende de gradientes contínuos (bordas),\n"
    "   mas o HHT é uma nuvem de pontos isolados."
)
ax3.text(0.5, 0.28, box_text, ha='center', va='center', fontsize=9.5, linespacing=1.4,
         bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#ced4da', linewidth=1.2))

# -------------------------------------------------------------------------
# PAINEL 4: POR QUE O RANDOM FOREST FUNCIONA NO HHT?
# -------------------------------------------------------------------------
ax4 = fig.add_subplot(gs[1, :2])
ax4.set_title("(D) Por que o Random Forest Vence no HHT? (Partição Ortogonal vs. Convolução 2D)", fontweight='bold', color='#2b5c8f', pad=10)
ax4.axis('off')

# Desenhar árvore de decisão esquemática
# Nó raiz
ax4.text(0.25, 0.85, "Energia IMF 1 > Limiar A?\n(Não depende de vizinhos 2D)", ha='center', va='center', fontweight='bold', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#e2eafc', edgecolor='#3a86ff', linewidth=1.5))
# Ramos
ax4.annotate('', xy=(0.12, 0.55), xytext=(0.20, 0.73), arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2))
ax4.annotate('', xy=(0.38, 0.55), xytext=(0.30, 0.73), arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2))
ax4.text(0.14, 0.65, "Sim", fontweight='bold', color='green', fontsize=9)
ax4.text(0.33, 0.65, "Não", fontweight='bold', color='red', fontsize=9)

# Nós filhos
ax4.text(0.12, 0.45, "Freq. Inst. Média > B?\n(Alta Rotação / Popa)", ha='center', va='center', fontsize=8.5,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#d8f3dc', edgecolor='#2d6a4f'))
ax4.text(0.38, 0.45, "Crest Factor < C?\n(Diesel Contínuo)", ha='center', va='center', fontsize=8.5,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffedd8', edgecolor='#ea580c'))

# Folhas finais
ax4.annotate('', xy=(0.06, 0.18), xytext=(0.10, 0.35), arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2))
ax4.annotate('', xy=(0.18, 0.18), xytext=(0.14, 0.35), arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2))
ax4.text(0.06, 0.10, "CLASSE:\nSMALL (Lancha)", ha='center', va='center', fontweight='bold', color='#155724', fontsize=9,
         bbox=dict(boxstyle='square,pad=0.4', facecolor='#b7efc5', edgecolor='#155724'))
ax4.text(0.18, 0.10, "CLASSE:\nMEDIUM", ha='center', va='center', fontsize=8.5,
         bbox=dict(boxstyle='square,pad=0.3', facecolor='#f1f1f1', edgecolor='gray'))

# Quadro explicativo ao lado direito
rf_explanation = (
    "A Vantagem Algorítmica das Árvores (Random Forest):\n"
    "• Independência de Topologia Visual:\n"
    "  O Random Forest NÃO procura 'linhas contínuas' ou 'texturas 2D'.\n"
    "• Partição Monotônica de Valores Isolados:\n"
    "  Ele faz perguntas pontuais: 'A energia na IMF 2 é alta?'\n"
    "• Imunidade à Esparsidade:\n"
    "  Ter 95% de zeros é irrelevante para uma árvore de decisão;\n"
    "  ela apenas divide o espaço onde há dados úteis.\n"
    "• Robustez a Cascos Inéditos:\n"
    "  Não sofre a sensibilidade geométrica de hiperplanos contínuos."
)
ax4.text(0.72, 0.50, rf_explanation, ha='center', va='center', fontsize=9.5, linespacing=1.35,
         bbox=dict(boxstyle='round,pad=0.6', facecolor='#edf2f4', edgecolor='#2b2d42', linewidth=1.2))

# -------------------------------------------------------------------------
# PAINEL 5: COMPARAÇÃO REAL NO DATASET IARA (SP INDEX E RECALL SMALL)
# -------------------------------------------------------------------------
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_title("(E) Prova Experimental no IARA:\nQueda Catastrófica da CNN no HHT", fontweight='bold', color='#1d3557', pad=10)

models_comp = ['Random Forest\n(Árvores)', 'MLP\n(Rede Densa)', 'SVM\n(Nyström RBF)', 'CNN 2D\n(Convolucional)']
sp_hht = [60.64, 56.97, 50.15, 40.15]
rec_small_hht = [52.2, 30.7, 41.7, 20.5]

x = np.arange(len(models_comp))
w = 0.35

bars_sp = ax5.bar(x - w/2, sp_hht, w, label='Índice SP Oficial (%)', color='#2b5c8f', alpha=0.9, edgecolor='black')
bars_rec = ax5.bar(x + w/2, rec_small_hht, w, label='Recall SMALL (%)', color='#e63946', alpha=0.9, edgecolor='black')

ax5.set_xticks(x)
ax5.set_xticklabels(models_comp, fontsize=8.5, fontweight='bold')
ax5.set_ylabel('Desempenho no HHT (%)', fontweight='bold')
ax5.set_ylim(0, 75)
ax5.legend(frameon=True, facecolor='white', loc='upper right', fontsize=8.5)

for bar, s in zip(bars_sp, sp_hht):
    ax5.text(bar.get_x() + bar.get_width()/2, s + 1.2, f'{s:.1f}%', ha='center', fontsize=8.5, fontweight='bold', color='#2b5c8f')
for bar, r in zip(bars_rec, rec_small_hht):
    ax5.text(bar.get_x() + bar.get_width()/2, r + 1.2, f'{r:.1f}%', ha='center', fontsize=8.5, fontweight='bold', color='#c1121f')

ax5.annotate('Colapso da CNN no HHT:\nSP cai para 40.1%\nSmall cai para 20.5%!', 
             xy=(3.15, 23), xytext=(2.2, 58),
             arrowprops=dict(facecolor='red', arrowstyle='->', lw=1.5),
             fontweight='bold', fontsize=8.5, color='red',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee', edgecolor='red'))

plt.suptitle("Por que a CNN Falha no HHT? Descasamento entre Convoluções 2D e Frequência Instantânea", 
             fontsize=14, fontweight='bold', y=0.98, color='#0b090a')

plt.savefig('reports/por_que_cnn_falha_hht.png', dpi=300, bbox_inches='tight')
print('Salvo reports/por_que_cnn_falha_hht.png com sucesso!')
