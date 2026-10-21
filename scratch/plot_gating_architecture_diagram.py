"""
Gera o Diagrama Arquitetural Corrigido do Processo de Gating Front-End (IARA).
Mostra explicitamente os 4 Ramos Especialistas (Extrator + Modelo Dedicado),
eliminando o conceito ambíguo de 'classificador downstream agnóstico'.
Salva em 'reports/diagrama_processo_gating.png'.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_gating_diagram(output_path=None):
    if output_path is None:
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reports', 'diagrama_processo_gating.png'))
    else:
        output_path = os.path.abspath(output_path)
    
    # 16:9 widescreen para apresentações, relatórios e teses
    fig = plt.figure(figsize=(16, 10.5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Paleta de cores corporativa e científica
    c_bg = '#f8fafc'         # Slate 50
    c_text_dark = '#0f172a'  # Slate 900
    c_text_muted = '#475569' # Slate 600
    
    # Cores de destaque dos 4 ramos especialistas
    c_mel = '#0284c7'        # Sky 600
    c_lofar = '#059669'      # Emerald 600
    c_hht = '#d97706'        # Amber 600
    c_demon = '#7c3aed'      # Violet 600
    c_gate = '#0d9488'       # Teal 600

    fig.patch.set_facecolor(c_bg)
    ax.set_facecolor(c_bg)

    # ------------------ CABEÇALHO DO DIAGRAMA ------------------
    ax.text(50, 97.5, "PROJETO IARA — ARQUITETURA DO SISTEMA DE GATING FRONT-END (MoE)", 
            ha='center', va='center', fontsize=17.5, fontweight='heavy', color='#0f172a')
    ax.text(50, 94.8, "Roteador Inteligente por Competência Acústica com 4 Especialistas Dedicados Pré-Treinados", 
            ha='center', va='center', fontsize=10.8, fontweight='medium', color='#64748b')

    # ------------------ 1. CAMADA DE ENTRADA (SINAL DE ÁUDIO) ------------------
    box_audio = FancyBboxPatch((24, 83.2), 52, 8.2, boxstyle="round,pad=0.5,rounding_size=1.2",
                               fc='#ffffff', ec='#0284c7', lw=2.2, zorder=2)
    ax.add_patch(box_audio)
    
    badge_in = FancyBboxPatch((25.5, 88.5), 12, 2.3, boxstyle="round,pad=0.2,rounding_size=0.5",
                              fc='#0284c7', ec='none', zorder=3)
    ax.add_patch(badge_in)
    ax.text(31.5, 89.65, "ENTRADA ACÚSTICA", ha='center', va='center', fontsize=7.5, fontweight='bold', color='white', zorder=4)

    ax.text(50, 87.5, "Sinal de Áudio Submarino Bruto (.wav)", ha='center', va='center', fontsize=12, fontweight='bold', color=c_text_dark, zorder=3)
    ax.text(50, 85.3, "Janela temporal CPA de 10 s • Taxa de amostragem: fs = 48 kHz (decimado p/ 16 kHz)",
            ha='center', va='center', fontsize=8.4, color=c_text_muted, zorder=3)

    # Mini-onda decorativa
    np.random.seed(42)
    t_wave = np.linspace(0, 10, 250)
    sig_wave = (0.28 * np.sin(2 * np.pi * 1.5 * t_wave) * np.exp(-0.15 * t_wave) +
                0.18 * np.sin(2 * np.pi * 4.2 * t_wave) +
                0.08 * np.random.randn(len(t_wave)))
    x_wave = np.linspace(26.5, 73.5, len(t_wave))
    y_wave = 84.1 + sig_wave * 0.9
    ax.plot(x_wave, y_wave, color='#0284c7', lw=1.1, alpha=0.55, zorder=3)

    # Seta 1: Entrada -> Roteador de Gating
    arr1 = FancyArrowPatch((50, 83.2), (50, 78.4), arrowstyle='-|>', mutation_scale=18,
                           color='#0284c7', lw=2.4, zorder=5)
    ax.add_patch(arr1)
    
    tag_arr1 = FancyBboxPatch((40.0, 80.0), 20, 1.8, boxstyle="round,pad=0.1,rounding_size=0.3",
                              fc='#f0f9ff', ec='#0284c7', lw=0.8, zorder=6)
    ax.add_patch(tag_arr1)
    ax.text(50, 80.9, "Extração de 7 meta-features (~ms)", ha='center', va='center', 
            fontsize=7.4, fontweight='bold', color='#0369a1', zorder=7)

    # ------------------ 2. CAMADA DE GATING FRONT-END (ROTEADOR POR COMPETÊNCIA) ------------------
    box_gate = FancyBboxPatch((7, 57.5), 86, 20.8, boxstyle="round,pad=0.7,rounding_size=1.4",
                              fc='#f0fdfa', ec=c_gate, lw=2.4, zorder=2)
    ax.add_patch(box_gate)

    badge_gate = FancyBboxPatch((9, 74.8), 34, 2.9, boxstyle="round,pad=0.2,rounding_size=0.6",
                                fc=c_gate, ec='none', zorder=3)
    ax.add_patch(badge_gate)
    ax.text(26, 76.25, "ACOUSTIC FRONT-END GATING (ROUTER)", ha='center', va='center', 
            fontsize=9.5, fontweight='heavy', color='white', zorder=4)

    ax.text(78, 76.25, "Estimador de Probabilidade de Acerto de Cada Especialista", 
            ha='right', va='center', fontsize=9.2, fontweight='bold', color='#115e59', zorder=4)

    # Sub-bloco A: Meta-features
    box_meta = FancyBboxPatch((9.2, 59.0), 48.5, 14.8, boxstyle="round,pad=0.4,rounding_size=0.8",
                              fc='#ffffff', ec='#99f6e4', lw=1.2, zorder=3)
    ax.add_patch(box_meta)
    ax.text(11, 71.9, "Propriedades Físicas Avaliadas no Sinal Bruto (10s):", 
            fontsize=8.8, fontweight='bold', color='#0f766e', zorder=4)

    meta_items = [
        ("Crest Factor (< 250 Hz)", "Razão pico/média espectral (indica raias harmônicas diesel)", "#059669"),
        ("Spectral Flatness", "Planicidade espectral (1 = ruído banda larga, 0 = harmônicos)", "#0284c7"),
        ("Modulation Index (2-8 kHz)", "Índice de modulação de envelope (cavitação de pás de hélice)", "#7c3aed"),
        ("Spectral Flux (Flux Std)", "Variação de energia em blocos de 1 s (transientes/manobra)", "#d97706"),
        ("High / Low Ratio", "Razão [2-8 kHz] / [< 250 Hz] (separa lancha vs cargueiro)", "#e11d48"),
        ("RMS (dB) & ZCR", "Potência acústica e taxa de cruzamento por zero (ruído de fundo)", "#475569"),
    ]
    y_m = 69.4
    for name, desc, dot_col in meta_items:
        ax.plot([11.2], [y_m], marker='o', markersize=4.5, color=dot_col, zorder=4)
        ax.text(12.6, y_m, name + ": ", fontsize=7.8, fontweight='bold', color='#1e293b', zorder=4)
        ax.text(30.0, y_m, desc, fontsize=7.4, color='#475569', zorder=4)
        y_m -= 1.85

    # Sub-bloco B: Lógica de Roteamento
    box_logic = FancyBboxPatch((60.0, 59.0), 31.0, 14.8, boxstyle="round,pad=0.4,rounding_size=0.8",
                               fc='#ffffff', ec='#99f6e4', lw=1.2, zorder=3)
    ax.add_patch(box_logic)
    ax.text(61.5, 71.9, "Critério de Seleção de Rota (1-de-4):", 
            fontsize=8.8, fontweight='bold', color='#0f766e', zorder=4)

    tag_form = FancyBboxPatch((62.0, 67.8), 27.0, 3.2, boxstyle="round,pad=0.15,rounding_size=0.4",
                             fc='#f0fdfa', ec='#5eead4', lw=0.9, zorder=4)
    ax.add_patch(tag_form)
    ax.text(75.5, 69.4, r"$\hat{m} = \arg\max_{m} \; P(\text{Especialista } m \text{ acerta} \mid \mathbf{x}_{\text{meta}})$",
            ha='center', va='center', fontsize=9.2, fontweight='bold', color='#0f172a', zorder=5)

    logic_notes = [
        ("• O Roteador:", "Prevê qual especialista terá maior chance de acertar"),
        ("• Comutador:", "Aciona exclusivamente o ramo do especialista eleito"),
        ("• Eficiência:", "Economiza ~70% de FLOPS computacionais"),
        ("• Sem Vazamento:", "Treinado em trn_fold = fold ^ 1 (sem leakage)"),
        ("  (As outras 3 representações pesadas NEM são extraídas!)", "")
    ]
    y_l = 66.2
    for title, note in logic_notes:
        if note:
            ax.text(61.8, y_l, title + " ", fontsize=7.3, fontweight='bold', color='#0f172a', zorder=4)
            ax.text(72.0, y_l, note, fontsize=7.1, color='#334155', zorder=4)
        else:
            ax.text(61.8, y_l, title, fontsize=6.9, fontweight='bold', color='#0d9488', zorder=4)
        y_l -= 1.45

    # Barra multiplexadora
    bar_mux = FancyBboxPatch((32, 54.8), 36, 2.2, boxstyle="round,pad=0.1,rounding_size=0.3",
                            fc='#0f766e', ec='#115e59', lw=1.0, zorder=5)
    ax.add_patch(bar_mux)
    ax.text(50, 55.9, "CHAVE MULTIPLEXADORA: DISPARA APENAS 1 ENTRE OS 4 RAMOS", 
            ha='center', va='center', fontsize=7.4, fontweight='heavy', color='#ffffff', zorder=6)

    ax.plot([50, 50], [57.5, 57.0], color=c_gate, lw=2.4, zorder=5)

    # ------------------ 3. OS 4 RAMOS ESPECIALISTAS (EXTRATOR + MODELO DEDICADO) ------------------
    # Cada coluna contém a representação física E o especialista treinado especificamente para ela
    branches_info = [
        {
            'name': 'RAMO MEL',
            'extractor': 'Extrator MEL (256-D)',
            'extractor_desc': 'Banco Psicoacústico 0–8 kHz',
            'specialist': 'Especialista MEL',
            'specialist_desc': 'Treinado c/ entradas de 256-D',
            'domain_focus': 'Foco: Ruído Contínuo & Fundo',
            'routing_rate': 'Roteamento: ~38.8%',
            'color': c_mel,
            'bg': '#f0f9ff',
            'border': '#bae6fd',
            'x': 6.5,
            'out_x': 15.5
        },
        {
            'name': 'RAMO LOFAR',
            'extractor': 'Extrator LOFAR (1024-D)',
            'extractor_desc': 'Banda Estreita FFT 0–2 kHz',
            'specialist': 'Especialista LOFAR',
            'specialist_desc': 'Treinado c/ entradas de 1024-D',
            'domain_focus': 'Foco: Raias Tonais de Diesel',
            'routing_rate': 'Roteamento: ~36.3%',
            'color': c_lofar,
            'bg': '#ecfdf5',
            'border': '#a7f3d0',
            'x': 29.5,
            'out_x': 38.5
        },
        {
            'name': 'RAMO HHT',
            'extractor': 'Extrator HHT (256-D)',
            'extractor_desc': 'EMD + Hilbert Marginal 0–1 kHz',
            'specialist': 'Especialista HHT',
            'specialist_desc': 'Treinado c/ entradas de 256-D',
            'domain_focus': 'Foco: Manobras & Lanchas Small',
            'routing_rate': 'Roteamento: ~23.0%',
            'color': c_hht,
            'bg': '#fffbeb',
            'border': '#fde68a',
            'x': 52.5,
            'out_x': 61.5
        },
        {
            'name': 'RAMO DEMON',
            'extractor': 'Extrator DEMON (253-D)',
            'extractor_desc': 'Demodulação Envelope 0–25 Hz',
            'specialist': 'Especialista DEMON',
            'specialist_desc': 'Treinado c/ entradas de 253-D',
            'domain_focus': 'Foco: Cavitação / Pás de Hélice',
            'routing_rate': 'Roteamento: ~1.9%',
            'color': c_demon,
            'bg': '#faf5ff',
            'border': '#ddd6fe',
            'x': 75.5,
            'out_x': 84.5
        }
    ]

    for b in branches_info:
        cx = b['x'] + 9.0
        
        # Seta do MUX para o topo do ramo
        arr = FancyArrowPatch((cx, 54.8), (cx, 49.5), arrowstyle='-|>', mutation_scale=16,
                              color=b['color'], lw=2.2, zorder=5)
        ax.add_patch(arr)

        # Caixa do Ramo Especialista Completo
        box_branch = FancyBboxPatch((b['x'], 17.5), 18.0, 32.0, boxstyle="round,pad=0.5,rounding_size=1.0",
                                    fc=b['bg'], ec=b['color'], lw=2.0, zorder=3)
        ax.add_patch(box_branch)

        # Cabeçalho do Ramo
        badge_b = FancyBboxPatch((b['x'] + 1.0, 45.8), 16.0, 2.7, boxstyle="round,pad=0.2,rounding_size=0.5",
                                 fc=b['color'], ec='none', zorder=4)
        ax.add_patch(badge_b)
        ax.text(cx, 47.15, b['name'], ha='center', va='center', fontsize=9.2, fontweight='heavy', color='white', zorder=5)

        # 1. Bloco do Extrator de Características dentro do ramo
        box_ext = FancyBboxPatch((b['x'] + 1.2, 36.2), 15.6, 8.2, boxstyle="round,pad=0.2,rounding_size=0.5",
                                 fc='#ffffff', ec=b['border'], lw=1.2, zorder=4)
        ax.add_patch(box_ext)
        ax.text(cx, 42.6, "1. EXTRAÇÃO ESPECTRAL", ha='center', va='center', fontsize=6.8, fontweight='heavy', color=b['color'], zorder=5)
        ax.text(cx, 40.4, b['extractor'], ha='center', va='center', fontsize=7.6, fontweight='bold', color=c_text_dark, zorder=5)
        ax.text(cx, 38.3, b['extractor_desc'], ha='center', va='center', fontsize=6.8, color='#475569', zorder=5)

        # Seta interna: Extrator -> Especialista
        arr_inner = FancyArrowPatch((cx, 36.2), (cx, 32.5), arrowstyle='-|>', mutation_scale=12,
                                    color=b['color'], lw=1.8, zorder=5)
        ax.add_patch(arr_inner)

        # 2. Bloco do Classificador Especialista Dedicado dentro do ramo
        box_esp = FancyBboxPatch((b['x'] + 1.2, 22.8), 15.6, 9.4, boxstyle="round,pad=0.2,rounding_size=0.5",
                                 fc='#ffffff', ec=b['color'], lw=1.5, zorder=4)
        ax.add_patch(box_esp)
        ax.text(cx, 30.5, "2. MODELO ESPECIALISTA", ha='center', va='center', fontsize=6.8, fontweight='heavy', color=b['color'], zorder=5)
        ax.text(cx, 28.3, b['specialist'], ha='center', va='center', fontsize=7.8, fontweight='heavy', color=c_text_dark, zorder=5)
        ax.text(cx, 26.3, b['specialist_desc'], ha='center', va='center', fontsize=6.7, color='#334155', zorder=5)
        ax.text(cx, 24.5, b['domain_focus'], ha='center', va='center', fontsize=6.5, fontweight='bold', color='#0f172a', zorder=5)

        # Tag de frequência de roteamento
        tag_share = FancyBboxPatch((b['x'] + 1.5, 18.5), 15.0, 2.3, boxstyle="round,pad=0.15,rounding_size=0.4",
                                   fc='#ffffff', ec=b['color'], lw=1.0, zorder=4)
        ax.add_patch(tag_share)
        ax.text(cx, 19.65, b['routing_rate'], ha='center', va='center', fontsize=6.8, fontweight='heavy', color=b['color'], zorder=5)

        # Seta de saída do ramo em direção ao resultado final
        arr_out = FancyArrowPatch((cx, 17.5), (50, 11.8),
                                  connectionstyle="angle3,angleA=90,angleB=0",
                                  arrowstyle='-|>', mutation_scale=16,
                                  color=b['color'], lw=1.8, alpha=0.9, zorder=5)
        ax.add_patch(arr_out)

    # ------------------ BARRA EXPLICATIVA SOBRE AS FAMÍLIAS DE ESPECIALISTAS ------------------
    # Esta caixa esclarece por que os notebooks falavam em "agnóstico":
    # A arquitetura suporta montar os 4 especialistas com RF, MLP, SVM ou H-MoE!
    box_info = FancyBboxPatch((15, 12.8), 70, 3.4, boxstyle="round,pad=0.2,rounding_size=0.4",
                             fc='#1e293b', ec='#0f172a', lw=1.2, zorder=6)
    ax.add_patch(box_info)
    ax.text(50, 14.8, "CONFIGURAÇÃO DOS 4 ESPECIALISTAS: Podem ser da mesma família ou combinados em H-MoE",
            ha='center', va='center', fontsize=7.6, fontweight='heavy', color='#38bdf8', zorder=7)
    ax.text(50, 13.6, "Testados no Projeto: 4x Random Forest (69.8% SP) | 4x MLP (69.0% SP) | 4x SVM (67.7% SP) | H-MoE Misto (70.7% SP)",
            ha='center', va='center', fontsize=6.9, color='#e2e8f0', zorder=7)

    # Seta: Decisão do Especialista Selecionado -> Predição Final
    arr_final = FancyArrowPatch((50, 12.8), (50, 9.8), arrowstyle='-|>', mutation_scale=18,
                                color='#0f172a', lw=2.4, zorder=6)
    ax.add_patch(arr_final)

    # ------------------ 4. CAMADA DE SAÍDA FINAL (4 CLASSES IARA) ------------------
    box_classes = FancyBboxPatch((10, 1.8), 80, 7.8, boxstyle="round,pad=0.6,rounding_size=1.0",
                                 fc='#ffffff', ec='#94a3b8', lw=1.5, zorder=2)
    ax.add_patch(box_classes)

    ax.text(50, 8.0, "PREDIÇÃO FINAL DO CONTATO ACÚSTICO (DATASET IARA)", 
            ha='center', va='center', fontsize=9.2, fontweight='bold', color='#1e293b', zorder=3)

    classes_info = [
        ("SMALL", "Lanchas e Patrulhas", "< 50 metros", "#ef4444", "#fef2f2", 13.0),
        ("MEDIUM", "Rebocadores / Médio Porte", "50 a 100 metros", "#f59e0b", "#fffbeb", 32.0),
        ("LARGE", "Cargueiros & Petroleiros", "> 100 metros", "#3b82f6", "#eff6ff", 51.0),
        ("BACKGROUND", "Ruído Oceânico Ambiente", "Sem alvo", "#10b981", "#ecfdf5", 70.0),
    ]

    for c_title, c_desc, c_size, c_c, c_b, c_pos in classes_info:
        c_box = FancyBboxPatch((c_pos, 2.5), 17.0, 4.4, boxstyle="round,pad=0.2,rounding_size=0.5",
                               fc=c_b, ec=c_c, lw=1.2, zorder=3)
        ax.add_patch(c_box)
        ax.text(c_pos + 8.5, 5.6, c_title, ha='center', va='center', fontsize=8.8, fontweight='heavy', color=c_c, zorder=4)
        ax.text(c_pos + 8.5, 4.2, c_desc, ha='center', va='center', fontsize=6.6, fontweight='medium', color='#334155', zorder=4)
        ax.text(c_pos + 8.5, 3.1, c_size, ha='center', va='center', fontsize=6.3, color='#64748b', zorder=4)

    # Rodapé com nota científica
    ax.text(50, 0.5, "Projeto IARA • Laboratório de Processamento de Sinais (LPS / COPPE / UFRJ) • Avaliação rigorosa sob protocolo 5x2cv",
            ha='center', va='center', fontsize=7.5, color='#94a3b8')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Diagrama atualizado e salvo com sucesso em: {output_path}")

if __name__ == '__main__':
    create_gating_diagram()
