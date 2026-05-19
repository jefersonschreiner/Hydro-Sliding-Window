import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
 

# Configurações gerais
cor_laranja  = '#F4A460'   # calibração inicial MFB
cor_verde    = '#90C4A0'   # recalibrações MOF
cor_azul     = '#A8C4E0'   # recalibrações MJD
cor_cinza    = '#C8C8C8'   # simulação
cor_texto    = '#2C2C2C'
 
ano_inicio_dados = 1931
ano_fim_dados    = 2023
ano_corte        = 1975
ano_inicio_MJD   = 1946
janela_MJD       = 30
 
 
def blocos_mof(frequencia):
    blocos = []
    ano = ano_corte + 1
    i = 1
    while ano <= ano_fim_dados:
        sim_fim = min(ano + frequencia - 1, ano_fim_dados)
        blocos.append({
            'label'  : f'MOF({frequencia},{i})',
            'cal_ini': ano_inicio_dados,
            'cal_fim': ano - 1,
            'sim_ini': ano,
            'sim_fim': sim_fim,
            'tipo'   : 'MOF',
        })
        ano = sim_fim + 1
        i += 1
    return blocos
 
 
def blocos_mjd(frequencia):
    blocos = []
    ano = ano_corte + 1
    i = 1
    while ano <= ano_fim_dados:
        cal_fim = ano - 1
        cal_ini = cal_fim - janela_MJD + 1
        sim_fim = min(ano + frequencia - 1, ano_fim_dados)
        blocos.append({
            'label'  : f'MJD({janela_MJD},{frequencia},{i})',
            'cal_ini': cal_ini,
            'cal_fim': cal_fim,
            'sim_ini': ano,
            'sim_fim': sim_fim,
            'tipo'   : 'MJD',
        })
        ano = sim_fim + 1
        i += 1
    return blocos
 
 
def desenhar_timeline(ax, blocos, tipo, anos_ticks):
    total = ano_fim_dados - ano_inicio_dados
 
    for i, b in enumerate(blocos):
        y = len(blocos) - i - 1
 
        if tipo == 'MOF':
            cor_cal = cor_laranja    # Cor do MOF
        elif tipo == 'MJD':
            cor_cal = cor_azul     # Cor do MJD
 
        x_cal = b['cal_ini'] - ano_inicio_dados
        w_cal = b['cal_fim'] - b['cal_ini'] + 1
        ax.barh(y, w_cal, left=x_cal, height=0.6,
                color=cor_cal, edgecolor='white', linewidth=0.5)
        ax.text(x_cal + w_cal / 2, y, b['label'],
                ha='center', va='center', fontsize=7.5,
                color=cor_texto, fontweight='bold')
 
        # Bloco de simulação
        x_sim = b['sim_ini'] - ano_inicio_dados
        w_sim = b['sim_fim'] - b['sim_ini'] + 1
        ax.barh(y, w_sim, left=x_sim, height=0.6,
                color=cor_cinza, edgecolor='white', linewidth=0.5)
        ax.text(x_sim + w_sim / 2, y, 'Simulação',
                ha='center', va='center', fontsize=7.5, color=cor_texto)
 
    ax.set_xlim(0, total + 1)
    ax.set_ylim(-0.5, len(blocos) - 0.5)
    ax.set_xticks([a - ano_inicio_dados for a in anos_ticks])
    ax.set_xticklabels([str(a) for a in anos_ticks], fontsize=8)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
 
    titulo = 'MOF — Modelo de Origem Fixa' if tipo == 'MOF' \
             else f'MJD — Modelo de Janela Deslizante (W={janela_MJD} anos)'
    ax.set_title(titulo, fontsize=9, fontweight='bold',
                 loc='left', pad=6, color=cor_texto)
 
 
def gerar_grafico(frequencia, pasta_saida='resultados'):
    os.makedirs(pasta_saida, exist_ok=True)
 
    lista_mof = blocos_mof(frequencia)
    lista_mjd = blocos_mjd(frequencia)
 
    anos_ticks = [ano_inicio_dados, ano_inicio_MJD, ano_corte]
    ano = ano_corte + frequencia
    while ano <= ano_fim_dados:
        anos_ticks.append(ano)
        ano += frequencia
    if ano_fim_dados not in anos_ticks:
        anos_ticks.append(ano_fim_dados)
    anos_ticks = sorted(set(anos_ticks))
 
    mfb = {
        'cal_ini': ano_inicio_dados,
        'cal_fim': ano_corte,
        'sim_ini': ano_corte + 1,
        'sim_fim': ano_fim_dados,
    }
 
    n_mof = len(lista_mof)
    n_mjd = len(lista_mjd)
    total = ano_fim_dados - ano_inicio_dados
 
    altura_total = 1.2 + (n_mof + n_mjd) * 0.55
    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, altura_total),
        gridspec_kw={'height_ratios': [1, n_mof, n_mjd], 'hspace': 0.6}
    )
 
    # --- MFB ---
    ax0 = axes[0]
    x_cal = mfb['cal_ini'] - ano_inicio_dados
    w_cal = mfb['cal_fim'] - mfb['cal_ini'] + 1
    x_sim = mfb['sim_ini'] - ano_inicio_dados
    w_sim = mfb['sim_fim'] - mfb['sim_ini'] + 1
 
    ax0.barh(0, w_cal, left=x_cal, height=0.6,
             color=cor_verde, edgecolor='white', linewidth=0.5)
    ax0.text(x_cal + w_cal / 2, 0, 'MFB',
             ha='center', va='center', fontsize=8, fontweight='bold', color=cor_texto)
    ax0.barh(0, w_sim, left=x_sim, height=0.6,
             color=cor_cinza, edgecolor='white', linewidth=0.5)
    ax0.text(x_sim + w_sim / 2, 0, 'Simulação',
             ha='center', va='center', fontsize=8, color=cor_texto)
 
    ax0.set_xlim(0, total + 1)
    ax0.set_ylim(-0.5, 0.5)
    ax0.set_xticks([a - ano_inicio_dados for a in anos_ticks])
    ax0.set_xticklabels([str(a) for a in anos_ticks], fontsize=8)
    ax0.set_yticks([])
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    ax0.set_title('MFB — Modelo Fixo Base', fontsize=9, fontweight='bold',
                  loc='left', pad=6, color=cor_texto)
 
    # --- MOF ---
    desenhar_timeline(axes[1], lista_mof, 'MOF', anos_ticks)
 
    # --- MJD ---
    desenhar_timeline(axes[2], lista_mjd, 'MJD', anos_ticks)
 
    # Legenda
    legenda = [
        mpatches.Patch(color=cor_verde, label='Calibração inicial (1931–1975)'),
        mpatches.Patch(color=cor_laranja,   label='Recalibração MOF'),
        mpatches.Patch(color=cor_azul,    label='Recalibração MJD'),
        mpatches.Patch(color=cor_cinza,   label='Simulação'),
    ]
    fig.legend(handles=legenda, loc='lower center', ncol=4,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
 
    nome_freq = {1: 'Anual', 5: '5 Anos', 10: '10 Anos'}
    fig.suptitle(f'Estratégias de Atualização — {nome_freq[frequencia]}',
                 fontsize=11, fontweight='bold', y=1.01, color=cor_texto)
 
    nome_arquivo = {1: 'grafico_anual.png', 5: 'grafico_5anos.png', 10: 'grafico_10anos.png'}
    caminho = os.path.join(pasta_saida, nome_arquivo[frequencia])
    fig.savefig(caminho, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Salvo: {caminho}')
 
 
if __name__ == '__main__':
    gerar_grafico(10, pasta_saida='resultados')