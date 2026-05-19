import numpy as np
import pandas as pd
import os
import time
from .sams import Sams
from .banco import periodo

# Config inicial
ANO_INICIO_DADOS = 1931
ANO_CORTE = 1975
ANO_FIM_DADOS = 2023
ANO_INICIO_MJD = 1946
JANELA_MJD = 30
N_SERIES = 500
SEMENTE_BASE = 42

# FUNÇÕES AUXILIARES

def gerarData(ano_ini: int, ano_fim: int) -> pd.DatetimeIndex:
    # Gera o indice dde datas diárias

    datas = pd.date_range(
        start=f'{ano_ini}-01-01',
        end=f'{ano_fim}-12-31',
        freq='D'
    )
    return datas[~((datas.month == 2) & (datas.day == 29))]

def montarDataFrame(ensemble: np.ndarray, datas: pd.DatetimeIndex) -> pd.DataFrame:
    # Conversor da array para colunas
    
    n_series = ensemble.shape[0]
    colunas = [f'serie_{i+1:03d}' for i in range(n_series)]
    df = pd.DataFrame(ensemble.T, index=datas, columns=colunas)
    df.index.name = 'data'

    return df

def salvar(df: pd.DataFrame, nome: str, pasta: str) -> None:
    #Salva o banco em CSV

    os.makedirs(pasta, exist_ok=True)
    caminho = os.path.join(pasta, f'{nome}.csv')
    df.to_csv(caminho, sep=';', decimal=',', date_format='%d/%m/%Y')
    print(f'Salvo: {caminho} ({df.shape[0]} dias x {df.shape[1]} series)')

def calibrarEGerar(df_obs:pd.DataFrame, ano_ini_cal: int, ano_fim_cal: int,
                   n_anos_sim:int, semente_base: int, log:list, ciclo:int) -> np.ndarray:
    
    janela = periodo(df_obs, ano_ini_cal, ano_fim_cal)
    tempo_ini = time.time()
    sams = Sams(janela)
    sams.selecionar(verbose=False)
    tempo_ciclo = time.time() - tempo_ini

    janela_str = f'{ano_ini_cal}-{ano_fim_cal}'
    sams.registrarLog(log, ciclo=ciclo, janela=janela_str, modelo=sams.modelo_vencedor,
                      aic=sams.aic_vencedor, bic=sams.bic_vencedor, tempo=tempo_ciclo)
    
    print(f'Modelo: {sams.modelo_vencedor} / AIC:{sams.aic_vencedor} / {tempo_ciclo:.1f}s')

    return sams.gerar(n_anos=n_anos_sim, n_series=N_SERIES, semente_base=semente_base)

# METODOLOGIAS DE ATUALIZAÇÕES PROPOSTAS

# Modelo Fixo Base
def gerar_MFB(df_obs: pd.DataFrame, log:list, pasta: str = 'resultados') -> None:
    # Calibra com 1935-1975 e gera as séries de 1976-2023 de uma vez
    print('\n[MFB] Calibrando e gerando')

    n_anos = ANO_FIM_DADOS - ANO_CORTE # 48 anos
    ensemble = calibrarEGerar(df_obs, ANO_INICIO_DADOS, ANO_CORTE,n_anos,SEMENTE_BASE, log, ciclo=0)
    datas = gerarData(ANO_CORTE + 1, ANO_FIM_DADOS)
    ensemble = ensemble[:,:len(datas)]
    salvar(montarDataFrame(ensemble,datas), 'MFB', pasta)
    

# Modelo de Origem Fixa
def gerar_MOF(df_obs: pd.DataFrame, frequencia: int, log:list, pasta:str='resultados') -> None:
    # Calibra com origem no ano 1931 até o ano final da janela e simula os próximos N anos - Atualiza conforme dados observados !!!!!
    nome = f'MOF_{frequencia}ano{"s" if frequencia > 1 else ""}'
    print(f'\n[{nome}] Iniciando...')

    partes = []
    ano_atual = ANO_CORTE + 1 # 1976
    ciclo = 0

    while ano_atual <= ANO_FIM_DADOS:
        ano_fim_sim =  min(ano_atual + frequencia-1, ANO_FIM_DADOS)
        n_anos_sim = ano_fim_sim - ano_atual + 1

        print(f'  Ciclo {ciclo + 1}: calibra {ANO_INICIO_DADOS}–{ano_atual - 1}'
              f'→ simula {ano_atual}–{ano_fim_sim}')
        
        ensemble = calibrarEGerar(df_obs, ANO_INICIO_DADOS, ano_atual-1, n_anos_sim,
                                  SEMENTE_BASE+ciclo*100, log, ciclo)
        datas = gerarData(ano_atual, ano_fim_sim)
        ensemble = ensemble[:,:len(datas)]
        partes.append(montarDataFrame(ensemble, datas))

        ano_atual = ano_fim_sim+1
        ciclo += 1
    
    salvar(pd.concat(partes), nome, pasta)

# Modelo de Janela Deslizante
def gerar_MJD(df_obs: pd.DataFrame, frequencia: int, log:list, pasta:str='resultados') -> None:
    # Calibra com janela de 30 anos e desliza a cada ciclo
    nome = f'MJD_{frequencia}ano{"s" if frequencia > 1 else ""}'
    print(f'\n[{nome}] Iniciando...')

    partes = []
    ano_atual = ANO_CORTE+1 #1976
    ciclo = 0

    while ano_atual <= ANO_FIM_DADOS:
        ano_fim_sim = min(ano_atual + frequencia - 1, ANO_FIM_DADOS)
        n_anos_sim  = ano_fim_sim - ano_atual + 1
 
        # Janela deslizante de 30 anos
        ano_fim_cal = ano_atual - 1
        ano_ini_cal = ano_fim_cal - JANELA_MJD + 1
 
        print(f'  Ciclo {ciclo + 1}: calibra {ano_ini_cal}-{ano_fim_cal} '
              f'→ simula {ano_atual}–{ano_fim_sim}')
 
        ensemble = calibrarEGerar(df_obs, ano_ini_cal, ano_fim_cal, n_anos_sim,
                                  SEMENTE_BASE+ciclo*100, log, ciclo)
        datas = gerarData(ano_atual, ano_fim_sim)
        ensemble = ensemble[:,:len(datas)]
        partes.append(montarDataFrame(ensemble, datas))
        ano_atual = ano_fim_sim + 1
        ciclo    += 1
 

    salvar(pd.concat(partes), nome, pasta)

def gerarTodos(df_obs: pd.DataFrame, pasta:str='resultados') -> None:
    #Gera todos os bancos
    t_total = time.time()
    log=[]

    print('Gerando os 7 bancos de dados')

    gerar_MFB(df_obs, log, pasta)

    for freq in [1, 5, 10]:
        gerar_MOF(df_obs, freq, log, pasta)
    
    for freq in [1, 5, 10]:
        gerar_MJD(df_obs, freq, log, pasta)
    
    # Salva log dos modelos vencedores
    df_log = pd.DataFrame(log)
    caminho_log = os.path.join(pasta, 'log_modelos.csv')
    df_log.to_csv(caminho_log, sep=';', decimal=',', index=False)
    print(f'\nLog salvo: {caminho_log}')
    print(f'Concluído em {time.time() - t_total:.1f}s')

# ------ FIM DO CÓDIGO ------

'''
#Teste
if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from banco import carregar

    df = carregar(os.path.join('dados', 'Banco.csv'))
    gerar_todos(df, pasta='resultados')
'''