import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
import warnings
warnings.filterwarnings('ignore')

from .modelos import (ajustarArmaWorker, gerarArma,
                     ajustarGarWorker, gerarGar,
                     ajustarParmaWorker, gerarParma, 
                     MESES_INICIO)

class Sams:
    """
    Seleciona automaticamente o modelo estocástico mais adequado
    para uma janela de calibração, usando AIC como critério principal.
 
    Modelos candidatos: ARMA(p,q), GAR(1), PARMA(p,q)
 
    Fluxo:
        1. Padroniza a série (remove sazonalidade diária)
        2. Ajusta os três modelos em paralelo
        3. Seleciona o modelo com menor AIC
        4. Gera ensemble re-adicionando a sazonalidade
    """
 
    MESES_INICIO = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

    def __init__(self, df:pd.DataFrame):

        self.df = df[~((df['data'].dt.month == 2) & (df['data'].dt.day == 29))].copy()
        self.df = self.df.sort_values('data').reset_index(drop=True)

        if np.any(self.df['vazao'].values <= 0):
            raise ValueError("Série contém valores <= 0. Verifique os dados.")

        # Serie bruta comum
        self.serie=self.df['vazao'].values
        self.df['dia_ano'] = self.df['data'].apply(self.calcularDiaAno)

        #Variáveis
        self.modelo_vencedor = None
        self.aic_vencedor = np.inf
        self.bic_vencedor = np.inf
        self.params_vencedor = None
        self.resultados = {}
    
    @staticmethod
    def calcularDiaAno(data: pd.Timestamp) -> int:
        # Ignora o dia do ano bissexto
        dia = data.timetuple().tm_yday
        if data.is_leap_year and dia > 59:
            dia -= 1
        return dia
    
    def extrairMes(self, mes: int) -> np.ndarray:
        # Extrai os valores histórios de um mês específico (PARMA)
        ini = self.MESES_INICIO[mes]
        fim = self.MESES_INICIO[mes+1] if mes < 11 else 365
        dias_mes = list(range(ini+1,fim+1))
        return self.df[self.df['dia_ano'].isin(dias_mes)]['vazao'].values

    def ajustarArma(self) -> dict:
        # Testa ARMA (p,q) em paralelo

        combinacoes = [(self.serie, p, q)
                       for p in range (4) for q in range (4) if not (p==0 and q==0)]
        
        melhor_aic = np.inf
        melhor = None
        
        for c in combinacoes:
            resultado = ajustarArmaWorker(c)

            if resultado and resultado['aic']<melhor_aic:
                melhor_aic=resultado['aic']
                melhor = resultado
        
        if melhor is None:
            return None
        
        return{
            'modelo':'ARMA',
            'ordem': (melhor['p'], melhor['q']),
            'ajuste': melhor['ajuste'],
            'aic': melhor['aic'],
            'bic': melhor['bic'],
            }
    
    def ajustarGar(self) -> dict:
        #Ajusta GAR(1) na série padronizada

        resultado = ajustarGarWorker(self.serie)
        if resultado is None:
            return None
        
        return {
            'modelo': 'GAR',
            'ordem' : (1,),
            'params': resultado['params'],
            'aic'   : resultado['aic'],
            'bic'   : resultado['bic'],
        }
    
    def ajustarParma(self) -> dict:
        #Ajusta PARMA - um ARMA por mês, em paralelo entre os 12 meses

        tarefas=[(self.extrairMes(m),m) for m in range(12)]

        aic_total = 0
        bic_total = 0
        params_mensais=[None]*12

        for t in tarefas:
            resultado = ajustarParmaWorker(t)
            if resultado:
                aic_total += resultado['aic']
                bic_total += resultado['bic']
                params_mensais[resultado['mes']] = resultado['params']

        return {
            'modelo': 'PARMA',
            'ordem': 'periódico',
            'params': params_mensais,
            'aic': aic_total,
            'bic': bic_total,
        }
    
    def selecionar(self, verbose: bool = True) -> str:
        #Ajusta os três modelos em paralelo e seleciona o melhor pelo AIC

        with ThreadPoolExecutor(max_workers=3) as executor:
            futuro_arma = executor.submit(self.ajustarArma)
            futuro_gar = executor.submit(self.ajustarGar)
            futuro_parma = executor.submit(self.ajustarParma)

            candidatos = {'ARMA':futuro_arma.result(),'GAR':futuro_gar.result(),'PARMA':futuro_parma.result()}

        for nome, resultado in candidatos.items():
            if resultado is None:
                continue
            self.resultados[nome]=resultado
            if resultado['aic']<self.aic_vencedor:
                self.aic_vencedor=resultado['aic']
                self.bic_vencedor=resultado['bic']
                self.modelo_vencedor=nome
                self.params_vencedor=resultado

        if verbose:
            self.imprimirResumo()
        
        return self.modelo_vencedor
    
    def gerar(self, n_anos:int, n_series:int, semente_base:int=42) -> np.ndarray:
        # Isso aqui vai gerar o ensemble usando o modelo vencedor.

        if self.modelo_vencedor is None:
            raise RuntimeError("Executar a funsa selecionar() antes vilão")
        
        ensemble = np.zeros((n_series,n_anos*365))
        for i in range(n_series):
            np.random.seed(semente_base+i)
            ensemble[i]=self.gerarSerie(n_anos)
        
        return ensemble
    
    def gerarSerie(self,n_anos:int)->np.ndarray:
        # Gera uma série padronizada usando o modelo vencedor
        n_dias = n_anos*365

        if self.modelo_vencedor == 'ARMA':
            return gerarArma(self.params_vencedor['ajuste'], n_dias)
        
        elif self.modelo_vencedor == 'GAR':
            return gerarGar(self.params_vencedor['params'], n_dias)
        
        elif self.modelo_vencedor == 'PARMA':
            return gerarParma(self.params_vencedor['params'], n_dias, self.extrairMes)
        
        return np.zeros(n_dias)
    
    def imprimirResumo(self) -> None:
        #Imprime a tabela
        print("   SELEÇÃO DO MODELO - SAMS   ")
        print(f"{'Modelo':<10} {'AIC':>12} {'BIC':>12}")
        for nome, res in self.resultados.items():
            marca = " ←" if nome == self.modelo_vencedor else ""
            print(f" {nome:<10} {res['aic']:>12.2f} {res['bic']:>12.2f}{marca}")
        print(f"Vencedor: {self.modelo_vencedor} (AIC = {self.aic_vencedor:.2f})")
    
    @staticmethod
    def registrarLog(log:list,ciclo:int,janela:str,modelo:str,aic:float,bic:float,tempo:float) -> None:
        # Registra o resultado de cada calibração de uma lista de log
        log.append({
            'ciclo': ciclo,
            'janela': janela,
            'modelo': modelo,
            'aic': aic,
            'bic': bic,
            'tempo': tempo,
        })

    @staticmethod
    def estimarTempo(tempo_ciclo:float,ciclos_restante:int) -> str:
        # Estimativa de tempo como diz o nome da função né

        segundos = tempo_ciclo*ciclos_restante
        horas = int(segundos // 3600)
        minutos = int((segundos%3600)//60)
        return f"{horas}h {minutos}min"
    
"-------------------------------- FIM DO CÓDIGO --------------------------------"

'''
# Teste
if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from banco import carregar, periodo

    df     = carregar(os.path.join('dados', 'Banco.csv'))
    janela = periodo(df, 1931, 1975)

    print("Rodando SAMS na janela 1931–1975...")
    tempo_inicio = time.time()
    sams = Sams(janela)
    vencedor = sams.selecionar()
    tempo_ciclo = time.time() - tempo_inicio

    print(f"Tempo: {tempo_ciclo:.2f}s")
    print(f"Estimativa para 127 ciclos: {Sams.estimarTempo(tempo_ciclo, 127)}")

    print(f"\nGerando ensemble com modelo {vencedor}...")
    ensemble = sams.gerar(n_anos=48, n_series=10, semente_base=42)
    print(f"Shape: {ensemble.shape}")
    print(f"Média gerada: {ensemble.mean():.2f}")
    print(f"Média histórica: {janela['vazao'].mean():.2f}")

    log = []
    Sams.registrarLog(log, ciclo=1, janela='1931-1975',
                      modelo=vencedor, aic=sams.aic_vencedor,
                      bic=sams.bic_vencedor, tempo=tempo_ciclo)
    print(f"\nLog registrado: {log}")
'''