import pandas as pd
import numpy as np
import pymannkendall as mk
from scipy import stats


class NaoEstacionariedade:

    # Inicialização da Classe
    def __init__(self, df: pd.DataFrame):
        self.df_diario = df

        # Agrega para médias anuais — escala correta para detectar
        # mudanças de regime de longo prazo (mudanças climáticas)
        df_anual = (
            df.set_index('data')
            .resample('YE')['vazao']
            .mean()
            .reset_index()
        )
        df_anual.columns = ['data', 'vazao']

        self.df = df_anual
        self.vazoes = df_anual['vazao'].values

        self.resultado_mk = None
        self.resultado_rho = None
        self.resultado_pt = None

    def rodar(self):
        self.mannkendall()
        self.rho()
        self.pettitt()

    # Testes para Estacionariedade
    def mannkendall(self):
        # Detecta tendência na série
        resultado = mk.original_test(self.vazoes)

        self.resultado_mk = {
            'tendencia'    : resultado.trend,
            'p_valor'      : resultado.p,
            'significativo': resultado.p < 0.05
        }

    # Avalia correlação monotônica entre tempo e vazão
    def rho(self):
        tempo = np.arange(len(self.vazoes))
        correlacao, p_valor = stats.spearmanr(tempo, self.vazoes)

        self.resultado_rho = {
            'correlacao'   : correlacao,
            'p_valor'      : p_valor,
            'significativo': p_valor < 0.05
        }

    # Funções do Teste de Pettitt pelo método dos rankings
    def pettitt_segmento(self, indices):
        # Aplicação dos rankings em um segmento da série definido por índices
        segmento = self.vazoes[indices]
        n = len(segmento)

        if n < 4:
            return None

        # Converte os valores para ranks dentro do segmento
        ranks = stats.rankdata(segmento)

        # Calcula estatística U de forma incremental (sem matriz)
        u = np.zeros(n)
        for t in range(1, n):
            u[t] = u[t - 1] + (2 * ranks[t] - n - 1)

        klocal = int(np.argmax(np.abs(u)))
        kstat  = np.max(np.abs(u))

        # Aproximação do p-valor
        pvalor = 2 * np.exp((-6 * kstat**2) / (n**3 + n**2))
        pvalor = min(pvalor, 1.0)

        # Converte índice local para índice global da série completa
        kglobal = indices[klocal]  # ← colchetes, não parênteses

        return kglobal, kstat, pvalor

    def pettitt_recursivo(self, indices, quebras):
        # Aplica o Pettitt recursivamente para encontrar todas as quebras
        resultado = self.pettitt_segmento(indices)
        if resultado is None:
            return

        kglobal, kstat, pvalor = resultado

        if pvalor < 0.05:
            data_quebra = self.df['data'].iloc[kglobal].date()

            quebras.append({
                'ponto_quebra' : kglobal,
                'data_quebra'  : data_quebra,
                'k_stat'       : kstat,
                'p_valor'      : pvalor,
                'significativo': True
            })

            # Divide e aplica nos dois subsegmentos
            self.pettitt_recursivo(indices[:kglobal - indices[0] + 1], quebras)
            self.pettitt_recursivo(indices[kglobal - indices[0] + 1:], quebras)

    def pettitt(self):
        # Identifica todos os pontos de ruptura
        indices = np.arange(len(self.vazoes))
        quebras = []
        self.pettitt_recursivo(indices, quebras)

        # Ordena cronologicamente
        quebras = sorted(quebras, key=lambda x: x['ponto_quebra'])

        if len(quebras) == 0:
            self.resultado_pt = {
                'quebras'      : [],
                'significativo': False
            }
        else:
            self.resultado_pt = {
                'quebras'      : quebras,
                'significativo': True
            }

    # Avaliação Final e Resultados
    def serie_estacionaria(self) -> bool:
        if self.resultado_mk is None:
            raise RuntimeError("Eita")

        mk_est  = not self.resultado_mk['significativo']
        rho_est = not self.resultado_rho['significativo']
        pt_est  = not self.resultado_pt['significativo']

        return mk_est and rho_est and pt_est

    # Imprime os resultados
    def resumo(self):
        if self.resultado_mk is None:
            raise RuntimeError("Eita")

        print("Análise da Não-Estacionariedade")

        print("\n[ Mann-Kendall ]")
        print(f"  Tendência    : {self.resultado_mk['tendencia']}")
        print(f"  p-valor      : {self.resultado_mk['p_valor']:.4f}")
        print(f"  Significativo: {self.resultado_mk['significativo']}")

        print("\n[ Spearman Rho ]")
        print(f"  Correlação   : {self.resultado_rho['correlacao']:.4f}")
        print(f"  p-valor      : {self.resultado_rho['p_valor']:.4f}")
        print(f"  Significativo: {self.resultado_rho['significativo']}")

        print("\n[ Pettitt ]")
        if not self.resultado_pt['significativo']:
            print("  Nenhum ponto de quebra significativo encontrado.")
        else:
            print(f"  Total de quebras encontradas: {len(self.resultado_pt['quebras'])}")
            for i, q in enumerate(self.resultado_pt['quebras'], 1):
                print(f"\n  Quebra {i}:")
                print(f"    Data         : {q['data_quebra']}")
                print(f"    K estatístico: {q['k_stat']:.4f}")
                print(f"    p-valor      : {q['p_valor']:.4f}")

        if self.serie_estacionaria():
            print("\nSérie ESTACIONÁRIA - Reveja seus conceitos parça")
        else:
            print("\nSérie NÃO ESTACIONÁRIA - Marcha")


"-------------------------------- FIM DO CÓDIGO --------------------------------"

'''
# Teste
if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from modulos.banco import carregar

    df = carregar(os.path.join('dados', 'Banco.csv'))

    testes = NaoEstacionariedade(df)
    testes.rodar()
    testes.resumo()
'''