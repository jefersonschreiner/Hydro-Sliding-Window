import os
import time
import pandas as pd
from modulos.banco import carregar
from modulos.parametros import NaoEstacionariedade
from modulos.metodologias import gerarTodos
 
# Configurações
CAMINHO_BANCO    = os.path.join('dados', 'Banco.csv')
PASTA_RESULTADOS = 'resultados'

# Main 
def main():
 
    tempo_inicio = time.time()
    print("  HYDRO-SLIDING-WINDOW")
    print("  Análise de Não-Estacionariedade e Geração")
    print("  de Séries Sintéticas de Vazão")
 
    # 1 — Carregar dados
    print("\n[1/3] Carregando dados...")
    df = carregar(CAMINHO_BANCO)
    print(f"Registros: {len(df)}")
    print(f"Período: {df['data'].min().date()} → {df['data'].max().date()}")
 
    # 2 — Análise de não-estacionariedade
    print("\n[2/3] Analisando estacionariedade...")
    testes = NaoEstacionariedade(df)
    testes.rodar()
    testes.resumo()
 
    if testes.serie_estacionaria():
        print("\n Série ESTACIONÁRIA detectada.")
        print("A metodologia de janelas deslizantes não se aplica.")
        print("Encerrando o programa.")
        return
 
    # 3 — Gerar os 7 bancos de dados
    print("\n[3/3] Gerando os 7 bancos de dados...")
    gerarTodos(df, pasta=PASTA_RESULTADOS)
 
    # 4 — Resumo final
    tempo_total = time.time() - tempo_inicio
    horas       = int(tempo_total // 3600)
    minutos     = int((tempo_total % 3600) // 60)
    segundos    = int(tempo_total % 60)
 
    print("  RESUMO FINAL")
    print(f"  Tempo total : {horas}h {minutos}min {segundos}s")
 
    # Lê e imprime o log de modelos vencedores
    caminho_log = os.path.join(PASTA_RESULTADOS, 'log_modelos.csv')
    if os.path.exists(caminho_log):
        df_log = pd.read_csv(caminho_log, sep=';', decimal=',')
 
        print(f"\n  Calibrações realizadas: {len(df_log)}")
        print(f"\n  Modelos selecionados:")
 
        contagem = df_log['modelo'].value_counts()
        for modelo, count in contagem.items():
            pct = count / len(df_log) * 100
            print(f"    {modelo:<8}: {count:>3} vezes ({pct:.1f}%)")
 
        print(f"\n  Tempo médio por calibração: {df_log['tempo'].mean():.1f}s")
        print(f"  Tempo mínimo: {df_log['tempo'].min():.1f}s")
        print(f"  Tempo máximo: {df_log['tempo'].max():.1f}s")
    print("  Concluído! Resultados salvos em:", PASTA_RESULTADOS)
 
if __name__ == '__main__':
    main()