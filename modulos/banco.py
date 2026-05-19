import pandas as pd
import os

# Função de leitura e ordenação dos dados
def carregar(caminho_arq: str) -> pd.DataFrame:
    
    # Carrega os dados
    df_raw = pd.read_csv(
        caminho_arq,
        sep=';',
        decimal=',',
        encoding='utf-8'
    )

    # Transforma de wide para long (um registro por dia)
    registros = []

    for _, linha in df_raw.iterrows():
        mes_ano = pd.to_datetime(linha['Data'], format='%d/%m/%Y')

        for dia in range(1, 32):
            coluna = f'Vazao{dia:02d}'

            # Ignora colunas que não existem ou valores vazios
            if coluna not in df_raw.columns:
                continue
            if pd.isna(linha[coluna]):
                continue

            # Monta a data correta para o dia
            try:
                data= pd.Timestamp(year=mes_ano.year, month=mes_ano.month, day=dia)
            except ValueError:
                # Dias inválidos
                continue

            registros.append({'data': data, 'vazao': linha[coluna]})
    
    # Cria o novo banco e ordena do mais antigo para o mais recente
    df = pd.DataFrame(registros)
    df = df.sort_values('data').reset_index(drop=True)

    return df

# Função do filtro para um período específico
def periodo(df: pd.DataFrame, ano_inicio: int, ano_fim: int) -> pd.DataFrame:

    mask = (df['data'].dt.year >= ano_inicio) & (df['data'].dt.year <= ano_fim)
    return df[mask].reset_index(drop=True)


"-------------------------------- FIM DO CÓDIGO --------------------------------"
'''
# Teste
if __name__ == '__main__':
    caminho = os.path.join('dados', 'Banco.csv')
    df = carregar(caminho)

    print(f"Total de registros: {len(df)}")
    print(f"Período: {df['data'].min().date()} até {df['data'].max().date()}")
    print(f"\nPrimeiras linhas:")
    print(df.head())
    print(f"\nÚltimas linhas:")
    print(df.tail())
'''