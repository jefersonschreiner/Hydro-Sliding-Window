import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


# ARMA - Autoregresivo de Médias Móveis
def ajustarArmaWorker(args):
    # Ajusta o ARMA (p,q) e retorna AIC, BIC e parâmetros
    serie, p, q = args
    try:
        ajuste = ARIMA(serie, order=(p, 0, q)).fit()
        return{'p':p,'q':q,'aic':ajuste.aic,'bic':ajuste.bic,'params':ajuste.params.tolist(),'ajuste':ajuste}
    except Exception:
        return None

def gerarArma(ajuste, n_dias:int) -> np.ndarray:
    serie = np.array(ajuste.simulate(n_dias))
    return np.maximum(serie, 0.001)

# GAR - Autoregressivo Generalizado em ordem 1
def ajustarGarWorker(serie):
    #Ajusta GAR(1) e retorna AIC, BIC eo parâmetro S
    
    try:
        # Garante valores positivos para ajuste Gamma
        serie_pos = serie - serie.min() + 1e-6
        # Ajusta distribuição Gamma à serie
        alpha, loc, beta = stats.gamma.fit(serie_pos, floc=0)
        # Coeficiente AR(1) via correlação lag-1
        phi = np.corrcoef(serie_pos[:-1], serie_pos[1:])[0,1]
        # Log-verossimilhanã e critérios de informação
        log_verossimilhanca = np.sum(stats.gamma.logpdf(serie_pos, a=alpha, scale=beta))
        k = 3 # Parâmetros: alpha, beta, phi
        n = len(serie_pos)
        aic = 2*k-2*log_verossimilhanca 
        bic = k*np.log(n)-2*log_verossimilhanca

        return{'aic':aic,'bic':bic,'params':{'alpha':alpha,'beta':beta,'phi':phi,'minimo':serie.min()}}
    
    except Exception:
        return None

def gerarGar(params:dict, n_dias:int) -> np.ndarray:
    alpha, beta, phi, minimo = params['alpha'], params['beta'], params['phi'], params['minimo']
    serie_pos = np.zeros(n_dias)
    serie_pos[0] = stats.gamma.rvs(a=alpha, scale=beta)
    for t in range(1, n_dias):
        ruido = stats.gamma.rvs(a=alpha, scale=beta)
        serie_pos[t] = phi * serie_pos[t-1]+(1-phi)*ruido
    return np.maximum(serie_pos+minimo-1e-6, 0.001)

# PARMA - Autoregressivo de Médias Móveis Periódicos
MESES_INICIO = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
def ajustarParmaWorker(args:tuple) -> dict:
    # Ajusta PARMA para um mês específico e retorna AIC, BIC e parâmetros]

    serie_mes, mes = args
    melhor_aic = np.inf
    melhor_bic = np.inf
    melhor_params = None
    for p in range(3):
        for q in range(3):
            if p == 0 and q == 0:
                continue
            try:
                ajuste = ARIMA(serie_mes, order=(p, 0, q)).fit()
                if ajuste.aic < melhor_aic:
                    melhor_aic = ajuste.aic
                    melhor_bic = ajuste.bic
                    melhor_params = {'p':p, 'q':q, 'params': ajuste.params.tolist(), 'ajuste':ajuste}
            except Exception:
                continue
    
    if melhor_params is None:
        return None
    
    return {'mes':mes, 'aic': melhor_aic, 'bic': melhor_bic, 'params': melhor_params}

def gerarParma(params_mensais: list, n_dias: int, extrairMes) -> np.ndarray:
    serie = np.zeros(n_dias)
    for t in range(n_dias):
        dia_ano = t % 365
        mes = next(m for m in range(11, -1, -1) if dia_ano >= MESES_INICIO[m])
        params_mes = params_mensais[mes]
        if params_mes and t > 0:
            serie[t] = params_mes['ajuste'].simulate(1)[-1]
        else:
            serie[t] = np.random.choice(extrairMes(mes))
    return np.maximum(serie, 0.001)