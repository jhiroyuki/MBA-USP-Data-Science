# Geral
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadr
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.metrics import log_loss

# Só pra múltiplas
from scipy.stats import pearsonr
#%%

def confint_logistical(X, y, lm, significance =0.05):
    index = ['(Intercept)'] + list(X.columns.values)
    
    if isinstance(X, pd.DataFrame):
        if len(X.columns)==1:
            X = X.values.reshape(-1,1)
            y = y.values
        else:
            X = X.values
            y = y.values
    
    #k = len(X[0,:] ) +1 # Graus de liberade de regressão (graus de inclinação + intercepto)
    #n = len(X[:, 0]) # Numero de observações

    params = np.append(lm.intercept_,lm.coef_)
    
    X_matrix = np.insert(X, 0, 1, axis = 1)
    V_matrix = np.diag(lm.predict_proba(X)[:, 0]*lm.predict_proba(X)[:, 1])
    C_diag = np.diag(np.linalg.inv(np.matmul(X_matrix.T,np.matmul(V_matrix, X_matrix))))
    std_error = np.sqrt(C_diag)
    
    
    z_critical = np.abs(stats.norm.ppf(significance/2)) # Valor em x da distriuição normal que corresponde ao valor de significancia
    
    params_correction = z_critical*std_error
    params_low = params - params_correction
    params_high = params + params_correction


    df = pd.DataFrame({str(significance/2*100) +'%': params_low,str((1-significance/2)*100) +'%': params_high }, index= index)
    return df


def chi2_test_LogisticRegression(X, y, blm):
    
    if isinstance(X, pd.DataFrame):
        if len(X.columns)==1:
            X = X.values.reshape(-1,1)
            y = y.values
        else:
            X = X.values
            y = y.values
    
    k = len(X[0,:] ) # Numero de variáveis
    #n = len(X[:, 0]) # Numero de observações
    
    loglikelihood = np.sum(np.ravel(y) * np.log(blm.predict_proba(X)[:, 1]) + (1- np.ravel(y))* np.log(blm.predict_proba(X)[:, 0]))
    null_coef = np.mean(y)
    loglikelihood0 = np.sum(np.ravel(y) * np.log(null_coef) + (1- np.ravel(y))* np.log(1-null_coef))

    chi2_value = -2*(loglikelihood0 - loglikelihood)

    p_value = 1-stats.chi2.cdf(chi2_value, k)
    df = pd.DataFrame()
    df['chi2-statistic'] = [chi2_value]
    df['p-value'] =  [p_value]
    
    print('\n############## CHI SQUARED TEST ##############')
    print(df)
    return df


    
    
def z_test_LogisticRegression(X, y, lm):
    
    index = ['(Intercept)'] + list(X.columns.values)
    
    if isinstance(X, pd.DataFrame):
        if len(X.columns)==1:
            X = X.values.reshape(-1,1)
            y = y.values
        else:
            X = X.values
            y = y.values
    
    #k = len(X[0,:] ) +1 # Graus de liberade de regressão (graus de inclinação + intercepto)
    #n = len(X[:, 0]) # Numero de observações

    params = np.append(lm.intercept_,lm.coef_)
    
    
    X_matrix = np.insert(X, 0, 1, axis = 1)
    V_matrix = np.diag(lm.predict_proba(X)[:, 0]*lm.predict_proba(X)[:, 1])
    C_diag = np.diag(np.linalg.inv(np.matmul(X_matrix.T,np.matmul(V_matrix, X_matrix))))
    std_error = np.sqrt(C_diag)
    
    z_values = params/std_error


    p_values =[2*(1-stats.norm.cdf(abs(i))) for i in z_values]
    
    #df = pd.DataFrame({'Coefficients': params, "Standard Errors": sd_b,
    #                      "t-statistics": ts_b, 'p-values': p_values}, index = index)
    df = pd.DataFrame({'Coefficients': params, "Standard Errors": std_error,
                          "z-statistics": z_values, 'p-values': p_values}, index = index)

    print('\n######################## z TEST ########################')
    print(df)
    return df



def OtherTest_LogisticRegression(X, y, blm):
    if isinstance(X, pd.DataFrame):
        if len(X.columns)==1:
            X = X.values.reshape(-1,1)
            y = y.values.score(X, y)
        else:
            X = X.values
            y = y.values
    
    k = len(X[0,:] ) # Numero de variáveis
    n = len(X[:, 0]) # Numero de observações
    
    loglikelihood = np.sum(np.ravel(y) * np.log(blm.predict_proba(X)[:, 1]) + (1- np.ravel(y))* np.log(blm.predict_proba(X)[:, 0]))
    null_coef = np.mean(y)
    loglikelihood0 = np.sum(np.ravel(y) * np.log(null_coef) + (1- np.ravel(y))* np.log(1-null_coef))

    chi2_value = -2*(loglikelihood0 - loglikelihood)
    AIC = -2*loglikelihood + 2*(k+1)
    BIC = -2*loglikelihood + (k + 1)*np.log(n)
    pR2_McFadden = chi2_value/(-2*loglikelihood0)
    pR2_Cragg = (1 - (np.exp(loglikelihood0)/np.exp(loglikelihood))**(2/n))/(1 - np.exp(loglikelihood0)**(2/n))
    
    values_array = [loglikelihood, AIC, BIC, pR2_McFadden, pR2_Cragg]
    index = ['Log Likelihood','AIC', 'BIC', 'pseudo-R2 (McFadden)', 'pseudo-R2 (Cragg)']
    
    df = pd.DataFrame({'Values': values_array}, index = index)
    print('\n######################## OTHER TESTS ########################')
    print(df)
    return df


def stepwise_LogisticRegression(X, y, significancia = 0.05):
        
    while True:
        
        blm = LogisticRegression(penalty='none',  solver='newton-cg' ) #binary logistic model
        blm = blm.fit(X.values,y.values)
        
    
        df_chi2 = chi2_test_LogisticRegression(X, y, blm)
        df_z = z_test_LogisticRegression(X, y, blm)
    
        p_values = df_z[['p-values']].drop(['(Intercept)'])
        p_values_delete = p_values.loc[p_values['p-values'] > 0.05]
        
        if p_values_delete.empty == True:
            break
        
        X = X.drop(p_values_delete.index.values, axis = 1)
    return blm, X


#%%
# Carregamento da base de dados (challenger)
#result = pyreadr.read_r('/home/hiro/Documents/3. MBA/2. Aulas/06. Supervised Machine Learning: Modelos Logísticos Binários e Multinomiais/Python/Atrasado.RData') 
result = pyreadr.read_r('challenger.RData') 
print(result.keys()) # Checando o nome do DataFrame

#%%
# Carregando o DataFrame
df = result["challenger"]

#%%
# Visualização da base de dados
print('################### DATAFRAME ####################')
print(df)

# Fazendo parecido com o "kable" do R (só funciona no Jupyter)
df.style

#%%
# Estatísticas descritivas
summary = df.describe(include='all')
print('############ ESTATÍSTICAS DESCRITIVAS ############')
print(summary)



#%%
#Não há uma variável binária para servir como uma variável dependente, certo?
#Então vamos criá-la considerando a ocorrência de desgastes de peças como a
#ocorrência de um evento que chamaremos de 'falha':
df['falha']  = np.where(df[['desgaste']].values >=1, 1, 0)    

#%%
# Estimação de modelo logístico binário
X = df[['temperatura', 'pressão']]
y = df[['falha']]

blm = LogisticRegression(penalty='none',  solver='newton-cg' ) #binary logistic model
blm = blm.fit(X.values,y.values)

df_chi2 = chi2_test_LogisticRegression(X, y, blm)
df_z = z_test_LogisticRegression(X, y, blm)
df_other = OtherTest_LogisticRegression(X, y, blm)

#%%

blm_step, X_step = stepwise_LogisticRegression(X, y, significancia=0.05)

#%%

#Fazendo predições para o modelo step_challenger:
#Exemplo 1: qual a probabilidade média de falha a 70ºF (~21ºC)?
print(blm_step.predict_proba([[70]])[:, 1])

#Exemplo 2: qual a probabilidade média de falha a 77ºF (25ºC)?
print(blm_step.predict_proba([[77]])[:, 1])

#Exemplo 3: qual a probabilidade média de falha a 34ºF (~1ºC) - manhã do lançamento?
print(blm_step.predict_proba([[34]])[:, 1])
