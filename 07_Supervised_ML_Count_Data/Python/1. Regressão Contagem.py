# Geral
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.discrete
import pyreadr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.metrics import log_loss
from scipy.special import gamma
# Só pra múltiplas
from scipy.stats import pearsonr
import math

#%%

def z_test_PoissonRegression(X, y, plm):
    
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

    params = np.append(plm.intercept_,plm.coef_)
    
    
    X_matrix = np.insert(X, 0, 1, axis = 1)
    W_matrix = np.diag(plm.predict(X))
    C_diag = np.diag(np.linalg.inv(np.matmul(X_matrix.T,np.matmul(W_matrix, X_matrix))))
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



def OtherTest_PoissonRegression(X, y, plm):
    if isinstance(X, pd.DataFrame):
        if len(X.columns)==1:
            X = X.values.reshape(-1,1)
            y = y.valueslm.score(X, y)
        else:
            X = X.values
            y = y.values
    
    k = len(X[0,:] ) # Numero de variáveis
    n = len(X[:, 0]) # Numero de observações
    
    loglikelihood = np.sum(np.ravel(y) * np.log(plm.predict(X)) - plm.predict(X) - np.ravel(np.log(gamma(y + 1))))
    
    X_null = np.ones_like(y).reshape(-1, 1)
    null_model = PoissonRegressor(fit_intercept=True)
    null_model = null_model.fit(X_null, y)
    
    loglikelihood0 = np.sum(np.ravel(y) * np.log(null_model.predict(X_null)) - null_model.predict(X_null) - np.ravel(np.log(gamma(y + 1))))

    chi2_value = -2*(loglikelihood0 - loglikelihood)
    AIC = -2*loglikelihood + 2*(k+1)
    BIC = -2*loglikelihood + (k + 1)*np.log(n)
    pR2_McFadden = chi2_value/(-2*loglikelihood0)
    pR2_Cragg = (1 - (np.exp(loglikelihood0)/np.exp(loglikelihood))**(2/n))/(1 - np.exp(loglikelihood0)**(2/n))
    
    values_array = [loglikelihood, AIC, BIC, pR2_McFadden, pR2_Cragg]
    index = ['Log Likelihood','AIC', 'BIC', 'pseudo-R2 (McFadden)', 'pseudo-R2 (Cragg)']
    
    
    #values_array = [loglikelihood, chi2_value]

    
    df = pd.DataFrame({'Values': values_array}, index = index)
    print('\n######################## OTHER TESTS ########################')
    print(df)
    return df


def OtherTest_NegativeBinomialRegression(X, y, nbm, nbm_results):
    # if isinstance(X, pd.DataFrame):
    #     if len(X.columns)==1:
    #         X = X.values.reshape(-1,1)
    #         y = y.valueslm.score(X, y)
    #     else:
    #         X = X.values
    #         y = y.values
    
    #k = len(X[0,:] ) # Numero de variáveis
    #n = len(X[:, 0]) # Numero de observações
    
    
    
    alpha = nbm_results.params[['alpha']].values[0]
    predict = nbm.predict(nbm_results.params.values)
    loglikelihood = np.sum(np.ravel(y.values)* np.log((alpha*predict)/(1 + alpha* predict)) - np.log(1 + alpha*predict)/alpha + np.log(gamma(np.ravel(y.values) + 1/alpha)) - np.log(gamma(np.ravel(y.values) + 1)) - np.log(gamma(1/alpha)))    
    



    X_null = np.ones_like(y).reshape(-1, 1)
    #X_null['const'] = 0*X_null['const']
    

    null_model = statsmodels.discrete.discrete_model.NegativeBinomial(endog=y, exog=X_null)
    null_model_results = null_model.fit()
    
    alpha_null = null_model_results.params[['alpha']].values[0]
    predict_null = null_model.predict(null_model_results.params.values)
    loglikelihood0 = np.sum(np.ravel(y.values)* np.log((alpha_null*predict_null)/(1 + alpha_null* predict_null)) - np.log(1 + alpha_null*predict_null)/alpha_null + np.log(gamma(np.ravel(y.values) + 1/alpha_null)) - np.log(gamma(np.ravel(y.values) + 1)) - np.log(gamma(1/alpha_null)))    


    chi2_value = -2*(loglikelihood0 - loglikelihood)
    AIC = -2*loglikelihood + 2*(k+1)
    BIC = -2*loglikelihood + (k + 1)*np.log(n)
    pR2_McFadden = chi2_value/(-2*loglikelihood0)
    pR2_Cragg = (1 - (np.exp(loglikelihood0)/np.exp(loglikelihood))**(2/n))/(1 - np.exp(loglikelihood0)**(2/n))
    
    values_array = [loglikelihood, chi2_value, AIC, BIC, pR2_McFadden, pR2_Cragg]
    index = ['Log Likelihood', 'chi2_value', 'AIC', 'BIC', 'pseudo-R2 (McFadden)', 'pseudo-R2 (Cragg)']
    
    
    #values_array = [loglikelihood, chi2_value]

    
    df = pd.DataFrame({'Values': values_array}, index = index)
    print('\n######################## OTHER TESTS ########################')
    print(df)
    return df






def chi2_test_PoissonRegression(X, y, plm):
    
    if isinstance(X, pd.DataFrame):
        if len(X.columns)==1:
            X = X.values.reshape(-1,1)
            y = y.values
        else:
            X = X.values
            y = y.values
    
    k = len(X[0,:] ) # Numero de variáveis
    #n = len(X[:, 0]) # Numero de observações
    
    loglikelihood = np.sum(np.ravel(y) * np.log(plm.predict(X)) - plm.predict(X) - np.ravel(np.log(gamma(y + 1))))
    
    X_null = np.ones_like(y).reshape(-1, 1)
    null_model = PoissonRegressor(fit_intercept=True)
    null_model = null_model.fit(X_null, y)
    
    loglikelihood0 = np.sum(np.ravel(y) * np.log(null_model.predict(X_null)) - null_model.predict(X_null) - np.ravel(np.log(gamma(y + 1))))

    chi2_value = -2*(loglikelihood0 - loglikelihood)

    p_value = 1-stats.chi2.cdf(chi2_value, k)
    df = pd.DataFrame()
    df['chi2-statistic'] = [chi2_value]
    df['p-value'] =  [p_value]
    
    print('\n############## CHI SQUARED TEST ##############')
    print(df)
    return df



def t_test_LinearRegression(X, y, lm):
     
    index = ['(Intercept)'] + list(X.columns.values)
    
    if isinstance(X, pd.DataFrame):
        if len(X.columns)==1:
            X = X.values.reshape(-1,1)
            y = y.values
        else:
            X = X.values
            y = y.values
    
    k = len(X[0,:] ) +1 # Graus de liberade de regressão (graus de inclinação + intercepto)
    n = len(X[:, 0]) # Numero de observações

    params = np.append(lm.intercept_,lm.coef_)
    predictions = lm.predict(X)
    
    X_matrix = np.insert(X, 0, 1, axis = 1)
    MSE = sum((y-predictions)**2)/(n - k )
    C_matrix = MSE*(np.linalg.inv(np.dot(X_matrix.T,X_matrix)))
    C_diag = np.diag(C_matrix)
    std_error = np.sqrt(C_diag)
    t_values = params/std_error


    p_values =[2*(1-stats.t.cdf(np.abs(i),(n - k))) for i in t_values]
    
    df = pd.DataFrame({'Coefficients': params, "Standard Errors": std_error,
                          "t-statistics": t_values, 'p-values': p_values}, index = index)

    print('\n######################## t TEST ########################')
    print(df)
    return df

def t_test_LinearRegression_NoIntercept(X, y, lm):
     
    index =  list(X.columns.values)
    
    if isinstance(X, pd.DataFrame):
        if len(X.columns)==1:
            X = X.values.reshape(-1,1)
            y = y.values
        else:
            X = X.values
            y = y.values
    
    k = len(X[0,:] ) # Graus de liberade de regressão (graus de inclinação + intercepto)
    n = len(X[:, 0]) # Numero de observações

    params = lm.coef_[0]
    predictions = lm.predict(X)
    
    X_matrix = X
    MSE = sum((y-predictions)**2)/(n - k )
    C_matrix = MSE*(np.linalg.inv(np.dot(X_matrix.T,X_matrix)))
    C_diag = np.diag(C_matrix)
    std_error = np.sqrt(C_diag)
    t_values = params/std_error


    p_values =[2*(1-stats.t.cdf(np.abs(i),(n - k))) for i in t_values]
    
    df = pd.DataFrame({'Coefficients': params, "Standard Errors": std_error,
                          "t-statistics": t_values, 'p-values': p_values}, index = index)

    print('\n######################## t TEST ########################')
    print(df)
    return df

#%%
# Carregamento da base de dados (corruption)
#result = pyreadr.read_r('/home/hiro/Documents/3. MBA/2. Aulas/06. Supervised Machine Learning: Modelos Logísticos Binários e Multinomiais/Python/Atrasado.RData') 
result = pyreadr.read_r('corruption.RData') 
print(result.keys()) # Checando o nome do DataFrame

#%%
# Carregando o DataFrame
df = result["corruption"]

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
#Diagnóstico preliminar para observação de eventual igualdade entre a média e
#a variância da variável dependente 'violations'

# Calculate the mean and variance
mean = np.mean(df[['violations']].values)
variance = np.var(df[['violations']].values)

# Create a Pandas DataFrame
print(pd.DataFrame({'mean': [mean], 'variance': [variance]}))

#%%
# Arrumando o Dataframe
df['post'] = np.where(df['post'].values == 'yes', 1, 0)

#%%
################################################################################
#                        ESTIMAÇÃO DO MODELO POISSON                           #
################################################################################
#Estimação do modelo
X = df[['staff', 'post', 'corruption']]
y = df[['violations']]

plm = PoissonRegressor(alpha = 0, solver='newton-cholesky')  #Poisson Linear Model
plm = plm.fit(X.values, np.ravel(y.values))

params = np.append(plm.intercept_, plm.coef_)
index = ['(Intercept)'] + list(X.columns.values)

df_params = pd.DataFrame({'Coefficients': params}, index = index)
print(df_params)
#%%
z_test_PoissonRegression(X, y, plm)

OtherTest_PoissonRegression(X, y, plm)

chi2_test_PoissonRegression(X, y, plm)


#%%
################################################################################
#            TESTE DE SUPERDISPERSÃO DE CAMERON E TRIVEDI (1990)               #
################################################################################
#CAMERON, A. C.; TRIVEDI, P. K. Regression-based tests for overdispersion in
#the Poisson model. Journal of Econometrics, v. 46, n. 3, p. 347-364, 1990.

#1º Passo: estimar um modelo Poisson;
#2º Passo: criar uma nova variável (Y*) utilizando os fitted values do modelo
#Poisson estimado anteriormente;
#3º Passo: estimar um modelo auxiliar OLS, com a variável Y* como variável
#dependente, os fitted values do modelo Poisson como única variável preditora e 
#sem o intercepto;
#4º Passo: Observar a significância do parâmetro beta.

df['lambda_poisson'] = plm.predict(X.values)
df['ystar'] = (((np.ravel(y.values) -plm.predict(X.values))**2)- np.ravel(y.values)) / plm.predict(X.values)


X_ct= df[['lambda_poisson']]
y_ct = df[['ystar']]

lm = LinearRegression(fit_intercept=False)
lm = lm.fit(X_ct, y_ct)


t_test_LinearRegression_NoIntercept(X_ct, y_ct, lm)

#Caso o p-value do parâmetro do lambda_poisson seja maior que 0.05,
#verifica-se a existência de equidispersão nos dados.
#Caso contrário, diagnostica-se a existência de superdispersão nos dados, fato
#que favorecerá a estimação de um modelo binomial negativo.

#%%
#Apenas para fins didáticos, caso considerássemos a estimação Poisson como a
#mais adequada, qual seria a quantidade média esperada de violações de trânsito
#para um país cujo corpo diplomático fosse composto por 23 membros, considerando
#o período anterior à vigência da lei e cujo índice de corrupção seja
#igual a 0.5?
print(plm.predict(np.array([23, 0, 0.5]).reshape(1, -1)))

#Qual seria a quantidade média esperada de violações de trânsito para o mesmo
#país, porém agora considerando a vigência da lei?
print(plm.predict(np.array([23, 1, 0.5]).reshape(1, -1)))

#%%
################################################################################
#                   ESTIMAÇÃO DO MODELO BINOMIAL NEGATIVO                      #
################################################################################
#Estimação do modelo
# X = df[['staff', 'post', 'corruption']]
# y = df[['violations']]
# X_with_constant = sm.add_constant(X, prepend= True)

# nbm = sm.GLM(endog=y, exog=X_constant,
#                family=sm.families.NegativeBinomial(alpha = 2.096338)).fit() # Negative Binomial Model

# print(nbm.summary())
# # params = np.append(plm.intercept_, plm.coef_)
# # index = ['(Intercept)'] + list(X.columns.values)

# # df_params = pd.DataFrame({'Coefficients': params}, index = index)
# # print(df_params)

#%%
X = df[['staff', 'post', 'corruption']]
y = df[['violations']]
X_with_constant = sm.tools.tools.add_constant(X, prepend= True)

nbm = statsmodels.discrete.discrete_model.NegativeBinomial(endog=y, exog=X_with_constant)
nbm_results = nbm.fit()
nbm_results.summary()

#%%
loglike = nbm.loglike(nbm_results.params.values)

alpha = nbm_results.params[['alpha']].values[0]
predict = nbm.predict(nbm_results.params.values)
loglike_analytical = np.sum(np.ravel(y.values)* np.log((alpha*predict)/(1 + alpha* predict)) - np.log(1 + alpha*predict)/alpha + np.log(gamma(np.ravel(y.values) + 1/alpha)) - np.log(gamma(np.ravel(y.values) + 1)) - np.log(gamma(1/alpha)))
print(loglike)
print(loglike_analytical)

#%% 

#Qual seria a quantidade média esperada de violações de trânsito para um país
#cujo corpo diplomático seja composto por 23 membros, considerando o período
#anterior à vigência da lei e cujo índice de corrupção seja igual 0.5?

print(nbm.predict(nbm_results.params, exog = np.array([[1, 23,0,0.5]])))
# tem que adicionar o 1 antes, pois para gerar o modelo usamos a matriz de design

#%%

OtherTest_NegativeBinomialRegression(X, y, nbm, nbm_results)
#%%
X = df[['staff', 'post', 'corruption']]
y = df[['violations']]

index = ['(Intercept)'] + list(X.columns.values) 
#index = ['(Intercept)'] + list(X.columns.values) + ['(Alpha)']

if isinstance(X, pd.DataFrame):
    if len(X.columns)==1:
        X = X.values.reshape(-1,1)
        y = y.values
    else:
        X = X.values
        y = y.values

k = len(X[0,:] ) +1 # Graus de liberade de regressão (graus de inclinação + intercepto)
n = len(X[:, 0]) # Numero de observações


#params = nbm_results.params.values
params = nbm_results.params.drop(['alpha']).values
alpha = nbm_results.params[['alpha']].values[0]

#MSE = np.sum((np.ravel(y) - np.exp(nbm.fittedvalues.values))**2)/(n)
X_matrix = np.insert(X, 0, 1, axis = 1)
W_matrix = np.diag(1/((np.exp(nbm_results.fittedvalues.values)) +(1/    alpha)* (np.exp(nbm_results.fittedvalues.values))**2))
C_matrix = np.linalg.inv(np.matmul(X_matrix.T,np.matmul(W_matrix, X_matrix)))
C_diag = np.diag(C_matrix)
std_error = np.sqrt(C_diag)
print(C_matrix)
z_values = params/std_error


p_values =[2*(1-stats.norm.cdf(abs(i))) for i in z_values]

#df = pd.DataFrame({'Coefficients': params, "Standard Errors": sd_b,
#                      "t-statistics": ts_b, 'p-values': p_values}, index = index)
df0 = pd.DataFrame({'Coefficients': params, "Standard Errors": std_error,
                      "z-statistics": z_values, 'p-values': p_values}, index = index)

print('\n######################## z TEST ########################')
print(df0)
