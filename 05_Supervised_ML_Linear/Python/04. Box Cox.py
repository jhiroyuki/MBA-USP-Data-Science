# Geral
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadr

from sklearn.linear_model import LinearRegression
from scipy import stats

# Só pra múltiplas
from scipy.stats import pearsonr


#%%

def F_test_LinearRegression(X, y, lm):
    
    if isinstance(X, pd.DataFrame):
        if len(X.columns)==1:
            X = X.values.reshape(-1,1)
            y = y.values
        else:
            X = X.values
            y = y.values
        
    Rsqr = lm.score(X, y)
    k = len(X[0,:] ) +1 # Graus de liberade de regressão (graus de inclinação + intercepto)
    n = len(X[:, 0]) # Numero de observações

    F_value = (Rsqr/(k -1))/((1- Rsqr)/(n - k )) 
    p_value = 1-stats.f.cdf(F_value, k-1, n-k)
    F_df = pd.DataFrame()
    F_df['F-statistic'] = [F_value]
    F_df['p-value'] =  [p_value]
    
    print('\n############## F TEST ##############')
    print(F_df)
    
def t_test_LinearRegression(X, y, lm):
    
    index = ['(Intercept)'] + list(X.columns.values)
    
    if isinstance(X, pd.DataFrame):
        if len(X.columns)==1:
            X = X.values.reshape(-1,1)
            y = y.values
        else:
            X = X.values
            y = y.values
        
    
    params = np.append(lm.intercept_,lm.coef_)
    predictions = lm.predict(X)

    newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
    MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))

    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b

    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX) - len(newX.iloc[0, :])))) for i in ts_b]
    
    myDF3 = pd.DataFrame({'Coefficients': params, "Standard Errors": sd_b,
                          "t-statistics": ts_b, 'p-values': p_values}, index = index)

    print('\n######################## t TEST ########################')
    print(myDF3)
    
    
def pearson(df1, df2):

    coeffmat = np.zeros((df1.shape[1], df2.shape[1]))
    pvalmat = np.zeros((df1.shape[1], df2.shape[1]))
    
    for i in range(df1.shape[1]):    
        for j in range(df2.shape[1]):        
            corrtest = pearsonr(df1[df1.columns[i]], df2[df2.columns[j]])  
    
            coeffmat[i,j] = corrtest[0]
            pvalmat[i,j] = corrtest[1]

    corr_coef = pd.DataFrame(coeffmat, columns=df2.columns, index=df1.columns)
    corr_sig = pd.DataFrame(pvalmat, columns=df2.columns, index=df1.columns)
    return corr_coef, corr_sig



def confint(X, y, lm, significance =0.05):
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
    
    std_error = np.sqrt(np.sum((y - lm.predict(X))**2)/(n- k))
    X_matrix = np.insert(X, 0, 1, axis = 1)
    C_diag = np.diag(std_error**2 * np.linalg.inv(np.matmul(X_matrix.T, X_matrix)))
    t_critical = stats.t.ppf(1- significance/2, n - k) # Valor em x da distriuição em t que corresponde ao valor de significancia
    
    params_correction = t_critical*np.sqrt(C_diag)
    params_low = params - params_correction
    params_high = params + params_correction

    df = pd.DataFrame({str(significance/2*100) +'%': params_low,str((1-significance/2)*100) +'%': params_high }, index= index)
    return df

#%%
def ppoints(n, a):   
    try:
        n = float(len(n))
    except TypeError:
        n = float(n)
    return (np.arange(n) + 1 - a)/(n + 1 - 2*a)

def removerDiagonal(x):
    x_sem_diagonais = np.ndarray.flatten(x)
    x_sem_diagonais = np.delete(x_sem_diagonais, range(0, len(x_sem_diagonais), len(x) + 1), 0)
    x_sem_diagonais = x_sem_diagonais.reshape(len(x), len(x) - 1)
    return x_sem_diagonais     

def shapiroFrancia(x):
  x = np.sort(x)
  n = len(x)
  y = stats.norm.ppf(ppoints(n,a = 3/8))
  W = np.corrcoef(x, y)**2
  u = np.log(n)
  v = np.log(u)
  mu = -1.2725 + 1.0521 * (v - u)
  sig = 1.0308 - 0.26758 * (v + 2/u)
  z = (np.log(1 - W) - mu)/sig
  pval = stats.norm.sf(z)
  
  SF_df = pd.DataFrame()
  SF_df['W'] = [removerDiagonal(W)[0][0]]
  SF_df['p-value'] =  [removerDiagonal(pval)[0][0]]
  
  print('\n######## SHAPIRO FRANCIA ##########')
  print(SF_df)
#%%
################################################################################
#           REGRESSÃO NÃO LINEAR SIMPLES E TRANSFORMAÇÃO DE BOX-COX            #
#                 EXEMPLO 04 - CARREGAMENTO DA BASE DE DADOS                   #
################################################################################
#%%
# Carregamento da base de dados
result = pyreadr.read_r('/home/hiro/Documents/3. MBA/2. Aulas/05. Supervised Machine Learning: Análise de Regressão Simples e Múltipla/Python/bebes.RData') 
print(result.keys()) # Checando o nome do DataFrame
#%%
# Carregando o DataFrame
df = result["bebes"]
#%%
# Estatísticas descritivas
summary = df.describe(include='all')
print('############ ESTATÍSTICAS DESCRITIVAS ############')
print(summary)

#%%
# Gráfico de dispersão
X = df[['idade']]
y = df[['comprimento']]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X, 
           y,
           color = 'black',
           alpha = 0.6)


ax.set_xlabel('Idade em semanas')
ax.set_ylabel('Comprimento em cm')
ax.grid(alpha = 0.3)

plt.show()
#%%
#Estimação do modelo OLS linear
X = df[['idade']]
y = df[['comprimento']]

lm = LinearRegression()
lm = lm.fit(X.values.reshape(-1,1), y.values)


F_test_LinearRegression(X, y, lm)

t_test_LinearRegression(X, y, lm)

#%%
# Teste de verificação da aderência dos resíduos à normalidade
# Shapiro-Francia
# A implementação feita no preâmbulo é similar aquela que é feita no R
residuals = y - lm.predict(X.values.reshape(-1,1))
residuals = residuals.values.reshape(1,-1)[0]
shapiroFrancia(residuals)

#%%
# Histograma dos resíduos do modelo OLS linear

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.hist(residuals, bins = 30, ec = 'k', density = True)
x_ = np.linspace(min(residuals), max(residuals), 100)
ax.plot(x_, stats.norm.pdf(x_, np.mean(residuals), np.std(residuals)), lw = 8)
ax.set_xlabel('Resíduos')
ax.set_ylabel('Frequência')
ax.grid(alpha = 0.3)

plt.show()

#%%
# Transformação de Box-Cox
y_boxcox, lambda_bc = stats.boxcox(y.values.reshape(1,-1)[0])
print('\nEstimated transformation parameter ')
print(lambda_bc)

#Inserindo o lambda de Box-Cox na base de dados para a estimação de um novo modelo
df['bc_comprimento'] = y_boxcox

#%%
#Estimando um novo modelo OLS com variável dependente transformada por Box-Cox
X = df[['idade']]
y = df[['bc_comprimento']]

lm_bc = LinearRegression()
lm_bc = lm_bc.fit(X.values.reshape(-1,1), y.values)

F_test_LinearRegression(X, y, lm)
t_test_LinearRegression(X, y, lm)

#%%
#Teste de Shapiro-Francia para os resíduos do modelo_bc
residuals_bc = y - lm.predict(X.values.reshape(-1,1))
residuals_bc = residuals_bc.values.reshape(1,-1)[0]
shapiroFrancia(residuals_bc)

#%%
#Histograma dos resíduos do modelo_bc
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)

ax.hist(residuals_bc, bins = 30, ec = 'k', density = True)
x_ = np.linspace(min(residuals_bc), max(residuals_bc), 100)
ax.plot(x_, stats.norm.pdf(x_, np.mean(residuals_bc), np.std(residuals_bc)), lw = 8)
ax.set_xlabel('Resíduos')
ax.set_ylabel('Frequência')
ax.grid(alpha = 0.3)

plt.show()

#%%
#Salvando os fitted values dos dois modelos (modelo_linear e modelo_bc) no
#dataset 'bebes'
df['yhat_linear'] = lm.predict(X.values.reshape(-1,1))
df['yhat_modelo_bc'] = (lm_bc.predict(X.values.reshape(-1,1))*lambda_bc + 1)**(1/lambda_bc)

#%%
# Plotando os dados e os fits
X = df[['idade']]
y = df[['comprimento']]
x_ = np.linspace(min(X.values.reshape(1,-1)[0]), max(X.values.reshape(1,-1)[0]), 100)
y_linear = lm.predict(x_.reshape(-1,1))
y_bc = (lm_bc.predict(x_.reshape(-1,1))*lambda_bc + 1)**(1/lambda_bc)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)


ax.plot(x_, y_linear, lw = 3, label ='Linear')
ax.plot(x_, y_bc, lw = 3, label='Box-Cox')
ax.scatter(X, 
           y,
           color = 'black',
           alpha = 0.6)

ax.set_xlabel('Idade em semanas')
ax.set_ylabel('Comprimento em cm')
ax.grid(alpha = 0.3)
ax.legend()
plt.show()