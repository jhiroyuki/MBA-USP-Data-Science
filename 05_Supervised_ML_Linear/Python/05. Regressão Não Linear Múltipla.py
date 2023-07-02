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
    return myDF3
    
    
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
  
def stepwise(X, y, significancia = 0.05):
        
    while True:
        lm = LinearRegression()
        lm = lm.fit(X.values, y.values)
    
        F_test_LinearRegression(X, y, lm)
        df_t = t_test_LinearRegression(X, y, lm)
    
        p_values = df_t[['p-values']].drop(['(Intercept)'])
        p_values_delete = p_values.loc[p_values['p-values'] > 0.05]
        
        if p_values_delete.empty == True:
            break
        
        X = X.drop(p_values_delete.index.values, axis = 1)
    return lm, X

#%%
# Carregamento da base de dados
result = pyreadr.read_r('/home/hiro/Documents/3. MBA/2. Aulas/05. Supervised Machine Learning: Análise de Regressão Simples e Múltipla/Python/empresas.RData') 
print(result.keys()) # Checando o nome do DataFrame
#%%
# Carregando o DataFrame
df = result["empresas"]
#%%
# Estatísticas descritivas
summary = df.describe(include='all')
print('############ ESTATÍSTICAS DESCRITIVAS ############')
print(summary)

#%%
# Estudo das correlações
corr_coef, corr_sig = pearson(df.iloc[:,1:], df.iloc[:,1:])
print('##### MATRIZ DE CORRELAÇÃO ######') #OK
print(corr_coef)


# Elaboração de um mapa de calor das correlações de Pearson entre as variáveis
ax = sns.heatmap(
    corr_coef, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot=True, 
    fmt='.2f')

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

#%%
# Estimando a Regressão Múltipla
X = df[['disclosure', 'endividamento', 'ativos', 'liquidez']]
y = df[['retorno']]
lm = LinearRegression()
lm = lm.fit(X.values, y.values)

F_test_LinearRegression(X, y, lm)
t_test_LinearRegression(X, y, lm)

# Note que o parâmetro da variável 'endividamento' não é estatisticamente
# significante ao nível de significância de 5% (nível de confiança de 95%)

#%%
# Primeiro indicio de que uma variável não ser estatisticamente significante num modelo de regressão múltipla
# é que esta variável não apresenta correlação estatisticamente significante com a Y. Olha a matriz de correlações e a matriz de pvalues das correlações

ax = sns.heatmap(
    corr_sig, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot=True, 
    fmt='.2f')

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

# Também, ao fazer um modelo de regressão da y só com essa variável, ela não é estat. signf.
X_aux = df[['endividamento']]
lm_aux = LinearRegression()
lm_aux = lm_aux.fit(X_aux.values.reshape(-1,1), y.values)

F_test_LinearRegression(X_aux, y, lm_aux)
t_test_LinearRegression(X_aux, y, lm_aux)

#%% 
# Procedimento Stepwise

X = df[['disclosure', 'endividamento', 'ativos', 'liquidez']]
y = df[['retorno']]

lm_step, X_step = stepwise(X, y, significancia=0.05)

#%%
# Intervalo de confiança
confint(X_step, y, lm_step, significance =0.05)

#%%
# Shapiro Francia
residuals = y - lm_step.predict(X_step)
residuals = residuals.values.reshape(1,-1)[0]
shapiroFrancia(residuals)

#%%
#Histograma dos resíduos 
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
# Transformação de Box Cox
y_boxcox, lambda_bc = stats.boxcox(y.values.reshape(1,-1)[0])
print('\nEstimated transformation parameter ')
print(lambda_bc)

#Inserindo o lambda de Box-Cox na base de dados para a estimação de um novo modelo
df['bcretorno'] = y_boxcox

#%%
#Estimando um novo modelo múltiplo com variável dependente transformada por Box-Cox
X = df[['disclosure', 'endividamento', 'ativos', 'liquidez']]
y_bc = df[['bcretorno']]

lm_bc = LinearRegression()
lm_bc = lm_bc.fit(X.values, y_bc.values)

F_test_LinearRegression(X, y_bc, lm_bc)
t_test_LinearRegression(X, y_bc, lm_bc)


#%%
# Stepwise
X = df[['disclosure', 'endividamento', 'ativos', 'liquidez']]
y_bc = df[['bcretorno']]

lm_stepbc, X_stepbc = stepwise(X, y_bc, significancia=0.05)

# Shapiro Francia
residuals = y_bc - lm_stepbc.predict(X_stepbc)
residuals = residuals.values.reshape(1,-1)[0]
shapiroFrancia(residuals)


#Histograma dos resíduos 
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)

ax.hist(residuals, bins = 30, ec = 'k', density = True)
x_ = np.linspace(min(residuals), max(residuals), 100)
ax.plot(x_, stats.norm.pdf(x_, np.mean(residuals), np.std(residuals)), lw = 8)
ax.set_xlabel('Resíduos')
ax.set_ylabel('Frequência')
ax.grid(alpha = 0.3)

plt.show()

# Intervalo de confiança
confint(X_stepbc, y, lm_stepbc, significance =0.05)

#%%
#Salvando os fitted values dos modelos
df['yhat_step_empresas'] = lm_step.predict(X_step)
df['yhat_step_modelo_bc'] = (lm_stepbc.predict(X_stepbc)*lambda_bc + 1)**(1/lambda_bc)

#%%
# Plotando (fitted values) X valores reais
X = df[['retorno']].values.reshape(1,-1)[0]
y_step = df[['yhat_step_empresas']].values.reshape(1,-1)[0]
y_stepbc = df[['yhat_step_modelo_bc']].values.reshape(1,-1)[0]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)

x_ = np.linspace(min([min(y_step),min(y_stepbc)]), max([max(y_step),max(y_stepbc)]), 100)
x_ = np.linspace(min(X), max(X), 100)
ax.scatter(X, y_step, alpha = 0.6)
ax.scatter(X, y_stepbc, alpha = 0.6)
ax.plot(x_, x_, lw =4, color = 'grey')

ax.set_xlabel('Retorno')
ax.set_ylabel('Fitted values')
ax.grid(alpha = 0.3)
ax.legend()
plt.show()
