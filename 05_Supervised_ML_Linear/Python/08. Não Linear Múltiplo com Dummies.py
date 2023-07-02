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

from scipy.stats import chi2
from statsmodels.formula.api import ols
import statsmodels.api as sm


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
  
def stepwise(X, y, significancia=0.05):

    while True:

        lm = LinearRegression()
        lm = lm.fit(X.values, y.values)

        F_test_LinearRegression(X, y, lm)
        df_t = t_test_LinearRegression(X, y, lm)

        p_values = df_t[['p-values']].drop(['(Intercept)'])
        p_values_delete = p_values.loc[p_values['p-values'] > 0.05]

        if p_values_delete.empty == True:
            break

        #p_values_delete = [max(p_values_delete)]
        print(p_values_delete)

        X = X.drop(p_values_delete.idxmax(), axis=1)

    return lm, X


def breusch_pagan_test(X, y, lm):

    X = X.values
    y = y.values
    n_samples = len(y)
    resid = (y - lm.predict(X))
    up = n_samples*resid**2/sum(resid**2)
    
    
    df_aux = pd.DataFrame({'yhat': lm.predict(X).reshape(1,-1)[0], 'up': up.reshape(1,-1)[0]})
    model = ols('up ~ yhat', data=df_aux).fit()
    aov_table = sm.stats.anova_lm(model, typ=2)
    sqreg = aov_table.loc['yhat']['sum_sq']
    chisq = sqreg/2
    p_value = chi2.sf(chisq, 1)
    
    df_results = pd.DataFrame({'Values': [chisq, p_value]}, index=['Chi2', 'p-value'])
    print(df_results)

#%%
# Carregamento da base de dados
result = pyreadr.read_r('/home/hiro/Documents/3. MBA/2. Aulas/05. Supervised Machine Learning: Análise de Regressão Simples e Múltipla/Python/planosaude.RData') 
print(result.keys()) # Checando o nome do DataFrame
#%%
# Carregando o DataFrame
df = result["planosaude"]
#%%
# Estatísticas descritivas
summary = df.describe(include='all')
print('############ ESTATÍSTICAS DESCRITIVAS ############')
print(summary)

#%%
# Estudo das correlações
corr_coef, corr_sig = pearson(df.iloc[:,1:-1], df.iloc[:,1:-1])
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
# PROCEDIMENTO N-1 DUMMIES

# Dummizando a variável regiao. O código abaixo, automaticamente, fará: 
# dados;

# a) o estabelecimento de dummies que representarão cada uma das regiões da base de 
dummies = pd.get_dummies(df['plano'])

# b) estabelecerá como categoria de referência a dummy mais frequente.
dummies = dummies.drop(columns = np.sum(dummies).idxmax())

# c) incluir no dataframe
df_dummies = df.join(dummies)
#%%
# ESTIMAÇÃO DO MODELO DE REGRESSÃO

# Modelagem com todas as variáveis
# Estimando a Regressão Múltipla
X = df_dummies[[ 'idade', 'dcron', 'renda','esmeralda', 'ouro']]
y = df_dummies[['despmed']]
lm = LinearRegression()
lm = lm.fit(X.values, y.values)

F_test_LinearRegression(X, y, lm)
t_test_LinearRegression(X, y, lm)

# %%

# Stepwise
X = df_dummies[[ 'idade', 'dcron', 'renda','esmeralda', 'ouro']]
y = df_dummies[['despmed']]

lm_step, X_step = stepwise(X, y, significancia=0.05)

#%%
# Shapiro Francia
residuals = y - lm_step.predict(X_step)
residuals = residuals.values.reshape(1,-1)[0]   
shapiroFrancia(residuals)

#%%
#Histograma dos resíduos 
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)

ax.hist(residuals, bins = 15, ec = 'k', density = True)
x_ = np.linspace(min(residuals), max(residuals), 100)
ax.plot(x_, stats.norm.pdf(x_, np.mean(residuals), np.std(residuals)), lw = 8)
ax.set_xlabel('Resíduos')
ax.set_ylabel('Frequência')
ax.grid(alpha = 0.3)

plt.show()

#%%

breusch_pagan_test(X_step, y, lm_step)

#%%
#Salvando fitted values e residuals no dataframe
df_dummies['fitted_step'] = lm_step.predict(X_step.values)
df_dummies['residuos_step'] = y - lm_step.predict(X_step.values)

#Gráfico Fitted Values x Residuos
X = df_dummies['fitted_step']
y = df_dummies['residuos_step']

fig = plt.figure(figsize=(12, 8))

ax = fig.add_subplot(1, 1, 1)
ax.scatter(X.values, y.values)


ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')

ax.grid(alpha = 0.3)

plt.show()

#%%
# Transformação de Box Cox
y = df_dummies[['despmed']]
y_boxcox, lambda_bc = stats.boxcox(y.values.reshape(1,-1)[0])
print('\nEstimated transformation parameter ')
print(lambda_bc)

#Inserindo o lambda de Box-Cox na base de dados para a estimação de um novo modelo
df_dummies['bcdespmed'] = y_boxcox


#%%
#Estimando um novo modelo múltiplo com variável dependente transformada por Box-Cox
X = df_dummies[[ 'idade', 'dcron', 'renda','esmeralda', 'ouro']]
y_bc = df_dummies[['bcdespmed']]

lm_bc = LinearRegression()
lm_bc = lm_bc.fit(X.values, y_bc.values)

F_test_LinearRegression(X, y_bc, lm_bc)
t_test_LinearRegression(X, y_bc, lm_bc)

#%%
# Stepwise
lm_bcstep, X_bcstep = stepwise(X, y_bc, significancia=0.05)

#%%
# Shapiro Francia
residuals = y_bc - lm_bcstep.predict(X_bcstep)
residuals = residuals.values.reshape(1,-1)[0]   
shapiroFrancia(residuals)

#%%
#Histograma dos resíduos 
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)

ax.hist(residuals, bins = 15, ec = 'k', density = True)
x_ = np.linspace(min(residuals), max(residuals), 100)
ax.plot(x_, stats.norm.pdf(x_, np.mean(residuals), np.std(residuals)), lw = 8)
ax.set_xlabel('Resíduos')
ax.set_ylabel('Frequência')
ax.grid(alpha = 0.3)

plt.show()

#%%

breusch_pagan_test(X_bcstep, y_bc, lm_bcstep)

#%%
#Salvando fitted values e residuals no dataframe
df_dummies['fitted_step_novo'] = lm_bcstep.predict(X_bcstep.values)
df_dummies['residuos_step_novo'] = y_bc - lm_bcstep.predict(X_bcstep.values)

#Gráfico Fitted Values x Residuos
X = df_dummies['fitted_step_novo']
y = df_dummies['residuos_step_novo']

fig = plt.figure(figsize=(12, 8))

ax = fig.add_subplot(1, 1, 1)
ax.scatter(X.values, y.values)


ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')

ax.grid(alpha = 0.3)

plt.show()