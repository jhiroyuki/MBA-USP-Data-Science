# Geral
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
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
################################################################################
#            REGRESSÃO COM UMA VARIÁVEL EXPLICATIVA (X) QUALITATIVA            #
#                EXEMPLO 03 - CARREGAMENTO DA BASE DE DADOS                    #
################################################################################
#%%
# Carregamento da base de dados (corrupcao)
result = pyreadr.read_r('/home/hiro/Documents/3. MBA/2. Aulas/05. Supervised Machine Learning: Análise de Regressão Simples e Múltipla/Python/corrupcao.RData') 
print(result.keys()) # Checando o nome do DataFrame

#%%
# Carregando o DataFrame
df = result["corrupcao"]

#%%
# Visualização da base de dados
print('################### DATAFRAME ####################')
print(df)


#%%
# Exploração visual do Corruption Perception Index para cada um dos países

X = df[['regiao']]
y = df[['cpi']]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X.values.flatten(), 
           y.values,
           color = 'black')


ax.set_xlabel('Região')
ax.set_ylabel('Corruption Perception Index')
ax.grid(alpha = 0.3)

plt.show()

#%%
#Exploração visual do Corruption Perception Index para cada um dos países, com valores médios por região

X = df[['regiao']]
y = df[['cpi']]


fig = plt.figure(figsize=(12, 8))

ax = fig.add_subplot(1, 1, 1)
ax.scatter(X.values.flatten(), 
           y,
           color = 'black')
ax.plot(X.iloc[:,0].unique(), df.groupby(['regiao']).mean().reindex(index=X.iloc[:,0].unique()))


ax.set_xlabel('Região')
ax.set_ylabel('Corruption Perception Index')

ax.grid(alpha = 0.3)

plt.show()

#%%
# PROCEDIMENTO N-1 DUMMIES

# Dummizando a variável regiao. O código abaixo, automaticamente, fará: 
# dados;

# a) o estabelecimento de dummies que representarão cada uma das regiões da base de 
corrupcao_dummies = pd.get_dummies(df['regiao'])

# b) estabelecerá como categoria de referência a dummy mais frequente.
corrupcao_dummies = corrupcao_dummies.drop(columns = np.sum(corrupcao_dummies).idxmax())

# c) incluir no dataframe
df_dummies = df.join(corrupcao_dummies)

#%%
# ESTIMAÇÃO DO MODELO DE REGRESSÃO

# Modelagem com todas as variáveis
# Estimando a Regressão Múltipla
X = df_dummies.iloc[:, -4:]
y = df_dummies['cpi']
lm = LinearRegression()
lm = lm.fit(X.values, y.values)

F_test_LinearRegression(X, y, lm)
t_test_LinearRegression(X, y, lm)

#%%
#Exploração visual do Corruption Perception Index para cada um dos países, com valores médios por região
X = df_dummies.iloc[:, -4:]
X_ = df[['regiao']]
y = df_dummies['cpi']

fig = plt.figure(figsize=(12, 8))

ax = fig.add_subplot(1, 1, 1)
ax.scatter(X_.values.flatten(), 
           y,
           color = 'black')
ax.scatter(X_.values.flatten(), lm.predict(X))


ax.set_xlabel('Região')
ax.set_ylabel('Corruption Perception Index')

ax.grid(alpha = 0.3)

plt.show()
