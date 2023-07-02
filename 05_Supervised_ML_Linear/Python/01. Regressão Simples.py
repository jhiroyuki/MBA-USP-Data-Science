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
#                             REGRESSÃO LINEAR SIMPLES                         #
#                     EXEMPLO 01 - CARREGAMENTO DA BASE DE DADOS               #
################################################################################
#%%
# Carregamento da base de dados (tempodist)
result = pyreadr.read_r('/home/hiro/Documents/3. MBA/2. Aulas/05. Supervised Machine Learning: Análise de Regressão Simples e Múltipla/Python/tempodist.RData') 
print(result.keys()) # Checando o nome do DataFrame

#%%
# Carregando o DataFrame
df = result["tempodist"]

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
# Gráfico de dispersão
X = df[['distancia']]
y = df[['tempo']]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X, 
           y,
           color = 'black')


ax.set_xlabel('Distância')
ax.set_ylabel('Tempo')
ax.grid(alpha = 0.3)

plt.show()

#%%
#Estimando o modelo
X = df[['distancia']]
y = df[['tempo']]
lm = LinearRegression()
lm = lm.fit(X.values.reshape(-1,1), y.values)


F_test_LinearRegression(X, y, lm)

t_test_LinearRegression(X, y, lm)
#%%
#Salvando fitted values (variável yhat) e residuals (variável erro) no dataset
df['yhat'] = lm.predict(X.values.reshape(-1,1))
df['residuals'] = y - lm.predict(X.values.reshape(-1,1))

#%% 
# Plot da regressão linear
X = df[['distancia']]
y = df[['tempo']]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X, 
           y,
           color = 'black')
ax.plot(X, lm.predict(X.values.reshape(-1,1)))
ax.annotate('$R^2 = $' + str(lm.score(X.values.reshape(-1,1), y)),  xy=(0.5, 0), 
            xycoords=('axes fraction', 'figure fraction'),
            xytext=(0, 10),  
            textcoords='offset points',)
ax.set_xlabel('Distância')
ax.set_ylabel('Tempo')
ax.grid(alpha = 0.3)

plt.show()

#%% 
# Intervalo de confiança 95%
# O cálculo do intervalo de confiança não está claro pelas aulas e também pelo código do que é gerado pela função regplot do Seaborn.
# É necessário investigar com mais calma, entretanto, plotamos os do Seaborn

sns.regplot(x = "distancia",
            y = "tempo",
            data = df,
            ci = 95)
plt.show()
#%% 
# Intervalo de confiança 99%

sns.regplot(x = "distancia",
            y = "tempo",
            data = df,
            ci = 99)
plt.show()

#%% 
# Intervalo de confiança 99,999%

sns.regplot(x = "distancia",
            y = "tempo",
            data = df,
            ci = 99.999)
plt.show()
#%%
# Cálculo do intervalo de confiança
X = df[['distancia']]
y = df[['tempo']]

confint_df = confint(X, y, lm, significance=0.05)
print(confint_df)
alfa_interval = confint_df.iloc[0, :].values
beta_interval = confint_df.iloc[1, :].values
#%%
# Plot do nível de significância

X = df[['distancia']]
y = df[['tempo']]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X.values.reshape(-1,1), 
            y.values,
            color = 'black', s = 50)
ax.plot(X.values.reshape(-1,1), lm.predict(X.values.reshape(-1,1)))


alfa_array = np.linspace(alfa_interval[0], alfa_interval[1], 15)
beta_array = np.linspace(beta_interval[0], beta_interval[1], 15)
x_array = np.linspace(min(X.values.reshape(-1,1))[0], max(X.values.reshape(-1,1))[0], 100)
for alfa_ in alfa_array:
    for beta_ in beta_array:
        reta = alfa_ + beta_*x_array
        predict = lm.predict(x_array.reshape(-1,1))
        predict = predict.reshape(len(predict))
        ax.fill_between(x_array, reta,predict, color='b', alpha=.01)


ax.set_xlabel('Distância')
ax.set_ylabel('Tempo')
ax.grid(alpha = 0.3)

plt.show()

#%%
#  NOVA MODELAGEM PARA O EXEMPLO 01, COM NOVO DATASET QUE CONTÉM REPLICAÇÕES   #

# Quantas replicações de cada linha você quer? 
newdf = pd.DataFrame(np.repeat(df.values, 3, axis=0), columns=df.columns)

# Reestimando o modelo
X = newdf[['distancia']]
y = newdf[['tempo']]
lm = LinearRegression()
lm = lm.fit(X.values.reshape(-1,1), y.values)

#Observando os parâmetros do modelo novo
F_test_LinearRegression(X, y, lm)
t_test_LinearRegression(X, y, lm)

#Calculando os novos intervalos de confiança
confint_df = confint(X, y, lm, significance=0.05)
print(confint_df)
alfa_interval = confint_df.iloc[0, :].values
beta_interval = confint_df.iloc[1, :].values

#Plotando o Novo Gráfico com Intervalo de Confiança de 95%
#Note o estreitamento da amplitude dos intervalos de confiança!
X = newdf[['distancia']]
y = newdf[['tempo']]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(X.values.reshape(-1,1), 
            y.values,
            color = 'black', s = 50)
ax.plot(X.values.reshape(-1,1), lm.predict(X.values.reshape(-1,1)))


alfa_array = np.linspace(alfa_interval[0], alfa_interval[1], 15)
beta_array = np.linspace(beta_interval[0], beta_interval[1], 15)
x_array = np.linspace(min(X.values.reshape(-1,1))[0], max(X.values.reshape(-1,1))[0], 100)
for alfa_ in alfa_array:
    for beta_ in beta_array:
        reta = alfa_ + beta_*x_array
        predict = lm.predict(x_array.reshape(-1,1))
        predict = predict.reshape(len(predict))
        ax.fill_between(x_array, reta,predict, color='b', alpha=.01)


ax.set_xlabel('Distância')
ax.set_ylabel('Tempo')
ax.grid(alpha = 0.3)

plt.show()
ax.set_xlabel('Distância')
ax.set_ylabel('Tempo')
ax.grid(alpha = 0.3)

plt.show()



