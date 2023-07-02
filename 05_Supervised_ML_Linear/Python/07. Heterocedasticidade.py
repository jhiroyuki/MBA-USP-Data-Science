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

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
from scipy.stats import chi2
from statsmodels.formula.api import ols
import statsmodels.api as sm

# %%


def F_test_LinearRegression(X, y, lm):

    if isinstance(X, pd.DataFrame):
        if len(X.columns) == 1:
            X = X.values.reshape(-1, 1)
            y = y.values
        else:
            X = X.values
            y = y.values

    Rsqr = lm.score(X, y)
    # Graus de liberade de regressão (graus de inclinação + intercepto)
    k = len(X[0, :]) + 1
    n = len(X[:, 0])  # Numero de observações

    F_value = (Rsqr/(k - 1))/((1 - Rsqr)/(n - k))
    p_value = 1-stats.f.cdf(F_value, k-1, n-k)
    F_df = pd.DataFrame()
    F_df['F-statistic'] = [F_value]
    F_df['p-value'] = [p_value]

    print('\n############## F TEST ##############')
    print(F_df)


def t_test_LinearRegression(X, y, lm):
    while True:
        try:
            index = ['(Intercept)'] + list(X.columns.values)

            if isinstance(X, pd.DataFrame):
                if len(X.columns) == 1:
                    X = X.values.reshape(-1, 1)
                    y = y.values
                else:
                    X = X.values
                    y = y.values

            params = np.append(lm.intercept_, lm.coef_)
            predictions = lm.predict(X)

            newX = pd.DataFrame(
                {"Constant": np.ones(len(X))}).join(pd.DataFrame(X))
            MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))

            var_b = MSE*(np.linalg.inv(np.dot(newX.T, newX)).diagonal())
            sd_b = np.sqrt(var_b)
            ts_b = params / sd_b

            p_values = [
                2*(1-stats.t.cdf(np.abs(i), (len(newX) - len(newX.iloc[0, :])))) for i in ts_b]

            myDF3 = pd.DataFrame({'Coefficients': params, "Standard Errors": sd_b,
                                  "t-statistics": ts_b, 'p-values': p_values}, index=index)

            print('\n######################## t TEST ########################')
            print(myDF3)
            return myDF3
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print(
                    'Matriz sigular, talvez tenha uma multicolinearidade perfeita aqui!')
                break
            else:
                raise


def pearson(df1, df2):

    coeffmat = np.zeros((df1.shape[1], df2.shape[1]))
    pvalmat = np.zeros((df1.shape[1], df2.shape[1]))

    for i in range(df1.shape[1]):
        for j in range(df2.shape[1]):
            corrtest = pearsonr(df1[df1.columns[i]], df2[df2.columns[j]])

            coeffmat[i, j] = corrtest[0]
            pvalmat[i, j] = corrtest[1]

    corr_coef = pd.DataFrame(coeffmat, columns=df2.columns, index=df1.columns)
    corr_sig = pd.DataFrame(pvalmat, columns=df2.columns, index=df1.columns)
    return corr_coef, corr_sig


def confint(X, y, lm, significance=0.05):
    index = ['(Intercept)'] + list(X.columns.values)

    if isinstance(X, pd.DataFrame):
        if len(X.columns) == 1:
            X = X.values.reshape(-1, 1)
            y = y.values
        else:
            X = X.values
            y = y.values

    # Graus de liberade de regressão (graus de inclinação + intercepto)
    k = len(X[0, :]) + 1
    n = len(X[:, 0])  # Numero de observaçõ#%%


    params = np.append(lm.intercept_, lm.coef_)

    std_error = np.sqrt(np.sum((y - lm.predict(X))**2)/(n - k))
    X_matrix = np.insert(X, 0, 1, axis=1)
    C_diag = np.diag(
        std_error**2 * np.linalg.inv(np.matmul(X_matrix.T, X_matrix)))
    # Valor em x da distriuição em t que corresponde ao valor de significancia
    t_critical = stats.t.ppf(1 - significance/2, n - k)

    params_correction = t_critical*np.sqrt(C_diag)
    params_low = params - params_correction
    params_high = params + params_correction

    df = pd.DataFrame({str(significance/2*100) + '%': params_low,
                      str((1-significance/2)*100) + '%': params_high}, index=index)
    return df


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



def anova_table(aov):
    aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']

    #aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])

    #aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])

    #cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)']
    aov = aov[cols]
    return aov



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



# %%
# Carregamento da base de dados
result = pyreadr.read_r(
    '/home/hiro/Documents/3. MBA/2. Aulas/05. Supervised Machine Learning: Análise de Regressão Simples e Múltipla/Python/saeb_rend.RData')
print(result.keys())  # Checando o nome do DataFrame
# %%
# Carregando o DataFrame
df = result["saeb_rend"]
# %%
# Estatísticas descritivas
summary = df.describe(include='all')
print('############ ESTATÍSTICAS DESCRITIVAS ############')
print(summary)

#%%

# Estimando a Regressão Múltipla
X = df.dropna()[['rendimento']]
y = df.dropna()[['saeb']]
lm = LinearRegression()
lm = lm.fit(X.values, y.values)

F_test_LinearRegression(X, y, lm)
t_test_LinearRegression(X, y, lm)

#%%
# Plotando os dados e os fits
X = df.dropna()[['rendimento']]
y = df.dropna()[['saeb']]
x_ = np.linspace(min(X.values.reshape(1,-1)[0]), max(X.values.reshape(1,-1)[0]), 100)
y_linear = lm.predict(x_.reshape(-1,1))

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)


ax.plot(x_, y_linear, lw = 3, label ='Linear', color = 'black')
ax.scatter(X, 
           y,
           color = 'yellow',
           alpha = 0.15)

ax.set_xlabel('rendimento')
ax.set_ylabel('saeb')
ax.grid(alpha = 0.3)
ax.legend()
plt.show()

#%%

breusch_pagan_test(X, y, lm)

#%%
# PROCEDIMENTO N-1 DUMMIES

# Dummizando a variável regiao. O código abaixo, automaticamente, fará: 
# dados;

# a) o estabelecimento de dummies que representarão cada uma das regiões da base de 
saeb_dummies = pd.get_dummies(df['uf'])

# b) estabelecerá como categoria de referência a dummy mais frequente.
saeb_dummies = saeb_dummies.drop(columns = np.sum(saeb_dummies).idxmax())

# c) incluir no dataframe
df_dummies = df.join(saeb_dummies)

#%%
# ESTIMAÇÃO DO MODELO DE REGRESSÃO

# Modelagem com todas as variáveis
# Estimando a Regressão Múltipla
X = df_dummies.dropna().iloc[:, 6:]
y = df_dummies.dropna()[['saeb']]
lm_dummies = LinearRegression()
lm_dummies = lm_dummies.fit(X.values, y.values)

F_test_LinearRegression(X, y, lm_dummies)
t_test_LinearRegression(X, y, lm_dummies)

#%%

breusch_pagan_test(X, y, lm_dummies)
#ccint@eesc.usp.br