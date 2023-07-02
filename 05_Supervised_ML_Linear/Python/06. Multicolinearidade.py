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
    n = len(X[:, 0])  # Numero de observações

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


# %%
# Carregamento da base de dados
result = pyreadr.read_r(
    '/home/hiro/Documents/3. MBA/2. Aulas/05. Supervised Machine Learning: Análise de Regressão Simples e Múltipla/Python/salarios.RData')
print(result.keys())  # Checando o nome do DataFrame
# %%
# Carregando o DataFrame
df = result["salarios"]
# %%
# Estatísticas descritivas
summary = df.describe(include='all')
print('############ ESTATÍSTICAS DESCRITIVAS ############')
print(summary)

# %%
# Estudo das correlações
corr_coef, corr_sig = pearson(df.iloc[:, 1:], df.iloc[:, 1:])
print('##### MATRIZ DE CORRELAÇÃO ######')  # OK
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


# %%
# Estudo para correlação perfeita (rh1 e econometria1)

X = df[['rh1', 'econometria1']]
y = df[['salario']]
lm = LinearRegression()
lm = lm.fit(X.values, y.values)

F_test_LinearRegression(X, y, lm)
t_test_LinearRegression(X, y, lm)

# calculating VIF for each feature
vif_data = pd.DataFrame(index=X.columns)

vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]
vif_data["Tolerance"] = [1/variance_inflation_factor(X.values, i)
                         for i in range(len(X.columns))]
print(vif_data)
# %%
# Estudo para correlação baixa (rh3 e econometria3)

X = df[['rh3', 'econometria3']]
y = df[['salario']]
lm = LinearRegression()
lm = lm.fit(X.values, y.values)

F_test_LinearRegression(X, y, lm)
t_test_LinearRegression(X, y, lm)

# calculating VIF for each feature
vif_data = pd.DataFrame(index=X.columns)

vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]
vif_data["Tolerance"] = [1/variance_inflation_factor(X.values, i)
                         for i in range(len(X.columns))]

print(vif_data)

# %%
# Estudo para correlação muto alta porém não perfeita (rh2 e econometria2)

X = df[['rh2', 'econometria2']]
y = df[['salario']]
lm = LinearRegression()
lm = lm.fit(X.values, y.values)

F_test_LinearRegression(X, y, lm)
t_test_LinearRegression(X, y, lm)

# calculating VIF for each feature
vif_data = pd.DataFrame(index=X.columns)

vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]
vif_data["Tolerance"] = [1/variance_inflation_factor(X.values, i)
                         for i in range(len(X.columns))]

print(vif_data)
# %%

# Stepwise
X = df[['rh2', 'econometria2']]
y = df[['salario']]

lm_step, X_step = stepwise(X, y, significancia=0.05)
