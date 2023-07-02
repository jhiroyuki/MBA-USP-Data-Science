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


def params_LogisticRegressionMultinomial(X, y, blm, reference):
    alfa = np.delete(blm.intercept_, reference) - blm.intercept_[reference]
    beta = np.delete(blm.coef_, reference, axis = 0) - blm.coef_[reference, :]

    
    df = pd.DataFrame({'(Intercept)': alfa}, index = np.delete(pd.factorize(np.ravel(y.values))[1], reference))
    df[X.columns.values] = beta
    
    print(df)
    return df

def chi2_test_LogisticRegressionMultinomial(X, y, blm):
    
    # Não consegui desenvolver mais isso
    dummies = pd.get_dummies(y)
    
    ordered_categories = pd.factorize(np.ravel(y))[1].tolist()
    
    new_dummies = pd.DataFrame()
    for categories in ordered_categories:
        new_dummies[categories] = dummies[[y.columns[0] + str('_') +categories]]
    
    dummies = new_dummies

    y_new = pd.factorize(np.ravel(y))[0]
    
    if isinstance(X, pd.DataFrame):
        if len(X.columns)==1:
            X = X.values.reshape(-1,1)
            y = y.values
        else:
            X = X.values
            y = y.values

    
    
    k = len(X[0,:] ) # Numero de variáveis
    #n = len(X[:, 0]) # Numero de observações
    
    
    loglikelihood = (dummies *np.log(blm.predict_proba(X))).values.sum()
    
    null_coef = np.mean(y_new)
    loglikelihood0 = np.sum(y_new * np.log(null_coef) + (1- y_new)* np.log(1- null_coef))
    print((y_new))
    chi2_value = -2*(loglikelihood0 - loglikelihood)

    p_value = 1-stats.chi2.cdf(chi2_value, k)
    df = pd.DataFrame()
    df['chi2-statistic'] = [chi2_value]
    df['p-value'] =  [p_value]
    
    print('\n############## CHI SQUARED TEST ##############')
    print(df)
    return df

    
    
def z_test_LogisticRegressionMultinomial(X, y, blm, reference):
    
    
    # Não consegui resolver isso
    df_params = params_LogisticRegressionMultinomial(X, y, blm, reference)
    
    index =  list(df_params.columns.values)
    ordered_categories = pd.factorize(np.ravel(y))[1].tolist()
    remain_categories = np.delete(ordered_categories, reference)
    predicted_proba = pd.DataFrame(blm.predict_proba(X), columns = ordered_categories)
    
    
    if isinstance(X, pd.DataFrame):
        if len(X.columns)==1:
            X = X.values.reshape(-1,1)
            y = y.values
        else:
            X = X.values
            y = y.values
    
    k = len(X[0,:] ) +1 # Graus de liberade de regressão (graus de inclinação + intercepto)
    n = len(X[:, 0]) # Numero de observações
    
    #df_z = pd.DataFrame()
    
    X_matrix = np.insert(X, 0, 1, axis = 1)
    C_matrix = []
    for category_h in ordered_categories:
        for category_l in ordered_categories:
            if category_h == category_l:
                V_matrix = np.diag(np.ravel(predicted_proba[[category_h]].values)*(1 -np.ravel(predicted_proba[[category_h]].values)))
            else:
                V_matrix = -np.diag(np.ravel(predicted_proba[[category_h]].values)*np.ravel(predicted_proba[[category_l]].values))
            C_hl = np.matmul(X_matrix.T,np.matmul(V_matrix, X_matrix))
            C_matrix.append(C_hl)
    
    C_matrix = [C_matrix[start::k] for start in range(k)]
    #C_matrix_ravel = np.ravel(C_matrix)
    C_matrix_reshape = np.block(C_matrix)
    covariance_matrix = np.linalg.inv(C_matrix_reshape)
    covariance_diag = np.diag(covariance_matrix)
    std_error = np.sqrt(covariance_diag)


        
    #     z_values = params/std_error
    
    
    #     p_values =[2*(1-stats.norm.cdf(abs(i))) for i in z_values]
    
    # #df = pd.DataFrame({'Coefficients': params, "Standard Errors": sd_b,
    # #                      "t-statistics": ts_b, 'p-values': p_values}, index = index)
    
        
    #     df = pd.DataFrame({'Coefficients': params, "Standard Errors": std_error,
    #                           "z-statistics": z_values, 'p-values': p_values}, index = index)
    
    #     print('\n######################## z TEST ########################')
    #     print('############ ' +category+ ' ###########')
    #     print(df)
    return 



def OtherTest_LogisticRegressionMultinomial(X, y, blm):
    
    
    dummies = pd.get_dummies(y)
    
    ordered_categories = pd.factorize(np.ravel(y))[1].tolist()
    
    new_dummies = pd.DataFrame()
    for categories in ordered_categories:
        new_dummies[categories] = dummies[[y.columns[0] + str('_') +categories]]
    
    dummies = new_dummies
    
    
    
    if isinstance(X, pd.DataFrame):
        if len(X.columns)==1:
            X = X.values.reshape(-1,1)
            y = y.valueslm.score(X, y)
        else:
            X = X.values
            y = y.values
    
    #k = len(X[0,:] ) # Numero de variáveis
    #n = len(X[:, 0]) # Numero de observações
    
    
    
    
    
    loglikelihood = (dummies *np.log(blm.predict_proba(X))).values.sum()
    # null_coef = np.mean(y)
    # loglikelihood0 = np.sum(np.ravel(y) * np.log(null_coef) + (1- np.ravel(y))* np.log(1-null_coef))

    # chi2_value = -2*(loglikelihood0 - loglikelihood)
    # AIC = -2*loglikelihood + 2*(k+1)
    # BIC = -2*loglikelihood + (k + 1)*np.log(n)
    # pR2_McFadden = chi2_value/(-2*loglikelihood0)
    # pR2_Cragg = (1 - (np.exp(loglikelihood0)/np.exp(loglikelihood))**(2/n))/(1 - np.exp(loglikelihood0)**(2/n))
    
    # values_array = [loglikelihood, AIC, BIC, pR2_McFadden, pR2_Cragg]
    # index = ['Log Likelihood','AIC', 'BIC', 'pseudo-R2 (McFadden)', 'pseudo-R2 (Cragg)']
    
    # df = pd.DataFrame({'Values': values_array}, index = index)
    df = pd.DataFrame({'Values': [loglikelihood]})
    print('\n######################## OTHER TESTS ########################')
    print(df)
    return df


#%%
# Carregamento da base de dados (AtrasadoMultinomial)
#result = pyreadr.read_r('/home/hiro/Documents/3. MBA/2. Aulas/06. Supervised Machine Learning: Modelos Logísticos Binários e Multinomiais/Python/Atrasado.RData') 
result = pyreadr.read_r('AtrasadoMultinomial.RData') 
print(result.keys()) # Checando o nome do DataFrame

#%%
# Carregando o DataFrame
df = result["AtrasadoMultinomial"]

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
# Estimação de modelo logístico multinomial




X = df[['dist', 'sem']]
y = df[['atrasado']]


y_new = pd.factorize(df['atrasado'])[0] 
reference = pd.factorize(df['atrasado'])[1].tolist().index('nao chegou atrasado')



blm = LogisticRegression(penalty='none',  solver='newton-cg', multi_class='multinomial' ) #binary logistic model
blm = blm.fit(X.values,y_new)
#%%

params_LogisticRegressionMultinomial(X, y, blm, reference)

# df_chi2 = chi2_test_LogisticRegression(X, y, blm)
# df_z = z_test_LogisticRegression(X, y, blm)
# df_other = OtherTest_LogisticRegression(X, y, blm)
#%%
#z_test_LogisticRegressionMultinomial(X, y, blm, reference)
