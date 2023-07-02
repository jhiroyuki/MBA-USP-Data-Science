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
# Carregamento da base de dados (dados_fidelidade)
result = pyreadr.read_r('dados_fidelidade.RData') 
print(result.keys()) # Checando o nome do DataFrame

#%%
# Carregando o DataFrame
df = result["dados_fidelidade"]

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
##############################################################################
#                    EXEMPLO 03 -  PROCEDIMENTO N-1 DUMMIES                  #
##############################################################################
#Dummizando as variáveis atendimento, sortimento, acessibilidade e preço. O 
#código abaixo, automaticamente, fará: a) a dummização das variáveis originais;
#b)removerá as variáveis dummizadas originais; c) estabelecerá como categorias 
#de referência as categorias de label 1 de cada variável original.

to_dummy = ['atendimento', 'sortimento', 'acessibilidade', 'preço']
df_dummies = df
for dummy in to_dummy:
    
    # a) o estabelecimento de dummies que representarão cada uma das regiões da base de 
    dummies = pd.get_dummies(df[[dummy]])
    
    # b) estabelecerá como categoria de referência a dummy mais frequente.
    # aqui na verdade vamos remover o primeiro dummy
    dummies = dummies.iloc[: , 1:]
    
    # c) incluir no dataframe
    df_dummies = df_dummies.join(dummies)

df_dummies = df_dummies.drop(columns = to_dummy)
df_dummies['sexo'] = np.where(df_dummies['sexo'].values == 'masculino', 1, 0)
df_dummies['fidelidade'] = np.where(df_dummies['fidelidade'].values == 'nao', 0, 1)
#%%
# Estimação de modelo logístico binário
X = df_dummies[df_dummies.columns.difference(['fidelidade', 'id'])]
y = df_dummies[['fidelidade']]

blm = LogisticRegression(penalty='none',  solver='newton-cg' ) #binary logistic model
blm = blm.fit(X.values,np.ravel(y.values))
from sklearn.linear_model import PoissonRegressor
df_chi2 = chi2_test_LogisticRegression(X, y, blm)
df_z = z_test_LogisticRegression(X, y, blm)
df_other = OtherTest_LogisticRegression(X, y, blm)

#%%

blm_step, X_step = stepwise_LogisticRegression(X, y, significancia=0.05)

OtherTest_LogisticRegression(X_step, y, blm_step)

#%%
#Matriz de confusão para cutoff = 0.5
cutoff = 0.5
y_true = y.values
y_prob = blm_step.predict_proba(X_step)[:, 1]
y_pred = np.where(y_prob >= cutoff, 1, 0)
confusion = confusion_matrix(y_true, y_pred)

# Indicadores
accuracy = np.sum(np.diag(confusion))/np.sum(confusion)
sensitivity = confusion[1,1]/np.sum(confusion[1, :])
specificity = confusion[0,0]/np.sum(confusion[0, :])
indicadores = [accuracy, sensitivity, specificity]
df_confusion = pd.DataFrame({'Values': indicadores}, index = ['Accuracy', 'Sensitivity', 'Specificity'])

print(df_confusion)


#%%

cutoff_array = np.linspace(0,1, 200)
y_prob = blm_step.predict_proba(X_step)[:, 1]
sensitivity_array = []
specificity_array = []
for cutoff in cutoff_array:
    y_pred = np.where(y_prob >= cutoff, 1, 0)
    confusion = confusion_matrix(y_true, y_pred)
    sensitivity = confusion[1,1]/np.sum(confusion[1, :])
    specificity = confusion[0,0]/np.sum(confusion[0, :])
    sensitivity_array.append(sensitivity)
    specificity_array.append(specificity)
    
df_roc = pd.DataFrame({'Cutoff': cutoff_array, 'Sensitivity': sensitivity_array, 'Specificity': specificity_array})


# Cutoff x Sensitividade/Especificidade 
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(df_roc[['Cutoff']].values, 
           df_roc[['Sensitivity']].values,
           color = 'blue')

ax.scatter(df_roc[['Cutoff']].values, 
           df_roc[['Specificity']].values,
           color = 'red')

ax.set_xlabel('Cutoff')
ax.set_ylabel('Sensitividade/Especificade')
ax.grid(alpha = 0.3)

plt.show()

# (1 - Especificdade) x Sensitividade (Curva ROC)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(1 - df_roc[['Specificity']].values, 
           df_roc[['Sensitivity']].values,
           color = 'blue')


ax.set_xlabel('Cutoff')
ax.set_ylabel('Sensitividade/Especificade')
ax.grid(alpha = 0.3)

plt.show()

# Área curva ROC
A_ROC = np.abs(np.trapz( np.ravel(df_roc[['Sensitivity']].values), x=np.ravel(1 - df_roc[['Specificity']].values)))
print('Area ROC: ', A_ROC)
gini = (A_ROC - 0.5)/0.5
print('Gini: ', gini)
