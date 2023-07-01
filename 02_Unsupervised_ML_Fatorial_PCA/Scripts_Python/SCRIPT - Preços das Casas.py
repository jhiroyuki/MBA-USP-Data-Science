import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
import seaborn as sns
import scipy.linalg as sla
from scipy.stats import zscore

#%%
# Carregamento da base de dados (preco_casas)
df = pd.read_excel('preco_casas.xlsx') 
print(df.keys()) # Checando o nome do DataFrame

#%%
# Visualização da base de dados
print('################### DATAFRAME ####################')
print(df)

#%%
# Estatísticas descritivas
summary = df.describe()
print('############ ESTATÍSTICAS DESCRITIVAS ############')
print(summary)


#%%
# Coeficientes de correlação de Pearson para cada par de variáveis
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



corr_coef, corr_sig = pearson(df.iloc[:,:-1], df.iloc[:,:-1])
print('##### MATRIZ DE CORRELAÇÃO ######') #OK
print(corr_coef)
print('##### P-VALUE ######') #OK
print(corr_sig)

#%%
# Elaboração de um mapa de calor das correlações de Pearson entre as variáveis
fig = plt.subplots(figsize=(10,10)) 
ax = sns.heatmap(
    corr_coef, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot=True, 
    fmt='.3f')

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

#%%
### Elaboração a Análise Fatorial Por Componentes Principais ###

#%%

def bartlett_sphericity(dataset, corr_method="pearson"):
    
    r"""
    
    Parameters
    ----------
    dataset : dataframe, mandatory (numerical or ordinal variables)
        
    corr_method : {'pearson', 'spearman'}, optional
        
    Returns
    -------
    out : namedtuple
        The function outputs the test value (chi2), the degrees of freedom (ddl)
        and the p-value.
        It also delivers the n_p_ratio if the number of instances (n) divided 
        by the numbers of variables (p) is more than 5. A warning might be issued.
        
        Ex:
        chi2:  410.27280642443156
        ddl:  45.0
        p-value:  8.73359410503e-61
        n_p_ratio:    20.00
        
        Out: Bartlett_Sphericity_Test_Results(chi2=410.27280642443156, ddl=45.0, pvalue=8.7335941050291506e-61)
    
    References
    ----------
    
    [1] Bartlett,  M.  S.,  (1951),  The  Effect  of  Standardization  on  a  chi  square  Approximation  in  Factor
    Analysis, Biometrika, 38, 337-344.
    [2] R. Sarmento and V. Costa, (2017)
    "Comparative Approaches to Using R and Python for Statistical Data Analysis", IGI-Global.
    
    Examples
    --------
    illustration how to use the function.
    
    >>> bartlett_sphericity(survey_data, corr_method="spearman")
    chi2:  410.27280642443145
    ddl:  45.0
    p-value:  8.73359410503e-61
    n_p_ratio:    20.00
    C:\Users\Rui Sarmento\Anaconda3\lib\site-packages\spyderlib\widgets\externalshell\start_ipython_kernel.py:75: 
    UserWarning: NOTE: we advise  to  use  this  test  only  if  the number of instances (n) divided by the number of variables (p) is lower than 5. Please try the KMO test, for example.
    backend_o = CONF.get('ipython_console', 'pylab/backend', 0)
    Out[12]: Bartlett_Sphericity_Test_Results(chi2=410.27280642443156, ddl=45.0, pvalue=8.7335941050291506e-61)
    """
    
    import numpy as np
    import math as math
    import scipy.stats as stats
#    import warnings as warnings
    import collections

    #Dimensions of the Dataset
    n = dataset.shape[0]
    p = dataset.shape[1]
#    n_p_ratio = n / p
    
    #Several Calculations
    chi2 = - (n - 1 - (2 * p + 5) / 6) * math.log(np.linalg.det(dataset.corr(method=corr_method)))
    #Freedom Degree
    ddl = p * (p - 1) / 2
    #p-value
    pvalue = 1 - stats.chi2.cdf(chi2 , ddl)
    
    Result = collections.namedtuple("Bartlett_Sphericity_Test_Results", ["chi2", "ddl", "pvalue"],  rename=False)   
    
    #Output of the results - named tuple
    result = Result(chi2=chi2,ddl=ddl,pvalue=pvalue) 

    
    #Output of the function
    # if n_p_ratio > 5 :
    #     print("n_p_ratio: {0:8.2f}".format(n_p_ratio))
    #     warnings.warn("NOTE: we advise  to  use  this  test  only  if  the number of instances (n) divided by the number of variables (p) is lower than 5. Please try the KMO test, for example.")
        
    
    return result

bartlett_chisq, bartlett_degrees, bartlett_pvalue = bartlett_sphericity(df.iloc[:, :-1], 'pearson')

print('##### CHI-SQUARED ######') #OK
print(bartlett_chisq)
print('##### DEGREES OF FREEDOM ######') #OK
print(bartlett_degrees)
print('##### P-VALUE ######') #OK
print(bartlett_pvalue)

#%%
# Autovalores e autovetores
eigenvalues,eigenvectors = sla.eig(corr_coef)
eigenvalues = np.real(eigenvalues)

idx = eigenvalues.argsort()[::-1]   
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]

#%%
print('##### AUTOVALORES ######') #OK
print(eigenvalues)
print('\n')
print('##### VARIÂNCIAS COMPARTILHADAS ######') #OK
eigenvalues_variance = eigenvalues/np.sum(eigenvalues)
print(eigenvalues_variance)
#%%
print('##### AUTOVETORES ######') #OK, sinais diferentes mas não deveria mudar os resultados
print(eigenvectors)

#%%
# Scores fatoriais 
fatoriais = eigenvectors/np.sqrt(eigenvalues[None,:]) #OK, sinais diferentes mas não deveria mudar os resultados
print(fatoriais)

#%%
# Z-scores
df_zscore = zscore(df.iloc[:,:-1], ddof = 1)
print('##### Z-SCORES ######') #OK
print(df_zscore)

#%%
# Fatores
fatores = np.matmul(df_zscore, fatoriais)
print('##################### FATORES ####################') #OK, sinais diferentes mas não deveria mudar os resultados
print(fatores)
print('\n')
print('################# PEARSON FATORES ################') #OK
print(pearson(fatores, fatores)[0])

#%%
cargas_fatoriais, _ = pearson( df.iloc[:,:-1], fatores)
print('################ CARGAS FATORIAIS ################') #OK, sinais diferentes mas não deveria mudar os resultados
print(cargas_fatoriais)
print('\n')
print('################# COMUNALIDADES ##################') #OK
print(np.sum(cargas_fatoriais**2, axis = 1))
print('\n')
print('################## AUTOVALORES ###################') #OK
print(np.sum(cargas_fatoriais**2, axis = 0))
print(eigenvalues)

#%%
# Critério de Kaiser
kaiser = np.where(eigenvalues >1)[0]
cargas_fatoriais_kaiser = cargas_fatoriais.iloc[:, kaiser.tolist()]
print('############# CARGAS FATORIAIS KAISER ############')
print(cargas_fatoriais_kaiser)
print('\n')
print('############## COMUNALIDADES KAISER ##############')
print(np.sum(cargas_fatoriais_kaiser**2, axis = 1))
#%%
# Loading plot com as cargas dos 2 primeiros fatores
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
x =  cargas_fatoriais_kaiser.iloc[:,0]
y = cargas_fatoriais_kaiser.iloc[:,1]
ax.scatter(x, 
           y,
           color = 'black',
           s = 60)
ax.axhline(0, color = 'grey', alpha = 0.5)
ax.axvline(0, color = 'grey', alpha = 0.5)


ax.set_xlabel('Principal Component 1',
              fontsize = 20)
ax.set_ylabel('Principal Component 2',
              fontsize = 20)

ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)

text =  cargas_fatoriais_kaiser.index.values
for i in range(len(x)):
    plt.annotate(text[i], (x[i], y[i] + 0.02))
  

plt.show()

#%%
# Loading plot com as cargas do 1° e 3° fator
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
x =  cargas_fatoriais_kaiser.iloc[:,0]
y = cargas_fatoriais_kaiser.iloc[:,2]
ax.scatter(x, 
           y,
           color = 'black',
           s = 60)
ax.axhline(0, color = 'grey', alpha = 0.5)
ax.axvline(0, color = 'grey', alpha = 0.5)


ax.set_xlabel('Principal Component 1',
              fontsize = 20)
ax.set_ylabel('Principal Component 2',
              fontsize = 20)

ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)

text =  cargas_fatoriais_kaiser.index.values
for i in range(len(x)):
    plt.annotate(text[i], (x[i], y[i] + 0.02))
  

plt.show()
#%%
# Ranking (ta dando meio ruim)
# ranking = np.sum(fatores.iloc[:, kaiser.tolist()]*eigenvalues_variance[kaiser], axis = 1)
# ranking_df = pd.DataFrame({'estudante':df['estudante'], 'ranking':ranking})
# ranking_df = ranking_df.sort_values(by='ranking', ascending= False)
