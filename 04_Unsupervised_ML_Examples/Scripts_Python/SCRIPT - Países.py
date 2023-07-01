# Geral
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# PCA
from scipy.stats import pearsonr
import scipy.linalg as sla
from scipy.stats import zscore
import prince


# Clustering
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as sch
import statsmodels.api as sm
from statsmodels.formula.api import ols



# PCA
# 01º Passo: Matriz de correlações de Pearson
# 02° Passo: Teste de esferecidade de Barllet
# 03° Passo: Autovalores e autovetores da matriz de correlação de Pearson. Da pra tirar a variância compartilhada, que é a porcentagem de cada autovalor da soma deles
# 04° Passo: Scores fatoriais (autovetores dividido pela raiz do seu autovalor)
# 05° Passo: Zscores (são as observações padronizadas)
# 06° Passo: Fatores (multiplicação matricial dos Zscores com os scores fatoriais)
# 07° Passo: Cargas fatoriais (correlaçaõ de Pearson entre os fatores e as variáveis originais)
# 08° Passo: Critério de Kaiser (autovalores maiores que 1), para definir o número de fatores
# 09° Passo: Comunalidades dos autovalores de Kaiser
# 10° Passo: Plotar as cargas fatoriais

# Clustering Hierarquico
# 01° Passo: Padronização das variáveis com o ZScore
# 02° Passo: Matriz de dissimilaridades    
# 03° Passo: Clusterização
# 04° Passo: Dendograma
# 05° Passo: Definir número de clusters e adicioná-los no dataframe
# 06° Passo: ANOVA
# 07° Passo: Estatísticas descritivas por cluster
#%%
# Carregamento da base de dados (notas_fatorial)
df = pd.read_csv('Países PCA Cluster.csv') 

#%%
# Visualização da base de dados
print(df)

#%%
# Estatísticas descritivas
summary = df.describe()
print('############ ESTATÍSTICAS DESCRITIVAS ############')
print(summary)


#%%

# INÍCIO ANÁLISE POR COMPONENTES PRINCIPAIS

#%%
# Scatter e ajuste entre as variáveis 'renda' e 'expectativa de vida'

x = df['income'].values.reshape(-1, 1)
y = df['life_expec'].values.reshape(-1, 1)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x, 
           y,
           color = 'black')


ax.grid()
ax.set_xlabel('Renda',
              fontsize = 20)
ax.set_ylabel('Expectativa de vida',
              fontsize = 20)
plt.show()
#%%
# Scatter e ajuste entre as variáveis 'exports' e 'gdpp de vida'

x = df['exports'].values.reshape(-1, 1)
y = df['gdpp'].values.reshape(-1, 1)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x, 
           y,
           color = 'black')

ax.grid()
ax.set_xlabel('exports',
              fontsize = 20)
ax.set_ylabel('gdpp',
              fontsize = 20)
plt.show()
#%%

# 01º Passo: Matriz de correlações de Pearson

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



corr_coef, corr_sig = pearson(df.iloc[:,1:], df.iloc[:,1:])
print('##### MATRIZ DE CORRELAÇÃO ######') #OK
print(corr_coef)
print('##### P-VALUE ######') #OK
print(corr_sig)

#%%
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

# 02° Passo: Teste de esferecidade de Barllet

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

bartlett_chisq, bartlett_degrees, bartlett_pvalue = bartlett_sphericity(df.iloc[:, 1:], 'pearson')

print('##### CHI-SQUARED ######') #OK
print(bartlett_chisq)
print('##### DEGREES OF FREEDOM ######') #OK
print(bartlett_degrees)
print('##### P-VALUE ######') #OK
print(bartlett_pvalue)


#%%

# 03° Passo: Autovalores e autovetores da matriz de correlação de Pearson. Da pra tirar a variância compartilhada, que é a porcentagem de cada autovalor da soma deles

# Autovalores e autovetores
eigenvalues,eigenvectors = sla.eig(corr_coef)
eigenvalues = np.real(eigenvalues)

idx = eigenvalues.argsort()[::-1]   
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]
eigenvectors[:, 1] = -eigenvectors[:, 1]
eigenvectors = -eigenvectors
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

# 04° Passo: Scores fatoriais (autovetores dividido pela raiz do seu autovalor)

# Scores fatoriais 
fatoriais = eigenvectors/np.sqrt(eigenvalues[None,:]) #OK, sinais diferentes mas não deveria mudar os resultados
print(fatoriais)

#%%

# 05° Passo: Zscores (são as observações padronizadas)

# Z-scores
df_zscore = zscore(df.iloc[:,1:], ddof = 1)
print('##### Z-SCORES ######') #OK
print(df_zscore)

#%%

# 06° Passo: Fatores (multiplicação matricial dos Zscores com os scores fatoriais)

# Fatores
fatores = np.matmul(df_zscore, fatoriais)
print('##################### FATORES ####################') #OK, sinais diferentes mas não deveria mudar os resultados
print(fatores)
print('\n')
print('################# PEARSON FATORES ################') #OK
print(pearson(fatores, fatores)[0])

#%%

# 07° Passo: Cargas fatoriais (correlaçaõ de Pearson entre os fatores e as variáveis originais)

cargas_fatoriais, _ = pearson( df.iloc[:,1:], fatores)
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

# 08° Passo: Critério de Kaiser (autovalores maiores que 1), para definir o número de fatores
# 09° Passo: Comunalidades dos autovalores de Kaiser

# Critério de Kaiser
kaiser = np.where(eigenvalues >1)[0]
cargas_fatoriais_kaiser = cargas_fatoriais.iloc[:, kaiser.tolist()]
print('############# CARGAS FATORIAIS KAISER ############')
print(cargas_fatoriais_kaiser)
print('\n')
print('############## COMUNALIDADES KAISER ##############')
print(np.sum(cargas_fatoriais_kaiser**2, axis = 1))

#%%

# 10° Passo: Plotar as cargas fatoriais (loading plot)

# Loading plot
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
# Adicionando os fatores extraídos no banco de dados original
for i in kaiser:
    column_name = 'fator_' + str(i + 1)
    df[column_name] = fatores.iloc[:,i].values


# FIM DA ANÁLISE POR COMPONENTES PRINCIPAIS    
#%%

# INÍCIO DA ANÁLISE DE CLUSTER


#%%

# 01° Passo: Padronização das variáveis com o ZScore
 
# Análise dos fatores (média e desvio padrão)
summary = df.iloc[:, -3:].describe()
print(summary)

## ATENÇÃO: os clusters serão formados a partir dos 3 fatores
## Não aplicaremos o Z-Score, pois os fatores já são padronizados

#%%

# 02° Passo: Matriz de dissimilaridades    

# Matriz de dissimilaridades
matrix_D = squareform(pdist(df.iloc[:, -3:],
                 metric = 'euclidean'))

#%%
# Elaboração da clusterização hierárquica

# 03° Passo: Clusterização

linkage_data = sch.linkage(df.iloc[:, -3:], 
                        method = 'complete',
                        metric = 'euclidean')

#%%
# Construção do dendrograma

# 04° Passo: Dendograma

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
sch.dendrogram(linkage_data, 
               ax = ax,
               color_threshold=0, 
               above_threshold_color= 'black')
# sch.fcluster(linkage_data, 
#                 0, 
#                 criterion = 'monocrit', 
#                 monocrit = 5)
plt.title("Dendrograma de Cluster",
          fontsize = 20)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.show()

#%%

# 05° Passo: Definir número de clusters e adicioná-los no dataframe

# # Criando variável categórica para indicação do cluster no banco de dados
## O argumento 'n_clusters' indica a quantidade de clusters, faremos com 10 clusters

cutree = sch.cut_tree(linkage_data, n_clusters = 10)
if 'cluster_H' not in df:
    df.insert(len(df.columns), 
              "cluster_H" ,
              pd.DataFrame(cutree, dtype='category'))
print(df)

#%%

# 06° Passo: ANOVA

# Análise de variância de um fator (ANOVA). Interpretação do output:

## Mean Sq do cluster_H: indica a variabilidade entre grupos
## Mean Sq dos Residuals: indica a variabilidade dentro dos grupos
## F value: estatística de teste (Sum Sq do cluster_H / Sum Sq dos Residuals)
## Pr(>F): p-valor da estatística 
## p-valor < 0.05: pelo menos um cluster apresenta média estatisticamente diferente dos demais
def anova_table(aov):
    aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']

    #aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])

    #aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])

    #cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)']
    aov = aov[cols]
    return aov

#%%
# ANOVA da variável 'fator_1'
model = ols('fator_1 ~ C(cluster_H)', data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print(anova_table(aov_table))
# ANOVA da variável 'fator_2'
model = ols('fator_2 ~ C(cluster_H)', data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print(anova_table(aov_table))
# ANOVA da variável 'fator_3'
model = ols('fator_3 ~ C(cluster_H)', data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print(anova_table(aov_table))


#%%

# 07° Passo: Estatísticas descritivas por cluster

agrupar_cluster = df.groupby('cluster_H')
for i in range(len(agrupar_cluster)):
    cluster = agrupar_cluster.get_group(i)
    summary = cluster.describe()
    print('############ CLUSTER ' + str(i) +' ############')
    print(summary)

# FIM!
#%%

# APÊNDICE: PCA PELO PRINCE

#%%
# Carregamento da base de dados (notas_fatorial)
df = pd.read_csv('Países PCA Cluster.csv') 

#%%

pca = prince.PCA(n_components=3, n_iter=3, rescale_with_mean=True,rescale_with_std=True,
                 copy=True,check_input=True, engine='auto', random_state=42)

pca = pca.fit(df.iloc[:, 1:])


print(pca.row_coordinates(df.iloc[:, 1:])) # Aqui são os que vão no dataframe

print(pca.column_correlations(df.iloc[:, 1:])) # Aqui são os que vai plotar no loading plot
