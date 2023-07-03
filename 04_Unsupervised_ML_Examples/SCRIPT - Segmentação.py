# Geral
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyreadr

# Clustering K-Means
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# ANACOR Múltipla
import prince
from scipy.stats import chi2_contingency

# Clustering Hierarquico
# 01° Passo: Padronização das variáveis com o ZScore
# 02° Passo: Elbow Method   
# 03° Passo: Clusterização
# 04° Passo: Adicionar os clusters no dataframe
# 05° Passo: ANOVA
# 06° Passo: Estatísticas descritivas por cluster

# ANACOR Múltipla
# 01° Passo: Tabelas de contingência
# 02° Passo: Número de dimensões (número de categorias - número de variáveis)
# 03° Passo: ANACOR pelo Prince
# 04° Passo: Mapa perceptual

#%%
# Carregamento da base de dados (segmenta)
result = pyreadr.read_r('datasets/segmenta.Rdata') 
print(result.keys()) # Checando o nome do DataFrame

#%%
# Carregando o DataFrame
df_ = result["segmenta"]

# Como existem variáveis com missing values (NAs), vamos excluir as observações

df_ = df_.dropna()

df = df_.copy()
#%%
# Visualização da base de dados
print(df)


#%%
# Algumas variáveis são qualitativas e outras são quantitativas
# Vamos separar o banco de dados em 2 partes (somente quali e quanti)

segmenta_quali = df[['Gender', 'Ever_Married', 'Graduated', 'Spending_Score']]
segmenta_quanti = df[['Age', 'Family_Size']]

#%%
print('############ ESTATÍSTICAS DESCRITIVAS QUALI ############')
print(segmenta_quali.describe())

print('############ ESTATÍSTICAS DESCRITIVAS QUANTI ############')
print(segmenta_quanti.describe())

#%%

# INÍCIO ANÁLISE DE CLUSTER PARA VARIÁVEIS QUANTITATIVAS

#%%

# 01° Passo: Padronização das variáveis com o ZScore

# Aplicando a padronização por ZScore
segm_pad = zscore(segmenta_quanti, ddof = 1)
print('##### Z-SCORES ######') #OK
print(segm_pad)

#%%

# 02° Passo: Elbow Method   

# Método de Elbow para identificação do número ótimo de clusters

df = segm_pad.copy()
n_clusters = 15

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, n_clusters)
  
for k in K:
    print(k)
    kmeanModel = KMeans(n_clusters=k).fit(df)
    distortions.append(sum(np.min(cdist(df, kmeanModel.cluster_centers_, metric= 'euclidean'), axis=1))/len(df.index))
    inertias.append(kmeanModel.inertia_)
    
    mapping1[k] = sum(np.min(cdist(df, kmeanModel.cluster_centers_, metric= 'euclidean'), axis =1)/len(df.index))
    mapping2[k] = kmeanModel.inertia_
    
plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()

#%%

# 03° Passo: Clusterização
# 04° Passo: Adicionar os clusters no dataframe

# Elaboração da clusterização não hieráquica k-means
# Criando variável categórica para indicação do cluster no banco de dados
kmeans = KMeans(
    init="random",
    n_clusters=5,
    n_init=10,
    max_iter=300,
    random_state=1,
    algorithm='full'
)

kmeans = kmeans.fit(df)
cluster_kmeans = kmeans.labels_
segm_pad['cluster_K'] = cluster_kmeans
df_['cluster_K'] = cluster_kmeans
df['cluster_K'] = cluster_kmeans


#%%


# 05° Passo: ANOVA

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
# ANOVA da variável 'Age'
model = ols('Age ~ C(cluster_K)', data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print(anova_table(aov_table))
# ANOVA da variável 'Family_Size'
model = ols('Family_Size ~ C(cluster_K)', data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print(anova_table(aov_table))

#%%

# 06° Passo: Estatísticas descritivas por cluster

agrupar_cluster = df.groupby('cluster_K')
for i in range(len(agrupar_cluster)):
    cluster = agrupar_cluster.get_group(i)
    summary = cluster.describe()
    print('############ CLUSTER ' + str(i) +' ############')
    print(summary)

#%%

# FIM ANÁLISE DE CLUSTER PARA VARIÁVEIS QUANTITATIVAS


#%%

# INÍCIO ANACOR PARA VARIÁVEIS QUALITATIVAS


#%%

# 01° Passo: Tabelas de contingência

# Tabela de contingência com frequências absolutas observadas
df = segmenta_quali.copy()
tabela_contingencia = pd.crosstab(index=df['Spending_Score'], columns=df['Gender'])
print('####### CONTINGENCY TABLE OBSERVED ##########')
print(tabela_contingencia)

# Estatística chi quadrado e teste
# Se o p valor é menor que 0.05 então se rejeita  hipótese nula, isto é, que as variáveis não tem associação estatisticamente relevantes.
chi2 = chi2_contingency(tabela_contingencia)
stat, pvalue, dof, expected = chi2 # Estatística do teste, p valor, degrees of freedom, tabela de frequências esperadas

print('####### CHI SQUARED TEST ##########')
teste_chi2 = pd.DataFrame({'stat': [stat], 'pvalues':[pvalue], 'dof': [dof]}, index = ['values'])
print(teste_chi2)

#%%
# Tabela de contingência com frequências absolutas observadas
df = segmenta_quali.copy()
tabela_contingencia = pd.crosstab(index=df['Spending_Score'], columns=df['Ever_Married'])
print('####### CONTINGENCY TABLE OBSERVED ##########')
print(tabela_contingencia)

# Estatística chi quadrado e teste
# Se o p valor é menor que 0.05 então se rejeita  hipótese nula, isto é, que as variáveis não tem associação estatisticamente relevantes.
chi2 = chi2_contingency(tabela_contingencia)
stat, pvalue, dof, expected = chi2 # Estatística do teste, p valor, degrees of freedom, tabela de frequências esperadas

print('####### CHI SQUARED TEST ##########')
teste_chi2 = pd.DataFrame({'stat': [stat], 'pvalues':[pvalue], 'dof': [dof]}, index = ['values'])
print(teste_chi2)

#%%
# Tabela de contingência com frequências absolutas observadas
df = segmenta_quali.copy()
tabela_contingencia = pd.crosstab(index=df['Spending_Score'], columns=df['Graduated'])
print('####### CONTINGENCY TABLE OBSERVED ##########')
print(tabela_contingencia)

# Estatística chi quadrado e teste
# Se o p valor é menor que 0.05 então se rejeita  hipótese nula, isto é, que as variáveis não tem associação estatisticamente relevantes.
chi2 = chi2_contingency(tabela_contingencia)
stat, pvalue, dof, expected = chi2 # Estatística do teste, p valor, degrees of freedom, tabela de frequências esperadas

print('####### CHI SQUARED TEST ##########')
teste_chi2 = pd.DataFrame({'stat': [stat], 'pvalues':[pvalue], 'dof': [dof]}, index = ['values'])
print(teste_chi2)
#%%
# Tabela de contingência com frequências absolutas observadas
df = segmenta_quali.copy()
tabela_contingencia = pd.crosstab(index=df['Graduated'], columns=df['Gender'])
print('####### CONTINGENCY TABLE OBSERVED ##########')
print(tabela_contingencia)

# Estatística chi quadrado e teste
# Se o p valor é menor que 0.05 então se rejeita  hipótese nula, isto é, que as variáveis não tem associação estatisticamente relevantes.
chi2 = chi2_contingency(tabela_contingencia)
stat, pvalue, dof, expected = chi2 # Estatística do teste, p valor, degrees of freedom, tabela de frequências esperadas

print('####### CHI SQUARED TEST ##########')
teste_chi2 = pd.DataFrame({'stat': [stat], 'pvalues':[pvalue], 'dof': [dof]}, index = ['values'])
print(teste_chi2)
#%%
# Tabela de contingência com frequências absolutas observadas
df = segmenta_quali.copy()
tabela_contingencia = pd.crosstab(index=df['Graduated'], columns=df['Ever_Married'])
print('####### CONTINGENCY TABLE OBSERVED ##########')
print(tabela_contingencia)

# Estatística chi quadrado e teste
# Se o p valor é menor que 0.05 então se rejeita  hipótese nula, isto é, que as variáveis não tem associação estatisticamente relevantes.
chi2 = chi2_contingency(tabela_contingencia)
stat, pvalue, dof, expected = chi2 # Estatística do teste, p valor, degrees of freedom, tabela de frequências esperadas

print('####### CHI SQUARED TEST ##########')
teste_chi2 = pd.DataFrame({'stat': [stat], 'pvalues':[pvalue], 'dof': [dof]}, index = ['values'])
print(teste_chi2)

#%%
# Tabela de contingência com frequências absolutas observadas
df = segmenta_quali.copy()
tabela_contingencia = pd.crosstab(index=df['Gender'], columns=df['Ever_Married'])
print('####### CONTINGENCY TABLE OBSERVED ##########')
print(tabela_contingencia)

# Estatística chi quadrado e teste
# Se o p valor é menor que 0.05 então se rejeita  hipótese nula, isto é, que as variáveis não tem associação estatisticamente relevantes.
chi2 = chi2_contingency(tabela_contingencia)
stat, pvalue, dof, expected = chi2 # Estatística do teste, p valor, degrees of freedom, tabela de frequências esperadas

print('####### CHI SQUARED TEST ##########')
teste_chi2 = pd.DataFrame({'stat': [stat], 'pvalues':[pvalue], 'dof': [dof]}, index = ['values'])
print(teste_chi2)

#%%

# 02° Passo: Número de dimensões (número de categorias - número de variáveis)

# Primeiro encontrar a quantidade de dimensões, isto é a quantidade de categorias menos a quantidade de variáveis
numero_categorias = 0
for variable in df.keys():
    numero_categorias = numero_categorias + df[variable].nunique()
    
numero_variaveis = len(df.keys())

numero_dimensoes = numero_categorias - numero_variaveis

#%%

# 03° Passo: ANACOR pelo Prince

# Análise de correspondência múltipla feita pelo "prince". Os valores das coordenadas são diferentes dos encontrados no R.

mca = prince.MCA(n_components=numero_dimensoes, n_iter=3, copy=True, check_input=True,engine='auto', random_state=42)
mca = mca.fit(df)
#%%

# 04° Passo: Mapa perceptual

ax = mca.plot_coordinates(df, ax=None, figsize=(6, 6),show_row_points=False,
                             show_column_points=True, column_points_size=30, show_column_labels=True, legend_n_cols=2)

ax.legend(bbox_to_anchor=(1.3, 1.0))
ax.get_figure()

print(mca.row_coordinates(df))
print(mca.column_coordinates(df))


print(mca.eigenvalues_)
print(mca.total_inertia_)
print(mca.explained_inertia_)
