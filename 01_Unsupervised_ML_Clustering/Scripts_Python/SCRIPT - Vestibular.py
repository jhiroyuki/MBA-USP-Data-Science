import pandas as pd
import numpy as np
import pyreadr
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as sch
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
#%%
# Carregamento da base de dados (Vestibular)
result = pyreadr.read_r('datasets/Vestibular.RData') 
print(result.keys()) # Checando o nome do DataFrame

#%%
# Carregando o DataFrame
df = result["Vestibular"]

#%%
# Visualização da base de dados
print(df)

#%%
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection='3d')

cs = df['quimica']
# s = pointsize
# c = sequencia de cores
# cmap = colormap

scatter = ax.scatter3D(df['fisica'], 
             df['matematica'], 
             df['quimica'],
             s = 200,
             c=cs,
             cmap='jet')

ax.set_xlabel('Física',
              fontsize = 20)
ax.set_ylabel('Matemática',
              fontsize = 20)
ax.set_zlabel('Química',
              fontsize = 20)
plt.title('Vestibular',
              fontsize = 20)

# Escrever nome nos pontos
for i, txt in enumerate(df['estudante']):
    ax.text(df['fisica'][i],  
            df['matematica'][i],  
            df['quimica'][i], 
            '%s' % (txt), 
            size = 20)

cbar = plt.colorbar(scatter)
cbar.ax.set_xlabel('Nota de química', 
                   fontsize = 20)
plt.show()

#%%
# Estatísticas descritivas
summary = df.describe(include = 'all')
print(summary)

#%%
# Padronizar variáveis
# Aparentemente tem uma diferença entre o scale aqui e o scale do R
newdf = pd.DataFrame(scale(df.iloc[:, 1:4]),
                     index=df.iloc[:, 1:4].index,
                     columns=df.iloc[:, 1:4].columns)
newdf.insert(0, "estudante" ,df['estudante'])
print(newdf)
print(newdf.describe(include = 'all'))

#%%
#---------- Esquema de aglomeração hierárquico ---------------------------------

#%%
# Matriz de dissimilaridades

matrix_D = squareform(pdist(df.iloc[:, 1:],
                 metric = 'euclidean'))

#%%
# Elaboração da clusterização hierárquica

linkage_data = sch.linkage(df.iloc[:, 1:], 
                        method = 'single',
                        metric = 'euclidean')

#%%
# As duas primeiras colunas da matriz de linkage são o esquema do clustering e a última coluna são os coeficientes
print(linkage_data)

#%%
# Construção do dendrograma

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
# # Criando variável categórica para indicação do cluster no banco de dados
## O argumento 'n_clusters' indica a quantidade de clusters

cutree = sch.cut_tree(linkage_data, n_clusters = 3)
if 'cluster_H' not in df:
    df.insert(len(df.columns), 
              "cluster_H" ,
              pd.DataFrame(cutree, dtype='category'))
print(df)

#%%
# Não consegui fazer o dendrograma com as cores diferentes para cada cluster

#%%
# Estatísticas descritivas dos clusters por variável

print(df.groupby('cluster_H').agg({'matematica': ['mean', 'std', 'min', 'max']}))
print(df.groupby('cluster_H').agg({'fisica': ['mean', 'std', 'min', 'max']}))
print(df.groupby('cluster_H').agg({'quimica': ['mean', 'std', 'min', 'max']}))

#%%
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
# ANOVA da variável 'matematica'
model = ols('matematica ~ C(cluster_H)', data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print(anova_table(aov_table))
# ANOVA da variável 'fisica'
model = ols('fisica ~ C(cluster_H)', data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print(anova_table(aov_table))
# ANOVA da variável 'quimica'
model = ols('quimica ~ C(cluster_H)', data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print(anova_table(aov_table))

#%%
#---------- Esquema de aglomeração não hierárquico K-MEANS ---------------------

kmeans = KMeans(n_clusters=3).fit(df.iloc[:, 1:4])

# Criando variável categórica para indicação do cluster no banco de dados
kmeans_labels = kmeans.labels_
if 'cluster_K' not in df:
    df.insert(len(df.columns), 
              "cluster_K" ,
              pd.DataFrame(kmeans_labels, dtype='category'))
print(df)

#%%
# ANOVA da variável 'matematica'
model = ols('matematica ~ C(cluster_K)', data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print(anova_table(aov_table))
# ANOVA da variável 'fisica'
model = ols('fisica ~ C(cluster_K)', data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print(anova_table(aov_table))
# ANOVA da variável 'quimica'
model = ols('quimica ~ C(cluster_K)', data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
print(anova_table(aov_table))

#%%
# Elbow Method´
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, len(df.index))
  
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(df.iloc[:, 1:4])
    distortions.append(sum(np.min(cdist(df.iloc[:,1:4], kmeanModel.cluster_centers_, metric= 'euclidean'), axis=1))/len(df.index))
    inertias.append(kmeanModel.inertia_)
    
    mapping1[k] = sum(np.min(cdist(df.iloc[:,1:4], kmeanModel.cluster_centers_, metric= 'euclidean'), axis =1)/len(df.index))
    mapping2[k] = kmeanModel.inertia_
    
plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()

