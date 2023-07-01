import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyreadr
import prince
from scipy.stats import chi2_contingency
from scipy.stats.contingency import margins
import seaborn as sns
#%%
# Carregamento da base de dados (conceito_enade_2016 e conceito_enade_2021)
df1 = pd.read_excel('conceito_enade_2016.xlsx') 
df2 = pd.read_excel('conceito_enade_2021.xlsx') 


#%%
# Visualização da base de dados
print('################### DATAFRAME 1 ####################')
print(df1)

print('################### DATAFRAME 2 ####################')
print(df1)

#%%
# Tipos das variáveis
print('################### DATAFRAME TYPES 1 ####################')
print(df1.dtypes)

print('################### DATAFRAME TYPES 2 ####################')
print(df2.dtypes)

#%%
# Vamos excluir o ano e o id
for variable in ['Ano', 'Id_Curso']:
    df1.drop(variable, inplace=True, axis=1)
    df2.drop(variable, inplace=True, axis=1)


#%%
# Tabelas de contingência
# 2016
tabela_contingencia = pd.crosstab(index=df1['Categoria_Adm'], columns=df1['Conceito_Enade'])
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
# Tabelas de contingência
# 2021
tabela_contingencia = pd.crosstab(index=df2['Categoria_Adm'], columns=df2['Conceito_Enade'])
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

# Anacor para 2016

df = df1
tabela_contingencia = pd.crosstab(index=df['Categoria_Adm'], columns=df['Conceito_Enade'])

#%%
# Resíduos – diferenças entre frequências absolutas observadas e esperadas

# Resíduos padronizados
std_residuals = (tabela_contingencia - expected) / np.sqrt(expected)

# Resíduos padronizados ajustados

n = tabela_contingencia.sum().sum()
rsum, csum = margins(tabela_contingencia)
rsum = rsum.astype(np.float64)
csum = csum.astype(np.float64)
v = csum * rsum * (n - rsum) * (n - csum) / n**3
adjusted_residuals = (tabela_contingencia - expected) / np.sqrt(v)


#%%
# Definição da matriz A
# Resíduos padronizados (qui2$residuals) divididos pela raiz quadrada do tamanho da amostra (n)

matrizA = std_residuals/np.sqrt(n)
matrizA = matrizA.values

#%%
# Definição da matriz W
# Multiplicação da matriz A transposta pela matriz A

matrizW = np.matmul(matrizA.T, matrizA)


#%%

# Definição da quantidade de dimensões
nrow = len(matrizW)
ncol = len(matrizW[0])
qtde_dimensoes = min(nrow - 1, ncol - 1)


# Singular Value Decomposition
u, valores_singulares, vt = np.linalg.svd(matrizA)
v = vt.T
idx = valores_singulares.argsort()[::-1] # Arrumar valores singulares de maneira decrescente
valores_singulares = valores_singulares[idx]
u = u[:,idx]
v = v[:,idx] 


# Autovalores
eigenvalues = valores_singulares**2


# Valores singulares de cada dimensão
eigenvalues_truncated = eigenvalues[:qtde_dimensoes]
singular_truncated = valores_singulares[:qtde_dimensoes]


#%%
# Cálculo da inércia principal total (a partir do qui-quadrado)
# Esse valor é a soma dos autovalores
inercia_total = stat/n


# Cálculo da variância explicada em cada dimensão
# Esse o valor da porcentagem dos autovalores
variancia_explicada = eigenvalues_truncated / inercia_total


# Cálculo das massas das colunas (column profiles)
soma_colunas = np.sum(tabela_contingencia, axis = 1)
massa_colunas = soma_colunas / n



# Cálculo das massas das linhas (row profiles)
soma_linhas  = np.sum(tabela_contingencia, axis = 0)
massa_linhas = soma_linhas / n

#%%
# Autovetores u e v das dimensões
autovetor_u = u[:,:qtde_dimensoes]
autovetor_v = v[:,:qtde_dimensoes]


#%%
# Calculando as coordenadas para plotar as categorias no mapa perceptual

# Variável em linha na tabela de contingência 
# Coordenadas das abcissas
coord_abcissas_1 = np.sqrt(valores_singulares[0]) * (massa_colunas**(-0.5)) * autovetor_u[:,0]
# Coordenadas das ordenadas
coord_ordenadas_1 = np.sqrt(valores_singulares[1]) * (massa_colunas**(-0.5)) * autovetor_u[:,1]

# Variável em coluna na tabela de contingência 
# Coordenadas das abcissas
coord_abcissas_2 = np.sqrt(valores_singulares[0]) * (massa_linhas**(-0.5)) * autovetor_v[:,0]
# Coordenadas das ordenadas
coord_ordenadas_2 = np.sqrt(valores_singulares[1]) * (massa_linhas**(-0.5)) * autovetor_v[:,1]


coord_1 = pd.DataFrame({'x': coord_abcissas_1.values, 'y' : coord_ordenadas_1.values},
                            index = coord_abcissas_1.index.values)

coord_2 = pd.DataFrame({'x': coord_abcissas_2.values, 'y' : coord_ordenadas_2.values},
                            index = coord_abcissas_2.index.values)

#%%
# Mapa perceptual

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(coord_abcissas_1, 
           coord_ordenadas_1,
           color = 'black')
ax.scatter(coord_abcissas_2, 
           coord_ordenadas_2,
           color = 'red')

ax.axhline(0, ls='--', color = 'grey')
ax.axvline(0, ls='--', color = 'grey')
ax.set_xlabel('Dim 1 ('+  str(round(variancia_explicada[0]*100,2)) +'%)')
ax.set_ylabel('Dim 1 ('+  str(round(variancia_explicada[1]*100,2)) +'%)')
ax.grid(alpha = 0.3)


# for i, txt in enumerate(coord_abcissas_2.index.values):
#     ax.annotate(txt, (coord_abcissas_2[i], coord_ordenadas_2[i]))
    
for i, txt in enumerate(coord_abcissas_1.index.values):
    ax.annotate(txt, (coord_abcissas_1[i], coord_ordenadas_1[i]))
    
plt.show()

#%%

# A análise de correspondencia pode ser feito com o pacote "prince", mas o valores que ele gera para as coordenadas são diferentes do que fazemos antes, apesar do mapa perceptual ser qualitativamente idêntico.

ca = prince.CA(n_components=2, n_iter=10, copy=True, check_input=True, engine='auto', random_state=42)
ca = ca.fit(tabela_contingencia)
print(ca.row_coordinates(tabela_contingencia))
print(ca.column_coordinates(tabela_contingencia))

ax = ca.plot_coordinates(tabela_contingencia, ax=None, figsize=(6, 6), x_component=0, y_component=1, show_row_labels=True, show_col_labels=True )
ax.get_figure()

print(ca.eigenvalues_)
print(ca.total_inertia_)
print(ca.explained_inertia_)

#%%

# Anacor para 2021

df = df2
tabela_contingencia = pd.crosstab(index=df['Categoria_Adm'], columns=df['Conceito_Enade'])



#%%
# Resíduos – diferenças entre frequências absolutas observadas e esperadas

# Resíduos padronizados
std_residuals = (tabela_contingencia - expected) / np.sqrt(expected)

# Resíduos padronizados ajustados

n = tabela_contingencia.sum().sum()
rsum, csum = margins(tabela_contingencia)
rsum = rsum.astype(np.float64)
csum = csum.astype(np.float64)
v = csum * rsum * (n - rsum) * (n - csum) / n**3
adjusted_residuals = (tabela_contingencia - expected) / np.sqrt(v)


#%%
# Definição da matriz A
# Resíduos padronizados (qui2$residuals) divididos pela raiz quadrada do tamanho da amostra (n)

matrizA = std_residuals/np.sqrt(n)
matrizA = matrizA.values

#%%
# Definição da matriz W
# Multiplicação da matriz A transposta pela matriz A

matrizW = np.matmul(matrizA.T, matrizA)


#%%

# Definição da quantidade de dimensões
nrow = len(matrizW)
ncol = len(matrizW[0])
qtde_dimensoes = min(nrow - 1, ncol - 1)


# Singular Value Decomposition
u, valores_singulares, vt = np.linalg.svd(matrizA)
v = vt.T
idx = valores_singulares.argsort()[::-1] # Arrumar valores singulares de maneira decrescente
valores_singulares = valores_singulares[idx]
u = u[:,idx]
v = v[:,idx] 


# Autovalores
eigenvalues = valores_singulares**2


# Valores singulares de cada dimensão
eigenvalues_truncated = eigenvalues[:qtde_dimensoes]
singular_truncated = valores_singulares[:qtde_dimensoes]


#%%
# Cálculo da inércia principal total (a partir do qui-quadrado)
# Esse valor é a soma dos autovalores
inercia_total = stat/n


# Cálculo da variância explicada em cada dimensão
# Esse o valor da porcentagem dos autovalores
variancia_explicada = eigenvalues_truncated / inercia_total


# Cálculo das massas das colunas (column profiles)
soma_colunas = np.sum(tabela_contingencia, axis = 1)
massa_colunas = soma_colunas / n



# Cálculo das massas das linhas (row profiles)
soma_linhas  = np.sum(tabela_contingencia, axis = 0)
massa_linhas = soma_linhas / n

#%%
# Autovetores u e v das dimensões
autovetor_u = u[:,:qtde_dimensoes]
autovetor_v = v[:,:qtde_dimensoes]


#%%
# Calculando as coordenadas para plotar as categorias no mapa perceptual

# Variável em linha na tabela de contingência 
# Coordenadas das abcissas
coord_abcissas_1 = np.sqrt(valores_singulares[0]) * (massa_colunas**(-0.5)) * autovetor_u[:,0]
# Coordenadas das ordenadas
coord_ordenadas_1 = np.sqrt(valores_singulares[1]) * (massa_colunas**(-0.5)) * autovetor_u[:,1]

# Variável em coluna na tabela de contingência 
# Coordenadas das abcissas
coord_abcissas_2 = np.sqrt(valores_singulares[0]) * (massa_linhas**(-0.5)) * autovetor_v[:,0]
# Coordenadas das ordenadas
coord_ordenadas_2 = np.sqrt(valores_singulares[1]) * (massa_linhas**(-0.5)) * autovetor_v[:,1]


coord_1 = pd.DataFrame({'x': coord_abcissas_1.values, 'y' : coord_ordenadas_1.values},
                            index = coord_abcissas_1.index.values)

coord_2 = pd.DataFrame({'x': coord_abcissas_2.values, 'y' : coord_ordenadas_2.values},
                            index = coord_abcissas_2.index.values)

#%%
# Mapa perceptual

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(coord_abcissas_1, 
           coord_ordenadas_1,
           color = 'black')
ax.scatter(coord_abcissas_2, 
           coord_ordenadas_2,
           color = 'red')

ax.axhline(0, ls='--', color = 'grey')
ax.axvline(0, ls='--', color = 'grey')
ax.set_xlabel('Dim 1 ('+  str(round(variancia_explicada[0]*100,2)) +'%)')
ax.set_ylabel('Dim 1 ('+  str(round(variancia_explicada[1]*100,2)) +'%)')
ax.grid(alpha = 0.3)


# for i, txt in enumerate(coord_abcissas_2.index.values):
#     ax.annotate(txt, (coord_abcissas_2[i], coord_ordenadas_2[i]))
    
for i, txt in enumerate(coord_abcissas_1.index.values):
    ax.annotate(txt, (coord_abcissas_1[i], coord_ordenadas_1[i]))
    
plt.show()

#%%

# A análise de correspondencia pode ser feito com o pacote "prince", mas o valores que ele gera para as coordenadas são diferentes do que fazemos antes, apesar do mapa perceptual ser qualitativamente idêntico.

ca = prince.CA(n_components=2, n_iter=10, copy=True, check_input=True, engine='auto', random_state=42)
ca = ca.fit(tabela_contingencia)
print(ca.row_coordinates(tabela_contingencia))
print(ca.column_coordinates(tabela_contingencia))

ax = ca.plot_coordinates(tabela_contingencia, ax=None, figsize=(6, 6), x_component=0, y_component=1, show_row_labels=True, show_col_labels=True )
ax.get_figure()

print(ca.eigenvalues_)
print(ca.total_inertia_)
print(ca.explained_inertia_)