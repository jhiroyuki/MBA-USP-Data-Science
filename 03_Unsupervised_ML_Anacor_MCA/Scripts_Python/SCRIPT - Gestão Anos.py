import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyreadr
import prince
from scipy.stats import chi2_contingency
from scipy.stats.contingency import margins
import seaborn as sns
#%%
# Carregamento da base de dados (gestao_municipal)
result = pyreadr.read_r('gestao_municipal.RData') 
print(result.keys()) # Checando o nome do DataFrame

#%%
# Carregando o DataFrame
df = result["gestao_municipal"]
df = df.rename(columns={'avaliaÃ§Ã£o': 'avaliação'})

#%%
# Tipos das variáveis
print('################### DATAFRAME TYPES ####################')
print(df.dtypes)


#%%
# Estatísticas descritivas
summary = df.describe(include='all')
print('############ ESTATÍSTICAS DESCRITIVAS ############')
print(summary)

#%%
# Tabela de contingência com frequências absolutas observadas
tabela_contingencia = pd.crosstab(index=df['avaliação'], columns=df['ano'])
print('####### CONTINGENCY TABLE OBSERVED ##########')
print(tabela_contingencia)

# Tamanho da amostra
n = tabela_contingencia.sum().sum()
print(n)

#%%
# Estatística chi quadrado e teste
# Se o p valor é menor que 0.05 então se rejeita  hipótese nula, isto é, que as variáveis não tem associação estatisticamente relevantes.
chi2 = chi2_contingency(tabela_contingencia)
stat, pvalue, dof, expected = chi2 # Estatística do teste, p valor, degrees of freedom, tabela de frequências esperadas
expected = pd.DataFrame(expected, 
                        index = tabela_contingencia.index.values, 
                        columns = tabela_contingencia.columns.values)
print('####### CHI SQUARED TEST ##########')
teste_chi2 = pd.DataFrame({'stat': [stat], 'pvalues':[pvalue], 'dof': [dof]}, index = ['values'])
print(teste_chi2)

#%%
# Resíduos – diferenças entre frequências absolutas observadas e esperadas
print('####### RESIDUES ############')
print(tabela_contingencia  -expected)

# Valores de qui-quadrado por célula
print('####### CHI SQUARED FOR EACH CELL ############')
print((tabela_contingencia  -expected)**2/expected)


# Resíduos padronizados
std_residuals = (tabela_contingencia - expected) / np.sqrt(expected)
print('####### STANDARDISED RESIDUALS ############')
print(std_residuals)

# Resíduos padronizados ajustados

n = tabela_contingencia.sum().sum()
rsum, csum = margins(tabela_contingencia)
rsum = rsum.astype(np.float64)
csum = csum.astype(np.float64)
v = csum * rsum * (n - rsum) * (n - csum) / n**3
adjusted_residuals = (tabela_contingencia - expected) / np.sqrt(v)


print('####### ADJUSTED STANDARDISED RESIDUALS ############')
print(adjusted_residuals)

#%%
# Mapa de calor dos resíduos padronizados ajustados  
# Não é tão necessaŕio assim, o mais interessante aqui é conseguir ajustar o midpoint do colormap pra conseguir vizualisar os valores maiores que 1.96, em que há associação significativa entre as categorias que interagem na célula.
# A referência de 1.96 é o valorcrítico da normal padrão para o nível de significância de 5%
fig = plt.subplots(figsize=(10,10)) 
ax = sns.heatmap(
    adjusted_residuals, 
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
## 2ª Parte: Análise da associação por meio do mapa perceptual

#%%
# Definição da matriz A
# Resíduos padronizados (qui2$residuals) divididos pela raiz quadrada do tamanho da amostra (n)

matrizA = std_residuals/np.sqrt(n)
matrizA = matrizA.values
print('####### MATRIZ A ############')
print(matrizA)

#%%
# Definição da matriz W
# Multiplicação da matriz A transposta pela matriz A

matrizW = np.matmul(matrizA.T, matrizA)
print('####### MATRIZ W ############')
print(matrizW)

#%%

# Definição da quantidade de dimensões
nrow = len(matrizW)
ncol = len(matrizW[0])
qtde_dimensoes = min(nrow - 1, ncol - 1)
print('#### NUMBER OF DIMENSIONS ####')
print(qtde_dimensoes)

# Singular Value Decomposition
u, valores_singulares, vt = np.linalg.svd(matrizA)
v = vt.T
idx = valores_singulares.argsort()[::-1] # Arrumar valores singulares de maneira decrescente
valores_singulares = valores_singulares[idx]
u = u[:,idx]
v = v[:,idx] 


# Definição dos valores singulares
print('#### SINGULAR VALUES ####')
print(valores_singulares)

# Autovalores
eigenvalues = valores_singulares**2
print('#### EIGENVALUES####')
print(eigenvalues)

# Valores singulares de cada dimensão
eigenvalues_truncated = eigenvalues[:qtde_dimensoes]
singular_truncated = valores_singulares[:qtde_dimensoes]


#%%
# Cálculo da inércia principal total (a partir do qui-quadrado)
# Esse valor é a soma dos autovalores
inercia_total = stat/n
print('#### TOTAL INERTIA ####')
print(inercia_total)


# Cálculo da variância explicada em cada dimensão
# Esse o valor da porcentagem dos autovalores
variancia_explicada = eigenvalues_truncated / inercia_total
print('#### EXPLAINED VARIATION ####')
print(variancia_explicada)

# Cálculo das massas das colunas (column profiles)
soma_colunas = np.sum(tabela_contingencia, axis = 1)
massa_colunas = soma_colunas / n
print('#### COLUMN MASSES ####')
print(massa_colunas)


# Cálculo das massas das linhas (row profiles)
soma_linhas  = np.sum(tabela_contingencia, axis = 0)
massa_linhas = soma_linhas / n
print('#### ROW MASSES ####')
print(massa_linhas)

#%%
# Autovetores u e v das dimensões
autovetor_u = u[:,:qtde_dimensoes]
autovetor_v = v[:,:qtde_dimensoes]

print('#### EIGENVECTORS U ####')
print(autovetor_u)

print('#### EIGENVECTORS V ####')
print(autovetor_v)


#%%
# Calculando as coordenadas para plotar as categorias no mapa perceptual

# Variável em linha na tabela de contingência ('perfil')
# Coordenadas das abcissas
coord_abcissas_perfil = np.sqrt(valores_singulares[0]) * (massa_colunas**(-0.5)) * autovetor_u[:,0]
# Coordenadas das ordenadas
coord_ordenadas_perfil = np.sqrt(valores_singulares[1]) * (massa_colunas**(-0.5)) * autovetor_u[:,1]

# Variável em coluna na tabela de contingência ('aplicacao')
# Coordenadas das abcissas
coord_abcissas_aplicacao = np.sqrt(valores_singulares[0]) * (massa_linhas**(-0.5)) * autovetor_v[:,0]
# Coordenadas das ordenadas
coord_ordenadas_aplicacao = np.sqrt(valores_singulares[1]) * (massa_linhas**(-0.5)) * autovetor_v[:,1]


coord_perfil = pd.DataFrame({'x': coord_abcissas_perfil.values, 'y' : coord_ordenadas_perfil.values},
                            index = coord_abcissas_perfil.index.values)
print('#### COORDINATES OF CATEGORIES OF VARIABLE "PERFIL" ####')
print(coord_perfil)
1

coord_aplicacao = pd.DataFrame({'x': coord_abcissas_aplicacao.values, 'y' : coord_ordenadas_aplicacao.values},
                            index = coord_abcissas_aplicacao.index.values)
print('#### COORDINATES OF CATEGORIES OF VARIABLE "APLICACAO" ####')
print(coord_aplicacao)

#%%
# Mapa perceptual

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(coord_abcissas_perfil, 
           coord_ordenadas_perfil,
           color = 'black')
ax.scatter(coord_abcissas_aplicacao, 
           coord_ordenadas_aplicacao,
           color = 'red')

ax.axhline(0, ls='--', color = 'grey')
ax.axvline(0, ls='--', color = 'grey')
ax.set_xlabel('Dim 1 ('+  str(round(variancia_explicada[0]*100,2)) +'%)')
ax.set_ylabel('Dim 1 ('+  str(round(variancia_explicada[1]*100,2)) +'%)')
ax.grid(alpha = 0.3)


for i, txt in enumerate(coord_abcissas_aplicacao.index.values):
    ax.annotate(txt, (coord_abcissas_aplicacao[i], coord_ordenadas_aplicacao[i]))
    
for i, txt in enumerate(coord_abcissas_perfil.index.values):
    ax.annotate(txt, (coord_abcissas_perfil[i], coord_ordenadas_perfil[i]))
    
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
