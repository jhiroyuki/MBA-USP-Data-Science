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

# ANACOR Múltipla
import prince
from scipy.stats import chi2_contingency


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

# ANACOR Múltipla
# 01° Passo: Tabelas de contingência
# 02° Passo: Número de dimensões (número de categorias - número de variáveis)
# 03° Passo: ANACOR pelo Prince
# 04° Passo: Mapa perceptual



#%%
# Carregamento da base de dados (Preco Casas)
df = pd.read_excel('datasets/Preco Casas.xlsx') 

#%%
# Visualização da base de dados
print('################### DATAFRAME ####################')
print(df)

#%%
# Tipos das variáveis
print('################### DATAFRAME TYPES ####################')
print(df.dtypes)

#%%
# Algumas variáveis são qualitativas e outras são quantitativas
# Vamos separar o banco de dados em 2 partes (somente quali e quanti)

var_quali = df[['large_living_room', 'parking_space', 'front_garden', 'swimming_pool', 'wall_fence', 'water_front', 'room_size_class']]
var_quanti = df[['land_size_sqm', 'house_size_sqm', 'no_of_rooms', 'no_of_bathrooms', 'distance_to_school', 'house_age', 'distance_to_supermarket_km', 'crime_rate_index']]

## Nota: vamos deixar a variável "valor da casa" fora da análise por enquanto
## O objetivo é criar um ranking que capture os valores das casas
#%%
print('############ ESTATÍSTICAS DESCRITIVAS QUALI ############')
print(var_quali.describe())

print('############ ESTATÍSTICAS DESCRITIVAS QUANTI ############')
print(var_quanti.describe())
#%%

# INÍCIO ANACOR PARA VARIÁVEIS QUALITATIVAS

#%%
# 01° Passo: Tabelas de contingência
def contingencia_chi(df, column1, column2):
    # Tabela de contingência com frequências absolutas observadas

    tabela_contingencia = pd.crosstab(index=df[column1], columns=df[column2])
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
df  = var_quali.copy()
keys = df.keys()
for i in range(len(keys)):
    for j in range(len(keys)):
        column1 = keys[i]
        column2 = keys[j]
        if j>i:
            print(column1 + ' x ' + column2)
            contingencia_chi(df, column1, column2)
            print('\n')
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

#%%

# Adicionando as coordenadas ao banco de dados de variáveis quantitativas
coord_obs = mca.row_coordinates(df)
var_quanti['Axis1'] = coord_obs.iloc[:, 0]
var_quanti['Axis2'] = coord_obs.iloc[:, 1]


#%%

# FIM ANACOR PARA VARIÁVEIS QUALITATIVAS

#%%

# INÍCIO PCA PARA VARIÁVEIS QUANTITATIVAS

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


df =  var_quanti.copy()
corr_coef, corr_sig = pearson(df, df)
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

bartlett_chisq, bartlett_degrees, bartlett_pvalue = bartlett_sphericity(df, 'pearson')

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
df_zscore = zscore(df, ddof = 1)
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

cargas_fatoriais, _ = pearson( df, fatores)
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
        

