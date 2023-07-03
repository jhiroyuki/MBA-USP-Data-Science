import pandas as pd
import numpy as np
import prince
from scipy.stats import chi2_contingency
import math

#%%
# Carregamento da base de dados (estudantes_adapta)
df = pd.read_csv('datasets/estudantes_adapta.csv') 


#%%
# Visualização da base de dados
print('################### DATAFRAME ####################')
print(df)

#%%
# Tipos das variáveis
print('################### DATAFRAME TYPES ####################')
print(df.dtypes)


#%%
# Tabelas de contingência
# Adaptivity x Education

tabela_contingencia = pd.crosstab(index=df['Adaptivity'], columns=df['Education'])
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
# Adaptivity x Institution

tabela_contingencia = pd.crosstab(index=df['Adaptivity'], columns=df['Institution'])
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
# Adaptivity x Financial

tabela_contingencia = pd.crosstab(index=df['Adaptivity'], columns=df['Financial'])
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
# Adaptivity x Financial

tabela_contingencia = pd.crosstab(index=df['Adaptivity'], columns=df['Internet'])
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
# Institution x Financial

tabela_contingencia = pd.crosstab(index=df['Institution'], columns=df['Internet'])
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
# Institution x Education

tabela_contingencia = pd.crosstab(index=df['Institution'], columns=df['Education'])
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
# Institution x Internet

tabela_contingencia = pd.crosstab(index=df['Institution'], columns=df['Internet'])
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
# Education x Financial

tabela_contingencia = pd.crosstab(index=df['Education'], columns=df['Financial'])
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
# Education x Internet

tabela_contingencia = pd.crosstab(index=df['Education'], columns=df['Internet'])
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

# Análise de correspondência múltipla feita pelo "prince". Os valores das coordenadas são diferentes dos encontrados no R.

# Primeiro encontrar a quantidade de dimensões, isto é a quantidade de categorias menos a quantidade de variáveis
numero_categorias = 0
for variable in df.keys():
    numero_categorias = numero_categorias + df[variable].nunique()
    
numero_variaveis = len(df.keys())

numero_dimensoes = numero_categorias - numero_variaveis

#%%
mca = prince.MCA(n_components=numero_dimensoes, n_iter=3, copy=True, check_input=True,engine='auto', random_state=42)
mca = mca.fit(df)

ax = mca.plot_coordinates(df, ax=None, figsize=(6, 6),show_row_points=False,
                             show_column_points=True, column_points_size=30, show_column_labels=True, legend_n_cols=2)


ax.legend(bbox_to_anchor=(1.3, 0.7))
ax.get_figure()

print(mca.row_coordinates(df))
print(mca.column_coordinates(df))


print(mca.eigenvalues_)
print(mca.total_inertia_)
print(mca.explained_inertia_)
