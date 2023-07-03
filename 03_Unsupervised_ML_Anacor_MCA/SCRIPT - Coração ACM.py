import pandas as pd
import numpy as np
import prince
from scipy.stats import chi2_contingency
import math

#%%
# Carregamento da base de dados (dados_cor_acm)
df = pd.read_excel('datasets/dados_cor_acm.xlsx') 


#%%
# Visualização da base de dados
print('################### DATAFRAME ####################')
print(df)

#%%
# Tipos das variáveis
print('################### DATAFRAME TYPES ####################')
print(df.dtypes)



#%%

## Algumas variáveis são qualitativas e outras quantitativas

# Vamos categorizar as variáveis quanti (por critério estatístico)
# Vamos remover as variáveis que não utilizaremos (quantitativas)
to_categorize = ['Idade', 'PS_Descanso', 'Colesterol','BC_Max']
for variable in to_categorize:
    values = df[variable].values
    categorizado = pd.cut(df[variable], bins=[-math.inf, np.quantile(values, 0.25),np.quantile(values, 0.75), math.inf], include_lowest=True,labels=[variable + '_baixa',variable + '_média', variable + '_alta']).to_frame()
    categorizado = categorizado.rename(columns={variable: str('Categ_') + variable}  )
    df.insert(len(df.keys()), str('Categ_') + variable,categorizado.values)
    df.drop(variable, inplace=True, axis=1)
# #%%
# for variable in df.keys():
#     df[variable] = df[variable].astype('string')
    
# df.dtypes
#%%
# Tabelas de contingência
# Doença_Card x Sexo

tabela_contingencia = pd.crosstab(index=df['Doença_Card'], columns=df['Sexo'])
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
# Doença_Card x Tipo_Dor_Peito

tabela_contingencia = pd.crosstab(index=df['Doença_Card'], columns=df['Tipo_Dor_Peito'])
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
# Doença_Card x Açucar_Sangue

tabela_contingencia = pd.crosstab(index=df['Doença_Card'], columns=df['Açucar_Sangue'])
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
# Doença_Card x ECG_Descanso

tabela_contingencia = pd.crosstab(index=df['Doença_Card'], columns=df['ECG_Descanso'])
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
# Doença_Card x Angina_Exerc

tabela_contingencia = pd.crosstab(index=df['Doença_Card'], columns=df['Angina_Exerc'])
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
# Doença_Card x Categ_Idade

tabela_contingencia = pd.crosstab(index=df['Doença_Card'], columns=df['Categ_Idade'])
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
# Doença_Card x Categ_PS_Descanso

tabela_contingencia = pd.crosstab(index=df['Doença_Card'], columns=df['Categ_PS_Descanso'])
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
# Doença_Card x Categ_Colesterol

tabela_contingencia = pd.crosstab(index=df['Doença_Card'], columns=df['Categ_Colesterol'])
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
# Doença_Card x Categ_BC_Max

tabela_contingencia = pd.crosstab(index=df['Doença_Card'], columns=df['Categ_BC_Max'])
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
#%%
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
# Este plot precisa da versão nova do prince, mas essa versão nova da problema para a celula acima
ax = mca.plot_coordinates(df, ax=None, figsize=(6, 6),show_row_points=True,
                             show_column_points=False, row_points_size=30, show_column_labels=False, legend_n_cols=1, show_row_labels=False, row_groups = list(df['Doença_Card']))
ax.get_figure()
