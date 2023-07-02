# MBA DATA SCIENCE & ANALYTICS USP/Esalq
# SUPERVISED MACHINE LEARNING: MODELAGEM MULTINÍVEL
# Prof. Dr. Luiz Paulo Fávero

#!/usr/bin/env python
# coding: utf-8


# In[ ]: Importação dos pacotes necessários

import pandas as pd #manipulação de dados em formato de dataframe
import seaborn as sns #biblioteca de visualização de informações estatísticas
import matplotlib.pyplot as plt #biblioteca de visualização de dados
import statsmodels.api as sm #biblioteca de modelagem estatística
from scipy import stats #estatística chi2
# !pip install -q pymer4
from pymer4.models import Lmer #estimação de modelos HLM3 neste código
from statstests.process import stepwise #procedimento Stepwise


#%%
##############################################################################
##############################################################################
#                ESTIMAÇÃO DE MODELOS HIERÁRQUICOS LINEARES                  #
#                    DE DOIS NÍVEIS COM DADOS AGRUPADOS                      #
##############################################################################
##############################################################################

##############################################################################
#        DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'desempenho_aluno_escola'         #
##############################################################################

# Carregamento da base de dados 'desempenho_aluno_escola'
df_aluno_escola = pd.read_csv('desempenho_aluno_escola.csv', delimiter=',')

# Visualização da base de dados 'desempenho_aluno_escola'
df_aluno_escola

# Atribuição de categorias para as variáveis 'estudante' e 'escola'
df_aluno_escola['estudante'] = df_aluno_escola['estudante'].astype('category')
df_aluno_escola['escola'] = df_aluno_escola['escola'].astype('category')

# Características das variáveis do dataset
df_aluno_escola.info()

# Estatísticas univariadas
df_aluno_escola.describe()

#%%
# Estudo sobre o desbalanceamento dos dados por escola
df_aluno_escola.groupby('escola')['estudante'].count().reset_index()

#%%
# Desempenho médio dos estudantes por escola
desempenho_medio = df_aluno_escola.groupby('escola')['desempenho'].mean().reset_index()
desempenho_medio

#%%
# Gráfico do desempenho escolar médio dos estudantes por escola

plt.figure(figsize=(15,10))
plt.plot(desempenho_medio['escola'], desempenho_medio['desempenho'],
         linewidth=5, color='indigo')
plt.scatter(df_aluno_escola['escola'], df_aluno_escola['desempenho'],
            alpha=0.5, color='orange', s = 40)
plt.xlabel('Escola j (nível 2)', fontsize=14)
plt.ylabel('Desempenho Escolar', fontsize=14)
plt.xticks(desempenho_medio.escola)
plt.show()

#%%
# Boxplot da variável dependente ('desempenho')

plt.figure(figsize=(15,10))
sns.boxplot(data=df_aluno_escola, y='desempenho',
            linewidth=2, orient='v', color='deepskyblue')
sns.stripplot(data=df_aluno_escola, y='desempenho',
              color='orange', jitter=0.1, size=8, alpha=0.5)
plt.ylabel('Desempenho', fontsize=16)
plt.show()

#%%
# Kernel density estimation (KDE) - função densidade de probabilidade da
#variável dependente ('desempenho'), com histograma

plt.figure(figsize=(15,10))
sns.histplot(data=df_aluno_escola['desempenho'], kde=True,
             bins=30, color='deepskyblue')
plt.xlabel('Desempenho', fontsize=17)
plt.ylabel('Frequência', fontsize=17)
plt.show()

#%%
# Boxplot da variável dependente ('desempenho') por escola

plt.figure(figsize=(15,10))
sns.boxplot(data=df_aluno_escola, x='escola', y='desempenho',
            linewidth=2, orient='v', palette='viridis')
sns.stripplot(data=df_aluno_escola, x='escola', y='desempenho',
              palette='viridis', jitter=0.01, size=8, alpha=0.5)
plt.ylabel('Desempenho', fontsize=16)
plt.xlabel('escola', fontsize=16)
plt.show()

#%%
# Kernel density estimation (KDE) - função densidade de probabilidade da
#variável dependente ('desempenho') por escola

plt.figure(figsize=(15,10))
sns.pairplot(df_aluno_escola[['escola','desempenho']], hue='escola', height=8,
             aspect=2, palette='viridis')
plt.xlabel('Desempenho', fontsize=17)
plt.ylabel('Frequência', fontsize=17)
plt.show()

#%%
# Kernel density estimation (KDE) - função densidade de probabilidade da
#variável dependente ('desempenho'), com histograma e por escola separadamente
#(função 'FacetGrid' do pacote 'seaborn')

g = sns.FacetGrid(df_aluno_escola, col='escola', col_wrap=8, hue='escola')
g.map_dataframe(sns.histplot, kde=True, x='desempenho')
plt.show()

#%%
# Gráfico de desempenho x horas (MQO)

plt.figure(figsize=(15,10))
sns.lmplot(x='horas', y='desempenho', data=df_aluno_escola, ci=False, height=8)
plt.ylabel('Desempenho escolar', fontsize=14)
plt.xlabel('Quantidade semanal de horas de estudo do aluno', fontsize=14)
plt.show()

#%%
# Gráfico de desempenho escolar em função da variável 'horas'
# Variação entre estudantes de uma mesma escola e entre escolas diferentes
# Visualização do contexto!
# NOTE QUE A PERSPECTIVA MULTINÍVEL NATURALMENTE CONSIDERA O COMPORTAMENTO
#HETEROCEDÁSTICO NOS DADOS!

sns.lmplot(x='horas', y='desempenho', data=df_aluno_escola, hue='escola',
           ci=False, height=12, palette='viridis')
plt.ylabel('Desempenho escolar', fontsize=14)
plt.xlabel('Quantidade semanal de horas de estudo do aluno',fontsize=14)
plt.show()


#%%
##############################################################################
#                        ESTIMAÇÃO DO MODELO NULO HLM2                       #
##############################################################################

# Estimação do modelo nulo (função 'MixedLM' do pacote 'statsmodels')
modelo_nulo_hlm2 = sm.MixedLM.from_formula(formula='desempenho ~ 1',
                                           groups='escola',
                                           re_formula='1',
                                           data=df_aluno_escola).fit()

# Parâmetros do 'modelo_nulo_hlm2'
modelo_nulo_hlm2.summary()


#%%
##############################################################################
#                   COMPARAÇÃO DO HLM2 NULO COM UM MQO NULO                  #
##############################################################################

# Estimação de um modelo MQO nulo
modelo_ols_nulo = sm.OLS.from_formula(formula='desempenho ~ 1',
                                      data=df_aluno_escola).fit()

# Parâmetros do 'modelo_ols_nulo'
modelo_ols_nulo.summary()

#%%
# Teste de razão de verossimilhança entre o 'modelo_nulo_hlm2' e o 'modelo_ols_nulo'

# Definição da função 'lrtest'
def lrtest(modelos):
    modelo_1 = modelos[0]
    llk_1 = modelo_1.llf
    llk_2 = modelos[1].llf
    LR_statistic = -2*(llk_1-llk_2)
    p_val = stats.chi2.sf(LR_statistic, 1)
    return round(LR_statistic,2), round(p_val,2)

# Teste de razão de verossimilhança propriamente dito
lrtest([modelo_ols_nulo, modelo_nulo_hlm2])

#%%
# Gráfico para comparação visual dos logLiks dos modelos estimados até o momento

df_llf = pd.DataFrame({'modelo':['MQO Nulo','HLM2 Nulo'],
                      'loglik':[modelo_ols_nulo.llf,modelo_nulo_hlm2.llf]})

fig, ax = plt.subplots(figsize=(15,10))

c = ['black','grey']

ax1 = ax.barh(df_llf.modelo,df_llf.loglik, color = c)
ax.bar_label(ax1, label_type='center', color='white', fontsize=22)
ax.set_ylabel("Modelo Proposto", fontsize= 17)
ax.set_xlabel("LogLik", fontsize=17)
plt.show()

#%%
##############################################################################
#           ESTIMAÇÃO DO MODELO COM INTERCEPTOS ALEATÓRIOS HLM2              #
##############################################################################

# Estimação do modelo com interceptos aleatórios
modelo_intercept_hlm2 = sm.MixedLM.from_formula(formula='desempenho ~ horas',
                                                groups='escola',
                                                re_formula='1',
                                                data=df_aluno_escola).fit()

# Parâmetros do 'modelo_intercept_hlm2'
modelo_intercept_hlm2.summary()

#%%
# Gráfico para comparação visual dos logLiks dos modelos estimados até o momento

df_llf = pd.DataFrame({'modelo':['MQO Nulo','HLM2 Nulo',
                                 'HLM2 com Interceptos Aleatórios'],
                      'loglik':[modelo_ols_nulo.llf,modelo_nulo_hlm2.llf,
                                modelo_intercept_hlm2.llf]})

fig, ax = plt.subplots(figsize=(15,10))

c = ['black','grey','saddlebrown']

ax1 = ax.barh(df_llf.modelo,df_llf.loglik, color = c)
ax.bar_label(ax1, label_type='center', color='white', fontsize=22)
ax.set_ylabel("Modelo Proposto", fontsize=17)
ax.set_xlabel("LogLik", fontsize=17)
plt.show()

#%%
##############################################################################
#     ESTIMAÇÃO DO MODELO COM INTERCEPTOS E INCLINAÇÕES ALEATÓRIAS HLM2      #
##############################################################################

# Estimação do modelo com interceptos e inclinações aleatórias
modelo_intercept_inclin_hlm2 = sm.MixedLM.from_formula(formula='desempenho ~ horas',
                                                       groups='escola',
                                                       re_formula='horas',
                                                       data=df_aluno_escola).fit()

# Parâmetros do 'modelo_intercept_inclin_hlm2'
modelo_intercept_inclin_hlm2.summary()

#%%
# Gráfico para comparação visual dos logLiks dos modelos estimados até o momento

df_llf = pd.DataFrame({'modelo':['MQO Nulo','HLM2 Nulo',
                                 'HLM2 com Interceptos Aleatórios',
                                 'HLM2 com Interceptos e Inclinações Aleatórias'],
                      'loglik':[modelo_ols_nulo.llf,modelo_nulo_hlm2.llf,
                                modelo_intercept_hlm2.llf,
                                modelo_intercept_inclin_hlm2.llf]})

fig, ax = plt.subplots(figsize=(15,10))

c = ['black','grey','saddlebrown','chocolate']

ax1 = ax.barh(df_llf.modelo,df_llf.loglik, color = c)
ax.bar_label(ax1, label_type='center', color='white', fontsize=22)
ax.set_ylabel("Modelo Proposto", fontsize=17)
ax.set_xlabel("LogLik", fontsize=17)
plt.show()

#%%
# Teste de razão de verossimilhança para comparar as estimações dos modelos
#com interceptos aleatórios e com interceptos e inclinações aleatórias
#(função 'lrtest' definida anteriormente)
lrtest([modelo_intercept_hlm2, modelo_intercept_inclin_hlm2])

#%%
##############################################################################
#         ESTIMAÇÃO DO MODELO FINAL COM INTERCEPTOS ALEATÓRIOS HLM2          #
##############################################################################

# Estimação do modelo final com interceptos aleatórios
modelo_final_hlm2 = sm.MixedLM.from_formula(formula='desempenho ~ horas + texp +\
                                            horas:texp',
                                            groups='escola',
                                            re_formula='horas',
                                            data=df_aluno_escola).fit()

# Parâmetros do modelo 'modelo_final_hlm2'
modelo_final_hlm2.summary()

#%%
# Gráfico para comparação visual dos logLiks dos modelos estimados até o momento

df_llf = pd.DataFrame({'modelo':['MQO Nulo','HLM2 Nulo',
                                 'HLM2 com Interceptos Aleatórios',
                                 'HLM2 com Interceptos e Inclinações Aleatórias',
                                 'HLM2 Modelo Final'],
                      'loglik':[modelo_ols_nulo.llf,modelo_nulo_hlm2.llf,
                                modelo_intercept_hlm2.llf,
                                modelo_intercept_inclin_hlm2.llf,
                                modelo_final_hlm2.llf]})

fig, ax = plt.subplots(figsize=(15,10))

c = ['black','grey','saddlebrown','chocolate','deepskyblue']

ax1 = ax.barh(df_llf.modelo,df_llf.loglik, color = c)
ax.bar_label(ax1, label_type='center', color='white', fontsize=22)
ax.set_ylabel("Modelo Proposto", fontsize=17)
ax.set_xlabel("LogLik", fontsize=17)
plt.show()

#%%
# Visualização dos interceptos aleatórios por escola, para o
#'modelo_final_completo_hlm2'

pd.DataFrame(modelo_final_hlm2.random_effects)

#%%
# Melhor visualização dos interceptos aleatórios por escola, para o
#'modelo_final_completo_hlm2'

efeitos_aleatorios = pd.DataFrame(modelo_final_hlm2.random_effects).T
efeitos_aleatorios = efeitos_aleatorios.rename(columns = {'escola':'u0j'})
efeitos_aleatorios = efeitos_aleatorios.reset_index().rename(columns={'index': 'escola'})
efeitos_aleatorios

#%%
# Gráfico para visualização do comportamento dos valores de u0j, ou seja,
#dos interceptos aleatórios por escola

colors = ['limegreen' if x>0 else 'red' for x in efeitos_aleatorios['u0j']]

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(0, point['y'], str(round(point['x'],2)),fontsize=10)

plt.figure(figsize=(15,10))
plt.barh(efeitos_aleatorios['escola'], efeitos_aleatorios['u0j'], color=colors)

label_point(x = efeitos_aleatorios['u0j'],
            y = efeitos_aleatorios['escola'],
            val = efeitos_aleatorios['u0j'],
            ax = plt.gca()) 
plt.ylabel('Escola', fontsize=14)
plt.xlabel('u0j', fontsize=14)
plt.show()

#%%
# Visualização dos fitted values do 'modelo_final_hlm2', por estudante
#e por escola

fixed = modelo_final_hlm2.predict(df_aluno_escola.drop(columns=['desempenho']))

df_aluno_escola['predict.fixed'] = fixed
df_aluno_escola['predict.escola'] = modelo_final_hlm2.fittedvalues
df_aluno_escola['rij'] = modelo_final_hlm2.resid
df_aluno_escola

#%%
# Elaboração de previsões para o 'modelo_final_completo_hlm2':
# Exemplo: Quais os valores previstos de desempenho escolar, para dado
#aluno que estuda na escola "1", sabendo-se que estuda 11 horas por semana e
#que a escola oferece tempo médio de experiência de seus professores igual a
#3,6 anos?

modelo_final_hlm2.predict(pd.DataFrame({'horas':[11],
                                        'texp':[3.6],
                                        'escola':['1']}))

# O resultado obtido por meio da função 'predict' só considera efeitos fixos.
# Criação do objeto 'resultado_fixo' apenas com o efeito fixo
resultado_fixo = modelo_final_hlm2.predict(pd.DataFrame({'horas':[11],
                                                         'texp':[3.6],
                                                         'escola':['1']}))

# A função 'predict' não considera os efeitos aleatórios de intercepto por
#'escola'. Neste sentido, precisamos adicioná-los a partir dos parâmetros do
#'modelo_final_hlm2', conforme segue.

#%%
# Efeitos aleatórios de intercepto por 'escola'
pd.DataFrame(modelo_final_hlm2.random_effects)

#%%
# Criação do objeto 'resultado_aleat_escola1' com o efeito aleatório de intercepto
#da escola 1
resultado_aleat_escola1 = modelo_final_hlm2.random_effects[1]
resultado_aleat_escola1

#%%
# Predição completa para o enunciado anterior, com efeitos fixos e aleatórios
#para a escola 1
resultado_total = resultado_fixo.iloc[0] + resultado_aleat_escola1.iloc[0]
resultado_total

#%%
# Gráfico com valores previstos do desempenho escolar em função da variável 'horas'
#para o 'modelo_final_completo_hlm2'

plt.figure(figsize=(15,10))
escolas = df_aluno_escola.escola.unique()
[plt.plot(df_aluno_escola[df_aluno_escola['escola']==escola].horas,
          df_aluno_escola[df_aluno_escola['escola']==escola]['predict.escola']) for escola in escolas]
[plt.scatter(df_aluno_escola[df_aluno_escola['escola']==escola].horas,
             df_aluno_escola[df_aluno_escola['escola']==escola]['predict.escola']) for escola in escolas]
plt.ylabel('Desempenho Escolar (Fitted Values)', fontsize=17)
plt.xlabel('Quantidade Semanal de Horas de Estudo do Aluno', fontsize=17)
plt.show()


#%%
##############################################################################
##############################################################################
#                ESTIMAÇÃO DE MODELOS HIERÁRQUICOS LINEARES                  #
#                   DE TRÊS NÍVEIS COM MEDIDAS REPETIDAS                     #
##############################################################################
##############################################################################

##############################################################################
#     DESCRIÇÃO E EXPLORAÇÃO DO DATASET 'desempenho_tempo_aluno_escola'      #
##############################################################################

# Carregamento da base de dados 'desempenho_tempo_aluno_escola'
df_tempo_aluno_escola = pd.read_csv('desempenho_tempo_aluno_escola.csv',
                                    delimiter=',')

# Visualização da base de dados 'desempenho_tempo_aluno_escola'
df_tempo_aluno_escola

# Atribuição de categorias para as variáveis 'estudante' e 'escola'
df_tempo_aluno_escola['estudante'] = df_tempo_aluno_escola['estudante'].astype('category')
df_tempo_aluno_escola['escola'] = df_tempo_aluno_escola['escola'].astype('category')

# Características das variáveis do dataset
df_tempo_aluno_escola.info()

# Estatísticas univariadas
df_tempo_aluno_escola.describe()

#%%
# Estudo sobre o balanceamento dos dados em relação à quantidade de alunos 
#por período analisado
# Quantidade de estudantes monitorados em cada período
df_tempo_aluno_escola.groupby('mes')['estudante'].count()

#%%
# Estudo sobre o desbalanceamento da quantidade de alunos aninhados em 
#escolas
df_tempo_aluno_escola.groupby('escola')['estudante'].count()/4

#%%
# Gráfico com a evolução temporal do desempenho escolar dos 50 primeiros
#estudantes da amostra (50 estudantes em razão da visualização no gráfico)
df_tempo_aluno_escola['estudante'] = df_tempo_aluno_escola['estudante'].astype('int')
df = df_tempo_aluno_escola[df_tempo_aluno_escola['estudante'] <= 50]
df_tempo_aluno_escola['estudante'] = df_tempo_aluno_escola['estudante'].astype('category')

plt.figure(figsize=(15,10))
sns.lineplot(x='mes', y='desempenho', data=df,
             hue='estudante', marker="o", palette='viridis')
plt.ylabel('Desempenho Escolar',fontsize=14)
plt.xlabel('Mês',fontsize=14)
plt.show()

#%%
# Desempenho escolar médio dos estudantes em cada período
desempenho_medio_periodo = df_tempo_aluno_escola.groupby('mes')['desempenho'].mean().reset_index()
desempenho_medio_periodo

#%%
# Gráfico com a evolução do desempenho escolar médio dos estudantes em cada período
#(ajuste linear)

plt.figure(figsize=(15,10))
sns.regplot(df_tempo_aluno_escola['mes'], df_tempo_aluno_escola['desempenho'],
            data=df_tempo_aluno_escola, ci=None, marker='o',
            scatter_kws={'color':'gold', 's':170, 'alpha':0.2},
            line_kws={'color':'darkorchid', 'linewidth':7})
plt.xlabel('Mês', fontsize=17)
plt.ylabel('Desempenho Escolar', fontsize=17)
plt.legend(fontsize=17)
plt.show

#%%
# Kernel density estimation (KDE) - função densidade de probabilidade da
#variável dependente ('desempenho'), com histograma

plt.figure(figsize=(15,10))
sns.histplot(data=df_tempo_aluno_escola.desempenho, kde=True,
             bins=30, color='deepskyblue')
plt.xlabel('Desempenho', fontsize=17)
plt.ylabel('Frequência', fontsize=17)
plt.show()

#%%
# Kernel density estimation (KDE) - função densidade de probabilidade da
#variável dependente ('desempenho') por escola
plt.figure(figsize=(15,10))
sns.pairplot(df_tempo_aluno_escola[['escola','desempenho']], hue='escola', height=8,
             aspect=2, palette='viridis')
plt.xlabel('Desempenho', fontsize=17)
plt.ylabel('Frequência', fontsize=17)
plt.show()

#%%
# Kernel density estimation (KDE) - função densidade de probabilidade da
#variável dependente ('desempenho'), com histograma e por escola separadamente
#(função 'FacetGrid' do pacote 'seaborn')

g = sns.FacetGrid(df_tempo_aluno_escola, col="escola", col_wrap=5, hue='escola', palette='viridis')
g.map_dataframe(sns.histplot, kde=True, x='desempenho')
plt.show()

#%%
# Gráfico da evolução temporal do desempenho médio por escola (ajustes lineares)
# NOTE QUE A PERSPECTIVA MULTINÍVEL NATURALMENTE CONSIDERA O COMPORTAMENTO
#HETEROCEDÁSTICO NOS DADOS!

sns.lmplot(x='mes', y='desempenho', hue='escola', data=df_tempo_aluno_escola,
           height=8, palette='viridis', ci=False)
plt.ylabel('Desempenho Escolar', fontsize=17)
plt.xlabel('Mês', fontsize=17)
plt.show()

#%%
##############################################################################
#                        ESTIMAÇÃO DO MODELO NULO HLM3                       #
##############################################################################

# Estimação do modelo nulo (função 'MixedLM' do pacote 'statsmodels')
modelo_nulo_hlm3 = sm.MixedLM.from_formula(formula='desempenho ~ 1',
                                           groups='escola',
                                           re_formula='1',
                                           vc_formula={'estudante': '0 + C(estudante)'},
                                           data=df_tempo_aluno_escola).fit()

# Parâmetros do 'modelo_nulo_hlm3'
modelo_nulo_hlm3.summary()

#%%
# A partir deste momento, iremos estimar os modelos multinível HLM3 com medidas
#repetidas por meio da função 'Lmer' do pacote 'pymer4.models', já que esta
#função permite que sejam considerados efeitos aleatórios de inclinação para
#os níveis 2 e 3 simultaneamente, ao contrário da função 'MixedLM' do pacote
#'statsmodels', que permite a inclusão de tais efeitos em apenas um dos níveis
#contextuais

# Transformação das variáveis 'estudante' e 'escola' para 'int64', a fim de que
#seja possível estimar os modelos multinível por meio da função 'Lmer' do
#pacote 'pymer4.models'
df_tempo_aluno_escola['estudante'] = df_tempo_aluno_escola['estudante'].astype('int64')
df_tempo_aluno_escola['escola'] = df_tempo_aluno_escola['escola'].astype('int64')

# Estimação do modelo nulo (função 'Lmer' do pacote 'pymer4.models')
modelo_nulo_hlm3 = Lmer(formula='desempenho ~ 1 + (1|escola) + (1|estudante)',
                        data=df_tempo_aluno_escola)

# Parâmetros do 'modelo_nulo_hlm3'
modelo_nulo_hlm3.fit(old_optimizer=(True))

modelo_nulo_hlm3.grps

#%%
##############################################################################
#                   COMPARAÇÃO DO HLM3 NULO COM UM MQO NULO                  #
##############################################################################

# Estimação de um modelo MQO nulo
modelo_ols_nulo = sm.OLS.from_formula(formula='desempenho ~ 1',
                                      data=df_tempo_aluno_escola).fit()

# Parâmetros do 'modelo_ols_nulo'
modelo_ols_nulo.summary()

#%%
# Teste de razão de verossimilhança entre o 'modelo_nulo_hlm3' e o 'modelo_ols_nulo'

# Definição da função 'lrtest'
# Note que o valor de log-likelihood de modelos estimados por meio da função
#'Lmer' do pacote 'pymer4.models' é obtido pelo argumento '.logLike'
def lrtest(modelos):
    modelo_1 = modelos[0]
    llk_1 = modelo_1.llf
    llk_2 = modelos[1].logLike
    LR_statistic = -2*(llk_1-llk_2)
    p_val = stats.chi2.sf(LR_statistic, 1)
    return round(LR_statistic,2), round(p_val,2)

# Teste de razão de verossimilhança propriamente dito
lrtest([modelo_ols_nulo, modelo_nulo_hlm3])

#%%
# Gráfico para comparação visual dos logLiks dos modelos estimados até o momento

df_llf = pd.DataFrame({'modelo':['MQO Nulo','HLM3 Nulo'],
                      'loglik':[modelo_ols_nulo.llf,modelo_nulo_hlm3.logLike]})

fig, ax = plt.subplots(figsize=(15,10))

c = ['black','grey']

ax1 = ax.barh(df_llf.modelo,df_llf.loglik, color = c)
ax.bar_label(ax1, label_type='center', color='white', fontsize=22)
ax.set_ylabel("Modelo Proposto", fontsize= 17)
ax.set_xlabel("LogLik", fontsize=17)
plt.show()

#%%
##############################################################################
#              ESTIMAÇÃO DO MODELO HLM3 COM TENDÊNCIA LINEAR E               #
#                   INTERCEPTOS E INCLINAÇÕES ALEATÓRIAS                     #
##############################################################################

# Estimação do modelo com tendência linear e interceptos e inclinações aleatórias
modelo_intercept_inclin_hlm3 = Lmer('desempenho ~ mes + (mes|escola) + (mes|estudante)',
                                    data = df_tempo_aluno_escola)

# Parâmetros do 'modelo_intercept_inclin_hlm3'
modelo_intercept_inclin_hlm3.fit(old_optimizer=(True))

#%%
# Gráfico para comparação visual dos logLiks dos modelos estimados até o momento

df_llf = pd.DataFrame({'modelo':['MQO Nulo','HLM3 Nulo',
                                 'HLM3 com Interceptos e Inclinações Aleatórias'],
                      'loglik':[modelo_ols_nulo.llf,modelo_nulo_hlm3.logLike,
                                modelo_intercept_inclin_hlm3.logLike]})

fig, ax = plt.subplots(figsize=(15,10))

c = ['black','grey','navy']

ax1 = ax.barh(df_llf.modelo,df_llf.loglik, color = c)
ax.bar_label(ax1, label_type='center', color='white', fontsize=22)
ax.set_ylabel("Modelo Proposto", fontsize=17)
ax.set_xlabel("LogLik", fontsize=17)
plt.show()

#%%
# Teste de razão de verossimilhança para comparar as estimações do modelo HLM3
#nulo e do presente modelo

# Definição da função 'lrtest2'
# Note que o valor de log-likelihood de modelos estimados por meio da função
#'Lmer' do pacote 'pymer4.models' é obtido pelo argumento '.logLike'
def lrtest2(modelos):
    modelo_1 = modelos[0]
    llk_1 = modelo_1.logLike
    llk_2 = modelos[1].logLike
    LR_statistic = -2*(llk_1-llk_2)
    p_val = stats.chi2.sf(LR_statistic, 1)
    return round(LR_statistic,2), round(p_val,2)

lrtest2([modelo_nulo_hlm3, modelo_intercept_inclin_hlm3])

#%%
##############################################################################
#              ESTIMAÇÃO DO MODELO HLM3 COM TENDÊNCIA LINEAR,                #
#                   INTERCEPTOS E INCLINAÇÕES ALEATÓRIAS                     #
#          E AS VARIÁVEIS 'sexo' DE NÍVEL 2 E 'text' DE NÍVEL 3              #
##############################################################################

# Dummização da variável preditora qualitativa 'ativ', a fim de que seja possível
#estabelecer, adiante, as funções para a definição dos efeitos aleatórios dos
#níveis contextuais.
df_tempo_aluno_escola = pd.get_dummies(df_tempo_aluno_escola,
                                       columns=['ativ'],
                                       drop_first=True)

# Estimação do modelo com tendência linear, interceptos e inclinações aleatórias
#e as variáveis 'ativ' de nível 2 e 'texp' de nível 3
modelo_completo_final_hlm3 = Lmer('desempenho ~ mes + ativ_sim + texp +\
                                  ativ_sim:mes + texp:mes +\
                                      (mes|escola) + (mes|estudante)',
                                      data = df_tempo_aluno_escola)

# Parâmetros do 'modelo_completo_final_hlm3'
modelo_completo_final_hlm3.fit(old_optimizer=(True))

#%%
# Gráfico para comparação visual dos logLiks dos modelos estimados até o momento

df_llf = pd.DataFrame({'modelo':['MQO Nulo','HLM3 Nulo',
                                 'HLM3 com Interceptos e Inclinações Aleatórias',
                                 'HLM3 Completo Níveis 2 e 3'],
                      'loglik':[modelo_ols_nulo.llf,modelo_nulo_hlm3.logLike,
                                modelo_intercept_inclin_hlm3.logLike,
                                modelo_completo_final_hlm3.logLike]})

fig, ax = plt.subplots(figsize=(15,10))

c = ['black','grey','navy','dodgerblue']

ax1 = ax.barh(df_llf.modelo,df_llf.loglik, color = c)
ax.bar_label(ax1, label_type='center', color='white', fontsize=22)
ax.set_ylabel("Modelo Proposto", fontsize=17)
ax.set_xlabel("LogLik", fontsize=17)
plt.show()

#%%
# Teste de razão de verossimilhança para comparar as estimações do presente
#modelo e do modelo anterior
#(função 'lrtest2' definida anteriormente)
lrtest2([modelo_intercept_inclin_hlm3, modelo_completo_final_hlm3])

#%%
# Visualização dos interceptos e das inclinações aleatórias por escola,
#para o 'modelo_completo_final_hlm3'

# Visualização dos valores de r0jk e r1jk (efeitos aleatórios de intercepto e 
#de inclinação no nível estudante, respectivamente) e de u00k e u10k (efeitos
#aleatórios de intercepto e de inclinação no nível escola, respectivamente)

#%%
# Nível estudante
aleat_estudante = pd.DataFrame(modelo_completo_final_hlm3.ranef[0]).dropna()
aleat_estudante = aleat_estudante.rename(columns={'X.Intercept.':'r0jk','mes':'r1jk'})
aleat_estudante = aleat_estudante.reset_index().rename(columns={'index': 'estudante'})
aleat_estudante

#%%
# Nível escola
aleat_escola = pd.DataFrame(modelo_completo_final_hlm3.ranef[1]).dropna()
aleat_escola = aleat_escola.rename(columns={'X.Intercept.':'u00k','mes':'u10k'})
aleat_escola = aleat_escola.reset_index().rename(columns={'index': 'escola'})
aleat_escola

#%%
# Gráfico para visualização do comportamento dos valores de r0jk, ou seja,
#dos interceptos aleatórios por estudante

plt.figure(figsize=(15,10))
plt.barh(aleat_estudante['estudante'], aleat_estudante['r0jk'], color='gold')
plt.ylabel('Estudante', fontsize=14)
plt.yticks([])
plt.xlabel('r0jk', fontsize=14)
plt.show()

#%%
# Gráfico para visualização do comportamento dos valores de r1jk, ou seja,
#das inclinações aleatórias por estudante

plt.figure(figsize=(15,10))
plt.barh(aleat_estudante['estudante'], aleat_estudante['r1jk'], color='darkorchid')
plt.ylabel('Estudante', fontsize=14)
plt.yticks([])
plt.xlabel('r1jk', fontsize=14)
plt.show()

#%%
# Gráfico para visualização do comportamento dos valores de u10k, ou seja,
#das inclinações aleatórias por escola

colors = ['green' if x>0 else 'red' for x in aleat_escola['u10k']]

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(0, point['y'], str(round(point['x'],3)), fontsize=14)

plt.figure(figsize=(15,10))
plt.barh(aleat_escola['escola'], aleat_escola['u10k'], color=colors)

label_point(x = aleat_escola['u10k'],
            y = aleat_escola['escola'],
            val = aleat_escola['u10k'],
            ax = plt.gca()) 
plt.ylabel('Escola', fontsize=14)
plt.xlabel('u10k', fontsize=14)
plt.show()

#%%
# Visualização dos fitted values do 'modelo_completo_final_hlm3', por estudante
#e por escola

# Definição manual dos valores de 'fitted_fixed'
df_tempo_aluno_escola['fitted_fixed'] = modelo_completo_final_hlm3.coefs.iloc[0,0] +\
    modelo_completo_final_hlm3.coefs.iloc[1,0]*df_tempo_aluno_escola['mes'] +\
        modelo_completo_final_hlm3.coefs.iloc[2,0]*df_tempo_aluno_escola['ativ_sim'] +\
            modelo_completo_final_hlm3.coefs.iloc[3,0]*df_tempo_aluno_escola['texp'] +\
                modelo_completo_final_hlm3.coefs.iloc[4,0]*df_tempo_aluno_escola['mes']*df_tempo_aluno_escola['ativ_sim'] +\
                    modelo_completo_final_hlm3.coefs.iloc[5,0]*df_tempo_aluno_escola['mes']*df_tempo_aluno_escola['texp']

#%%
# Definição da função 'predict_fixed' para o estabelecimento alternativo dos
#valores de 'fitted_fixed'

def predict_fixed(df:pd.DataFrame, model:Lmer):
    
    coefs = list(model.coefs.index)
    intercept = model.coefs['Estimate'][coefs[0]]
    equation = [str(intercept)]
    
    for index, item in enumerate(coefs[1:]):
       
        if item.find(":") != -1:
            nested = item.split(":")
            equation.append(str(nested[0]) + " * " + str(nested[1]) + " * " + str(model.coefs['Estimate'][item]))
        
        else:
            equation.append(item + " * " + str(model.coefs['Estimate'][item]))
    
    final_equation = " + ".join(equation)
    
    final_equation = "fitted_fixed =" + final_equation
    
    return df.eval(final_equation)

#%%
# Dataframe com os valores de 'fitted_fixed'
df_tempo_aluno_escola = predict_fixed(df_tempo_aluno_escola,
                                      modelo_completo_final_hlm3)

#%%
# Dois modos para a definição direta dos valores de 'fitted_estudante'
df_tempo_aluno_escola['fitted_estudante'] =\
    modelo_completo_final_hlm3.predict(df_tempo_aluno_escola,
                                       skip_data_checks=True,
                                       verify_predictions=False)

df_tempo_aluno_escola['fitted_estudante'] = pd.DataFrame(modelo_completo_final_hlm3.fits)

#%%
# Definição da função 'predict_random' para o estabelecimento alternativo dos
#valores de 'fitted_estudante' e dos valores de 'fitted_escola'

def predict_random(df: pd.DataFrame, model: Lmer) -> pd.DataFrame:

    groups = model.grps

    for index, group in enumerate(groups.keys()):

        cols = model.ranef[index].columns[1:]

        for item in enumerate(model.ranef[index].iterrows()):

            intercept = item[1][1]['X.Intercept.']

            rnd = 0

            for col in cols:

                rnd += item[1][1][col]

            level = item[1][0]

            result = intercept + rnd

            df.loc[df[group] == int(level), 'random_' + group] = result

    return df

#%%
# Dataframe com os valores de 'fitted_estudante' e 'fitted_escola'
df_tempo_aluno_escola = predict_random(df_tempo_aluno_escola,
                                       modelo_completo_final_hlm3)

df_tempo_aluno_escola['fitted_estudante'] = df_tempo_aluno_escola['fitted_fixed'] +\
    df_tempo_aluno_escola['random_estudante'] +\
    df_tempo_aluno_escola['random_escola']

df_tempo_aluno_escola['fitted_escola'] = df_tempo_aluno_escola['fitted_fixed'] +\
    df_tempo_aluno_escola['random_escola']

#%%
# Definição dos valores de 'etjk'
df_tempo_aluno_escola['etjk'] = modelo_completo_final_hlm3.residuals

#%%
# Visualização do dataframe 'df_effects' com os valores de 'fitted_fixed',
#'fitted_escola', 'fitted_estudante' e 'etjk'
df_effects = df_tempo_aluno_escola[['escola','estudante','fitted_fixed',
                                    'fitted_escola','fitted_estudante',
                                    'mes','desempenho','etjk']]

df_effects

#%%
# Elaboração de previsões para o 'modelo_completo_final_hlm3':
# Exemplo: Quais os valores previstos de desempenho escolar no primeiro mes
#('mes' = 1) para o estudante "1" da escola "1", sabendo-se que esta escola
#oferece tempo médio de experiência de seus professores igual a 2 anos?

# Predict estudante (ativ_sim = 0)
pred_est_nao = modelo_completo_final_hlm3.predict(pd.DataFrame({'escola':[1],
                                                                'estudante':[1],
                                                                'mes':[1],
                                                                'ativ_sim':[0],
                                                                'texp':[2]}),
                                                  skip_data_checks=True)
pred_est_nao

# Predict estudante (ativ_sim = 1)
pred_est_sim = modelo_completo_final_hlm3.predict(pd.DataFrame({'escola':[1],
                                                                 'estudante':[1],
                                                                 'mes':[1],
                                                                 'ativ_sim':[1],
                                                                 'texp':[2]}),
                                                   skip_data_checks=True)
pred_est_sim

# Predict escola (ativ_sim = 0)
pred_esc_nao = pred_est_nao - df_tempo_aluno_escola['random_estudante'][0]
pred_esc_nao

# Predict escola (ativ_sim = 1)
pred_esc_sim = pred_est_sim - df_tempo_aluno_escola['random_estudante'][0]
pred_esc_sim

# Predict fixed (ativ_sim = 0)
pred_fixed_nao = pred_est_nao - df_tempo_aluno_escola['random_estudante'][0] -\
    df_tempo_aluno_escola['random_escola'][0]
pred_fixed_nao

# Predict fixed (ativ_sim = 1)
pred_fixed_sim = pred_est_sim - df_tempo_aluno_escola['random_estudante'][0] -\
    df_tempo_aluno_escola['random_escola'][0]
pred_fixed_sim

#%%
# Gráfico com os valores previstos do desempenho escolar ao longo do tempo para
#os 47 primeiros estudantes da amostra (47 estudantes que estão na escola 1)

df_tempo_aluno_escola['estudante'] = df_tempo_aluno_escola['estudante'].astype('int')
df = df_tempo_aluno_escola[df_tempo_aluno_escola['estudante'] <= 47]
df_tempo_aluno_escola['estudante'] = df_tempo_aluno_escola['estudante'].astype('category')

plt.figure(figsize=(15,10))
sns.lineplot(x='mes', y='fitted_estudante', data=df,
             hue='estudante', marker="o", palette='viridis')
plt.ylabel('Desempenho Escolar',fontsize=14)
plt.xlabel('Mês',fontsize=14)
plt.show()

#%%
##############################################################################
#       COMPARAÇÃO FINAL COM UM MODELO MQO COM DUMMIES PARA ESCOLAS          #
##############################################################################

# Procedimento para criação de n-1 dummies para as escolas
base = df_tempo_aluno_escola[['estudante','escola','desempenho','mes',
                              'texp','ativ_sim']]

base_dummizada = pd.get_dummies(base, columns=['escola'], drop_first=True)
base_dummizada

#%%
# Estimação de um modelo MQO com as mesmas variáveis do modelo HLM3

# Definição da expressão a ser utilizada no modelo
base_dummizada['ativ_mes'] = base_dummizada['ativ_sim'] * base_dummizada['mes']
base_dummizada['texp_mes'] = base_dummizada['texp'] * base_dummizada['mes']

lista_colunas = list(base_dummizada.drop(columns=['estudante',
                                                  'desempenho']).columns)
formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "desempenho ~ " + formula_dummies_modelo

# Estimação do 'modelo_ols_dummies'
modelo_ols_dummies = sm.OLS.from_formula(formula_dummies_modelo,
                                         base_dummizada).fit()

# Parâmetros do 'modelo_ols_dummies'
modelo_ols_dummies.summary()

#%%
# Procedimento Stepwise para o 'modelo_ols_dummies'

# Instalação e carregamento da função 'stepwise' do pacote
#'statstests.process'
#pip install statstests
# Autores do pacote: Helder Prado Santos e Luiz Paulo Fávero

# Estimação do modelo por meio do procedimento Stepwise
modelo_ols_dummies_step = stepwise(modelo_ols_dummies, pvalue_limit=0.05)

# Parâmetros do 'modelo_ols_dummies_step' (output já obtido no código anterior)
modelo_ols_dummies_step.summary()

#%%
# Gráfico para comparação visual dos logLiks dos modelos HLM3 completo e MQO
#com dummies e procedimento Stepwise

df_llf = pd.DataFrame({'modelo':['MQO Final com Stepwise',
                                 'HLM3 Completo Final'],
                      'loglik':[modelo_ols_dummies_step.llf,
                                modelo_completo_final_hlm3.logLike]})

fig, ax = plt.subplots(figsize=(15,10))

c = ['orange','purple']

ax1 = ax.barh(df_llf.modelo,df_llf.loglik, color = c)
ax.bar_label(ax1, label_type='center', color='white', fontsize=22)
ax.set_ylabel("Modelo Proposto", fontsize=17)
ax.set_xlabel("LogLik", fontsize=17)
plt.show()

#%%
# Teste de razão de verossimilhança para comparar as estimações dos modelos
#HLM3 completo e MQO com dummies e procedimento Stepwise
#(função 'lrtest' definida anteriormente)
lrtest([modelo_ols_dummies_step, modelo_completo_final_hlm3])

#%%
# Gráfico para a comparação dos fitted values dos modelos HLM3 completo e
#MQO com dummies e procedimento Stepwise

plt.figure(figsize=(15,10))
sns.regplot(df_tempo_aluno_escola['desempenho'],
            modelo_ols_dummies.fittedvalues,
            data=df_tempo_aluno_escola, ci=None, marker='o', order=4,
            scatter_kws={'color':'orange', 's':40, 'alpha':0.5},
            line_kws={'color':'orange', 'linewidth':5,
                      'label':'MQO'})
sns.regplot(df_tempo_aluno_escola['desempenho'],
            df_tempo_aluno_escola['fitted_estudante'],
            data=df_tempo_aluno_escola, ci=None, marker='s', order=4,
            scatter_kws={'color':'darkorchid', 's':40, 'alpha':0.5},
            line_kws={'color':'darkorchid', 'linewidth':5,
                      'label':'HLM3'})
plt.xlabel('Desempenho', fontsize=17)
plt.ylabel('Fitted Values', fontsize=17)
plt.legend(fontsize=17)
plt.show

##############################################################################