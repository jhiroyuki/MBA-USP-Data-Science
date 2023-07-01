
# Análise Fatorial PCA + Análise de Clusters Hierárquico

# Curso: MBA DSA (USP ESALQ)

# Prof. Wilson Tarantin Jr.

# Fonte dos dados: https://www.kaggle.com/datasets/vipulgohel/clustering-pca-assignment?resource=download&select=Country-data.csv

# Instalação e carregamento dos pacotes utilizados

pacotes <- c("plotly",
             "tidyverse",
             "ggrepel",
             "knitr", "kableExtra",
             "reshape2",
             "PerformanceAnalytics",
             "psych",
             "Hmisc",
             "readxl",
             "cluster",
             "factoextra") 

if(sum(as.numeric(!pacotes %in% installed.packages())) != 0){
  instalador <- pacotes[!pacotes %in% installed.packages()]
  for(i in 1:length(instalador)) {
    install.packages(instalador, dependencies = T)
    break()}
  sapply(pacotes, require, character = T) 
} else {
  sapply(pacotes, require, character = T) 
}

# Carregamento da base de dados
paises <- read.csv("Países PCA Cluster.csv", sep = ",", dec = ".")

# Estatísticas descritivas
summary(paises)

# Scatter e ajuste entre as variáveis 'renda' e 'expectativa de vida'
paises %>%
  ggplot() +
  geom_point(aes(x = income, y = life_expec),
             color = "darkorchid",
             size = 3) +
  geom_smooth(aes(x = income, y = life_expec),
              color = "orange", 
              method = "loess", 
              formula = y ~ x, 
              se = FALSE,
              size = 1.3) +
  labs(x = "income",
       y = "life_expec") +
  theme_bw()

# Scatter e ajuste entre as variáveis 'exports' e 'gdpp'
paises %>%
  ggplot() +
  geom_point(aes(x = exports, y = gdpp),
             color = "darkorchid",
             size = 3) +
  geom_smooth(aes(x = exports, y = gdpp),
              color = "orange", 
              method = "loess", 
              formula = y ~ x, 
              se = FALSE,
              size = 1.3) +
  labs(x = "exports",
       y = "gdpp") +
  theme_bw()

# Coeficientes de correlação de Pearson para cada par de variáveis
rho <- rcorr(as.matrix(paises[,2:10]), type="pearson")

correl <- rho$r # Matriz de correlações
sig_correl <- round(rho$P, 4) # Matriz com p-valor dos coeficientes

# Elaboração de um mapa de calor das correlações de Pearson entre as variáveis
ggplotly(
  paises[,2:10] %>%
    cor() %>%
    melt() %>%
    rename(Correlação = value) %>%
    ggplot() +
    geom_tile(aes(x = Var1, y = Var2, fill = Correlação)) +
    geom_text(aes(x = Var1, y = Var2, label = format(Correlação, digits = 1)),
              size = 5) +
    scale_fill_viridis_b() +
    labs(x = NULL, y = NULL) +
    theme_bw())

# Visualização das distribuições das variáveis, scatters, valores das correlações
chart.Correlation(paises[,2:10], histogram = TRUE, pch = "+")

### Elaboração a Análise Fatorial Por Componentes Principais ###

# Teste de esfericidade de Bartlett
cortest.bartlett(paises[,2:10])

# Elaboração da análise fatorial por componentes principais
fatorial <- principal(paises[,2:10],
                      nfactors = length(paises[,2:10]),
                      rotate = "none",
                      scores = TRUE)
fatorial

# Eigenvalues (autovalores)
eigenvalues <- round(fatorial$values, 5)
eigenvalues

# Soma dos eigenvalues = 9 (quantidade de variáveis na análise)
# Também representa a quantidade máxima de possíveis fatores na análise
round(sum(eigenvalues), 2)

# Identificação da variância compartilhada em cada fator
variancia_compartilhada <- as.data.frame(fatorial$Vaccounted) %>% 
  slice(1:3)

rownames(variancia_compartilhada) <- c("Autovalores",
                                       "Prop. da Variância",
                                       "Prop. da Variância Acumulada")

# Variância compartilhada pelas variáveis originais para a formação de cada fator
round(variancia_compartilhada, 3) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = FALSE, 
                font_size = 20)

# Cálculo dos scores fatoriais
scores_fatoriais <- as.data.frame(fatorial$weights)

# Visualização dos scores fatoriais
round(scores_fatoriais, 3) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = FALSE, 
                font_size = 20)

# Cálculo dos fatores propriamente ditos
fatores <- as.data.frame(fatorial$scores)

View(fatores)

# Coeficientes de correlação de Pearson para cada par de fatores (ortogonais)
rho <- rcorr(as.matrix(fatores), type="pearson")
round(rho$r, 4)

# Cálculo das cargas fatoriais
cargas_fatoriais <- as.data.frame(unclass(fatorial$loadings))

# Visualização das cargas fatoriaisX
round(cargas_fatoriais, 3) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = FALSE, 
                font_size = 20)

# Cálculo das comunalidades
comunalidades <- as.data.frame(unclass(fatorial$communality)) %>%
  rename(comunalidades = 1)

# Visualização das comunalidades (aqui são iguais a 1 para todas as variáveis)
# Foram extraídos 9 fatores neste primeiro momento
round(comunalidades, 3) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped",
                full_width = FALSE,
                font_size = 20)

### Elaboração da Análise Fatorial por Componentes Principais ###
### Fatores extraídos a partir de autovalores maiores que 1 ###

# Definição da quantidade de fatores com eigenvalues maiores que 1
k <- sum(eigenvalues > 1)
print(k)

# Elaboração da análise fatorial por componentes principais sem rotação
# Com quantidade 'k' de fatores com eigenvalues maiores que 1
fatorial2 <- principal(paises[,2:10],
                       nfactors = k,
                       rotate = "none",
                       scores = TRUE)
fatorial2

# Cálculo das comunalidades com apenas os 'k' ('k' = 3) primeiros fatores
comunalidades2 <- as.data.frame(unclass(fatorial2$communality)) %>%
  rename(comunalidades = 1)

# Visualização das comunalidades com apenas os 'k' ('k' = 3) primeiros fatores
round(comunalidades2, 3) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped",
                full_width = FALSE,
                font_size = 20)

# Loading plot com as cargas dos dois primeiros fatores
cargas_fatoriais[, 1:2] %>% 
  data.frame() %>%
  rownames_to_column("variáveis") %>%
  ggplot(aes(x = PC1, y = PC2, label = variáveis)) +
  geom_point(color = "darkorchid",
             size = 3) +
  geom_text_repel() +
  geom_vline(aes(xintercept = 0), linetype = "dashed", color = "orange") +
  geom_hline(aes(yintercept = 0), linetype = "dashed", color = "orange") +
  expand_limits(x= c(-1.25, 0.25), y=c(-0.25, 1)) +
  theme_bw()

# Adicionando os fatores extraídos no banco de dados original
paises <- bind_cols(paises,
                    "fator_1" = fatores$PC1, 
                    "fator_2" = fatores$PC2,
                    "fator_3" = fatores$PC3)

# Análise de Cluster Utilizando os 3 Fatores

# Análise dos fatores (média e desvio padrão)
summary(paises[,11:13])
sd(paises[,11])
sd(paises[,12])
sd(paises[,13])

## ATENÇÃO: os clusters serão formados a partir dos 3 fatores
## Não aplicaremos o Z-Score, pois os fatores já são padronizados

# Matriz de dissimilaridades
matriz_D <- paises[,11:13] %>% 
  dist(method = "euclidean")

# Elaboração da clusterização hierárquica
cluster_hier <- agnes(x = matriz_D, method = "complete")

# Definição do esquema hierárquico de aglomeração

# As distâncias para as combinações em cada estágio
coeficientes <- sort(cluster_hier$height, decreasing = FALSE) 
coeficientes

# Tabela com o esquema de aglomeração. Interpretação do output:

## As linhas são os estágios de aglomeração
## Nas colunas Cluster1 e Cluster2, observa-se como ocorreu a junção
## Quando for número negativo, indica observação isolada
## Quando for número positivo, indica cluster formado anteriormente (estágio)
## Coeficientes: as distâncias para as combinações em cada estágio

esquema <- as.data.frame(cbind(cluster_hier$merge, coeficientes))
names(esquema) <- c("Cluster1", "Cluster2", "Coeficientes")
esquema

# Construção do dendrograma
dev.off()
fviz_dend(x = cluster_hier, show_labels = FALSE)

# Dendrograma com visualização dos clusters (definição de 10 clusters)
fviz_dend(x = cluster_hier,
          h = 3.0,
          show_labels = FALSE,
          color_labels_by_k = F,
          rect = F,
          rect_fill = F,
          ggtheme = theme_bw())

# Criando variável categórica para indicação do cluster no banco de dados
## O argumento 'k' indica a quantidade de clusters
paises$cluster_H <- factor(cutree(tree = cluster_hier, k = 10))

# Análise de variância de um fator (ANOVA). Interpretação do output:

## Mean Sq do cluster_H: indica a variabilidade entre grupos
## Mean Sq dos Residuals: indica a variabilidade dentro dos grupos
## F value: estatística de teste (Sum Sq do cluster_H / Sum Sq dos Residuals)
## Pr(>F): p-valor da estatística 
## p-valor < 0.05: pelo menos um cluster apresenta média estatisticamente diferente dos demais

## A variável mais discriminante dos grupos contém maior estatística F (e significativa)

# ANOVA da variável 'fator 1'
summary(anova_fator_1 <- aov(formula = fator_1 ~ cluster_H,
                             data = paises))

# ANOVA da variável 'fator 2'
summary(anova_fator_2 <- aov(formula = fator_2 ~ cluster_H,
                             data = paises))

# ANOVA da variável 'fator 3'
summary(anova_fator_3 <- aov(formula = fator_3 ~ cluster_H,
                             data = paises))

# Algumas estatísticas descritivas entre clusters

# PIB
group_by(paises, cluster_H) %>%
  summarise(
    mean = mean(gdpp, na.rm = TRUE),
    sd = sd(gdpp, na.rm = TRUE),
    min = min(gdpp, na.rm = TRUE),
    max = max(gdpp, na.rm = TRUE),
    obs = n())

# Indicador de saúde
group_by(paises, cluster_H) %>%
  summarise(
    mean = mean(health, na.rm = TRUE),
    sd = sd(health, na.rm = TRUE),
    min = min(health, na.rm = TRUE),
    max = max(health, na.rm = TRUE),
    obs = n())

# Expectativa de vida
group_by(paises, cluster_H) %>%
  summarise(
    mean = mean(life_expec, na.rm = TRUE),
    sd = sd(life_expec, na.rm = TRUE),
    min = min(life_expec, na.rm = TRUE),
    max = max(life_expec, na.rm = TRUE),
    obs = n())

# FIM!

