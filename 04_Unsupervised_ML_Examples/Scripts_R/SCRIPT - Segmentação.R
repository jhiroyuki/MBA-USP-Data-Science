
# Análise de Clusters K-means + Análise de Correspondência Múltipla

# Curso: MBA DSA (USP ESALQ)

# Prof. Wilson Tarantin Jr.

# Fonte dos dados: https://www.kaggle.com/datasets/kaushiksuresh147/customer-segmentation

# Instalação e carregamento dos pacotes utilizados
pacotes <- c("plotly", 
             "tidyverse", 
             "ggrepel",
             "knitr", "kableExtra", 
             "sjPlot", 
             "FactoMineR", 
             "amap", 
             "ade4",
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

# Importando a base de dados
load("segmenta.Rdata")

## Em uma pré-visualização, há valores faltantes (NAs)

# Como existem variáveis com missing values (NAs), vamos excluir as observações
segmenta <- drop_na(segmenta)

# Algumas variáveis são qualitativas e outras são quantitativas
## Vamos separar o banco de dados em 2 partes (somente quali e quanti)
segmenta_quali <- segmenta[,c(1,2,4,5)]
segmenta_quanti <- segmenta[,c(3,6)]

# A função para a criação da ACM pede que sejam utilizados "fatores"
segmenta_quali <- as.data.frame(unclass(segmenta_quali), stringsAsFactors=TRUE)

# Estatísticas descritivas dos dados
summary(segmenta_quanti)
summary(segmenta_quali)

# Iniciando a Análise de Cluster nas variáveis quantitativas

# Aplicando a padronização por ZScore
segm_pad <- as.data.frame(scale(segmenta_quanti))

round(mean(segm_pad$Age), 3)
round(mean(segm_pad$Family_Size), 3)

round(sd(segm_pad$Age), 3)
round(sd(segm_pad$Family_Size), 3)

# Método de Elbow para identificação do número ótimo de clusters
fviz_nbclust(segm_pad, kmeans, method = "wss", k.max = 15)

# Elaboração da clusterização não hieráquica k-means
cluster_kmeans <- kmeans(segm_pad,
                         centers = 5)

# Criando variável categórica para indicação do cluster no banco de dados
segm_pad$cluster_K <- factor(cluster_kmeans$cluster)
segmenta$cluster_K <- factor(cluster_kmeans$cluster)

# Análise de variância de um fator (ANOVA). Interpretação do output:

## Mean Sq do cluster_K: indica a variabilidade entre grupos
## Mean Sq dos Residuals: indica a variabilidade dentro dos grupos
## F value: estatística de teste (Sum Sq do cluster_K / Sum Sq dos Residuals)
## Pr(>F): p-valor da estatística 
## p-valor < 0.05: pelo menos um cluster apresenta média estatisticamente diferente dos demais

## A variável mais discriminante dos grupos contém maior estatística F (e significativa)

# ANOVA da variável 'Age'
summary(anova_Age <- aov(formula = Age ~ cluster_K,
                         data = segm_pad))

# ANOVA da variável 'Family_Size'
summary(anova_Family_Size <- aov(formula = Family_Size ~ cluster_K,
                                 data = segm_pad))

# Estatísticas descritivas para as variáveis originais 

## 'Age'
group_by(segmenta, cluster_K) %>%
  summarise(
    mean = mean(Age, na.rm = TRUE),
    sd = sd(Age, na.rm = TRUE),
    min = min(Age, na.rm = TRUE),
    max = max(Age, na.rm = TRUE),
    obs = n())

## 'Family_Size'
group_by(segmenta, cluster_K) %>%
  summarise(
    mean = mean(Family_Size, na.rm = TRUE),
    sd = sd(Family_Size, na.rm = TRUE),
    min = min(Family_Size, na.rm = TRUE),
    max = max(Family_Size, na.rm = TRUE),
    obs = n())

## 'Gender'
group_by(segmenta, cluster_K) %>%
  count(Gender) %>%
  mutate(prop = prop.table(n))

## 'Ever_Maried'
group_by(segmenta, cluster_K) %>%
  count(Ever_Married) %>%
  mutate(prop = prop.table(n))

## 'Graduated'
group_by(segmenta, cluster_K) %>%
  count(Graduated) %>%
  mutate(prop = prop.table(n))

## 'Spending_Score'
group_by(segmenta, cluster_K) %>%
  count(Spending_Score) %>%
  mutate(prop = prop.table(n))

# Iniciando a Análise de Correspondência Múltipla nas variáveis qualitativas

# Adicionando variável qualitativa que indica o cluster
segmenta_quali$cluster_K <- factor(cluster_kmeans$cluster)

# Estatísticas descritivas
summary(segmenta_quali)

# Tabelas de contingência
sjt.xtab(var.row = segmenta_quali$Spending_Score,
         var.col = segmenta_quali$Gender,
         show.exp = TRUE,
         show.row.prc = TRUE,
         show.col.prc = TRUE, 
         encoding = "UTF-8")

sjt.xtab(var.row = segmenta_quali$Spending_Score,
         var.col = segmenta_quali$Ever_Married,
         show.exp = TRUE,
         show.row.prc = TRUE,
         show.col.prc = TRUE, 
         encoding = "UTF-8")

sjt.xtab(var.row = segmenta_quali$Spending_Score,
         var.col = segmenta_quali$Graduated,
         show.exp = TRUE,
         show.row.prc = TRUE,
         show.col.prc = TRUE,
         encoding = "UTF-8")

sjt.xtab(var.row = segmenta_quali$Spending_Score,
         var.col = segmenta_quali$cluster_K,
         show.exp = TRUE,
         show.row.prc = TRUE,
         show.col.prc = TRUE,
         encoding = "UTF-8")

# Vamos gerar a ACM (para 3 eixos)
ACM <- dudi.acm(segmenta_quali, scannf = FALSE, nf = 3)

# Analisando as variâncias de cada dimensão
perc_variancia <- (ACM$eig / sum(ACM$eig)) * 100
perc_variancia

# Quantidade de categorias por variável
quant_categorias <- apply(segmenta_quali,
                          MARGIN =  2,
                          FUN = function(x) nlevels(as.factor(x)))

# Consolidando as coordenadas-padrão obtidas por meio da matriz binária
df_ACM <- data.frame(ACM$c1, Variável = rep(names(quant_categorias),
                                            quant_categorias))

# Plotando o mapa perceptual
df_ACM %>%
  rownames_to_column() %>%
  rename(Categoria = 1) %>%
  ggplot(aes(x = CS1, y = CS2, label = Categoria, color = Variável)) +
  geom_point() +
  geom_label_repel() +
  geom_vline(aes(xintercept = 0), linetype = "longdash", color = "grey48") +
  geom_hline(aes(yintercept = 0), linetype = "longdash", color = "grey48") +
  labs(x = paste("Dimensão 1:", paste0(round(perc_variancia[1], 2), "%")),
       y = paste("Dimensão 2:", paste0(round(perc_variancia[2], 2), "%"))) +
  theme_bw()

# Mapa perceptual em 3D (3 primeiras dimensões)
ACM_3D <- plot_ly()

# Adicionando as coordenadas
ACM_3D <- add_trace(p = ACM_3D,
                    x = df_ACM$CS1,
                    y = df_ACM$CS2,
                    z = df_ACM$CS3,
                    mode = "text",
                    text = rownames(df_ACM),
                    textfont = list(color = "blue"),
                    marker = list(color = "red"),
                    showlegend = FALSE)

ACM_3D

# FIM!