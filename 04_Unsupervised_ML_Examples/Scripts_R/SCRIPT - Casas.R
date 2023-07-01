
# Análise de Correspondência Múltipla + Análise Fatorial PCA

# Curso: MBA DSA (USP ESALQ)

# Prof. Wilson Tarantin Jr.

# Instalação e carregamento dos pacotes utilizados

pacotes <- c("plotly",
             "tidyverse",
             "ggrepel",
             "knitr", 
             "kableExtra",
             "reshape2",
             "PerformanceAnalytics", 
             "psych",
             "ltm", 
             "Hmisc",
             "readxl",
             "sjPlot",
             "ade4")

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
casas <- read_xlsx("Preco Casas.xlsx")

# Fonte: https://www.kaggle.com/datasets/elakiricoder/jiffs-house-price-prediction-dataset

# Separação das variáveis em qualitativas e quantitativas
var_quali <- casas[,c(5,6,7,8,10,12,15)]
var_quanti <- casas[,c(1,2,3,4,9,11,13,14)]

## Nota: vamos deixar a variável "valor da casa" fora da análise por enquanto
## O objetivo é criar um ranking que capture os valores das casas

# A função para a criação da ACM pede que sejam utilizados "fatores"
var_quali <- as.data.frame(unclass(var_quali), stringsAsFactors=TRUE)

# Ajustando variáveis quantitativas que estão como textos
var_quanti$distance_to_school <- as.double(var_quanti$distance_to_school)
var_quanti$distance_to_supermarket_km <- as.double(var_quanti$distance_to_supermarket_km)
var_quanti$crime_rate_index <- as.double(var_quanti$crime_rate_index)

# Estatísticas descritivas
summary(var_quali)
summary(var_quanti)

# Iniciando a Análise de Correspondência Múltipla nas variáveis qualitativas

# Tabelas de contingência
sjt.xtab(var.row = var_quali$large_living_room,
         var.col = var_quali$parking_space,
         show.exp = TRUE,
         show.row.prc = TRUE,
         show.col.prc = TRUE, 
         encoding = "UTF-8")

sjt.xtab(var.row = var_quali$large_living_room,
         var.col = var_quali$front_garden,
         show.exp = TRUE,
         show.row.prc = TRUE,
         show.col.prc = TRUE, 
         encoding = "UTF-8")

sjt.xtab(var.row = var_quali$large_living_room,
         var.col = var_quali$swimming_pool,
         show.exp = TRUE,
         show.row.prc = TRUE,
         show.col.prc = TRUE, 
         encoding = "UTF-8")

sjt.xtab(var.row = var_quali$large_living_room,
         var.col = var_quali$wall_fence,
         show.exp = TRUE,
         show.row.prc = TRUE,
         show.col.prc = TRUE, 
         encoding = "UTF-8")

sjt.xtab(var.row = var_quali$large_living_room,
         var.col = var_quali$water_front,
         show.exp = TRUE,
         show.row.prc = TRUE,
         show.col.prc = TRUE, 
         encoding = "UTF-8")

sjt.xtab(var.row = var_quali$large_living_room,
         var.col = var_quali$room_size_class,
         show.exp = TRUE,
         show.row.prc = TRUE,
         show.col.prc = TRUE, 
         encoding = "UTF-8")

# Análise de Correspondência Múltipla
ACM <- dudi.acm(var_quali, scannf = FALSE)

# Analisando as variâncias de cada dimensão
perc_variancia <- (ACM$eig / sum(ACM$eig)) * 100
perc_variancia

# Quantidade de categorias por variável
quant_categorias <- apply(var_quali,
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

# Obtendo as coordenadas das observações
coord_obs <- ACM$li

# Adicionando as coordenadas ao banco de dados de variáveis quantitativas
var_quanti <- bind_cols(var_quanti, coord_obs)

# Coeficientes de correlação de Pearson para cada par de variáveis
rho <- rcorr(as.matrix(var_quanti), type="pearson")

corr_coef <- rho$r # Matriz de correlações
corr_sig <- round(rho$P, 5) # Matriz com p-valor dos coeficientes

# Elaboração de um mapa de calor das correlações de Pearson entre as variáveis
ggplotly(
    var_quanti %>%
    cor() %>%
    melt() %>%
    rename(Correlação = value) %>%
    ggplot() +
    geom_tile(aes(x = Var1, y = Var2, fill = Correlação)) +
    geom_text(aes(x = Var1, y = Var2, label = format(round(Correlação,3))),
              size = 3) +
    scale_fill_viridis_b() +
    labs(x = NULL, y = NULL) +
    theme_bw(base_size = 6))

### Elaboração a Análise Fatorial Por Componentes Principais ###

# Teste de esfericidade de Bartlett
cortest.bartlett(var_quanti)

# Elaboração da análise fatorial por componentes principais
fatorial <- principal(var_quanti,
                      nfactors = length(var_quanti),
                      rotate = "none",
                      scores = TRUE)
fatorial

# Eigenvalues (autovalores)
eigenvalues <- round(fatorial$values, 5)
eigenvalues
round(sum(eigenvalues), 2) # soma dos autovalores

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

### Elaboração da Análise Fatorial por Componentes Principais ###
### Fatores extraídos a partir de autovalores maiores que 1 ###

# Definição da quantidade de fatores com eigenvalues maiores que 1
k <- sum(eigenvalues > 1)
print(k)

# Elaboração da análise fatorial por componentes principais
fatorial_final <- principal(var_quanti,
                            nfactors = k,
                            rotate = "none",
                            scores = TRUE)

# Cálculo dos scores fatoriais
scores_fatoriais <- as.data.frame(fatorial_final$weights)

# Visualização dos scores fatoriais
round(scores_fatoriais, 3) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = FALSE, 
                font_size = 20)

# Cálculo dos fatores propriamente ditos
fatores <- as.data.frame(fatorial_final$scores)
View(fatores)

# Cálculo das cargas fatoriais
cargas_fatoriais <- as.data.frame(unclass(fatorial_final$loadings))

# Visualização das cargas fatoriais
round(cargas_fatoriais, 3) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped", 
                full_width = FALSE, 
                font_size = 20)

# Cálculo das comunalidades
comunalidades <- as.data.frame(unclass(fatorial_final$communality)) %>%
  rename(comunalidades = 1)

# Visualização das comunalidades para os 2 fatores extraídos
round(comunalidades, 3) %>%
  kable() %>%
  kable_styling(bootstrap_options = "striped",
                full_width = FALSE,
                font_size = 20)

# Loading plot com as cargas dos 2 primeiros fatores
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

# Criação de um ranking Critério da soma ponderada e ordenamento)
casas$ranking <- fatores$PC1 * variancia_compartilhada$PC1[2] +
                 fatores$PC2 * variancia_compartilhada$PC2[2] +
                 fatores$PC3 * variancia_compartilhada$PC3[2] + 
                 fatores$PC4 * variancia_compartilhada$PC4[2]

# Ranking e valor
corr_valor <- rcorr(as.matrix(casas[,16:17]))

valor_corr_coef <- corr_valor$r # Matriz de correlações
valor_corr_sig <- round(corr_valor$P, 5) # Matriz com p-valor dos coeficientes

# Fim!