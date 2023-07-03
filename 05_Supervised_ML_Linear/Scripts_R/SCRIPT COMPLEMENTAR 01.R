sd(tempodist$tempo)

library(scales)
show_col(viridis_pal()(20))

ggplotly(
  ggplot(tempodist, aes(x = distancia, y = tempo)) +
    geom_point(color = "#39568CFF", size = 2.5) +
    geom_smooth(aes(color = "Fitted Values"),
                method = "lm", formula = y ~ x, se = F, size = 2) +
    xlim(0, max(tempodist$distancia)) +
    ylim(0, max(tempodist$tempo)) +
    labs(x = "Distância",
         y = "Tempo",
         title = paste("R²:",
                       round(((cor(tempodist$tempo, tempodist$distancia))^2),4))) +
    scale_color_manual("Legenda:",
                       values = "grey50") +
    theme_classic()
)

anova(modelo_tempodist) #analysis of variance


