# Exemplo 1:

modelo_atrasos_errado <- lm(formula = atrasado ~ dist + sem,
                            data = Atrasado)
summary(modelo_atrasos_errado) # NÃO FAÇAM ISSO!

modelo_atrasos_errado2 <- glm(formula = atrasado ~ dist + sem, 
                              data = Atrasado, 
                              family = "gaussian")
summary(modelo_atrasos_errado2)

export_summs(modelo_atrasos, modelo_atrasos_errado, scale = F)



modelo_nulo <- glm(formula = atrasado ~ 1,
                   data = Atrasado,
                   family = "binomial")
logLik(modelo_nulo)

chi2 <- -2*(logLik(modelo_nulo)-logLik(modelo_atrasos))
chi2

pchisq(chi2, df=2, lower.tail = F)

AIC <- -2*logLik(modelo_atrasos) + 2*(2+1)
AIC

BIC <- -2*logLik(modelo_atrasos) + (2+1)*log(100)
BIC


pseudoR2MF <- (-2*logLik(modelo_nulo) - (-2*logLik(modelo_atrasos)))/
  (-2*logLik(modelo_nulo))
pseudoR2MF


objeto <- confusionMatrix(table(predict(modelo_atrasos, type = "response") >= 0.5,
                                Atrasado$atrasado == 1)[2:1, 2:1])
objeto
