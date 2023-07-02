# Poisson:
pois <- rpois(1000, lambda = 2)
hist(pois, breaks = 10)

####
by(data = corruption$staff, INDICES = corruption$post, FUN = mean)

by(data = corruption$violations, INDICES = corruption$post, FUN = mean)

#####
library(equatiomatic)

extract_eq(modelo_poisson, use_coefs = T, coef_digits = 4) %>% 
  kable() %>% 
  kable_styling(font_size = 18)


#######
# LRTEST:
modelo_poisson0 <- glm(formula = violations ~ 1,
                       data = corruption,
                       family = "poisson")
logLik(modelo_poisson0)

chi2 <- -2*(logLik(modelo_poisson0) - logLik(modelo_poisson))
chi2

lrtest(modelo_poisson)

logLik(modelo_poisson) / 298


#####
exp(2.2127 + 0.0219*23 - 4.2968*0 + 0.3418*0.5) #postyes = 0 (antes da lei)
exp(2.2127 + 0.0219*23 - 4.2968*1 + 0.3418*0.5) #postyes = 1 (depois da lei)
