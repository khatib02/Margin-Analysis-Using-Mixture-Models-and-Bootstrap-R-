library(gamlss)
library(robustbase)
library(ggplot2)
library(flexmix)
library(tree) # funcoes para estimar arvore de decisao
library(randomForest) 
library(gamlssbssn)
library(gbm)
library(rpart)
library(caret)
library(vip)
library(xgboost)
library(lightgbm)
library(parallel)
library(doParallel)
library(foreach)
library(dplyr)
library(car)
library(MASS)
library(purrr)
library(tidyr)

set.seed(123) 

dados <- read.csv("dados.csv")

# Transformacao do formato de variaveis
dados[] <- lapply(dados, function(x) {
  if(is.character(x)) factor(x) else x
})
dados$Postal.Code <- factor(dados$Postal.Code)
dados$Discount <- factor(dados$Discount)

# Criacao da resposta
dados$Margin <- dados$Profit/dados$Sales

# Distribuição das variaveis categoricas
table(dados$Ship.Mode) 
table(dados$Segment)
table(dados$Sub.Category)
table(dados$Discount)
table(dados$Quantity)

# Agrupamento de categorias raras
formata_df <- function(df) {
  
  df$division <- NA
  
  df$division[df$State %in% c(
    "Connecticut","Maine","Massachusetts","New Hampshire","Rhode Island","Vermont"
  )] <- "New England"
  
  df$division[df$State %in% c(
    "New Jersey","New York","Pennsylvania"
  )] <- "Middle Atlantic"
  
  df$division[df$State %in% c(
    "Illinois","Indiana","Michigan","Ohio","Wisconsin"
  )] <- "East North Central"
  
  df$division[df$State %in% c(
    "Iowa","Kansas","Minnesota","Missouri","Nebraska","North Dakota","South Dakota"
  )] <- "West North Central"
  
  df$division[df$State %in% c(
    "Delaware","District of Columbia","Florida","Georgia","Maryland","North Carolina",
    "South Carolina","Virginia","West Virginia"
  )] <- "South Atlantic"
  
  df$division[df$State %in% c(
    "Alabama","Kentucky","Mississippi","Tennessee"
  )] <- "East South Central"
  
  df$division[df$State %in% c(
    "Arkansas","Louisiana","Oklahoma","Texas"
  )] <- "West South Central"
  
  df$division[df$State %in% c(
    "Arizona","Colorado","Idaho","Montana","Nevada","New Mexico","Utah","Wyoming"
  )] <- "Mountain"
  
  df$division[df$State %in% c(
    "Alaska","California","Hawaii","Oregon","Washington"
  )] <- "Pacific"
  
  df$division <- factor(df$division)
  
  discount_num <- as.numeric(as.character(df$Discount))
  
  breaks <- seq(0, 0.8, by = 0.1)
  breaks <- round(breaks, 3)
  
  df$Discount_group <- cut(
    discount_num,
    breaks = breaks,
    include.lowest = TRUE,
    right = FALSE
  )
  
  df$Quantity_group <- cut(
    df$Quantity,
    breaks = c(0,1,3,5,9,Inf),
    labels = c("1","2-3","4-5","6-9","10+"),
    right = TRUE
  )
  
  df
}

dados <- formata_df(dados)

# fracoes de treino, validacao e teste
train_frac <- 0.7
valid_frac <- 0.15
test_frac  <- 0.15

# Geracao de índices aleatórios
n <- nrow(dados)
indices <- sample(1:n, n)

# Cria treino, validacao e teste
train_index <- indices[1:floor(train_frac*n)]
valid_index <- indices[(floor(train_frac*n)+1):(floor((train_frac+valid_frac)*n))]
test_index  <- indices[(floor((train_frac+valid_frac)*n)+1):n]
train_data <- dados[train_index, ]
valid_data <- dados[valid_index, ]
test_data  <- dados[test_index, ]

# Resumos do treino
summary(train_data)
str(train_data)
sum(is.na(train_data))

# Distribuicao das covariaveis importantes
table(train_data$Discount_group)
table(train_data$Ship.Mode)
table(train_data$Segment)
table(train_data$division)
table(train_data$Sub.Category)
table(train_data$Quantity_group)

# # Distribuicao condicional da resposta
par(mfrow=c(2,3))
adjbox(Margin ~ Discount_group, data=train_data, 
       main="Example of grouped variable: Discount", ylab = "Margin", xlab = "Discounts (groups)") 


axis(2)  # eixo y
axis(
  1,
  at = seq_along(levels(train_data$Discount_group)),
  labels = levels(train_data$Discount_group)
)
box()
adjbox(Margin ~ Ship.Mode, data=train_data, main="Entrega vs Margem") 
adjbox(Margin ~ Segment, data=train_data, main="Segmento vs Margem") 
adjbox(Margin ~ division, data=train_data, main="Divisao vs Margem") 
adjbox(Margin ~ Sub.Category, data=train_data, main="Subcategoria vs Margem") 
adjbox(Margin ~ Quantity_group, data=train_data, main="Quantidade vs Margem") 



# Distribuicao marginal da resposta
plot(density(train_data$Margin))
boxplot(train_data$Margin)
hist(train_data$Margin)

# Dataframe de metricas
res <- data.frame(modelo = character(14), rmse = numeric(14), mae=numeric(14), r2_adj=numeric(14))


############################### Modelos lineares ###############################

attach(train_data)
### modelo linear normal

# Ajuste
modelo1 <- lm(Margin ~ Discount_group + Ship.Mode + Segment + division + Sub.Category + Quantity_group)
modelo1 <- stepAIC(modelo1) # Selecao de variaveis por AIC

# Propriedades do modelo
summary(modelo1) 
plot(modelo1, 2, main="heavy tails suggests t-student distribution")

# previsoes
y_hat <- predict(modelo1, newdata = valid_data)
y_real <- valid_data$Margin

# Metricas
rmse <- sqrt(mean((y_hat - y_real)^2))
mae  <- mean(abs(y_hat - y_real))
R2_pred <- 1 - sum((y_real - y_hat)^2) /
  sum((y_real - mean(y_real))^2)

# Salvamento das metricas
res[1,] <- c("Linear Gaussian model without interactions", rmse, mae, R2_pred)
res

### modelo linear com interacoes

# Ajuste
modelo2 <- lm(Margin ~ (Discount_group + Sub.Category + Quantity_group)^2)
modelo2 <- stepAIC(modelo2)
summary(modelo2)
plot(modelo2)

# previsoes
y_hat <- predict(modelo2, newdata = valid_data)
y_real <- valid_data$Margin

# Métricas básicas
rmse <- sqrt(mean((y_hat - y_real)^2))
mae  <- mean(abs(y_hat - y_real))
R2_pred <- 1 - sum((y_real - y_hat)^2) /
  sum((y_real - mean(y_real))^2)

# Salvamento das metricas
res[2,] <- c("Linear Gaussian model with interactions", rmse, mae, R2_pred)
res

### Modelo aditivo generalizado t-student

# Ajuste
modelo3 <- gamlss(Margin ~ Discount_group + Sub.Category + Quantity_group,
                  family=TF, data=train_data)

# Resumos do modelo
summary(modelo3)
plot(modelo3, which=3)
plot(density(residuals(modelo3)), main="Presence of bimodality in the residuals’ density")
wp(modelo3, main="")
rqres.plot(modelo3)

# Previsoes
y_hat <- predict(modelo3, newdata = valid_data[, c("Discount_group", "Sub.Category",
                                                   "Quantity_group")])
y_real <- valid_data$Margin

# Metricas basicas
rmse <- sqrt(mean((y_hat - y_real)^2))
mae  <- mean(abs(y_hat - y_real))
R2_pred <- 1 - sum((y_real - y_hat)^2) /
  sum((y_real - mean(y_real))^2)

# Salvamento das metricas
res[3,] <- c("GAMLSS Student-t model without interactions", rmse, mae, R2_pred)
res

### Modelo aditivo generalizado t-student com interacoes

# Ajuste
modelo4 <- gamlss(Margin ~ (Discount_group + Sub.Category + Quantity_group)^2,
                  family=TF, data=train_data)

# Resumos do modelo
summary(modelo4)
plot(modelo4)
wp(modelo4)
rqres.plot(modelo4)

# Previsão da média (mu)
pred_gamlss <- predict(modelo4, newdata = valid_data[, c("Discount_group", "Sub.Category",
                                                         "Quantity_group")], what = "mu", type = "response")

# Previsoes
y_hat <- predict(modelo4, newdata = valid_data[, c("Discount_group", "Sub.Category",
                                                   "Quantity_group")])
y_real <- valid_data$Margin

# Métricas basicas
rmse <- sqrt(mean((y_hat - y_real)^2))
mae  <- mean(abs(y_hat - y_real))
R2_pred <- 1 - sum((y_real - y_hat)^2) /
  sum((y_real - mean(y_real))^2)

# Salvamento das metricas
res[4,] <- c("GAMLSS Student-t model with interactions", rmse, mae, R2_pred)
res

### Modelo linear generalizado com mistura de normais

# Ajuste
modelo5 <- initFlexmix(Margin ~ Discount_group + Ship.Mode + Segment + division + Sub.Category + Quantity_group,
                      data = train_data, k = 2, nrep = 20, model = FLXMRglm(family = "gaussian")) 

# Resumos do modelo
summary(modelo5)
plot(modelo5)

# Probabilidades de cada ponto pertencer a cada cluster
probs <- posterior(modelo5)  # n x k

# Coeficientes de cada cluster
coef1 <- modelo5@components$Comp.1[[1]]@parameters$coef
coef2 <- modelo5@components$Comp.2[[1]]@parameters$coef

# Matriz de covariaveis com intercepto
X_valid <- model.matrix(Margin ~ Discount_group + Ship.Mode + Segment + division + Sub.Category + Quantity_group,
                        data = valid_data)

# Previsao por componente
pred1 <- X_valid %*% coef1
pred2 <- X_valid %*% coef2
probs <- posterior(modelo5, newdata = valid_data)
pred_mix <- probs[,1]*pred1 + probs[,2]*pred2

# Metricas
real <- valid_data$Margin
MAE  <- mean(abs(pred_mix - real))
RMSE <- sqrt(mean((pred_mix - real)^2))
R2_pred <- 1 - sum((y_real - pred_mix)^2) /
  sum((y_real - mean(y_real))^2)

res[5,] <- c("Additive Gaussian mixture model", RMSE, MAE, R2_pred)
res

# Criacao de residuos
resid_mix <- valid_data$Margin - pred_mix
sigma_mix <- probs[,1]*modelo5@components$Comp.1[[1]]@parameters$sigma +
  probs[,2]*modelo5@components$Comp.2[[1]]@parameters$sigma  # ajuste se k > 2
resid_std <- resid_mix / sigma_mix

# Banda de 90% dos residuos
n <- length(resid_std)
resid_sorted <- sort(resid_std)
p <- ppoints(n)
theo_q <- qnorm(p)
se <- sqrt(p*(1-p)/n) / dnorm(theo_q)
upper <- theo_q + qnorm(0.95) * se
lower <- theo_q - qnorm(0.95) * se

# Plot dos residuos com banda
plot(theo_q, resid_sorted, main="Q-Q plot com banda de 90%",
     xlab="Quantis teóricos", ylab="Resíduos padronizados", pch=19)
polygon(c(theo_q, rev(theo_q)), c(upper, rev(lower)), col=rgb(1,0,0,0.2), border=NA)  # sombra
abline(0,1, col="blue", lwd=2)

# Densidade dos resíduos
plot(density(resid_sorted), main="Residual density approximates a normal distribution (good fit)")

# Classificacao de cada compra do conjunto de validacao nos clusters 1 e 2
# usando bayes classifier (apenas com 80% de "certeza")
post_valid <- posterior(modelo5, newdata = valid_data)
valid_data$max_prob <- apply(post_valid, 1, max)
valid_data$cluster_bayes <- ifelse(
  valid_data$max_prob < 0.8,
  NA,
  apply(post_valid, 1, which.max)
)

# Porcentagem de compras incertas: NA
sum(is.na(valid_data$cluster_bayes))/nrow(valid_data) 

# Margem por cluster 
adjbox(
  Margin ~ cluster_bayes,
  data = valid_data,
  main = "Margem por cluster (validação)"
)

# Probabilidade de margem negativa por cluster
valid_data$margin_neg <- valid_data$Margin < 0
risk_table <- valid_data %>%
  group_by(cluster_bayes) %>%   # cluster definido só por X
  summarise(
    n = n(),
    prob_margin_neg = mean(margin_neg),
    .groups = "drop"
  )
risk_table

# Margem esperada por cluster
impact_table <- valid_data %>%
  group_by(cluster_bayes) %>%
  summarise(
    mean_margin = mean(Margin),
    median_margin = median(Margin),
    sd_margin = sd(Margin),
    .groups = "drop"
  )
impact_table

### Modelo linear generalizado heterocedastico normal bimodal

# Ajuste
modelo6 <- gamlss(Margin ~ Discount_group + Sub.Category + Quantity_group,
                  sigma.formula = ~ Discount_group + Sub.Category + Quantity_group,
                  family=BSSN(), data=train_data)

# Resumos do modelo
summary(modelo6)
plot(modelo6)
wp(modelo6)
rqres.plot(modelo6)

# previsao
colunas <- c("Margin", "Discount_group", "Sub.Category", "Quantity_group")
y_hat <- predict(modelo6, newdata = valid_data[,colunas])
y_real <- valid_data$Margin

# Métricas básicas
rmse <- sqrt(mean((y_hat - y_real)^2))
mae  <- mean(abs(y_hat - y_real))
R2_pred <- 1 - sum((y_real - y_hat)^2) /
  sum((y_real - mean(y_real))^2)

res[6,] <- c("Heteroskedastic Gaussian mixture model", rmse, mae, R2_pred)
res

### Modelo linear generalizado two-part hurdle

# Criacao de variavel indicadora para margem positiva
train_data$pos <- as.integer(train_data$Margin > 0)

# Ajuste do parte binomial do modelo
modelo7_bin <- gamlss(
  pos ~ Discount_group + Ship.Mode + Segment +
    division + Sub.Category + Quantity_group,
  family = BI,   # Bernoulli
  data = train_data
)

# Selecao de variaveis por stepGAIC
modelo7_bin <- stepGAIC(modelo7_bin)

# Resumos do modelo binomial
plot(modelo7_bin)
rqres.plot(modelo7_bin)
histogram(residuals(modelo7_bin))
summary(modelo7_bin)

# Densidade da Margem positiva
plot(density(subset(train_data, Margin > 0)$Margin))

# Ajuste da parte positiva da margem
fit_pos <- gamlss(formula = Margin ~ Discount_group + Sub.Category + Quantity_group,
                  family = TF, data = subset(train_data, Margin > 0)) 

# Resumo do modelo continuo
summary(fit_pos)
plot(fit_pos)
rqres.plot(fit_pos)
histogram(residuals(fit_pos))
wp(fit_pos)
AIC(fit_pos)

# Densidade da margem negativa 
plot(density(subset(train_data, Margin < 0)$Margin))

Loss <- -subset(train_data, Margin<0)$Margin

# Ajuste da parte negativa da margem
fit_neg <- gamlss(
  Loss ~ Discount_group + Sub.Category + Quantity_group,
  family = TF,
  data = subset(train_data, Margin < 0)
)

# Resumos do modelo negativo
plot(fit_neg)
rqres.plot(fit_neg)
histogram(residuals(fit_neg))
wp(fit_neg)
AIC(fit_neg) 

# Probabilidade de cada observacao da validacao ser positiva
colunas <- c("Sub.Category", "Discount_group", "Ship.Mode")
p_hat <- predict(
  modelo7_bin,
  newdata = valid_data[,colunas],
  type = "response"
)

# Valores preditos da margem no validacao pelo modelo positivo
colunas <- c("Discount_group", "Sub.Category", "Quantity_group")
Epos_hat <- predict(
  fit_pos,
  newdata = valid_data[,colunas],
  type = "response"
)

# Valores preditos da margem no validacao pelo modelo positivo
Eneg_hat <- predict(
  fit_neg,
  newdata = valid_data[,colunas],
  type = "response"
)

# Predicao da margem calculada pela ponderacao dos modelos pelas probabilidades
Margin_hat <- p_hat * Epos_hat - (1 - p_hat) * Eneg_hat

# Metricas
real <- valid_data$Margin
MAE  <- mean(abs(Margin_hat - real))
RMSE <- sqrt(mean((Margin_hat - real)^2))
y_bar <- mean(train_data$Margin)
R2_pred <- 1 - sum((real - Margin_hat)^2) /
  sum((real - y_bar)^2)

# Salvamento das metricas
res[7,] <- c("Two-part hurdle model", RMSE, MAE, R2_pred)
res

############################# Modelos não lineares #############################

### Arvore de regressão

# Ajuste
modelo8 <- tree(Margin ~ Discount_group + Ship.Mode + Segment + division +
                  Sub.Category + Quantity_group, data = train_data)

# Grafico
plot(modelo8) 
text(modelo8) 

# Metricas
y_hat <- predict(modelo8, newdata = valid_data)
y_real <- valid_data$Margin
MAE  <- mean(abs(y_hat - y_real))
RMSE <- sqrt(mean((y_hat - y_real)^2))
R2_pred <- 1 - sum((y_real - y_hat)^2) /
  sum((y_real - mean(y_real))^2)

# Salva metricas
res[8,] <- c("Regression tree", RMSE, MAE, R2_pred)
res

### arvore podada

# Ajuste
modelo9 <- rpart(
  Margin ~ Discount_group + Sub.Category + Quantity_group,
  data = train_data,
  method = "anova",
  control = rpart.control(
    cp = 0.001,   # árvore cresce bastante
    xval = 10     # 10-fold cross-validation
  )
)

# Grafico
plot(modelo9)
text(modelo9) 

# Metricas
y_hat <- predict(modelo9, newdata = valid_data)
y_real <- valid_data$Margin
MAE  <- mean(abs(y_hat - y_real))
RMSE <- sqrt(mean((y_hat - y_real)^2))
R2_pred <- 1 - sum((y_real - y_hat)^2) /
  sum((y_real - mean(y_real))^2)

# Salva metricas
res[9,] <- c("Pruned regression tree", RMSE, MAE, R2_pred)
res

### Bagging

# Ajuste
modelo10 <- randomForest(
  Margin ~ Discount_group + Sub.Category + Quantity_group + division + Segment + 
    Ship.Mode,
  data = train_data,
  mtry=3,
  ntree = 1000
)

# Grafico
plot(modelo10)

# Encontra numero de arvores otimo
n_otimo <- which.min(modelo10$mse)

# Ajuste usando n_otimo
ctrl <- trainControl(method = "cv", number = 5, verboseIter = FALSE, allowParallel = TRUE)
modelo10 <- train(
  Margin ~ Discount_group + Sub.Category + Quantity_group + division + Segment + Ship.Mode,
  data = train_data,
  method = "rf",
  trControl = ctrl,
  tuneGrid = data.frame(mtry = 6),
  ntree = n_otimo
)

# Metricas
y_hat <- predict(modelo10, newdata = valid_data)
y_real <- valid_data$Margin
MAE  <- mean(abs(y_hat - y_real))
RMSE <- sqrt(mean((y_hat - y_real)^2))
R2_pred <- 1 - sum((y_real - y_hat)^2) /
  sum((y_real - mean(y_real))^2)

# Salva metricas
res[10,] <- c("Bagged regression tree", RMSE, MAE, R2_pred)
res

# Floresta Aleatoria

# Ajuste usando notimo
ctrl <- trainControl(method = "cv", number = 5, verboseIter = TRUE, allowParallel = FALSE)
modelo11 <- train(
  Margin ~ Discount_group + Sub.Category + Quantity_group + division + Segment + 
    Ship.Mode,
  data = train_data,
  method = "rf",
  ntree = n_otimo,
  trControl = ctrl,
  tuneGrid = expand.grid(mtry = 1:5)
)

# Metricas
y_hat <- predict(modelo11, newdata = valid_data)
y_real <- valid_data$Margin
MAE  <- mean(abs(y_hat - y_real))
RMSE <- sqrt(mean((y_hat - y_real)^2))
R2_pred <- 1 - sum((y_real - y_hat)^2) /
  sum((y_real - mean(y_real))^2)

# Salva metricas
res[11,] <- c("Random forest", RMSE, MAE, R2_pred)
res

### Boosting

# Ajuste
modelo12 <- gbm(Margin ~ Discount_group + Sub.Category + Quantity_group + division + Segment + 
                  Ship.Mode, # formula
           distribution = "gaussian", 
           n.trees = n_otimo, # numero de arvores
           interaction.depth = 2, # profundidade maxima da arvore
           shrinkage = 0.1, # taxa de aprendizagem # 0.01
           data = train_data,
           n.cores=9,
           )

# importancia das variaveis
summary(modelo12)
vip(modelo12)

# grafico de dependencia parcial de total_day_minutes
plot(modelo12, i="Discount_group")

# grafico de dependencia parcial de number_customer_service_calls
plot(modelo12, i="Sub.Category")

# Predicoes
y_hat <- predict(modelo12, newdata = valid_data)
y_real <- valid_data$Margin

# Metricas
MAE  <- mean(abs(y_hat - y_real))
RMSE <- sqrt(mean((y_hat - y_real)^2))
R2_pred <- 1 - sum((y_real - y_hat)^2) /
  sum((y_real - mean(y_real))^2)

# Salvamento das metricas
res[12,] <- c("Boosted regression tree", RMSE, MAE, R2_pred)
res

### XGBOOST

# Matriz design de treino
X_train <- model.matrix(
  ~ Discount_group + Ship.Mode + Segment + division +
    Sub.Category + Quantity_group,
  data = train_data
)[, -1]   # remove intercepto

# Matriz design de validacao
X_valid <- model.matrix(
  ~ Discount_group + Ship.Mode + Segment + division +
    Sub.Category + Quantity_group,
  data = valid_data
)[, -1]   # remove intercepto

# Ajuste
y_train <- train_data$Margin
y_valid <- valid_data$Margin
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dvalid <- xgb.DMatrix(data = X_valid, label = y_valid)
modelo13 <- xgb.train(
  data = dtrain,
  nrounds = 500,
  watchlist = list(train = dtrain, valid = dvalid),
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.05,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8,
  early_stopping_rounds = 30,
  verbose = 1
)

# Predicao
pred_xgb <- predict(modelo13, X_valid)
y_real <- valid_data$Margin

# Metricas
MAE  <- mean(abs(pred_xgb - y_real))
RMSE <- sqrt(mean((pred_xgb - y_real)^2))
R2_pred <- 1 - sum((y_real - pred_xgb)^2) /
  sum((y_real - mean(y_real))^2)

# Salvamento das metricas
res[13,] <- c("XGBoost", RMSE, MAE, R2_pred)
res

### LightGBM

#Ajuste
lgb_train <- lgb.Dataset(data = X_train, label = y_train)
lgb_valid <- lgb.Dataset(data = X_valid, label = y_valid)
colnames(lgb_train) <- make.names(colnames(lgb_train))
colnames(lgb_valid) <- make.names(colnames(lgb_valid))
lgb_fit <- lgb.train(
  params = list(
    objective = "regression",
    metric = "rmse",
    learning_rate = 0.05,
    num_leaves = 31,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq = 5
  ),
  data = lgb_train,
  nrounds = 1000,
  valids = list(valid = lgb_valid),
  early_stopping_rounds = 50,
  verbose = 1
)

# Predicao
pred_lgb <- predict(lgb_fit, X_valid)

# Metricas
y_real <- valid_data$Margin
MAE  <- mean(abs(pred_lgb - y_real), na.rm = TRUE)
RMSE <- sqrt(mean((pred_lgb - y_real)^2,na.rm = TRUE))
R2_pred <- 1 - sum((y_real - pred_lgb)^2) /
  sum((y_real - mean(y_real))^2)

# Salvamento das metricas
res[14,] <- c("LightGBM", RMSE, MAE, R2_pred)
res

sort_by.data.frame(res, res$r2_adj, decreasing=TRUE)

####################### Modelo final: Mistura de Normais #######################

# Juncao do treino e validacao
final_data <- rbind(train_data, valid_data)

# Ajuste do modelo de misturas
modelo_final <- initFlexmix(Margin ~ Discount_group + Ship.Mode + Segment + division + Sub.Category + Quantity_group,
                       data = final_data, k = 2, nrep =20, model = FLXMRglm(family = "gaussian")) 

# Verifica se o modelo convergiu
modelo_final@converged

# Resumo do modelo
summary(modelo_final)
plot(modelo_final)

# Probabilidades de cada ponto pertencer a cada cluster (1 e 2)
probs <- posterior(modelo_final)

# Extracao dos coeficientes estimados de cada cluster
coef1 <- modelo_final@components$Comp.1[[1]]@parameters$coef
coef2 <- modelo_final@components$Comp.2[[1]]@parameters$coef

# Construcao dos residuos do modelo
resid_mix <- final_data$Margin - rowSums(fitted(modelo_final) * posterior(modelo_final) )

# Calculo de sigma ponderado pelas propabilidades do modelo
sigma_mix <- probs[,1]*modelo_final@components$Comp.1[[1]]@parameters$sigma +
  probs[,2]*modelo_final@components$Comp.2[[1]]@parameters$sigma  

# Calculo dos residuos padronizados
resid_std <- resid_mix / sigma_mix

# Banda de 90% dos residuos
n <- length(resid_std)
resid_sorted <- sort(resid_std)
p <- ppoints(n)
theo_q <- qnorm(p)
se <- sqrt(p*(1-p)/n) / dnorm(theo_q)
upper <- theo_q + qnorm(0.95) * se
lower <- theo_q - qnorm(0.95) * se

# Matriz de design dos dados de teste
X_test <- model.matrix(Margin ~ Discount_group + Ship.Mode + Segment + division + Sub.Category + Quantity_group,
                       data = test_data)

# Previsao por cluster da mistura
pred1 <- X_test %*% coef1
pred2 <- X_test %*% coef2
probs <- posterior(modelo_final, newdata = test_data)
pred_mix <- probs[,1]*pred1 + probs[,2]*pred2

# Predicao
real <- test_data$Margin
MAE  <- mean(abs(pred_mix - real))
RMSE <- sqrt(mean((pred_mix - real)^2))
R2_pred <- 1 - sum((real - pred_mix)^2) /
  sum((real - mean(real))^2)

# qqplot dos residuos ordenados
plot(theo_q, resid_sorted, main="Q-Q plot com banda de 90%",
     xlab="Quantis teóricos", ylab="Resíduos padronizados", pch=19)
polygon(c(theo_q, rev(theo_q)), c(upper, rev(lower)), col=rgb(1,0,0,0.2), border=NA)  # sombra
abline(0,1, col="blue", lwd=2)

# Classificar cada compra do conjunto de validacao nos clusters 1 e 2
post_test <- posterior(modelo_final, newdata = test_data)
test_data$max_prob <- apply(post_test, 1, max)
test_data$cluster_bayes <- ifelse(
  test_data$max_prob < 0.5,
  NA,
  apply(post_test, 1, which.max)
)

# Boxplot da Margem por cluster
adjbox(
  Margin ~ cluster_bayes,
  data = test_data,
  main = "Margin por cluster (validação)"
)

# Gráfico combinado de densidades
df <- data.frame(
  valor = test_data$Margin,
  grupo = factor(test_data$cluster_bayes,
                 levels = c(1, 2),
                 labels = c("Cluster 1", "Cluster 2"))
)
ggplot(df, aes(x = valor, fill = grupo)) +
  geom_density(alpha = 0.5) +
  labs(
    title = "Margin density by cluster (Bayesian/MAP)",
    x = "Margin",
    y = "Density"
  ) +
  theme_minimal()

# Probabilidade de margem negativa e margem esperada por cluster
resumo <- resumo %>%
  mutate(
    scale_factor = max(abs(mean_margin)),        # para eixo secundário
    mean_scaled  = mean_margin / scale_factor,
    label_mean   = sprintf("%.2f", mean_margin)
  )

# --- Gráfico atualizado ---
ggplot(resumo, aes(x = factor(cluster_bayes))) +
  
  # Barra: probabilidade de margem negativa
  geom_col(aes(y = prob_neg, fill = "P(Margin < 0)"),
           width = 0.5, alpha = 0.7) +
  
  # Ponto: valor esperado da margem
  geom_point(aes(y = mean_scaled, color = "E[Margin]"),
             size = 3) +
  
  # Rótulo do valor esperado, ajustado para não cortar
  geom_text(aes(y = mean_scaled + 0.09, label = label_mean),
            color = "black",
            size = 3.5) +
  
  # Eixo primário e secundário
  scale_y_continuous(
    name = "Probability of Negative Margin",
    sec.axis = sec_axis(
      ~ . * resumo$scale_factor[1],
      name = "Margin Expected value"
    )
  ) +
  
  # Cores manuais
  scale_fill_manual(values = c("P(Margin < 0)" = "grey70")) +
  scale_color_manual(values = c("E[Margin]" = "black")) +
  
  # Labels e tema
  labs(
    x = "Cluster",
    title = "Risk (negative margin) and average return by cluster"
  ) +
  theme_minimal() +
  theme(
    legend.title = element_blank(),
    legend.position = "bottom"
  )

# Drivers estruturais do cluster
conc_model <- modelo_final@concomitant
coef_conc <- conc_model@coef
odds_ratio <- exp(coef_conc)
options(scipen = 999)
driver_table1 <- data.frame(
  variable   = rownames(coef_conc),
  log_odds   = round(coef_conc[2],2),
  odds_ratio = round(exp(coef_conc[2]),2),
  prob_cluster_2 = round(exp(coef_conc[2])/(exp(coef_conc[2])+1),2))
rownames(driver_table1) <- c()
driver_table1

"
Interpretacao do driver_table1:
odds_ratio.2 = P(cluster2/X)/P(cluster1/X)
A probabilidade de pertencer ao cluster 2 é odds_ratio.2 vezes a probabilidade de pertencer ao cluster 1
"

set.seed(123)

### Avaliacao de convergência para diferentes nrep por meio do conjunto de validacao
# nrep: numero de repeticoes do ajuste da mistura
# ATENCAO: O codigo abaixo pode demorar alguns minutos para conclusao

# Grade de nreps para avaliacao
nrep_grid <- c(1, 3, 5, 10, 20, 35, 60, 100, 150)

# Ajuste do modelo para diferentes nrep
fits <- lapply(nrep_grid, function(nr)
  initFlexmix(
    Margin ~ Discount_group + Ship.Mode + Segment + division +
      Sub.Category + Quantity_group,
    data = train_data,
    k = 2,
    nrep = nr,
    model = FLXMRglm(family = "gaussian")
    )
  )

# Rotulacao dos ajustes para referencia
names(fits) <- paste0("nrep_", nrep_grid)

# Extracao das probabilidades para todos ajustes
probs_full <- lapply(fits, function(f) {
  p <- posterior(f, newdata = test_data)
  stopifnot(is.matrix(p), ncol(p) == 2)
  p
})

# Calculo dos coeficientes para conjunto de teste para todos modelos
preds_comp <- lapply(fits, function(f) {
  pr <- predict(f, newdata = test_data)
  stopifnot(is.list(pr), all(c("Comp.1", "Comp.2") %in% names(pr)))
  pr
})


# Predicao dos modelos no conjunto de teste
preds_marginal <- Map(function(pred, p) {
  p[,1] * pred$Comp.1[,1] +
    p[,2] * pred$Comp.2[,1]
}, preds_comp, probs_full)
pred_mat <- do.call(cbind, preds_marginal)
colnames(pred_mat) <- names(fits)

# Avaliacao dos diferentes nrep no modelo
sd_pred <- apply(pred_mat, 1, sd)
ref <- pred_mat[, "nrep_20"]
delta_pred <- sweep(pred_mat, 1, ref, "-")
class_mat <- sapply(probs_full, function(p)
  ifelse(p[,2] > 0.5, 2, 1))
colnames(class_mat) <- names(fits)
ref_class <- class_mat[, "nrep_20"]
swap_rate <- sapply(colnames(class_mat), function(nm)
  mean(class_mat[, nm] != ref_class))
prob_mat <- sapply(probs_full, function(p) p[,2])
apply(prob_mat, 1, sd) |> summary()
df_compare <- data.frame(
  nrep = nrep_grid,
  mean_abs_delta = colMeans(abs(delta_pred)),
  cluster_swap = swap_rate
)
df_compare
"
mean_abs_delta: O quanto a classificacao muda em relacao a referencia (nrep=20)
cluster_swap: Mudança absoluta media na predicao em relacao a referencia (nrep=20)
melhor nrep = 20, pois equilibra predicao com performance computacional
"
help("initFlexmix")

### Boostrap do modelo
# ATENCAO: o codigo abaixo pode demorar horas para conclusao

set.seed(123)

# Configuracao
n <- nrow(final_data)
B <- 1000
log_dir <- "bootstrap_logs" 
dir.create(log_dir, showWarnings = FALSE) 
boot_dir <- "boot_results"
dir.create(boot_dir, showWarnings = FALSE)
n_cores <- max(1, floor(detectCores())) 
cl <- makeCluster(n_cores) 
registerDoParallel(cl)
X_test <- model.matrix(
  Margin ~ Discount_group + Ship.Mode + Segment + division +
    Sub.Category + Quantity_group,
  data = test_data
)
y_test <- test_data$Margin

# Bootstrap com paralelizacao e salvamento automatico no disco
boot_results <- foreach(
  b = 1:B,
  .packages = "flexmix"
) %dopar% {
  
  out_file <- file.path(boot_dir, paste0("boot_", sprintf("%04d", b), ".rds"))
  
  if (file.exists(out_file)) {
    res <- list(ok = FALSE, skipped = TRUE)
    saveRDS(res, out_file)
    res
  } else {
    
    log_file <- file.path(log_dir, paste0("log_pid_", Sys.getpid(), ".txt"))
    cat(paste0("Bootstrap ", b, " | START | ", Sys.time(), "\n"),
        file = log_file, append = TRUE)
    
    ## ------------------------
    ## Bootstrap sample
    ## ------------------------
    idx <- sample(seq_len(nrow(final_data)), replace = TRUE)
    df_boot <- final_data[idx, ]
    
    ## ------------------------
    ## Fit model
    ## ------------------------
    modelo_boot <- tryCatch(
      initFlexmix(
        Margin ~ Discount_group + Ship.Mode + Segment + division +
          Sub.Category + Quantity_group,
        data = df_boot,
        nrep = 20, 
        k = 2,
        model = FLXMRglm(family = "gaussian")
      ),
      error = function(e) NULL
    )
    
    if (is.null(modelo_boot) || !isTRUE(modelo_boot@converged)) {
      
      res <- list(ok = FALSE)
      
    } else {
      
      ## ------------------------
      ## Concomitant model
      ## ------------------------
      coef_conc_full <- modelo_boot@concomitant@coef
      
      # garante matriz
      coef_conc_full <- as.matrix(coef_conc_full)
      
      coef_conc <- as.numeric(coef_conc_full[, 2])
      names(coef_conc) <- paste0("conc_", rownames(coef_conc_full))
      
      ## ------------------------
      ## Component coefficients
      ## ------------------------
      coef_c1 <- modelo_boot@components$Comp.1[[1]]@parameters$coef
      coef_c2 <- modelo_boot@components$Comp.2[[1]]@parameters$coef
      
      coef_c1 <- as.numeric(coef_c1)
      coef_c2 <- as.numeric(coef_c2)
      
      names(coef_c1) <- names(modelo_boot@components$Comp.1[[1]]@parameters$coef)
      names(coef_c2) <- names(modelo_boot@components$Comp.2[[1]]@parameters$coef)
      
      ## ------------------------
      ## Anchor labels (label switching) — COMPLETO
      ## ------------------------
      int1 <- coef_c1["(Intercept)"]
      int2 <- coef_c2["(Intercept)"]
      
      swapped <- FALSE
      if (!is.na(int1) && !is.na(int2) && int1 < int2) {
        
        ## troca coeficientes dos componentes
        tmp <- coef_c1
        coef_c1 <- coef_c2
        coef_c2 <- tmp
        
        ## troca coeficientes do concomitant
        coef_conc <- -coef_conc
        
        swapped <- TRUE
      }
      
      ## ------------------------
      ## Align with X_test
      ## ------------------------
      beta_c1 <- coef_c1[colnames(X_test)]
      beta_c2 <- coef_c2[colnames(X_test)]
      
      if (any(is.na(beta_c1)) || any(is.na(beta_c2))) {
        res <- list(ok = FALSE)
      } else {
        
        ## ------------------------
        ## Component predictions
        ## ------------------------
        yhat_c1 <- as.vector(X_test %*% beta_c1)
        yhat_c2 <- as.vector(X_test %*% beta_c2)
        
        ## ------------------------
        ## Posterior probabilities
        ## ------------------------
        probs_test <- posterior(modelo_boot, newdata = test_data)
        
        if (!is.matrix(probs_test) || nrow(probs_test) != nrow(test_data)) {
          res <- list(ok = FALSE)
        }
        
        if (anyNA(probs_test)) {
          
          res <- list(ok = FALSE)
          
        } else {
          
          ## se houve troca, troca probabilidades
          if (swapped) {
            probs_test <- probs_test[, c(2, 1)]
          }
          
          ## ------------------------
          ## Mixture prediction
          ## ------------------------
          yhat_test <- probs_test[, 1] * yhat_c1 +
            probs_test[, 2] * yhat_c2
          
          ## ------------------------
          ## Output
          ## ------------------------
          res <- list(
            coef_m    = c(
              coef_conc,
              setNames(coef_c1, paste0("c1_", names(coef_c1))),
              setNames(coef_c2, paste0("c2_", names(coef_c2)))
            ),
            probs     = probs_test,
            yhat_test = yhat_test,
            ok        = TRUE
          )
        }
      }
    }
    
    saveRDS(res, out_file)
    
    cat(paste0("Bootstrap ", b, " | SAVED\n"),
        file = log_file, append = TRUE)
    
    res
  }
}

stopCluster(cl)

# Codigo para resolver Error in summary.connection(connection) : invalid connection
unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list=ls(name=env), pos=env)
}
unregister_dopar()

### Extracao de resultados do bootstrap que convergiram apenas
files <- list.files(boot_dir, full.names = TRUE)
boot_list <- lapply(files, readRDS)
boot_ok <- boot_list[sapply(boot_list, function(x) isTRUE(x$ok))]

# matriz de predicoes
pred_matrix <- do.call(
  cbind,
  lapply(boot_ok, function(x) x$yhat_test)
)


# Probs de cluster
probs_array <- array(
  NA,
  dim = c(
    nrow(boot_ok[[1]]$probs),
    ncol(boot_ok[[1]]$probs),
    length(boot_ok)
  )
)
for (b in seq_along(boot_ok)) {
  probs_array[, , b] <- boot_ok[[b]]$probs
}
dim(probs_array) # n_test × 2 × B

# Coeficientes do concomitant
coef_conc_mat <- do.call(
  rbind,
  lapply(boot_ok, function(x) {
    x$coef_m[grep("^conc_", names(x$coef_m))]
  })
)
colnames(coef_conc_mat) <- sub("^conc_", "", colnames(coef_conc_mat))

# Probabilidades de cluster 2
probs_mat <- do.call(
  rbind,
  lapply(boot_ok, function(x) {
    x$probs[,2]
  })
)

# Coefs do cluster 1
coef_c1_mat <- do.call(
  rbind,
  lapply(boot_ok, function(x) {
    x$coef_m[grep("^c1_", names(x$coef_m))]
  })
)
colnames(coef_c1_mat) <- sub("^c1_", "", colnames(coef_c1_mat))

# Coefs do cluster 1
coef_c2_mat <- do.call(
  rbind,
  lapply(boot_ok, function(x) {
    x$coef_m[grep("^c2_", names(x$coef_m))]
  })
)
colnames(coef_c2_mat) <- sub("^c2_", "", colnames(coef_c2_mat))

# Verificar label switching
int_c1 <- coef_c1_mat[, "(Intercept)"]
int_c2 <- coef_c2_mat[, "(Intercept)"]
plot(int_c1, int_c2)
abline(0, 1, col = "red")
mean(int_c1 > int_c2)

# Verificar dimensoes
dim(coef_c1_mat) # B × p
dim(coef_c2_mat) # B × p

### Probabilidade de margem negativa e margem esperada por cluster
ic_long <- boot_resumos %>%
  pivot_longer(
    cols = c(mean_margin, prob_neg),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    metric = ifelse(metric == "mean_margin",
                    "Expected Margin Value",
                    "Probability of Negative Margin")
  )

ic_boot <- ic_long %>%
  group_by(cluster, metric) %>%
  summarise(
    mean = mean(value),
    lower = quantile(value, 0.025),
    upper = quantile(value, 0.975),
    .groups = "drop"
  )

ggplot(ic_boot, aes(x = factor(cluster), y = mean, fill = metric)) +
  geom_col(position = position_dodge(width = 0.6), width = 0.5, alpha = 0.7) +
  geom_errorbar(aes(ymin = lower, ymax = upper),
                width = 0.2, position = position_dodge(width = 0.6)) +
  geom_text(aes(y = mean + 0.02, label = sprintf("%.2f", mean)),
            position = position_dodge(width = 0.6), size = 3.5) +
  labs(
    x = "Cluster",
    y = NULL,
    title = "Bootstrap: expected margin and probability of negative margin"
  ) +
  scale_fill_manual(values = c("Expected Margin Value" = "steelblue",
                               "Probability of Negative Margin" = "tomato")) +
  theme_minimal() +
  theme(legend.title = element_blank())

### Analise de metricas: RMSE, MAE e R2_pred estimadas a partir do bootstrap

# Definicao resposta de treino e teste
y_test <- test_data$Margin
y_train <- final_data$Margin

# RMSE
rmse_boot <- apply(pred_matrix, 2, function(yhat)
  sqrt(mean((y_test - yhat)^2))
)

# MAE
mae_boot <- apply(pred_matrix, 2, function(yhat)
  mean(abs(y_test - yhat))
)

# R² predito
y_train <- final_data$Margin
ybar_train <- mean(y_train)
r2_boot <- apply(pred_matrix, 2, function(yhat) {
  ss_res <- sum((y_test - yhat)^2)
  ss_tot <- sum((y_test - ybar_train)^2)
  1 - ss_res / ss_tot
})

# Funcao para plot das metricas
plot_metric_ci <- function(x, name, col_fill, col_line) {
  
  dens <- density(x, na.rm = TRUE)
  q <- quantile(x, c(0.025, 0.5, 0.975), na.rm = TRUE)
  
  plot(
    dens,
    main = paste0(name, " (Bootstrap)"),
    xlab = name,
    ylab = "Densidade",
    lwd  = 2
  )
  
  # banda cheia do IC
  polygon(
    c(dens$x[dens$x >= q[1] & dens$x <= q[3]],
      rev(dens$x[dens$x >= q[1] & dens$x <= q[3]])),
    c(dens$y[dens$x >= q[1] & dens$x <= q[3]],
      rep(0, sum(dens$x >= q[1] & dens$x <= q[3]))),
    col = col_fill,
    border = NA
  )
  
  # linhas
  abline(v = q[2], lwd = 2, col = col_line)  # mediana
  abline(v = q[c(1,3)], lty = 2, col = col_line)
}

par(mfrow=c(2,2))

# plot do RMSE
plot_metric_ci(
  rmse_boot,
  name = "RMSE",
  col_fill = rgb(0.2, 0.4, 0.8, 0.35),
  col_line = "blue4"
)

# plot do MAE
plot_metric_ci(
  mae_boot,
  name = "MAE",
  col_fill = rgb(0.2, 0.7, 0.4, 0.35),
  col_line = "darkgreen"
)

#plot do R2 predito
plot_metric_ci(
  r2_boot,
  name = "Predictive R^2",
  col_fill = rgb(0.8, 0.3, 0.3, 0.35),
  col_line = "darkred"
)

# Construir estimativas para o regime 2
ci_clust2 <- t(apply(coef_c2_mat, 2, quantile,
                     probs = c(0.025, 0.975),
                     na.rm = TRUE))
point_clust2 <- colMeans(coef_c2_mat, na.rm = TRUE)
results_clust2 <- data.frame(
  variable = names(point_clust2),
  estimate = point_clust2,
  ci_low   = ci_clust2[, 1],
  ci_high  = ci_clust2[, 2]
)
rownames(results_clust2) <- NULL

# Construir estimativas para o regime 1
ci_clust1 <- t(apply(coef_c1_mat, 2, quantile,
                     probs = c(0.025, 0.975),
                     na.rm = TRUE))
point_clust1 <- colMeans(coef_c1_mat, na.rm = TRUE)
results_clust1 <- data.frame(
  variable = names(point_clust1),
  estimate = point_clust1,
  ci_low   = ci_clust1[, 1],
  ci_high  = ci_clust1[, 2]
)
rownames(results_clust1) <- NULL

# Construir estimativas de probabilidade de pertencer a mistura 2
mean_prob_boot <- colMeans(probs_mat, na.rm = TRUE)

ci_prob <- quantile(
  mean_prob_boot,
  probs = c(0.025, 0.5, 0.975),
  na.rm = TRUE
)

results_probclust2 <- data.frame(
  variable = "Cluster 2 (médio)",
  estimate = ci_prob[2],
  ci_low   = ci_prob[1],
  ci_high  = ci_prob[3]
)
rownames(results_probclust2) <- NULL

results_clust1
results_clust2
results_probclust2

# Grafico de efeito das covariaveis na margem de cada cluster
ggplot(results_clust1, aes(x = reorder(variable, estimate), y = estimate)) +
  geom_point(color = "blue") +
  geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.2, color = "blue") +
  coord_flip() +
  labs(title = "Efeito das covariáveis na margem - Cluster 1",
       y = "Coeficiente da margem",
       x = "") +
  theme_minimal()
ggplot(results_clust2, aes(x = reorder(variable, estimate), y = estimate)) +
  geom_point(color = "red") +
  geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.2, color = "red") +
  coord_flip() +
  labs(title = "Efeito das covariáveis na margem - Cluster 2",
       y = "Coeficiente da margem",
       x = "") +
  theme_minimal()


# Grafico das probabilidades de pertencer ao cluster2
hist(
  mean_prob_boot,
  breaks = 30,
  freq = FALSE,
  xlim = c(0, 1),
  main = "Distribuição das probabilidades médias",
  xlab = "P(cluster 2)"
)
lines(density(mean_prob_boot))

# Erros vs probabilidade de cluster 2
plot(
  mean_prob_boot,
  abs(y_test - yhat_test),
  xlab = "P(cluster 2)",
  ylab = "|Prediction error|",
  main = "Prediction Error × Cluster 2 Probability"
)

# Entropia
entropy <- function(p) -p*log(p) - (1-p)*log(1-p)
mean(entropy(mean_prob_boot), na.rm=TRUE)

### Análise de resíduos 

# Matriz de resíduos bootstrap
resid_matrix <- sweep(
  pred_matrix,
  1,
  y_test,
  FUN = "-" 
)

# Resíduos médios e bandas ponto-a-ponto
resid_ci <- t(apply(
  resid_matrix,
  1,
  quantile,
  probs = c(0.025, 0.5, 0.975),
  na.rm = TRUE
))
colnames(resid_ci) <- c("low", "median", "high")

ord <- order(rowMeans(pred_matrix, na.rm = TRUE))
plot(
  resid_ci[ord, "median"],
  
  ylim = range(resid_ci),
  xlab = "Observações (ordenadas)",
  ylab = "Resíduo"
)

# Distribuição bootstrap dos resíduos
mean_resid_boot <- colMeans(resid_matrix, na.rm = TRUE)
dens <- density(mean_resid_boot, na.rm = TRUE)
q <- quantile(mean_resid_boot, c(0.025, 0.5, 0.975))
plot(
  dens,
  main = "Resíduo médio preditivo (bootstrap)",
  xlab = "Resíduo médio",
  ylab = "Densidade",
  lwd  = 2,
)
idx <- dens$x >= q[1] & dens$x <= q[3]
polygon(
  c(dens$x[idx], rev(dens$x[idx])),
  c(dens$y[idx], rep(0, sum(idx))),
  col = rgb(0.3, 0.3, 0.8, 0.35),
  border = NA
)
abline(v = q[2], lwd = 2)
abline(v = q[c(1,3)], lty = 2)
abline(v = 0, col = "darkred", lty = 2)

# Heterocedasticidade dos residuos
resid_width <- resid_ci[, "high"] - resid_ci[, "low"]
plot(
  rowMeans(pred_matrix),
  resid_width,
  pch = 16,
  col = rgb(0.3, 0.3, 0.8, 0.5),
  xlab = "Mean Prediction",
  ylab = "Residual Uncertainty",
  main="Bootstrap: Difference between residual quantiles"
)

# Outliers
mad_resid <- apply(
  resid_matrix,
  1,
  mad,
  na.rm = TRUE
)

plot(mad_resid,
     ylab = "MAD do resíduo preditivo",
     xlab = "Observação (teste)")


lim <- median(mad_resid) + 3 * mad(mad_resid)

outliers <- which(mad_resid > lim)
length(outliers)

plot(
  t(resid_matrix[outliers, ]),
  main = "Distribuição dos resíduos (outliers preditivos)",
  ylab = "Resíduo"
)

# Residuo condicional cluster 2
w2 <- probs_array[, 2, ]
resid_weighted <- resid_matrix * w2
mean_resid_c2 <- colMeans(resid_weighted, na.rm = TRUE)
plot(density(mean_resid_c2))
abline(v = 0, lwd = 2)

# Residuo condicional cluster 1
w1 <- probs_array[, 1, ]
resid_weighted <- resid_matrix * w1
mean_resid_c1 <- colMeans(resid_weighted, na.rm = TRUE)
plot(density(mean_resid_c1))


############################# Graficos importantes ############################# 

### O que caracteriza cada cluster

# Número de bootstraps
B <- length(boot_ok)
n_test <- nrow(test_data)

# Inicializa lista para armazenar resultados de cada bootstrap
boot_prob_list <- vector("list", B)

for(b in seq_len(B)) {
  # Probabilidades do bootstrap b
  probs_b <- boot_ok[[b]]$probs
  
  # Data frame temporário
  df_b <- test_data[, cat_vars_sel] %>%
    mutate(Cluster1 = probs_b[,1],
           Cluster2 = probs_b[,2],
           Bootstrap = b) %>%
    pivot_longer(cols = cat_vars_sel, names_to = "Variable", values_to = "Value") %>%
    pivot_longer(cols = c("Cluster1","Cluster2"), names_to = "Cluster", values_to = "Prob")
  
  boot_prob_list[[b]] <- df_b
}

posterior_boot_summary <- posterior_boot_long %>%
  group_by(Bootstrap, Variable, Value, Cluster) %>%
  summarise(mean_prob_boot = mean(Prob), .groups = "drop") %>%   # média por bootstrap
  group_by(Variable, Value, Cluster) %>%
  summarise(
    mean_prob = mean(mean_prob_boot),
    lower = quantile(mean_prob_boot, 0.025),
    upper = quantile(mean_prob_boot, 0.975),
    .groups = "drop"
  )

top10_per_var_boot <- posterior_boot_summary %>%
  group_by(Variable, Cluster) %>%
  slice_max(order_by = mean_prob, n = 10) %>%
  ungroup()

# Combina todos os bootstraps
posterior_boot_long <- bind_rows(boot_prob_list)

var_names <- unique(top10_per_var_boot$Variable)

plot_list_boot <- lapply(var_names, function(var){
  df <- top10_per_var_boot %>% filter(Variable == var)
  
  ggplot(df, aes(x = reorder(Value, mean_prob), y = mean_prob, fill = Cluster)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6, alpha = 0.8) +
    geom_errorbar(aes(ymin = lower, ymax = upper), position = position_dodge(width = 0.7), width = 0.2) +
    labs(
      x = var,
      y = "Mean posteriori probability",
      fill = "Cluster",
      title = paste("Top 10 levels of", var, "per cluster (Bootstrap)")
    ) +
    scale_fill_manual(values = c("Cluster1" = "steelblue", "Cluster2" = "darkorange")) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
})

# Visualizar o primeiro gráfico
plot_list_boot[[1]]

### Grafico esperado por cluster

n_test <- nrow(pred_matrix)
B <- ncol(pred_matrix)

mean_margin_boot <- data.frame(
  Cluster = c("Cluster1", "Cluster2"),
  mean_margin = c(
    colMeans(pred_matrix[,1:B] * probs_array[,1,1:B] + pred_matrix[,1:B] * (1 - probs_array[,1,1:B])),
    colMeans(pred_matrix[,1:B] * probs_array[,2,1:B] + pred_matrix[,1:B] * (1 - probs_array[,2,1:B]))
  )
)

mean_margin_boot <- data.frame(
  Cluster = rep(c("Cluster1", "Cluster2"), each = length(boot_ok)),
  mean_margin = c(
    sapply(boot_ok, function(x) mean(x$probs[,1] * x$yhat_test)),
    sapply(boot_ok, function(x) mean(x$probs[,2] * x$yhat_test))
  )
)

mean_margin_summary <- mean_margin_boot %>%
  group_by(Cluster) %>%
  summarise(
    mean = mean(mean_margin),
    lower = quantile(mean_margin, 0.025),
    upper = quantile(mean_margin, 0.975)
  )

ggplot(mean_margin_summary, aes(x = Cluster, y = mean, fill = Cluster)) +
  geom_col(width = 0.6, alpha = 0.8) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  labs(
    x = "Cluster",
    y = "Margem Esperada",
    title = "Margem Esperada por Cluster (Bootstrap)",
    fill = "Cluster"
  ) +
  scale_fill_manual(values = c("Cluster1" = "steelblue", "Cluster2" = "darkorange")) +
  theme_minimal()


### Grafico top 10 variaveis mais impactantes das variaveis categoricas por cluster

B_ok <- length(boot_ok)

pred_boot_long <- lapply(seq_along(boot_ok), function(b){
  
  data.frame(
    Bootstrap = b,
    test_data,
    margin_hat = boot_ok[[b]]$yhat_test
  )
  
}) %>% bind_rows()

cat_vars <- c(
  "Sub.Category",
  "Segment",
  "Ship.Mode",
  "division",
  "Discount_group",
  "Quantity_group"
)

margin_level_boot <- pred_boot_long %>%
  pivot_longer(
    cols = all_of(cat_vars),
    names_to = "Variable",
    values_to = "Level"
  ) %>%
  group_by(Bootstrap, Variable, Level) %>%
  summarise(
    mean_margin_boot = mean(margin_hat, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

margin_level_summary <- margin_level_boot %>%
  group_by(Variable, Level) %>%
  summarise(
    mean_margin = mean(mean_margin_boot),
    lower = quantile(mean_margin_boot, 0.025),
    upper = quantile(mean_margin_boot, 0.975),
    .groups = "drop"
  )

global_mean_margin <- mean(pred_boot_long$margin_hat)

margin_level_summary <- margin_level_summary %>%
  mutate(
    impact = mean_margin - global_mean_margin
  )

top10_global <- margin_level_summary %>%
  mutate(abs_impact = abs(impact)) %>%
  arrange(desc(abs_impact)) %>%
  slice_head(n = 10)

ggplot(top10_global,
       aes(x = reorder(Level, impact), y = impact)) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  geom_errorbar(
    aes(ymin = lower - global_mean_margin,
        ymax = upper - global_mean_margin),
    width = 0.2
  ) +
  coord_flip() +
  labs(
    x = "Nível",
    y = "Impacto na margem esperada",
    title = "Top 10 níveis mais impactantes na margem (bootstrap)",
    subtitle = "Barras = efeito médio | Intervalos = IC 95% bootstrap"
  ) +
  theme_minimal()
