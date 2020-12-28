library(tidyverse)
library(lme4)
library(pROC)
cars_train <- read.csv('carstrain_2.csv')
cars_test <- read.csv('carstest_2.csv')
m1 <- glmer(is_severe~std.boxcox.distance+Side+Day+std.temp+std.boxcox.humidity+std.boxcox.pressure+std.boxcox.windspeed+std.boxcox.precipitation+Amenity+Bump+Crossing+Give_Way+Junction+No_Exit+Railway+Roundabout+Station+Stop+Traffic_Calming+Traffic_Signal+(1|State), data = cars_train, family = binomial(link = "logit"))
m2 <- glmer(is_severe~std.boxcox.distance+Side+Day+std.temp+std.boxcox.humidity+std.boxcox.pressure+std.boxcox.windspeed+std.boxcox.precipitation+Amenity+Bump+Crossing+Junction+Railway+Station+Stop+Traffic_Signal+(1|State), data = cars_train, family = binomial(link = "logit"))

anova(m1, m2)
m3 <- glmer(is_severe~std.boxcox.distance+Side+Day+std.temp+std.boxcox.humidity+std.boxcox.pressure+std.boxcox.windspeed+std.boxcox.precipitation+Amenity+Crossing+Junction+Railway+Station+Stop+Traffic_Signal+(1|State), data = cars_train, family = binomial(link = "logit"))
anova(m2, m3)
#Prediction accuracy of final model
sum(cars_test$is_severe== ifelse(predict(m3, cars_test, type='response')>=0.5, 1, 0))/nrow(cars_test)

m3_norandom <- glm(is_severe~std.boxcox.distance+Side+Day+std.temp+std.boxcox.humidity+std.boxcox.pressure+std.boxcox.windspeed+std.boxcox.precipitation+Amenity+Crossing+Junction+Railway+Station+Stop+Traffic_Signal+State, data = cars_train, family = binomial(link = "logit"))

anova(m3, m3_norandom)

#Prediction accuracy of logistic regression
sum(cars_test$is_severe== ifelse(predict(m3_norandom, cars_test, type='response')>=0.5, 1, 0))/nrow(cars_test)

library(gt)


summary_m1 <- summary(m1)
coef_m1 <- data.frame(State = rownames(ranef(m1)$State), Intercept=unlist(ranef(m1)[['State']]), StDev = attributes(VarCorr(m1)$State)$stddev)
summary_m1$coefficients %>% data.frame() %>% gt()%>%
  tab_header(
    title = glue::glue("Full Model Fixed Effects || AIC: {round(summary_m1$AICtab[1], 3)}"),
    subtitle = glue::glue("{summary_m1$call}")
  )  %>% gtsave('m1_fix.png')

coef_m1 %>% gt() %>%
  tab_header(
    title = "Full Model Random Intercepts",
    subtitle = glue::glue("{summary_m1$call}")
  ) %>% gtsave('m1_random.png')


summary_m2 <- summary(m2)
coef_m2 <- data.frame(State = rownames(ranef(m2)$State), Intercept=unlist(ranef(m2)[['State']]), StDev = attributes(VarCorr(m2)$State)$stddev)
summary_m2$coefficients %>% data.frame() %>% gt()%>%
  tab_header(
    title = glue::glue("Reduced Model Fixed Effects || AIC: {round(summary_m2$AICtab[1], 3)}"),
    subtitle = glue::glue("{summary_m2$call}")
  )  %>% gtsave('m2_fix.png')

coef_m2 %>% gt() %>%
  tab_header(
    title = "Reduced Model Random Intercepts",
    subtitle = glue::glue("{summary_m2$call}")
  ) %>% gtsave('m2_random.png')


summary_m3 <- summary(m3)
coef_m3 <- data.frame(State = rownames(ranef(m3)$State), Intercept=unlist(ranef(m3)[['State']]), StDev = attributes(VarCorr(m3)$State)$stddev)
summary_m3$coefficients %>% data.frame() %>% gt()%>%
  tab_header(
    title = glue::glue("Final Model Fixed Effects || AIC: {round(summary_m3$AICtab[1], 3)}"),
    subtitle = glue::glue("{summary_m3$call}")
  )  %>% gtsave('m3_fix.png')

coef_m3 %>% gt() %>%
  tab_header(
    title = "Final Model Random Intercepts",
    subtitle = glue::glue("{summary_m3$call}")
  ) %>% gtsave('m3_random.png')


## ROC Curves

test_prob_m1 <- predict(m1, newdata =cars_test, type='response')
test_roc_m1 = roc(cars_test$is_severe ~ test_prob_m1)
test_prob_m2 <- predict(m2, newdata =cars_test, type='response')
test_roc_m2 = roc(cars_test$is_severe ~ test_prob_m2)
test_prob_m3 <- predict(m3, newdata =cars_test, type='response')
test_roc_m3 = roc(cars_test$is_severe ~ test_prob_m3)
test_prob_m3_norandom <- predict(m3_norandom, newdata =cars_test, type='response')
test_roc_m3_norandom = roc(cars_test$is_severe ~ test_prob_m3_norandom)

plot(test_roc_m1, print.auc=FALSE, col="black", lty=1, lwd=2, legacy.axes = TRUE, print.auc.y=.35, grid=TRUE)
plot(test_roc_m2, print.auc=FALSE, col="black", lty=3, lwd=2, legacy.axes = TRUE, print.auc.y=.3, grid=TRUE, add=TRUE)
plot(test_roc_m3, print.auc=FALSE, col="black", lty=5, lwd=2, legacy.axes = TRUE, print.auc.y=.3, grid=TRUE, add=TRUE)
legend("bottomright",
       legend=c("Model 1 (Full)","Model 2", "Model 3 (Final)"),
       col=c("black", "black", "black"),
       lty=c(1,3,5),
       lwd=c(2,2,2))

text(0.4, 0.4, paste("AUC for m1:", round(test_roc_m1$auc, 3)))
text(0.4, 0.30, paste("AUC for m2:", round(test_roc_m2$auc, 3)))
text(0.4, 0.20, paste("AUC for m3:", round(test_roc_m3$auc, 3)))
## Sensitivity and Specificity
conf.matrix <- table(cars_test$is_severe,ifelse(test_prob_m3>=0.5, 1, 0))
conf.matrix
### Sensitivity
conf.matrix[2,2]/sum(conf.matrix[,2])

### Specificity
conf.matrix[1,1]/sum(conf.matrix[,1])

## Pearson Residual Plot
plot(m3) 


## Random Intercepts
m3_df <- data.frame(State=rownames(coef(m3)$State), Intercept=coef(m3)$State[,1], sd=as.numeric(attributes(VarCorr(m3)$State)$stddev))
pd <- position_dodge(0.78)
ggplot(m3_df, aes(x=State, y = Intercept, group = State)) +
  #draws the means
  geom_point(position=pd) +
  #draws the CI error bars
  geom_errorbar(aes(ymin=Intercept-2*sd, ymax=Intercept+2*sd, 
                    color=State), width=.1, position=pd)

