# Homework 2 Question 1 

library(MASS)
library(corrplot)
library(leaps)
library(glmnet)

data(Boston)
?Boston
names(Boston)

#This splits the data into a Training set and a Testing set.
#Here the training data set is 75% and the testing data set is the unused 25%
sampleIndex <- sample(nrow(Boston),nrow(Boston)*0.75)
BostonTrain <- Boston[sampleIndex,]
BostonTest <- Boston[-sampleIndex,]
summary(BostonTrain)

# 1. Boston Housing data. 
# Random sample a training data set that contains 75% of original data points. 
# Find a best model with training data of Boston Housing using linear regression.
# 
# a) Start with exploratory data analysis. 
dim(BostonTrain)
#    What is the nature of each variable? 
str(Boston)
summary(Boston)
apply(BostonTrain, 2, mean, na.rm = TRUE)
sd(Boston$indus)
#    How about pairwise correlations? 
bostonCor <- cor(BostonTrain)
round(bostonCor, 2)

corrplot(bostonCor, type = "lower", order = "hclust", 
         tl.col = "black", tl.srt = 20)

# Insignificant correlations are leaved blank
corrplot(bostonCor, type = "lower", order = "hclust", 
         p.mat = bostonCor, sig.level = 0.05, insig = "blank",
         tl.col = "black", tl.srt = 20)

#    Are there outliers? 
boxplot(BostonTrain, main = "Boxplots for All Variables in Boston Data")
# boxplot(Boston$crim, main = "Crime Rate by Town")
# boxplot(Boston$zn, main = "Residential land zoned for lots over 25,000 sq.ft")
# boxplot(Boston$indus, main = "Proportion of non-retail business acres")
# boxplot(Boston$nox, main = "Nitrogen Oxides Concentration")
# boxplot(Boston$rm, main = "Average numbers of rooms per dwelling")
# boxplot(Boston$age, main = "Owner-occupied units built prior to 1940")
# boxplot(Boston$dis, main = "Mean of distances to 5 Boston employment centers")
# boxplot(Boston$rad, main = "Accessibility to Radial Highways")
# boxplot(Boston$tax, main = "Property-tax Rate (per $10,000)")
# boxplot(Boston$ptratio, main = "Student-teacher Ratio by Town")
# boxplot(Boston$black, main = "Proportion of African Americans by Town")
# boxplot(Boston$lstat, main = "Lower status of the population")
# boxplot(Boston$medv, main = "Median value of owner-occupied homes (by $10,000)")

#    Any major comments about the data?
#    It may be helpful to go through Page 96-97 of the original paper



#standardizing the data
#optional for Linear models

# i <- 1
# for (i in 1:(ncol(BostonTrain) - 1)) {
#   BostonTrain[,i] <- scale(BostonTrain[,i])
# }

# b) Conduct linear regression on the data (no variable transformation)
model1 <- lm(medv ~ ., data = BostonTrain)
summary(model1)

model1Summary <- summary(model1)
model1Summary$adj.r.squared
AIC(model1)
BIC(model1)

model2 <- lm(medv ~ chas + zn, data = BostonTrain)
summary(model2)
model2Summary <- summary(model2)
model2Summary$adj.r.squared
AIC(model2)
BIC(model2)

# c) Conduct variable selection, including best subset, stepwise, and LASSO.

#regsubsets only takes data frame as input
subsetResult <- regsubsets(medv~.,data = BostonTrain, nbest = 2, nvmax = 14)
summary(subsetResult) #selects variables based on BIC
plot(subsetResult, scale = "bic")

modelSubset <- lm(medv ~ lstat + rm + ptratio + chas + black + crim + dis + nox + rad + tax, data = BostonTrain)
summary(modelSubset)
modelSubsetSummary <- summary(modelSubset)
mean(modelSubset$residuals^2)
modelSubsetSummary$adj.r.squared
AIC(modelSubset)
BIC(modelSubset)


#stepwise
#making an empty lm
emptyModel <- lm(medv ~ 1, data = BostonTrain)
#making a lm with every variable 
fullModel <- lm(medv ~ ., data = BostonTrain)

#backward
modelBackward <- step(fullModel, direction = 'backward')
#want AIC to be smallest also, so the top variable should be removed, bc it lowers the AIC the most 

#forward
modelForward <- step(emptyModel, scope = list(lower = emptyModel, upper = fullModel), direction = 'forward')

#both ways
modelStepwise <- step(emptyModel, scope = list(lower = emptyModel, upper = fullModel), direction = 'both')
summary(modelStepwise)
#it keeps going until the AIC does not lower anymore

lassoMatrix <- glmnet(x = as.matrix(Boston[, -c(which(colnames(Boston) == 'medv'))]), y = Boston$medv, alpha = 1)
#lambda = 0.5
coef(lassoMatrix, s = 0.5)
#lambda = 1
coef(lassoMatrix, s = 1)
#lambda = 0.3145908
coef(lassoMatrix, s = 0.3145908)

#use 5-fold cross validation to pick lambda
cvLassoMatrix = cv.glmnet(x = as.matrix(Boston[, -c(which(colnames(Boston) == 'medv'))]), y = Boston$medv, alpha = 1, nfolds = 5)
plot(cvLassoMatrix)
cvLassoMatrix$lambda.min
cvLassoMatrix$lambda.1se

BostonInsamplePrediction <- predict(lassoMatrix, as.matrix(Boston[, -c(which(colnames(Boston) == 'medv'))]), s = cvLassoMatrix$lambda.1se)

modelLASSO <- lm(medv ~ lstat + rm + ptratio + chas + black + crim + dis + nox + zn , data = BostonTrain)
summary(modelLASSO)
modelLASSOSummary <- summary(modelLASSO)
mean(modelLASSO$residuals^2)
modelLASSOSummary$adj.r.squared
AIC(modelLASSO)
BIC(modelLASSO)
#     Find the best linear model. 
#stepwise
medv ~ lstat + rm + ptratio + dis + nox + zn + chas + black + 
  rad + tax + crim
AIC(modelStepwise)

#      Show residual diagnosis. 
par(mfrow = c(2,2)) # Change the panel layout to 2 x 2
plot(modelLASSO)
#      For LASSO variable selection, you may use either (or both) lambda.min or (and) lambda.1se as optimal λ.
coef(lassoMatrix, s = 0.3145908)
# d) Report model mean squared error (model MSE), R2, and adjusted R2 for the selected model.

modelSubsetSummary <- summary(subsetResult)
mean(modelSubsetSummary$residuals^2)
modelSubsetSummary$r.squared
modelSubsetSummary$adj.r.squared
AIC(modelSubsetSummary)
BIC(modelSubsetSummary)


modelStepwiseSummary <- summary(modelStepwise)
mean(modelStepwiseSummary$residuals^2)
modelStepwiseSummary$r.squared
modelStepwiseSummary$adj.r.squared
AIC(modelStepwise)
BIC(modelStepwise)


modelLassoSummary <- summary(lassoMatrix)
mean(modelStepwiseSummary$residuals^2)
modelStepwiseSummary$r.squared
modelStepwiseSummary$adj.r.squared
AIC(modelStepwise)
BIC(modelStepwise)
# e) Interpret your results. 
#    Write a brief report. 
#    Please clearly label your figures and tables. 
#    No raw outputs please.
# 


