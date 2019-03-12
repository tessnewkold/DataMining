set.seed(11434445)
library(rpart)
library(rpart.plot)
library(MASS) #this data is in MASS package
library(glmnet)
library(boot)
#install.packages("DAAG")
library(DAAG)


# 1. Boston Housing data. Random sample a training data set that contains 75% of
#     the original data points. (You may stay with the same data set from HW2.)
BostonData <- data(Boston)
sample_index <- sample(nrow(Boston),nrow(Boston)*0.75)
BostonTrain <- Boston[sample_index,]
BostonTest <- Boston[-sample_index,]

# (i) Start with exploratory data analysis. Repeat linear regression as in HW2.
dim(BostonTrain)
str(BostonTrain)
summary(BostonTrain)
#To find mean and SD for all variables 
apply(BostonTrain, 2, mean, na.rm = TRUE)
apply(BostonTrain, 2, sd, na.rm = TRUE)

boxplot(BostonTrain, main = "Boxplots for All Variables in Boston Data")
#correlation plot
bostonCor <- cor(BostonTrain)
round(bostonCor, 2)
# Insignificant correlations are leaved blank
corrplot(bostonCor, type = "lower", order = "hclust", 
         p.mat = bostonCor, sig.level = 0.05, insig = "blank",
         tl.col = "black", tl.srt = 20)


# (ii) As in HW2, find a best model using linear regression with AIC and BIC and 
#       LASSO variable selection. Report model mean squared error (model MSE). 
#       Conduct some residual diagnosis.

#model selection
#stepwise
#making an empty lm
emptyModel <- lm(medv ~ 1, data = BostonTrain)
#making a lm with every variable 
fullModel <- lm(medv ~ ., data = BostonTrain)
#both ways
modelStepwise <- step(emptyModel, scope = list(lower = emptyModel, upper = fullModel), direction = 'both')
summary(modelStepwise)

#LASSO Selection
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

#Model MSE for Stepwise
modelStepwiseSummary <- summary(modelStepwise)
mean(modelStepwise$residuals^2)
modelStepwiseSummary$adj.r.squared
AIC(modelStepwise)
BIC(modelStepwise)

#Model MSE for LASSO
modelLASSOSummary <- summary(modelLASSO)
mean(modelLASSO$residuals^2)
modelLASSOSummary$adj.r.squared
AIC(modelLASSO)
BIC(modelLASSO)

#      Show residual diagnosis. 
par(mfrow = c(2,2)) # Change the panel layout to 2 x 2
plot(modelStepwise)
plot(modelLASSO)
#LASSO MSE; MSPE and CV
bostonTrainPredLASSO <- predict(modelLASSO)
#same thing except using testing data to predict Y
bostonTestPredLASSO <- predict(modelLASSO, BostonTest)
#this is taking all of the predicted values from above and 
#subracting the actual values from the response variable and 
# finding the mean squared error (MSE) of that
MSE_LASSO <- mean((bostonTrainPredLASSO - BostonTrain$medv)^2)
#this is taking the testing prediction minus the actual values
#and averageing them as well
MSPE_LASSO <- mean((bostonTestPredLASSO - BostonTest$medv)^2)
MSE_LASSO; MSPE_LASSO

#CV for LASSO
summary(modelLASSO)
modelLASSO_GLM <- glm(medv ~ lstat + rm + ptratio + black + dis + nox + zn + crim + chas, data = Boston)

cv.glm(data = Boston, glmfit = modelLASSO_GLM, K = 4)$delta[2]


# (iii) Test the out-of-sample performance. Using final linear model built from 
#       (i) on the 75% of original data, test with the remaining 25% testing data. 
#       (Try predict() function in R.) Report out-of-sample model MSE etc.
#this uses the data set used to make the model, and the model itself
#to predict the response variable (medv) for each observation
bostonTrainPredLM <- predict(modelStepwise)
#same thing except using testing data to predict Y
bostonTestPredLM <- predict(modelStepwise, BostonTest)
#this is taking all of the predicted values from above and 
#subracting the actual values from the response variable and 
# finding the mean squared error (MSE) of that
MSE_LM <- mean((bostonTrainPredLM - BostonTrain$medv)^2)
#this is taking the testing prediction minus the actual values
#and averageing them as well
MSPE_LM <- mean((bostonTestPredLM - BostonTest$medv)^2)
MSE_LM; MSPE_LM
# (iv) Cross validation. Use 4-fold cross validation. 
#       (Try cv.glm() function in R on the ORIGINAL 100% data.) 
#       Does (iv) yield similar answer as (iii)?
#make our stepwise model into a GLM
modelStepwiseGLM <- glm(medv ~ lstat + rm + ptratio + black + dis + nox + zn + crim + 
                          rad + tax + chas, data = Boston)

cv.glm(data = Boston, glmfit = modelStepwiseGLM, K = 4)$delta[2]

# (v) Fit a regression tree (CART) on the same data; repeat the above step (iii).
bostonRpart <- rpart(medv ~ lstat + rm + ptratio + black + dis + nox + zn + crim + 
                        rad + tax + chas, data = BostonTrain)
bostonRpart
par(mfrow = c(1,1))
prp(bostonRpart,digits = 4, extra = 1)
        # (iii) Test the out-of-sample performance. Using final linear model built from 
        #       (i) on the 75% of original data, test with the remaining 25% testing data. 
        #       (Try predict() function in R.) Report out-of-sample model MSE etc.
boston.train.pred.tree = predict(bostonRpart)
boston.test.pred.tree = predict(bostonRpart,BostonTest)
MSE.tree <- mean((boston.train.pred.tree - BostonTrain$medv)^2)
MSPE.tree <- mean((boston.test.pred.tree - BostonTest$medv)^2)
MSE.tree; MSPE.tree
# (vi) What do you find comparing CART to the linear regression model fits from HW2?
# CART MSE and MSPE is lower than the LM 

# > MSE.tree; MSPE.tree
# [1] 14.51898
# [1] 21.35488
# > MSE_LM; MSPE_LM
# [1] 21.1695
# [1] 25.55221

# (vii) Now repeat previous steps for another random sample 
#       (that is, to draw another training data set with 75% of original data, 
#       and the rest 25% as testing). Do you get similar results? 
#       What’s your conclusion?


set.seed(13023260)
# 1. Boston Housing data. Random sample a training data set that contains 75% of
#     the original data points. (You may stay with the same data set from HW2.)
BostonData <- data(Boston)
sample_index <- sample(nrow(Boston),nrow(Boston)*0.75)
BostonTrain <- Boston[sample_index,]
BostonTest <- Boston[-sample_index,]

# (i) Start with exploratory data analysis. Repeat linear regression as in HW2.
dim(BostonTrain)
str(BostonTrain)
summary(BostonTrain)
#To find mean and SD for all variables 
apply(BostonTrain, 2, mean, na.rm = TRUE)
apply(BostonTrain, 2, sd, na.rm = TRUE)

boxplot(BostonTrain, main = "Boxplots for All Variables in Boston Data")
#correlation plot
bostonCor <- cor(BostonTrain)
round(bostonCor, 2)
# Insignificant correlations are leaved blank
corrplot(bostonCor, type = "lower", order = "hclust", 
         p.mat = bostonCor, sig.level = 0.05, insig = "blank",
         tl.col = "black", tl.srt = 20)


# (ii) As in HW2, find a best model using linear regression with AIC and BIC and 
#       LASSO variable selection. Report model mean squared error (model MSE). 
#       Conduct some residual diagnosis.

#model selection
#stepwise
#making an empty lm
emptyModel <- lm(medv ~ 1, data = BostonTrain)
#making a lm with every variable 
fullModel <- lm(medv ~ ., data = BostonTrain)
#both ways
modelStepwise <- step(emptyModel, scope = list(lower = emptyModel, upper = fullModel), direction = 'both')
summary(modelStepwise)

#LASSO Selection
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

#Model MSE for Stepwise
modelStepwiseSummary <- summary(modelStepwise)
mean(modelStepwise$residuals^2)
modelStepwiseSummary$adj.r.squared
AIC(modelStepwise)
BIC(modelStepwise)

#Model MSE for LASSO
modelLASSOSummary <- summary(modelLASSO)
mean(modelLASSO$residuals^2)
modelLASSOSummary$adj.r.squared
AIC(modelLASSO)
BIC(modelLASSO)

#      Show residual diagnosis. 
par(mfrow = c(2,2)) # Change the panel layout to 2 x 2
plot(modelStepwise)
plot(modelLASSO)
#LASSO MSE; MSPE and CV
bostonTrainPredLASSO <- predict(modelLASSO)
#same thing except using testing data to predict Y
bostonTestPredLASSO <- predict(modelLASSO, BostonTest)
#this is taking all of the predicted values from above and 
#subracting the actual values from the response variable and 
# finding the mean squared error (MSE) of that
MSE_LASSO <- mean((bostonTrainPredLASSO - BostonTrain$medv)^2)
#this is taking the testing prediction minus the actual values
#and averageing them as well
MSPE_LASSO <- mean((bostonTestPredLASSO - BostonTest$medv)^2)
MSE_LASSO; MSPE_LASSO

#CV for LASSO
summary(modelLASSO)
modelLASSO_GLM <- glm(medv ~ lstat + rm + ptratio + black + dis + nox + zn + crim + chas, data = Boston)

cv.glm(data = Boston, glmfit = modelLASSO_GLM, K = 4)$delta[2]


# (iii) Test the out-of-sample performance. Using final linear model built from 
#       (i) on the 75% of original data, test with the remaining 25% testing data. 
#       (Try predict() function in R.) Report out-of-sample model MSE etc.
#this uses the data set used to make the model, and the model itself
#to predict the response variable (medv) for each observation
bostonTrainPredLM <- predict(modelStepwise)
#same thing except using testing data to predict Y
bostonTestPredLM <- predict(modelStepwise, BostonTest)
#this is taking all of the predicted values from above and 
#subracting the actual values from the response variable and 
# finding the mean squared error (MSE) of that
MSE_LM <- mean((bostonTrainPredLM - BostonTrain$medv)^2)
#this is taking the testing prediction minus the actual values
#and averageing them as well
MSPE_LM <- mean((bostonTestPredLM - BostonTest$medv)^2)
MSE_LM; MSPE_LM
# (iv) Cross validation. Use 4-fold cross validation. 
#       (Try cv.glm() function in R on the ORIGINAL 100% data.) 
#       Does (iv) yield similar answer as (iii)?
#make our stepwise model into a GLM
modelStepwiseGLM <- glm(medv ~ lstat + rm + ptratio + black + dis + nox + zn + crim + 
                          rad + tax + chas, data = Boston)

cv.glm(data = Boston, glmfit = modelStepwiseGLM, K = 4)$delta[2]

# (v) Fit a regression tree (CART) on the same data; repeat the above step (iii).
bostonRpart <- rpart(medv ~ lstat + rm + ptratio + black + dis + nox + zn + crim + 
                       rad + tax + chas, data = BostonTrain)
bostonRpart
par(mfrow = c(1,1))
prp(bostonRpart,digits = 4, extra = 1)
# (iii) Test the out-of-sample performance. Using final linear model built from 
#       (i) on the 75% of original data, test with the remaining 25% testing data. 
#       (Try predict() function in R.) Report out-of-sample model MSE etc.
boston.train.pred.tree = predict(bostonRpart)
boston.test.pred.tree = predict(bostonRpart,BostonTest)
MSE.tree <- mean((boston.train.pred.tree - BostonTrain$medv)^2)
MSPE.tree <- mean((boston.test.pred.tree - BostonTest$medv)^2)
MSE.tree; MSPE.tree
# (vi) What do you find comparing CART to the linear regression model fits from HW2?
# CART MSE and MSPE is lower than the LM 

# > MSE.tree; MSPE.tree
# [1] 14.51898
# [1] 21.35488
# > MSE_LM; MSPE_LM
# [1] 21.1695
# [1] 25.55221

# (vii) Now repeat previous steps for another random sample 
#       (that is, to draw another training data set with 75% of original data, 
#       and the rest 25% as testing). Do you get similar results? 
#       What’s your conclusion?
