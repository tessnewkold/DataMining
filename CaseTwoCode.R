#Tess Newkold
#03_26_2019
#CaseTwo Assignment
#Data Mining II - Spring 2019

# Load Libraries ----------------------------------------------------------
library(MASS)
library(rpart)
library(rpart.plot)
library(ipred)
library(randomForest)
library(gbm)
library(boot)	# cv.glm
library(tree)	# tree package
library(maptree)	# tree drawer
library(readxl)
library(mgcv)
library(neuralnet)
library(nnet)
library(caret)
library(NeuralNetTools)

# Set Training and Testing Data Set ---------------------------------------
#set seet
set.seed(11434445)

# load Boston data
data(Boston)
#setting up the testing and training set
index <- sample(nrow(Boston),nrow(Boston)*0.75)
bostonTrain <- Boston[index,]
bostonTest <- Boston[-index,]



# Boston Housing Data -----------------------------------------------------

# Generalized Linear Model  -----------------------------------------------
#stepwise
#making an empty lm
emptyModel <- lm(medv ~ 1, data = bostonTrain)
#making a lm with every variable 
fullModel <- lm(medv ~ ., data = bostonTrain)
#both ways
modelStepwise <- step(emptyModel, scope = list(lower = emptyModel, upper = fullModel), direction = 'both')
summary(modelStepwise)
#this uses the data set used to make the model, and the model itself
#to predict the response variable (medv) for each observation
bostonTrainPredLM <- predict(modelStepwise)
#same thing except using testing data to predict Y
bostonTestPredLM <- predict(modelStepwise, bostonTest)
#this is taking all of the predicted values from above and 
#subracting the actual values from the response variable and 
# finding the mean squared error (MSE) of that
MSE_LM <- mean((bostonTrainPredLM - bostonTrain$medv)^2)
#this is taking the testing predictibon minus the actual values
#and averageing them as well
MSPE_LM <- mean((bostonTestPredLM - bostonTest$medv)^2)
MSE_LM; MSPE_LM




# Tree Models -------------------------------------------------------------
#Random Forest
bostonRpart <- rpart(medv ~ ., data = bostonTrain)
bostonRpart
prp(bostonRpart,digits = 4, extra = 1)

#prune the model
plotcp(bostonRpart)
BostonCPTree <- prune.rpart(bostonRpart, cp=0.021)
prp(BostonCPTree, digits = 4, extra = 1)

bostonTrainPredTree = predict(bostonRpart)
bostonTestPredTree = predict(bostonRpart, bostonTest)
MSE_Tree <- mean((bostonTrainPredTree - bostonTrain$medv)^2)
MSPE_Tree <- mean((bostonTestPredTree - bostonTest$medv)^2)
MSE_Tree; MSPE_Tree

#Bagging
bostonBag<- bagging(medv~., data = bostonTrain, nbagg=50)
bostonBag
summary(bostonBag)

#Prediction on testing sample
bostonTrainBagPred<- predict(bostonBag)
bostonTestBagPred<- predict(bostonBag, newdata = bostonTest)
MSE_Bagging <- mean((bostonTrain$medv - bostonTrainBagPred)^2)
MSPE_Bagging <- mean((bostonTest$medv - bostonTestBagPred)^2)
MSE_Bagging; MSPE_Bagging 

#Boosting
bostonBoost <- gbm(medv~., data = bostonTrain, distribution = "gaussian", n.trees = 10000, shrinkage = 0.01, interaction.depth = 8)
summary(bostonBoost)

bostonTrainBoostPred <- predict(bostonBoost, n.trees = 10000)
bostonTestBoostPred <- predict(bostonBoost, bostonTest, n.trees = 10000)
MSE_Boosting <- mean((bostonTrain$medv - bostonTrainBoostPred)^2)
MSPE_Boosting <- mean((bostonTest$medv - bostonTestBoostPred)^2)
MSE_Boosting; MSPE_Boosting



# Generalized Additive Models (GAM) ---------------------------------------
#model 1 - not using s() on chas and rad, leaving them as integers
boston.gam <- gam(medv ~ s(crim) + s(zn) + s(indus) + s(nox) + s(rm) + s(age) + s(dis) + 
                    s(tax) + s(ptratio) + s(black) + s(lstat) + chas + rad, data = bostonTrain)
summary(boston.gam)
#model 2 - removing s() from functions which are linear
boston.gam <- gam(medv ~ s(crim) + zn + s(indus) + s(nox) + s(rm) + age + s(dis) + 
                    s(tax) + ptratio + s(black) + s(lstat) + chas + rad, data = bostonTrain)
summary(boston.gam)

bostonTrainPredGAM = predict(boston.gam)
bostonTestPredGAM = predict(boston.gam, bostonTest)
MSE_Boston_GAM <- mean((bostonTrainPredGAM - bostonTrain$medv)^2)
MSPE_Boston_GAM <- mean((bostonTestPredGAM - bostonTest$medv)^2)
#In-sample prediction; Out-of-sample prediction
MSE_Boston_GAM; MSPE_Boston_GAM

#plot
plot(boston.gam, shade = TRUE, seWithMean = TRUE, scale = 0, pages=1)

# Neural Network ----------------------------------------------------------
maxs <- apply(Boston, 2, max) 
mins <- apply(Boston, 2, min)

scaledBoston <- as.data.frame(scale(Boston, center = mins, scale = maxs - mins))

index <- sample(nrow(scaledBoston),nrow(scaledBoston)*0.75)
ScaledBostonTrain <- scaledBoston[index,]
ScaledBostonTest <- scaledBoston[-index,]

n <- names(ScaledBostonTrain)
f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))
nn <- neuralnet(f,data=ScaledBostonTrain,hidden=c(5,3),linear.output=T)
plot(nn)

pr.nn <- compute(nn,ScaledBostonTest[,1:13])

pr.nn_ <- pr.nn$net.result*(max(scaledBoston$medv)-min(scaledBoston$medv))+min(scaledBoston$medv)
test.r <- (ScaledBostonTest$medv)*(max(scaledBoston$medv)-min(scaledBoston$medv))+min(scaledBoston$medv)
# MSE of testing set
MSE.nn <- sum((test.r - pr.nn_)^2)/nrow(ScaledBostonTest)
MSE.nn




# German Credit Score Data ----------------------------------------------------------
germanData <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")

colnames(germanData) <- c("chk_acct", "duration", "credit_his", "purpose", 
                           "amount", "saving_acct", "present_emp", "installment_rate", "sex", "other_debtor", 
                           "present_resid", "property", "age", "other_install", "housing", "n_credits", 
                           "job", "n_people", "telephone", "foreign", "response")

#change response to 0s and 1s
germanData$response <- germanData$response- 1
#chanigng the response variable to a factor
germanData$response <- as.factor(germanData$response)
str(germanData)
summary(germanData)


#setting up the testing and training set
index <- sample(nrow(germanData),nrow(germanData)*0.75)
germanTrain <- germanData[index,]
germanTest <- germanData[-index,]



# German-Generalized Linear Model -----------------------------------------
#logit link model with eveyr variable
germanTrainGLM <- glm(response~., family = binomial, germanTrain)
summary(germanTrainGLM)
#use step wise to find best model 
step(germanTrainGLM)
#input best model from step function 
germanTrainGLM <- glm(formula = response ~ chk_acct + duration + credit_his + amount + 
                              saving_acct + installment_rate + sex + other_debtor + age + 
                              other_install + n_people + telephone + foreign, family = binomial, 
                            data = germanTrain)
summary(germanTrainGLM)

predictGLMTrain <- predict(germanTrainGLM, type = "response")
predictGLMTest <- predict(germanTrainGLM, newdata = germanTest, type = "response")
#residual deviance
germanTrainGLM$deviance

# Asymmetric Misclassification Rate, using  5:1 asymmetric cost
cost <- function(r, pi){
  weight1 = 5
  weight0 = 1
  c1 = (r == 1) & (pi == 0) #logical vector - true if actual 1 but predict 0
  c0 = (r == 0) & (pi == 1) #logical vector - true if actual 0 but predict 1
  return(mean(weight1 * c1 + weight0 * c0))
}
## Bayes estimate
pcut <-  1/6 

#training data stats
classPredictGLMTrain <- (predictGLMTrain > pcut) * 1
tableGLMTree <- table(germanTrain$response, classPredictGLMTrain, dnn = c("True", "Predicted"))
tableGLMTree
#misclassification rate in sample
MR.glm0 <- mean(germanTrain$response != classPredictGLMTrain)
costTrain <- cost(r = germanTrain$response, pi = classPredictGLMTrain) 
#testing data stats
classPredictGLMTest<- (predictGLMTest > pcut) * 1
table(germanTest$response, classPredictGLMTest, dnn = c("True", "Predicted"))
#misclassification rate for out of sample 
MR.glm0.test <- mean(germanTest$response != classPredictGLMTest)
costTest <- cost(r = germanTest$response, pi = classPredictGLMTest) 

#area under the curve
library(verification)
par(mfrow=c(1,1))

ROClogit <- roc.plot(x = (germanTrain$response == "1"), pred =predictGLMTrain)
ROClogit$roc.vol
ROClogitTest <- roc.plot(x = (germanTest$response == "1"), pred = predictGLMTest)
ROClogitTest$roc.vol



# German-ClassificationTreeModel -------------------------------------------------------
germanTree <- rpart(formula = response~., data = germanTrain, 
                              parms = list(loss = matrix(c(0, 5, 1, 0), nrow = 2)), cp=0.001)

prp(germanTree, extra = 1, nn.font=500,box.palette = "yellow")

plotcp(germanTree)

printcp(germanTree)

#library(rattle)
#fancyRpartPlot(Gcredit.tree1,cex=0.6)

#Pruning###
germanTreePruned <- rpart(response~., data = germanTrain, method = "class",
                           parms = list(loss = matrix(c(0, 5, 1, 0), nrow = 2)),cp=0.008)
prp(germanTreePruned, extra = 1, nn.font=500,box.palette = "yellow")

predictGermanTreeTrain <- predict(germanTreePruned, type = "prob")
predictGermanTreeTest <- predict(germanTreePruned, newdata=germanTest, type = "prob")

#training stats
predictGermanTreeTrainRpart = as.numeric(predictGermanTreeTrain[,2] > pcut)
table(germanTrain$response, predictGermanTreeTrainRpart, dnn=c("Truth","Predicted"))
mean(germanTrain$response!=predictGermanTreeTrainRpart)
cost(germanTrain$response,predictGermanTreeTrainRpart)
#testing stats
predictGermanTreeTestRpart = as.numeric(predictGermanTreeTest[,2] > pcut)
table(germanTest$response, predictGermanTreeTestRpart, dnn=c("Truth","Predicted"))
mean(germanTest$response!=predictGermanTreeTestRpart)
cost(germanTest$response, predictGermanTreeTestRpart)
#area under the curve
par(mfrow=c(1,2))
ROCTree <- roc.plot(x=(germanTrain$response == "1"), pred = predictGermanTreeTrain[,2])
ROCTree$roc.vol

ROCTreeTest <- roc.plot(x = (germanTest$response == "1"), pred = predictGermanTreeTest[,2])
ROCTreeTest$roc.vol



# German GAM --------------------------------------------------------------
###GAM Model###########################################################################################
str(germanTrain)
germanGAM <- gam(as.factor(response) ~ chk_acct + s(duration) + credit_his
                      +purpose + s(amount) + saving_acct + present_emp
                      + installment_rate + sex + other_debtor + present_resid + property
                      + s(age) + other_install + housing + n_credits + telephone
                      + foreign, family = binomial, data = germanTrain)

summary(germanGAM)
par(mfrow=c(1,3))
plot(germanGAM, shade=TRUE)

# Move age to partially linear term and refit gam() model
germanGAM <- gam(as.factor(response) ~ chk_acct + s(duration) + credit_his + purpose
                 +s(amount)+saving_acct+present_emp+installment_rate+sex
                 +other_debtor+present_resid+property
                 +(age)+other_install+housing+n_credits
                 +telephone+foreign , family=binomial,data = germanTrain)

summary(germanGAM)
plot(germanGAM, shade=TRUE)

#In sample performance
probGermandGAMin <- predict(germanGAM, germanTrain, type = "response")
predGermandGAMin<-(probGermandGAMin >= pcut) * 1
table(germanTrain$response, predGermandGAMin, dnn = c("Observed", "Predicted"))
mean(ifelse(germanTrain$response != predGermandGAMin, 1, 0))
cost(germanTrain$response, predGermandGAMin)

#Out-of-sample performance
probGermanGAMout<-predict(germanGAM, germanTest, type="response")
predGermanGAMout<-(probGermanGAMout >= pcut) * 1
table(germanTest$response, predGermanGAMout, dnn = c("Observed", "Predicted"))
mean(ifelse(germanTest$response != predGermanGAMout, 1, 0))
cost(germanTest$response, predGermanGAMout)

###ROC curve for GAM
par(mfrow=(c(1,2)))
ROC_GAM <- roc.plot(x=(germanTrain$response == "1"), pred = probGermandGAMin)
ROC_GAM$roc.vol

ROC_GAM_test <- roc.plot(x=(germanTest$response == "1"), pred = probGermanGAMout)
ROC_GAM_test$roc.vol



# Neural Net --------------------------------------------------------------
par(mfrow=c(1,1))
germanNN <- train(response~., data = germanTrain, method = "nnet")
print(germanNN)
plot(germanNN)
plotnet(germanNN$finalModel, y_names = "response")
title("Graphical Representation of our Neural Network")

#In sample
probNN = predict(germanNN,type='prob')
predNN = (probNN[,2] >=pcut)*1
table(germanTrain$response,predNN, dnn = c("Observed", "Predicted"))
mean(ifelse(germanTrain$response != predNN, 1, 0))
cost(germanTrain$response, predNN)

#Out of sample
probTestNN= predict(germanNN, germanTest, type='prob')
predTestNN = as.numeric(probTestNN[,2] > pcut)
table(germanTest$response, predTestNN, dnn = c("Observed","Predicted"))
mean(ifelse(germanTest$response != predTestNN, 1, 0))
cost(germanTest$response, predTestNN)


#Roc curve for nnet
par(mfrow = c(1,2))
ROC_NN <- roc.plot(x = (germanTrain$response == "1"), pred = probNN[,2])
ROC_NN$roc.vol

ROC_nnTest <- roc.plot(x = (germanTest$response == "1"), pred = probTestNN[ ,2])
ROC_nnTest$roc.vol




