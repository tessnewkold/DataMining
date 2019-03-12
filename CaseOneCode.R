#Tess Newkold
#Case 1 - Advanced Trees 
#03_12_2019


# Load Libraries ----------------------------------------------------------
library(MASS)
library(rpart)
library(rpart.plot)
library(ipred)
library(randomForest)
library(gbm)

#set seet
set.seed(11434445)

# load Boston data
data(Boston)
#setting up the testing and training set
index <- sample(nrow(Boston),nrow(Boston)*0.75)
bostonTrain <- Boston[index,]
bostonTest <- Boston[-index,]



# Linear Regression -------------------------------------------------------
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

# Regression Tree ---------------------------------------------------------
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

# Bagging -----------------------------------------------------------------
bostonBag<- bagging(medv~., data = bostonTrain, nbagg=50)
bostonBag
summary(bostonBag)

#Prediction on testing sample
bostonTrainBagPred<- predict(bostonBag)
bostonTestBagPred<- predict(bostonBag, newdata = bostonTest)
MSE_Bagging <- mean((bostonTrain$medv - bostonTrainBagPred)^2)
MSPE_Bagging <- mean((bostonTest$medv - bostonTestBagPred)^2)
MSE_Bagging; MSPE_Bagging 

#shows how many trees are necessary to make the model
ntree<- c(1, 3, 5, seq(10, 200, 10))
MSE_test<- rep(0, length(ntree))
for(i in 1:length(ntree)){
  bostonBag<- bagging(medv~., data = bostonTrain, nbagg=ntree[i])
  bostonTestBagPred<- predict(bostonBag, newdata = bostonTest)
  MSE_test[i]<- mean((bostonTest$medv-bostonTestBagPred)^2)
}
plot(ntree, MSE_test, type = 'l', col=2, lwd=2, xaxt="n", main = "Number of Trees to get Low MSE")
axis(1, at = ntree, las=1)
#50 trees should be sufficient 

# Random Forest -----------------------------------------------------------
bostonRandomForest <- randomForest(medv~., data = bostonTrain, importance=TRUE)
bostonRandomForest
bostonRandomForest$importance
plot(bostonRandomForest$mse, type='l', col=2, lwd=2, xlab = "ntree", ylab = "OOB Error", main = "Out of Bag Prediction Error for Number of Trees")
#Prediction
bostonTrainRandomPred <- predict(bostonRandomForest)
bostonTestRandomPred <- predict(bostonRandomForest, bostonTest)
MSE_RandomForest <- mean((bostonTrain$medv - bostonTrainRandomPred)^2)
MSPE_RandomForest <- mean((bostonTest$medv - bostonTestRandomPred)^2)
MSE_RandomForest; MSPE_RandomForest

# Boosting ----------------------------------------------------------------
bostonBoost <- gbm(medv~., data = bostonTrain, distribution = "gaussian", n.trees = 10000, shrinkage = 0.01, interaction.depth = 8)
summary(bostonBoost)

bostonTrainBoostPred <- predict(bostonBoost, n.trees = 10000)
bostonTestBoostPred <- predict(bostonBoost, bostonTest, n.trees = 10000)
MSE_Boosting <- mean((bostonTrain$medv - bostonTrainBoostPred)^2)
MSPE_Boosting <- mean((bostonTest$medv - bostonTestBoostPred)^2)
MSE_Boosting; MSPE_Boosting 

plot(bostonBoost, i = "lstat")
plot(bostonBoost, i = "rm")
plot(bostonBoost, i = "nox")
plot(bostonBoost, i = "dis")
plot(bostonBoost, i = "crim")




