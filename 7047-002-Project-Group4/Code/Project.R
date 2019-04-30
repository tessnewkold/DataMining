library(readxl)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(ROCR)
library(dplyr)
library(e1071)
library(pROC)
library(StatMeasures)


set.seed(012996)

cancer <- read_xlsx("/Users/MeganEckstein/Documents/2019SpringSemester/DataMiningII/BreastCancerWisconsin.xlsx")
cancer$Class <- ifelse(cancer$Class == 2, 0, 1)


str(cancer)
cancer$`Bare Nuclei` <- as.numeric((cancer$`Bare Nuclei`))
summary(glm(Class~., family = binomial, data = cancer))

ggplot(cancer, aes(x = cancer$`Clump Thickness`, fill = as.factor(Class))) + 
  geom_histogram(position = "dodge", binwidth = 1) +
  facet_grid(~Class) 

cancer <- select(cancer, -`ID Number`)

sampleIndex <- sample(nrow(cancer),nrow(cancer)*0.75)
CancerTrain <- cancer[sampleIndex,]
CancerTest <- cancer[-sampleIndex,]

cancer.tree <- rpart(Class~., CancerTrain, method = "class")
prp(cancer.tree,digits = 4, extra = 1)

train.pred.tree<- predict(cancer.tree, CancerTrain, type="class")
table(CancerTrain$Class, train.pred.tree, dnn=c("Truth","Predicted"))
MR.train <- mean(CancerTrain$Class!=train.pred.tree)
MR.train

test.pred.tree <- predict(cancer.tree, CancerTest, type = "class")
table(CancerTest$Class, test.pred.tree, dnn=c("Truth","Predicted"))
MR.test <- mean(CancerTest$Class!=test.pred.tree)
MR.test

cancer.tree.prob.rpart = predict(cancer.tree,CancerTrain, type="prob")
pred = prediction(cancer.tree.prob.rpart[,2], CancerTrain$Class)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
AUC.train <- slot(performance(pred, "auc"), "y.values")[[1]]
AUC.train




cancer.tree.prob.rpart.test = predict(cancer.tree,CancerTest, type="prob")
pred = prediction(cancer.tree.prob.rpart.test[,2], CancerTest$Class)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
AUC.test <- slot(performance(pred, "auc"), "y.values")[[1]]
AUC.test

# SUPPORT VECTOR MACHINE

# impute missing data
CancerTrain$`Bare Nuclei`[is.na(CancerTrain$`Bare Nuclei`)] = mean(CancerTrain$`Bare Nuclei`, na.rm=TRUE)
CancerTest$`Bare Nuclei`[is.na(CancerTest$`Bare Nuclei`)] = mean(CancerTest$`Bare Nuclei`, na.rm=TRUE)

str(CancerTrain)
dplyr::glimpse(CancerTrain)


CancerTrain$Class <- as.factor(CancerTrain$Class)
svm.mod <- svm(Class~., data = CancerTrain, type = "C-classification", cost = 1, probability = T)
summary(svm.mod)
svm.mod$index




pred.svm <- predict(svm.mod, CancerTrain, probability = T)
table(CancerTrain$Class,pred.svm,dnn=c("Obs","Pred"))
(MR.train.svm <- mean(CancerTrain$Class!=pred.svm))
pred.svm.train <- predict(svm.mod, CancerTrain, probability = T, type = "response")
train.roc <- roc(as.numeric(CancerTrain$Class), as.numeric(pred.svm.train))
plot(train.roc, col = 4)
pred.svm.test <- predict(svm.mod, CancerTest, probability = T)
table(CancerTest$Class,pred.svm.test,dnn=c("Obs","Pred"))
(MR.test.svm <- mean(CancerTest$Class!=pred.svm.test))
test.roc <- roc(as.numeric(CancerTest$Class), as.numeric(pred.svm.test))
test.roc
plot(test.roc, col = 4)

#plot(svm.mod, CancerTrain, `Marginal Adhesion`~`Bare Nuclei`)


# A 'data.frame' with y and yhat
df <- data.frame(y = as.numeric(CancerTrain$Class),
                 yhat = as.numeric(pred.svm.train))

# KS table and value
# ltKs <- ks(y = df[, 'y'], yhat = df[, 'yhat'])
# ksTable <- ltKs$ksTable
# KS <- ltKs$ks

library(InformationValue)
ks_plot(as.numeric(CancerTrain$Class), as.numeric(pred.svm.train))
ks_plot(as.numeric(CancerTest$Class), as.numeric(pred.svm.test))
