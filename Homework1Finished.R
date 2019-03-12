
# Load Packages -----------------------------------------------------------
library(MASS)
library(plyr)
library(dplyr)
library(ggplot2)
library(knitr)
library(datasets)
library(class)
library(fpc)

# Exploratory Data Analysis -----------------------------------------------
data(iris)
summary(iris)

sd(iris$Sepal.Width)
sd(iris$Sepal.Length)
sd(iris$Petal.Width)
sd(iris$Petal.Length)

head(iris)
str(iris)

#hist(iris$Sepal.Length, col ="purple", breaks = 15)
#plot(density(iris$Sepal.Length))

hist(iris$Sepal.Length, prob = T, col = "purple", breaks = 15, main = "Histogram and Density of Sepal Length", xlim = c(3,9), xlab = "Sepal Length")
lines(density(iris$Sepal.Length), col = "red", lwd = 2)

plot(iris$Sepal.Length, iris$Sepal.Width, xlab = "Length ", ylab = "Width", main = "Sepal Dimensions", col = "purple", pch = 8)




# Supervised Learning -----------------------------------------------------
#making the training set
set.seed(13003361)
allrows <- 1:nrow(iris)
trainrows <- sample(allrows, replace = F, size = 0.8*length(allrows))
train_iris <- iris[trainrows, 1:4]
train_label <- iris[trainrows, 5]
table(train_label)

#making the testing set
test_iris <- iris[-trainrows, 1:4]
test_label <- iris[-trainrows, 5]
table(test_label)



#K nearest neighbor
error.train <- replicate(0,50)
for (k in 1:50) {
  pred_iris <- knn(train = train_iris, test = train_iris, cl = train_label, k)
  error.train[k] <- 1 - mean(pred_iris == train_label)
}
error.train <- unlist(error.train, use.names = FALSE)

error.test <- replicate(0,50)
for (k in 1:50) {
  pred_iris <- knn(train = train_iris, test = test_iris, cl = train_label, k)
  error.test[k] <- 1 - mean(pred_iris == test_label)
}

error.test <- unlist(error.test, use.names = FALSE)

plot(error.train, type = "o", ylim = c(0,0.2), col = "blue", xlab = "K values", 
     ylab = "Misclassification errors")
lines(error.test, type = "o", col = "red")
legend("topright", legend = c("Training error","Test error"), 
     col = c("blue","red"), lty = 1:1)







# Unsupervised Learning ---------------------------------------------------

iris.new <- iris[,c(1,2,3,4)]
iris.class <- iris[,"Species"]
head(iris.new)
head(iris.class)


result <- kmeans(iris.new,5) 
result$size #

result$centers 

par(mfrow = c(2,2), mar = c(5,4,2,2))
#plot(iris.new[c(1,2)], col=iris.class)
plot(iris.new[c(1,2)], col = result$cluster)

 
#plot(iris.new[c(3,4)], col=iris.class)
plot(iris.new[c(3,4)], col = result$cluster)


table(result$cluster,iris.class,  dnn = c("True", "Predicted"))


#Hierarchical Clustering
par(mfrow = c(1,1), mar = c(5,4,2,2))

hc_result <- hclust(dist(iris[,1:4]))
plot(hc_result)

#Cut Dendrogram into 3 Clusters
rect.hclust(hc_result, k = 3)



