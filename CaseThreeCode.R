# Libraries ---------------------------------------------------------------
library(dplyr)
library(readxl)
library(tidyverse)
library(factoextra)
library(gridExtra)
library(fpc)
library(arules)
library(arulesViz)




# European Jobs - Clustering -----------------------------------------------------------
euroJobs <- read_excel("/Users/Tess/Documents/Tess/Ohio/UniversityOfCincinnati/BuisnessAnalytics/Spring2019/DataMining/DataMining2/CaseHomework/CaseThree/europeanJobs.xlsx", col_names = TRUE)
jobs <- euroJobs[-1]

distance <- get_dist(jobs)
fviz_dist(distance, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))

# K-Means Cluster Analysis
fit <- kmeans(jobs, 2) #2 cluster solution
#Display number of clusters in each cluster
table(fit$cluster)

fit

fviz_cluster(fit, data = jobs)

k3 <- kmeans(jobs, centers = 3, nstart = 25)
k4 <- kmeans(jobs, centers = 4, nstart = 25)
k5 <- kmeans(jobs, centers = 5, nstart = 25)

# plots to compare
p1 <- fviz_cluster(fit, geom = "point", data = jobs) + ggtitle("k = 2")
p2 <- fviz_cluster(k3, geom = "point",  data = jobs) + ggtitle("k = 3")
p3 <- fviz_cluster(k4, geom = "point",  data = jobs) + ggtitle("k = 4")
p4 <- fviz_cluster(k5, geom = "point",  data = jobs) + ggtitle("k = 5")


grid.arrange(p1, p2, p3, p4, nrow = 2)


plotcluster(jobs, fit$cluster)
#See exactly which item are in 1st group
jobs[fit$cluster==1,]
#get cluster means for scaled data
aggregate(jobs,by=list(fit$cluster),FUN=mean)

# Determine number of clusters
wss <- (nrow(jobs)-1)*sum(apply(jobs,2,var))
for (i in 2:12) wss[i] <- sum(kmeans(jobs,
                                     centers=i)$withinss)
plot(1:12, wss, type="b", xlab="Number of Clusters",ylab="Within groups sum of squares")

#prediction.strength(jobs, Gmin=2, Gmax=15, M=10,cutoff=0.8)

d = dist(jobs, method = "euclidean")
result = matrix(nrow = 14, ncol = 3)
for (i in 2:15){
  cluster_result = kmeans(jobs, i)
  clusterstat=cluster.stats(d, cluster_result$cluster)
  result[i-1,1]=i
  result[i-1,2]=clusterstat$avg.silwidth
  result[i-1,3]=clusterstat$dunn   
}
plot(result[,c(1,2)], type="l", ylab = 'silhouette width', xlab = 'number of clusters')
plot(result[,c(1,3)], type="l", ylab = 'dunn index', xlab = 'number of clusters')

#choosing the 3 clusters is the best



#Wards Method or Hierarchical clustering
#Calculate the distance matrix
jobsDist = dist(jobs)
#Obtain clusters using the Wards method
jobsHclust=hclust(jobsDist, method = "ward")
plot(jobsHclust)

#Cut dendrogram at the 3 clusters level and obtain cluster membership
jobs3clust = cutree(jobsHclust,k=3)
#See exactly which item are in third group
jobs[jobs3clust == 3,]

#get cluster means for raw data
#Centroid Plot against 1st 2 discriminant functions
#Load the fpc library needed for plotcluster function
plotcluster(jobs, jobs3clust)

# Cincinnati Zoo - Association --------------------------------------------
TransFood <- read.csv('https://xiaoruizhu.github.io/Data-Mining-R/data/food_4_association.csv')
TransFood <- TransFood[, -1]
# Find out elements that are not equal to 0 or 1 and change them to 1.
Others <- which(!(as.matrix(TransFood) == 1 | as.matrix(TransFood) == 0), arr.ind = T)
TransFood[Others] <- 1
TransFood <- as(as.matrix(TransFood), "transactions")

summary(TransFood)

x <- TransFood[size(TransFood) > 10]
inspect(x)

itemFrequencyPlot(TransFood, support = 0.1, cex.names=0.8)

# Run the apriori algorithm
basket_rules <- apriori(TransFood, parameter = list(sup = 0.003, conf = 0.5,target="rules"))
summary(basket_rules)
# Check the generated rules using inspect
inspect(head(basket_rules))
#Basket rules of size greater than 4
inspect(subset(basket_rules, size(basket_rules)>3))
#Greater lift values indicate stronger associations.
water.lhs <- subset(basket_rules, subset = lhs %in% "Bottled.WaterFood" & lift>1.8)
inspect(water.lhs)

hotDog.rhs <- subset(basket_rules, subset = rhs %in% "Hot.DogFood" & lift>1.8)
inspect(hotDog.rhs)

plot(basket_rules)
plot(basket_rules, interactive=TRUE)

plot(head(sort(basket_rules, by="lift"), 10), method = "graph")

plot(basket_rules, method="grouped")
