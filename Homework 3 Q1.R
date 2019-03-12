# 1
set.seed(11434445)
n <- 500
#to generate random uniform data
x1 <- runif(n,0,1)
x2 <- c()
for (i in 1:500) {
  if (i %% 2 == 0) {
    x2[i] <- 0
  }
  else{
    x2[i] <- 1
  }
}
z <- (-1 + (5.2 * x1) + ((-0.4) * x2))
pr <- 1/(1 + exp(-z))
y <- rbinom(n, 1, pr)

df <- data.frame(y = y, x1 = x1, x2 = x2)

model1Logit <- glm(y ~ x1 + x2, family = binomial(link = "logit"))
summary(model1Logit)
model1Probit <- glm( y~x1 + x2, family = binomial(link = "probit"))
summary(model1Probit)

#Part B
#calculate probability of CDF
pr <- pnorm(z)
y <- rbinom(n, 1, pr)
df <- data.frame(y = y, x1 = x1, x2 = x2)


model2Logit <- glm(y ~ x1 + x2, family = binomial(link = "logit"))
summary(model2Logit)
model2Probit <- glm( y~x1 + x2, family = binomial(link = "probit"))
summary(model2Probit)

#Residual deviance: is a measure of the goodness of fit of the model
      #higher the number indicates a bad fit 









