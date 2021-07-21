library(caTools)
library(randomForest)
library(caret)
 
#### data exploration ####
data<- read.csv('Train_upvotes.csv')
View(data)
str(data)
summary(data)

boxplot(data$ID)

boxplot(data$Reputation)
quantile(data$Reputation, 0.99)
data$Reputation[data$Reputation>3*quantile(data$Reputation,0.99)]
data$Reputation[data$Reputation>3*quantile(data$Reputation,0.99)]<- 3*quantile(data$Reputation,0.99)

boxplot(data$Answers)
data$Answers[data$Answers>3*quantile(data$Answers, 0.99)]
data$Answers[data$Answers>3*quantile(data$Answers, 0.99)]<- 3*quantile(data$Answers, 0.99)

boxplot(data$Username)

boxplot(data$Views)
data$Views[data$Views>3*quantile(data$Views, 0.99)]
data$Views[data$Views>3*quantile(data$Views, 0.99)]<-3*quantile(data$Views, 0.99)

boxplot(data$Upvotes)
###3*quantile(data$Upvotes, 0.99)
###data$Upvotes[data$Upvotes>200000]
data$Upvotes[data$Upvotes>200000]<- NA
sum(is.na(data))
data<- na.omit(data)
sum(is.na(data))


data$Tag<- as.numeric(data$Tag)

cor(data, method = "pearson")
#pairs(data$Upvotes~., data = data)

#### data splitting ####
split<- sample.split(data$Tag, SplitRatio = 0.70)
train <- subset(data, split==T)
test<- subset(data, split==F)

### data scaling ###
norm_train<- scale(train[,-c(7)], center = TRUE, scale = TRUE )
norm_test<- scale(test[,-c(7)], center = TRUE, scale = TRUE)
View(norm_train)
?scale

norm_train<- as.data.frame(norm_train)
norm_test<- as.data.frame(norm_test)

#### evaluation methods #### 
rsq<- function(x,y){
  rss= sum((x-y)^2)
  tss= sum((y-mean(y))^2)
  rsqre<- 1-(rss/tss)
  return(rsqre)
}

RMSE<-function(y,x){
  sum<-(y-x)^2
  m<- mean(sum)
  ols<- sqrt(m)
  return(ols)
}

#### model preparation####

### linear regression model ###
model_lm<- step(lm(train$Upvotes~., data = norm_train), direction = "both")  
summary(model_lm)
pred<- predict(model_lm, norm_test)
rsq(pred, test$Upvotes)  # r square obtained 0.28354
RMSE(pred,test$Upvotes)  # 2012.814

#### random forest ####
install.packages("randomForest")
library(randomForest)

?randomForest
model_rf<- randomForest(train$Upvotes~., data = norm_train, ntree = 40, nodesize = 50, maxnodes = 20)
pred2<- predict(model_rf, norm_test)
rsq(pred2, test$Upvotes)         # r squared obtained 0.582232
RMSE(pred2,test$Upvotes)         # 1537.024
View(pred2)
plot(model_rf)
print(model_rf)

model_rf2<- randomForest(train$Upvotes~., data = norm_train, ntree = 15, nodesize = 60, maxnodes = 25)
print(model_rf2)
pred3<- predict(model_rf2, norm_test)
View(pred3)
rsq(pred3, test$Upvotes)       # r squared obtained 0.5975
RMSE(pred3,test$Upvotes)      # 1508.505


model_rf3<- randomForest(train$Upvotes~., data = norm_train, ntree = 15, nodesize = 100, maxnodes = 40)
pred4<- predict(model_rf3, norm_test)
rsq(pred4, test$Upvotes)     # r squared obtained 0.647  # %var explained 6865
RMSE(pred4,test$Upvotes)     # 1331.418
print(model_rf3)
plot(model_rf3)

model_rf4<- randomForest(train$Upvotes~., data = norm_train, ntree = 13, nodesize = 200, maxnodes = 50)
pred5<- predict(model_rf4, norm_test)
rsq(pred5, test$Upvotes)   # r squared obtained 0.6237 # %var explained 61.86
RMSE(pred5,test$Upvotes)   # 1464
print(model_rf4)


### model tuning ####
#tune<- tuneRF(train[,-7], train[,7],
 #             stepFactor = 0.5,     #it will keep on changing the mtry value by this much
  #            ntreeTry = 40,        #seen after plotting the model
   #           plot = TRUE,
    #          improve = 10000,
     #         trace = TRUE)      ## for this much increase in error the search will continue



 ### decision tree ####
library(rpart)
library(rpart.plot)
?rpart
fulltree<- rpart(train$Upvotes~., data = norm_train, control = rpart.control( cp = 0))
pred6<- predict(fulltree, norm_test)
rsq(pred6, test$Upvotes)   # r squared obtained 0.8364
RMSE(pred6,test$Upvotes)   # 1044.81

printcp(fulltree)

## tree pruning ##
mincp<- fulltree$cptable[which.min(fulltree$cptable[,"xerror"]), "CP"]
mincp

## pruned tree ##
pruned_tree <- prune(fulltree, cp = mincp)
pred7<- predict(pruned_tree, norm_test)
rsq(pred7, test$Upvotes)   # r squared obtained 0.8372
RMSE(pred7,test$Upvotes)  # 1042.429

## gradient boosting  ##
install.packages("gbm")
library(gbm)
?gbm
boosting <- gbm(train$Upvotes~., data = train, distribution = "gaussian", n.trees = 500, interaction.depth = 10, shrinkage = 0.5, verbose = F)
pred8 <- predict(boosting, test, n.trees = 20)
rsq(pred8, test$Upvotes)   # r squared obtained 0.8561 
RMSE(pred8,test$Upvotes)  # 979.90

#######
boosting01 <- gbm(train$Upvotes~., data = norm_train, distribution = "gaussian", n.trees = 500, interaction.depth = 10, shrinkage = 0.5, verbose = F)
pred08 <- predict(boosting01, norm_test, n.trees = 20)
rsq(pred08, test$Upvotes)   # r squared obtained 0.8317 # 0.8027
RMSE(pred08,test$Upvotes)   # 1059.927
#######

boosting2 <- gbm(train$Upvotes~., data = norm_train, distribution = "gaussian", n.trees = 1000, interaction.depth = 15, shrinkage = 0.2, verbose = F)
pred9 <- predict(boosting2, norm_test, n.trees = 20)
rsq(pred9, test$Upvotes)   # r squared obtained 0.8622
RMSE(pred9,test$Upvotes)   # 959.00

##### max R square obtained ####
boosting20 <- gbm(train$Upvotes~., data = train, distribution = "gaussian", n.trees = 1000, interaction.depth = 15, shrinkage = 0.2, verbose = F)
pred90 <- predict(boosting20, test, n.trees = 20)
rsq(pred90, test$Upvotes)   # r squared obtained 0.88459
RMSE(pred90,test$Upvotes)   # 913.0815
summary(boosting20)


boosting3 <- gbm(train$Upvotes~., data = train, distribution = "gaussian", n.trees =2000, interaction.depth = 20, shrinkage = 1, verbose = F, cv.folds = 4)

pred10 <- predict(boosting3, test, n.trees = 20)
rsq(pred10, test$Upvotes)   # r squared obtained 0.7839
RMSE(pred10,test$Upvotes)   # 980.17

### bagging ###

model_bag<- randomForest(train$Upvotes~., data = norm_train, mtry = 6, ntree = 100)


#### saving the final file ####

test_final<- read.csv("test_upvotes.csv")
test_final$Tag<- as.numeric(test_final$Tag)
predicted<- predict(boosting20, test_final, n.trees = 20)
test_final["Predicted"]<- predicted
View(test_final)
write.csv(test_final,"final.csv")
