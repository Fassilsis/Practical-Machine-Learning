## Practical Machine learning Prediction assignment

setwd("~/Coursera/Data Science/PML")

# Packages to be used. 
library(caret)
library(knitr)
library(rpart)
library(rpart.plot)
library(randomForest)
library(corrplot)
library(rattle)


# Let us download the datasets

UrlTr <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
UrlTe  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"


training <- read.csv(url(UrlTr))
testing  <- read.csv(url(UrlTe))

# creating a partition  
inTrain  <- createDataPartition(training$classe, p=0.7, list=FALSE)
TrainSet <- training[inTrain, ]
TestSet  <- training[-inTrain, ]
dim(TrainSet)
dim(TestSet)

# Let us remove variables with NA
AllNA    <- sapply(TrainSet, function(x) mean(is.na(x))) > 0.95
TrainSet <- TrainSet[, AllNA==FALSE]
TestSet  <- TestSet[, AllNA==FALSE]
dim(TrainSet)
dim(TestSet)

TrainSet <- TrainSet[, -(1:5)]
TestSet  <- TestSet[, -(1:5)]
dim(TrainSet)
dim(TestSet)

# correlation

corMatrix <- cor(TrainSet[, -54])
corrplot(corMatrix, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))
# model fit
# 1.Random Forest

set.seed(12345)
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modFitRandForest <- train(classe ~ ., data=TrainSet, method="rf",
                          trControl=controlRF)
modFitRandForest$finalModel
# model fit
#2. Decision Trees

set.seed(12345)
modFitDecTree <- rpart(classe ~ ., data=TrainSet, method="class")
fancyRpartPlot(modFitDecTree)


predictDecTree <- predict(modFitDecTree, newdata=TestSet, type="class")
confMatDecTree <- confusionMatrix(predictDecTree, TestSet$classe)
confMatDecTree


plot(confMatDecTree$table, col = confMatDecTree$byClass, 
     main = paste("Decision Tree - Accuracy =",
                  round(confMatDecTree$overall['Accuracy'], 4)))
# model 
#3. Generalized Boosted Model

set.seed(12345)
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFitGBM  <- train(classe ~ ., data=TrainSet, method = "gbm",
                    trControl = controlGBM, verbose = FALSE)
modFitGBM$finalModel

predictGBM <- predict(modFitGBM, newdata=TestSet)
confMatGBM <- confusionMatrix(predictGBM, TestSet$classe)
confMatGBM


plot(confMatGBM$table, col = confMatGBM$byClass, 
     main = paste("GBM - Accuracy =", round(confMatGBM$overall['Accuracy'], 4)))



## Finally let us apply the Selected Model (i.e. Random Forest) to the Test Data
The accuracy of the 3 regression modeling methods above are:
  
predictTEST <- predict(modFitRandForest, newdata=testing)
predictTEST
