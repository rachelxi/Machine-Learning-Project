library(caret)
library(rpart)
library(rattle)
library(randomForest)
##Process training data
data <- read.csv("pml-training.csv",na.strings = c("NA", "","#DIV/0!"))
data <- data[,-(1:7)]
data$classe <- as.factor(data$classe)
summary(colSums(is.na(data)))
index <- colSums(is.na(data))==0
data <- data[index]
dim(data)
##Data partition
set.seed(2000)
inTrain <- createDataPartition(y=data$classe,p=0.75,list=FALSE)
training <- data[inTrain,]
validation <- data[-inTrain,]
##Remove zero covariates
nsv <- nearZeroVar(training,saveMetrics=TRUE)
sum(nsv$zeroVar=="TRUE")
##Model1, classification tree with rpart method
modelFit1 <- train(classe~.,method="rpart",data=training,trControl = trainControl(method = "cv"))
fancyRpartPlot(modelFit1$finalModel)
pred0 <- predict(modelFit1,newdata=training)
pred1 <- predict(modelFit1,newdata=validation)
confusionMatrix(pred0,training$classe)
confusionMatrix(pred1,validation$classe)
##Model2, pca with random forest
proc <- preProcess(training[,-53],method="pca")
length(proc)
trainPC <- predict(proc,newdata=training[,-53])
validPC <-predict(proc,newdata=validation[,-53])
modelFit2 <- train(training$classe~.,method="rf",data=trainPC,ntree=200,trControl=trainControl(method="cv"))
pred2 <- predict(modelFit2,newdata=trainPC)
pred3 <- predict(modelFit2,newdata=validPC)
confusionMatrix(pred2,training$classe)
confusionMatrix(pred3,validation$classe)
modelFit2$finalModel
##Model3, random forest
modelFit3 <- train(classe~.,method="rf",data=training,ntree=200,trControl=trainControl(method="cv"))
pred4 <- predict(modelFit3,newdata=training)
pred5 <- predict(modelFit3,newdata=validation)
confusionMatrix(pred4,training$classe)
confusionMatrix(pred5,validation$classe)
##Predict with model 2
##Process testing data
testing <- read.csv("pml-testing.csv",na.strings = c("NA", "","#DIV/0!"))
testing <- testing[,-(1:7)]
index <- colSums(is.na(testing))==0
testing <- testing[index]
testPC <- predict(proc,newdata=testing[,-53])
prediction <- predict(modelFit2,newdata=testPC)
##Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; 
##Fuks, H. Qualitative Activity Recognition of Weight Lifting 
##Exercises. Proceedings of 4th International Conference in 
##Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, 
##Germany: ACM SIGCHI, 2013.
##exactly according to the specification (Class A), 
##throwing the elbows to the front (Class B), 
##lifting the dumbbell only halfway (Class C), 
##lowering the dumbbell only halfway (Class D) and 
##throwing the hips to the front (Class E)

