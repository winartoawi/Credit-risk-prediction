#libraries 
library(mice)
library(ggplot2)
library(caTools)
library(caret)
library(e1071)
library(randomForest)
library(mlbench)
library(DataExplorer)
library(dplyr)
library(outliers)
library(rpart)
library(rpart.plot)


# loading data set
raw_dataset <- read.csv("D:/awi/Documents/UNI/AI Master Sem 1/AM/Dataset/csv file/give_some_credit_ori.csv")
str(raw_dataset)

# The ID Number column is dropped
raw_dataset <- raw_dataset[ , -1]
summary(raw_dataset)

# Columns are renamed to facilitate semantic interpretation
names(raw_dataset) <- c("class", "RUUL","Age","PD30.60","DR","MI","NOCL","DL90","NREL","PD60.80","NOD")

# Check for Missing Values
colSums(sapply(raw_dataset,is.na))
plot_missing(raw_dataset)
summary(raw_dataset)

# MICE Imputation - pmm method 
seed = 500
ds_pmm <- mice(raw_dataset,m=1,maxit=1,meth='pmm',seed=seed)
ds_pmm <- complete(ds_pmm,1)
colSums(sapply(ds_pmm,is.na))
plot_missing(ds_pmm)

# preprocess imputation - median method
ds_median <- preProcess(raw_dataset,method = "medianImpute")
ds_median <- predict(ds_median,raw_dataset)
colSums(sapply(ds_median,is.na))
plot_missing(ds_median)


# Check density plot 
ggplot(ds_pmm, aes(x=RUUL)) + 
  theme_bw()+
  geom_histogram(color = 'black', fill = 'blue', alpha = 0.2)+
  geom_density(aes(y=..count..), colour="red", adjust=4) +
  labs(title = "", x="RUUL", y="Count")

# checking the outlier
ggplot(ds_filter, aes(x=class, y=PD30.60,group=class)) +
  theme_bw()+
  geom_boxplot( fill = "blue", alpha = 0.2)+
  theme(legend.position='none')+ coord_flip()+
  labs(title = "Delinquency vs. Number of times late of 30 - 59 days", y="Number of times late of 30 - 59 days", x="non-default = yes, Default = no")

# checking the outlier
ggplot(ds_filter, aes(x=class, y=PD60.80,group=class)) +
  theme_bw()+
  geom_boxplot( fill = "blue", alpha = 0.2)+
  theme(legend.position='none')+ coord_flip()+
  labs(title = "Delinquency vs. Number of times late of 60 - 89 days", y="Number of times late of 60 - 89 days", x="non-default = yes, Default = no")

# checking the outlier
ggplot(ds_filter, aes(x=class, y=DL90 ,group=class)) +
  theme_bw()+
  geom_boxplot( fill = "blue", alpha = 0.2)+
  theme(legend.position='none')+ coord_flip()+
  labs(title = "Delinquency vs. Number of times late of 90 days", y="Number of times late of 90 days", x="non-default = yes, Default = no")

# Removing outlier - 2 different dataset differ by imputation method - ds 1 -> ds_median , ds 2 -> ds_pmm
#ds_filter <- ds_median
ds_filter <- ds_pmm

ds_filter$PD30.60 <- rm.outlier(ds_filter$PD30.60,fill=TRUE,median=FALSE,opposite=FALSE)
ds_filter$PD30.60 <- rm.outlier(ds_filter$PD30.60,fill=TRUE,median=FALSE,opposite=FALSE)
ds_filter$PD60.80 <- rm.outlier(ds_filter$PD30.60,fill=TRUE,median=FALSE,opposite=FALSE)
ds_filter$PD60.80 <- rm.outlier(ds_filter$PD30.60,fill=TRUE,median=FALSE,opposite=FALSE)
ds_filter$DL90 <- rm.outlier(ds_filter$PD30.60,fill=TRUE,median=FALSE,opposite=FALSE)
ds_filter$DL90 <- rm.outlier(ds_filter$PD30.60,fill=TRUE,median=FALSE,opposite=FALSE)

# checking the removed outlier
ggplot(ds_filter, aes(x=class, y=PD30.60,group=class)) +
  theme_bw()+
  geom_boxplot( fill = "blue", alpha = 0.2)+
  theme(legend.position='none')+ coord_flip()+
  labs(title = "Delinquency vs. Number of times late of 30 - 59 days", y="Number of times late of 30 - 59 days", x="non-default = yes, Default = no")

# checking the removed outlier
ggplot(ds_filter, aes(x=class, y=PD60.80,group=class)) +
  theme_bw()+
  geom_boxplot( fill = "blue", alpha = 0.2)+
  theme(legend.position='none')+ coord_flip()+
  labs(title = "Delinquency vs. Number of times late of 60 - 89 days", y="Number of times late of 60 - 89 days", x="non-default = yes, Default = no")

# checking the removed outlier
ggplot(ds_filter, aes(x=class, y=DL90 ,group=class)) +
  theme_bw()+
  geom_boxplot( fill = "blue", alpha = 0.2)+
  theme(legend.position='none')+ coord_flip()+
  labs(title = "Delinquency vs. Number of times late of 90 days", y="Number of times late of 90 days", x="non-default = yes, Default = no")


# Checking for class balance
table(ds_filter$class)
prop.table(table(ds_filter$class))
class_check <- ds_filter

# imbalance visualization on target/class variable
barplot(table(class_check$class),
        xlab="Delinquent (0 = Yes, 1 = No)", ylab="Count", col=c("darkblue","red"),
        legend = levels(ds_pmm$class), beside=TRUE)

# DOWNSAMPLING - CARET 
ds_filter$class <- factor(ds_pmm$class,levels = c(0,1), labels=c("Yes","No"))
down_sampled_dataset <- downSample(ds_filter,ds_filter$class,list=FALSE,yname="class")

# remove the target variable----------------------------------------------------------------------------
correlation <- down_sampled_dataset[,-1]
correlation$class <- as.numeric(correlation$class)

# Checking for correlation among independent variables
corrTable <- cor(correlation[,c("RUUL","Age","PD30.60","DR","MI","NOCL","DL90","NREL","PD60.80","NOD","class")])
plot_correlation(correlation,'continuous', cor_args = list("use" = "pairwise.complete.obs"))

# remove the correlated to avoid multi-collinearity 
correlation <- correlation[,c(-7)]
corrTable <- cor(correlation[,c("RUUL","Age","PD30.60","DR","MI","NOCL","NREL","PD60.80","NOD","class")])
plot_correlation(correlation,'continuous', cor_args = list("use" = "pairwise.complete.obs"))

# removal of correlated variable - to eliminate multi collinearity --
dataset <- down_sampled_dataset[ , c(-1,-8,-12)]

# Normalized --------------------------------------------------------
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
dataset.norm <-as.data.frame(lapply(dataset, normalize))
class <- down_sampled_dataset$class
dataset.norm <- cbind(dataset.norm,class)
str(dataset.norm)

# Data partition (SPLIT) - data set for test and validation
set.seed(seed)
split = sample.split(dataset.norm$class, SplitRatio = 0.65)
training_set = subset(dataset.norm, split == TRUE)
test_set = subset(dataset.norm, split == FALSE)
dim(training_set)
dim(test_set)
names(training_set)
names(test_set)

# MODEL CREATION ----------------------------------------------------
# Defining the training controls for multiple models
train_Control <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = 'final',
  classProbs = T)

# Training the knn model-----------------------------------------------------------------------------
model_knn<-train(class~.,
                 data = training_set,
                 method='knn', 
                 trControl=train_Control, 
                 tuneLength=3)

# k-NN model - on training set
training_set$pred_knn<-predict(object = model_knn,training_set)
confusionMatrix(training_set$class,training_set$pred_knn)

# k-NN model -  optimized parameter by cross validation with k-NN k = 9 on test set
test_set$opt_pred_knn<-predict(object = model_knn,test_set,k=9)
confusionMatrix(test_set$class,test_set$opt_pred_knn)

# k-NN ROC curve
knn_pred <- predict(model_knn,test_set,type ='prob',k=9)
knn_pred <- prediction(knn_pred[,1],test_set$class)
knn_perf <- performance(knn_pred,"tpr","fpr")
plot(knn_perf, colorize = T)
plot(knn_perf, colorize=T, 
     main = "k-NN ROC curve",
     ylab = "Sensitivity",
     xlab = "Specificity",
     print.cutoffs.at=seq(0,1.5,0.3),
     text.adj= c(-0.2,1.7))
abline(a=0,b=1)

# Area Under Curve
test_knn_pred <- predict(object = model_knn,test_set,type = 'prob',k=9)
test_knn_pred <- prediction(test_knn_pred[,1],test_set$class)
knn_auc = as.numeric(performance(test_knn_pred, "auc")@y.values)
knn_auc = round(knn_auc, 3)
knn_auc
legend(.4,.2,knn_auc,title="k-NN - AUC:",cex=1)

# Training the random forest model---------------------------------------------------------------------------
model_rf<-train(class~., 
                data = training_set, 
                method='rf', 
                trControl=train_Control, 
                tuneLength=3)

# Random forest model - on training set
training_set$pred_rf<-predict(object = model_rf,training_set)
confusionMatrix(training_set$class,training_set$pred_rf)

# Random forest model - optimized parameter by cross validation with mtry = 2 on test set
test_set$pred_rf<-predict(object = model_rf,test_set,mtry=2)
confusionMatrix(test_set$class,test_set$pred_rf)

# Random forest - ROC curve
rf_pred <- predict(model_rf,test_set,type ='prob',mtry=2)
rf_pred <- prediction(rf_pred[,1],test_set$class)
rf_perf <- performance(rf_pred,"tpr","fpr")
plot(rf_perf,add = TRUE, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "Specificity",
     print.cutoffs.at=seq(0,1.5,0.3),
     text.adj= c(-0.2,1.7))
abline(a = 0, b = 1)

# Area Under Curve
test_rf_pred <- predict(object = model_rf,test_set,type = 'prob',mtry =2)
test_rf_pred <- prediction(test_rf_pred[,1],test_set$class)

rf_auc = as.numeric(performance(test_rf_pred, "auc")@y.values)
rf_auc = round(rf_auc, 3)
rf_auc
legend(.6,.5,rf_auc,title="RF - AUC:",cex=1)

# Grid Search optimization 
control <- trainControl(method="cv", number=5, search="grid")
tunegrid <- expand.grid(.mtry=c(1:9))
rf_gridsearch <- train(class ~ ., data=training_set, method='rf', metric="Accuracy", tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)

# Optimized Random Forest with Grid search 
training_set$pred_opt_rf <- predict(rf_gridsearch,training_set)
confusionMatrix(training_set$class,training_set$pred_opt_rf)

# Optimized random forest with grid search - ROC curve
opt_rf_pred <- predict(rf_gridsearch,test_set,type ='prob')
opt_rf_pred <- prediction(opt_rf_pred[,1],test_set$class)

train_opt_rf_pred <- predict(rf_gridsearch,training_set,type ='prob')
train_opt_rf_pred <- prediction(train_opt_rf_pred[,1],training_set$class)

opt_rf_perf <- performance(opt_rf_pred,"tpr","fpr")
plot(opt_rf_perf,add = TRUE, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "Specificity",
     print.cutoffs.at=seq(0,1.5,0.3),
     text.adj= c(-0.2,1.7))
abline(a=0,b=1)

# Area Under Curve
opt_rf_auc = as.numeric(performance(train_opt_rf_pred, "auc")@y.values)
opt_rf_auc = round(opt_rf_auc, 3)
opt_rf_auc
legend(.6,.2,opt_rf_auc,title="Opt RF- AUC:",cex=1)


# Training the xgbtree model---------------------------------------------------------------------------
model_xgb <- train(class~.,
                   data=training_set,
                   method="xgbTree",
                   trControl=train_Control,
                   tuneLength=3)
model_xgb$bestTune

# Extreme Gradient Boosting model - on training set
training_set$pred_xgb<-predict(object = model_xgb,training_set,type = 'prob')
confusionMatrix(training_set$class,training_set$pred_xgb)

# Extreme Gradient Boosting model - optimized parameter by cross validation with on test set
test_set$pred_opt_xgb<-predict(object = model_xgb,test_set,eta=0.3,max_depth=37,colsample_bytree=0.6,subsample=0.5,nrounds=50)
confusionMatrix(test_set$class,test_set$pred_opt_xgb)

# ROC curve
xgb_pred <- predict(model_xgb,test_set,type ='prob')
xgb_pred <- prediction(xgb_pred[,1],test_set$class)
xgb_perf <- performance(xgb_pred,"tpr","fpr")
plot(xgb_perf,add = TRUE, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))
abline(a = 0, b = 1)

# Area Under Curve
train_xgb_pred <- predict(object = model_xgb,training_set,type = 'prob')
train_xgb_pred <- prediction(train_xgb_pred[,1],training_set$class)

test_xgb_pred <- predict(object = model_xgb,test_set,type = 'prob')
test_xgb_pred <- prediction(test_xgb_pred[,1],test_set$class)

xgb_auc = as.numeric(performance(test_xgb_pred, "auc")@y.values)
xgb_auc = round(xgb_auc, 3)
xgb_auc
legend(.6,.2,xgb_auc,title="xgb - AUC:",cex=1)

print(model_knn)
print(model_rf)
print(model_rf_gridsearch)
print(model_xgb)


# Optimization model 
# ~~~~~~~~~~~~~~~~~~~~ Averaging: 

#Predicting the probabilities
test_set$pred_rf_prob<-predict(object = model_rf,test_set,type='prob')
test_set$pred_knn_prob<-predict(object = model_knn,test_set,type='prob')
test_set$pred_xgb_prob<-predict(object = model_xgb,test_set,type='prob')
test_set$pred_avg
#Taking average of predictions
test_set$pred_avg<-(test_set$pred_rf_prob$Y+test_set$pred_knn_prob$Y+test_set$pred_xgb_prob$Y)/3

#Splitting into binary classes at 0.5
test_set$pred_avg<-as.factor(ifelse(test_set$pred_avg>0.5,'Y','N'))
cm = table(Predicted = test_set$pred_avg, Actual = test_set$class)
cm
accuracy =(1-sum(diag(cm))/sum(cm))*100
accuracy

