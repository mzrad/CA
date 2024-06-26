#1. load library#### 
library(dplyr)
library(readxl)
library(na.tools)
library(caret)
library(openxlsx)
library("glmnet")
library(pROC)
library(xgboost)
library(Matrix)
library(dplyr)
library(Boruta)
library(randomForestSRC)
library(rms)
# rm(list = ls())

#2. Read Radiomic Data####
# Load Radiomic Data
develop<-read.csv("develop_01.csv")
validation<-read.csv("validation_01.csv")
colnames(develop)[1] <-"id"
colnames(develop)[2] <-"Label"
colnames(validation)[1] <-"id"
colnames(validation)[2] <-"Label"

#3. DATA Tidying####
# Data Tidying based on CustomLabel
radiomics_end_index <- length(develop)
radiomics_full <- develop
radiomics_full_v<-validation
Feature_full<-develop[3:length(develop)]
Feature_full_v<-validation[3:length(validation)]
Feature_full_v[1,0:2]

#4. Extract Features####

Features_develop <- Feature_full[1:length(Feature_full)]
Features_validation <- Feature_full_v[1:length(Feature_full_v)]

Labels <- as.factor(radiomics_full$Label)
Labels_v <- as.factor(radiomics_full_v$Label)
id_v<-validation$id
id_d<-develop$id
which_na(develop)
which_na(validation)
data_Develop_ori <- cbind(Labels, Features_develop)
data_Develop_ori_full <- cbind(id_d,Labels, Features_develop)

data_validation_ori <- cbind(Labels_v, Features_validation)
data_validation_ori_full <- cbind(id_v,Labels_v, Features_validation)
total_deve_id<-data.frame(unique(data_Develop_ori_full$id_d))

sized1= round(0.7*length(total_deve_id[,1]))
sized2=length(total_deve_id[,1])-sized1

#5. Boruta+RandomForest: model construction####
library(caret)
library(openxlsx)
library("glmnet")
library(pROC)
library(xgboost)
library(Matrix)
library(dplyr)
library(Boruta)

seed <- NULL
counter <- 1
currtree <- NULL
currsize <- NULL
currdepth <- NULL
group <- NULL
train_result <- NULL
train_se <- NULL
train_sp <- NULL

test_result <- NULL
test_se <- NULL
test_sp <- NULL


currseed <- NULL
currboruta <- NULL
currsplit <- NULL

for (cboruta in c(floor(runif(10)*1000))){
  for (csplit in c(floor(runif(10)*1000))){
    
    set.seed(csplit)
    train_split_id<-sample(total_deve_id[,1], sized1, replace = FALSE, prob = NULL)
    train_split_id
    train_full<-data_Develop_ori_full[which(data_Develop_ori_full$id_d %in% train_split_id),]
    train_feature<-train_full[5:length(train_full)]
    y_train <- as.numeric(train_full$Labels)
    y_train <- train_full$Labels
    train_feature_full<-cbind(y_train,train_feature)
    test_full<-data_Develop_ori_full[-which(data_Develop_ori_full$id_d %in% train_split_id),]
    test_feature<-test_full[5:length(test_full)]
    y_test <- as.numeric(test_full$Labels)
    y_test <- test_full$Labels
    
    test_feature_full<-cbind(y_test,test_feature)
    set.seed(cboruta)
    boruta_train <- Boruta(y_train~., data = train_feature, doTrace = 0,maxRuns=15)
    boruta_df <- attStats(boruta_train)
    boruta_features<-boruta_df[order(abs(boruta_df$meanImp),decreasing=TRUE),]
    
    Selected_Features_train<-train_feature %>%dplyr:::select(rownames(boruta_features))
    Selected_Features_test <-test_feature %>%dplyr:::select(rownames(boruta_features))
    SELECTDATA_train <- cbind(y_train, Selected_Features_train)
    SELECTDATA_test <- cbind(y_test, Selected_Features_test)
    
    
    library(randomForestSRC)
    for (groupindex in c(1:5)){
      for (tree in 1:10){
        for (size in 1:10){
          for (depth in 1:5){
            currseed[counter]<-seedindex
            currsplit[counter] <- csplit
            currboruta[counter] <- cboruta
            currtree[counter] <- tree
            currsize[counter] <- size
            currdepth[counter] <- depth
            group[counter] <- groupindex
            set.seed(groupindex)
            rf <- rfsrc(y_train~., data =SELECTDATA_train, ntree=tree, nodesize =size,nodedepth = depth,importance  = TRUE)
            train_class <- predict.rfsrc(rf,SELECTDATA_train)$predicted[,1]
            
            Train_roc_data <- data.frame(y_train,train_class)
            train_result[counter] <- roc(Train_roc_data[,1], Train_roc_data[,2])$auc
            train_se[counter] <-coords (roc(Train_roc_data[,1], Train_roc_data[,2]),
                                        "best")$sensitivity
            train_sp[counter] <-coords (roc(Train_roc_data[,1], Train_roc_data[,2]),
                                        "best")$specificity
            test_class <- predict.rfsrc(rf,SELECTDATA_test)$predicted[,1]
            Test_roc_data <- data.frame(y_test,test_class)
            test_result[counter] <- roc(Test_roc_data[,1], Test_roc_data[,2])$auc
            test_se[counter] <-coords(roc(Test_roc_data[,1], Test_roc_data[,2]),coords (roc(Train_roc_data[,1], Train_roc_data[,2]),"best")$threshold)$sensitivity
            test_sp[counter] <-coords(roc(Test_roc_data[,1], Test_roc_data[,2]),coords (roc(Train_roc_data[,1], Train_roc_data[,2]),"best")$threshold)$specificity
            
            counter <- counter + 1
            
          }
        }
      }
    }
  }
}

summary <- data.frame(currsplit,currboruta,group, currtree,currsize,currdepth,train_result,test_result,train_se,test_se,train_sp,test_sp)


bestparam <- summary %>% filter(train_result > 0.9)%>%filter(test_result > 0.9)

bestparam[dim(bestparam)[1],] -> bestparam
bestparam

#6. Applying Boruta + RandomForest####

set.seed(bestparam$currsplit)

train_split_id<-sample(total_deve_id[,1], sized1, replace = FALSE, prob = NULL)
train_full<-data_Develop_ori_full[which(data_Develop_ori_full$id_d %in% train_split_id),]
train_feature<-train_full[5:length(train_full)]

y_train <- as.numeric(train_full$Labels)
y_train <- train_full$Labels

train_feature_full<-cbind(y_train,train_feature)
test_full<-data_Develop_ori_full[-which(data_Develop_ori_full$id_d %in% train_split_id),]
test_feature<-test_full[5:length(test_full)]
y_test <- as.numeric(test_full$Labels)
y_test <- test_full$Labels

test_feature_full<-cbind(y_test,test_feature)
y_validation <-as.numeric(data_validation_ori_full$Labels_v)
y_validation <-data_validation_ori_full$Labels_v

set.seed(bestparam$currboruta)

boruta_train <- Boruta(y_train~., data = train_feature, doTrace = 0,maxRuns=15)
boruta_df <- attStats(boruta_train)
boruta_features<-boruta_df[order(abs(boruta_df$meanImp),decreasing=TRUE),]

Selected_Features_train <-train_feature %>%dplyr:::select(rownames(boruta_features))
Selected_Features_test <-test_feature %>%dplyr:::select(rownames(boruta_features))
Selected_Features_v <- Features_validation%>% dplyr:::select(rownames(boruta_features))

SELECTDATA_validation <- cbind(y_validation, Selected_Features_v)
SELECTDATA_train <- cbind(y_train, Selected_Features_train)
SELECTDATA_test <- cbind(y_test, Selected_Features_test)

set.seed(bestparam$group)

rf <- rfsrc(y_train~ ., data =SELECTDATA_train, ntree=bestparam$currtree, nodesize=bestparam$currsize,nodedepth = bestparam$currdepth,importance  = TRUE)
rf$predicted
rf$class

library(ggRandomForests)
gg0 <- gg_vimp(rf)
gg<-gg0[gg0$vimp!=0,]
gg<-gg[gg$set=="all",]
plot(gg)




train_class <- predict.rfsrc(rf,SELECTDATA_train)$predicted[,1]
test_class <- predict.rfsrc(rf,SELECTDATA_test)$predicted[,1]
validation_class <- predict.rfsrc(rf,SELECTDATA_validation)$predicted[,1]

vali_full<-cbind("id_d"=validation$id,"Labels"=validation$Label,Feature_full_v)
data_train<-cbind(train_full,train_class)
data_test<-cbind(test_full,test_class)
data_vali<-cbind(vali_full,validation_class)
data_train <- transform(data_train, group = 0) 
data_test <- transform(data_test, group = 1) 
data_vali <- transform(data_vali, group = 2)  
write.csv(data_train,"Train data.csv")
write.csv(data_test,"Test_dataf.csv")
write.csv(data_vali,"validation_data.csv")

#Train roc data
library("pROC")
Train_roc_data <- data.frame(y_train,train_class)
names(Train_roc_data)<- c("y_train", "rfsrc_pred")
rocobj_train <-roc(Train_roc_data$y_train, Train_roc_data$rfsrc_pred)

print("Training Dataset")
rocobj_train$auc
ci.auc(rocobj_train)##CI
roc_summary_train <-coords (rocobj_train,
                            "best"
                            , ret=c ("threshold",  "specificity",  "sensitivity",  "accuracy","precision","recall","ppv","npv"),transpose = TRUE)
roc_summary_train
roc_summary_train1 <-coords (rocobj_train,
                             "best")$specificity  
# Test roc data 
library("pROC")
Test_roc_data <- data.frame(y_test,test_class)
names(Test_roc_data)<- c("y_test", "rfsrc_pred_test")
rocobj_test <- roc(Test_roc_data[,1], Test_roc_data[,2])
print("Testing Dataset")
rocobj_test$auc
ci.auc (rocobj_test)##CI
roc_summary_test<-coords (rocobj_test,roc_summary_train[1], ret=c ("threshold",  "specificity",  "sensitivity",  "accuracy","precision","recall","ppv","npv"),transpose = TRUE)
roc.test (rocobj_train,  rocobj_test, method= "delong")                                                                        
roc_summary_test 

# Validation roc data 
library("pROC")
Validation_roc_data <- data.frame(y_validation,validation_class)
names(Validation_roc_data)<- c("y_validation", "rfsrc_pred_validation")
rocobj_validation <- roc(Validation_roc_data$y_validation, Validation_roc_data$rfsrc_pred_validation)
print("Validation Dataset")
rocobj_validation$thresholds
roc_summary_vali<-coords (rocobj_validation,roc_summary_train[1], ret=c ("threshold",  "specificity",  "sensitivity",  "accuracy","precision","recall","ppv","npv"),transpose = TRUE)
roc.test (rocobj_train,  rocobj_validation, method= "delong")                                                                        

