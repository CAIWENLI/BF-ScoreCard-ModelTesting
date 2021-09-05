##清空变量+控制面板
rm(list=ls())
##Ctrl+l

##读取数据+看数据整体情况+看数据数据结构
bcard_org<-read.csv("C:/Users/hp/Desktop/R_DATA/bcard_woe.csv")
bcard_org$TARGET = as.factor(bcard_org$TARGET)
#检验空值占比
sapply(bcard_org,class)
sapply(bcard_org,function(x) sum(is.na(x))/length(x))

#随机抽样
bcard=bcard_org[,c(3,4:29)]
rows <- nrow (bcard)   
indexes <- sample (rows,20000, replace =TRUE)   
bcard_random<- bcard[indexes, ] 

#空值填补
library(missForest);
bcard_random=missForest(bcard_random)$ximp
sapply(bcard_random,function(x) sum(is.na(x))/length(x))
##生成训练、验证集 7:3
set.seed(18)
train_idx = sample(1:nrow(bcard_random),length(bcard_random$TARGET)*0.7)
train = bcard_random[train_idx,]
test = bcard_random[-train_idx,]


#神经网络---需要进行数据数值化处理，并要求数据进行归一化（权值会受到影响）
#建立数据结构--不同的函数对数据结构有严格的要求
train_y = data.frame(train)[,1]
test_y = na.omit(data.frame(test))[,1]
train_data = data.matrix(train)[,-1]
test_data = na.omit(data.matrix(test))[,-1]
#归一化处理--仅神经网络需要
train_s=scale(train_data [,1:25], center = TRUE, scale = TRUE)
test_s=scale(test_data [,1:25], center = TRUE, scale = TRUE)
#使用nnet函数
library(caret)
require(nnet)
bcard_nnet=nnet(train_y~., data =train_s,size = 2, rang = 0.1,
                decay = 5e-4, maxit = 200)
bcard_nnet_ptrain=predict(bcard_nnet,train_s, type ="raw")
bcard_nnet_ptest=predict(bcard_nnet,test_s, type = "raw")
#绘制ROC曲线
library(ROCR)
pred_nnet<- prediction(bcard_nnet_ptrain,train_y)
perf_nnet<- performance(pred_nnet,"tpr","fpr")
pred_nnet_test <- prediction(bcard_nnet_ptest,test_y)
perf_nnet_test <- performance(pred_nnet_test,"tpr","fpr") 
plot(perf_nnet,col='red',main="ROC of Net")
plot(perf_nnet_test, col='green',lty=2,add=TRUE);
abline(0,1,lty=2,col='blue')
nnet_m_auc<-round(as.numeric(performance(pred_nnet,'auc')@y.values),3)
nnet_m_str<-paste("Mode_train-AUC:",nnet_m_auc,sep="")
nnet_m_auc_test<-round(as.numeric(performance(pred_nnet_test,'auc')@y.values),3)
nnet_m_str_test<-paste("Mode_test-AUC:",nnet_m_auc_test,sep="")
legend("bottomright",0.2,c(nnet_m_str,nnet_m_str_test),2:4)

train_y = data.frame(train)[,1]
test_y = na.omit(data.frame(test))[,1]
train_data = data.frame(train)[,-1]
test_data = na.omit(data.frame(test))[,-1]
##随机森林(分类树)
library(randomForest)
train_data = train
test_data = test
rf<-randomForest(train_data$TARGET~.,data=train_data[,-1],xtest = test_data[,-1],ytest = test_data$TARGET,importance=TRUE,ntree=200,mtry = 6)
#Var Importance
rf$importance
varImpPlot(rf)
print(importance(rf))
# Make Probability Prediction for both Train and Test dataset
train_data$rf_p_prob = rf$votes[,2]
test_data$rf_p_prob = rf$test$votes[,2]
head(rf$votes)
#绘制ROC曲线图
library(ROCR)
pred_rf <- prediction(train_data$rf_p_prob,train_data$TARGET)
perf_rf <- performance(pred_rf,"tpr","fpr")
pred_rf_test <- prediction(test_data$rf_p_prob, test_data$TARGET)
perf_rf_test <- performance(pred_rf_test,"tpr","fpr")
plot(perf_rf,col='red',main="ROC of Random Forest")
plot(perf_rf_test, col='green',lty=2,add=TRUE);
abline(0,1,lty=2,col='blue')
rf_m_auc<-round(as.numeric(performance(pred_rf,'auc')@y.values),3)
rf_m_str<-paste("Mode_train-AUC:",rf_m_auc,sep="")
rf_m_auc_test<-round(as.numeric(performance(pred_rf_test,'auc')@y.values),3)
rf_m_str_test<-paste("Mode_test-AUC:",rf_m_auc_test,sep="")
legend("bottomright",0.2,c(rf_m_str,rf_m_str_test),2:4)

##Bagging
#建立数据结构--不同的函数对数据结构有严格的要求
train_y = data.matrix(train)[,1]
test_y = na.omit(data.matrix(test))[,1]
test_data = na.omit(data.frame(test))[,-1]
train_data = data.frame(train)[,-1]
library("ipred")
set.seed(500)
bag<-bagging(train_y~.,train_data,coob=TRUE)
bag_ptrain=predict(bag,train_data,type="prob")
bag_ptest=predict(bag,test_data,type="prob")
library(ROCR)
pred_bag<- prediction(bag_ptrain,train_y)
perf_bag<- performance(pred_bag,"tpr","fpr")
pred_bag_test <- prediction(bag_ptest,test_y)
perf_bag_test <- performance(pred_bag_test,"tpr","fpr")
plot(perf_bag,col='red',main="ROC of Bagging")
plot(perf_bag_test, col='green',lty=2,add=TRUE);
abline(0,1,lty=2,col='blue')
bag_m_auc<-round(as.numeric(performance(pred_bag,'auc')@y.values),3)
bag_m_str<-paste("Mode_train-AUC:",bag_m_auc,sep="")
bag_m_auc_test<-round(as.numeric(performance(pred_bag_test,'auc')@y.values),3)
bag_m_str_test<-paste("Mode_test-AUC:",bag_m_auc_test,sep="")
legend("bottomright",0.2,c(bag_m_str,bag_m_str_test),2:4)

#建立支持向量机模型
train_y = data.matrix(train)[,1]
test_y = na.omit(data.matrix(test))[,1]
train_data = data.matrix(train)[,-1]
test_data = na.omit(data.matrix(test))[,-1]

#使用svm函数
library(e1071)
set.seed(10)
bcard_svm=svm(train_y~.,kernel="sigmoid",
              data=train_data,probability=TRUE,cost=0.1,gamma=0.4,cross=10)
bcard_svm_ptrain=predict(bcard_svm,train_data,probability=F)
bcard_svm_ptest=predict(bcard_svm,test_data,probability=F)
library(ROCR)
pred_svm<- prediction(bcard_svm_ptrain,train_y)
perf_svm<- performance(pred_svm,"tpr","fpr")
pred_svm_test <- prediction(bcard_svm_ptest,test_y )
perf_svm_test <- performance(pred_svm_test,"tpr","fpr")
plot(perf_svm,col='red',main="ROC of SVM")
plot(perf_svm_test, col='green',lty=2,add=TRUE);
abline(0,1,lty=2,col='blue')
svm_m_auc<-round(as.numeric(performance(pred_svm,'auc')@y.values),3)
svm_m_str<-paste("Mode_train-AUC:",svm_m_auc,sep="")
svm_m_auc_test<-round(as.numeric(performance(pred_svm_test,'auc')@y.values),3)
svm_m_str_test<-paste("Mode_test-AUC:",svm_m_auc_test,sep="")
legend("bottomright",0.2,c(svm_m_str,svm_m_str_test),2:4)

#使用ksvm函数
library(kernlab)
#类型一
bcard_ksvm_r= ksvm(train_y~.,data=train_data,kernel="rbfdot",
                   kpar=list(sigma=0.05),C=5,cross=3)
bcard_ksvm_r_ptrain=predict(bcard_ksvm_r,train_data, type="decision")
bcard_ksvm_r_ptest=predict(bcard_ksvm_r,test_data, type="decision")
library(ROCR)
pred_ksvm<- prediction(bcard_ksvm_r_ptrain,train_y)
perf_ksvm<- performance(pred_ksvm,"tpr","fpr")
pred_ksvm_test <- prediction(bcard_ksvm_r_ptest,test_y )
perf_ksvm_test <- performance(pred_ksvm_test,"tpr","fpr")
plot(perf_ksvm,col='red',main="ROC of KSVM_1")
plot(perf_ksvm_test, col='green',lty=2,add=TRUE);
abline(0,1,lty=2,col='blue')
ksvm_m_auc<-round(as.numeric(performance(pred_ksvm,'auc')@y.values),3)
ksvm_m_str<-paste("Mode_train-AUC:",ksvm_m_auc,sep="")
ksvm_m_auc_test<-round(as.numeric(performance(pred_ksvm_test,'auc')@y.values),3)
ksvm_m_str_test<-paste("Mode_test-AUC:",ksvm_m_auc_test,sep="")
legend("bottomright",0.2,c(ksvm_m_str,ksvm_m_str_test),2:4)

#类型二
bcard_ksvm_c=ksvm(train_y~.,data=train_data,type="C-bsvc",
                  kernel="rbfdot",C=10,prob.model=TRUE)
bcard_ksvm_c_ptrain=predict(bcard_ksvm_c,train_data, type="decision")
bcard_ksvm_c_ptest=predict(bcard_ksvm_c,test_data, type="decision")
library(ROCR)
pred_ksvm_c<- prediction(bcard_ksvm_c_ptrain,train_y)
perf_ksvm_c<- performance(pred_ksvm_c,"tpr","fpr")
pred_ksvm_test_c <- prediction(bcard_ksvm_c_ptest,test_y )
perf_ksvm_test_c <- performance(pred_ksvm_test_c,"tpr","fpr")
plot(perf_ksvm_c,col='red',main="ROC of KSVM_2")
plot(perf_ksvm_test_c, col='green',lty=2,add=TRUE);
abline(0,1,lty=2,col='blue')
ksvm_m_auc_c<-round(as.numeric(performance(pred_ksvm_c,'auc')@y.values),3)
ksvm_m_str_c<-paste("Mode_train-AUC:",ksvm_m_auc_c,sep="")
ksvm_m_auc_test_c<-round(as.numeric(performance(pred_ksvm_test_c,'auc')@y.values),3)
ksvm_m_str_test_c<-paste("Mode_test-AUC:",ksvm_m_auc_test_c,sep="")
legend("bottomright",0.2,c(ksvm_m_str_c,ksvm_m_str_test_c),2:4)

##数据结构处理
train_y = data.frame(train)[,1]
test_y = na.omit(data.frame(test))[,1]
train_data = data.frame(train)[,-1]
test_data = na.omit(data.frame(test))[,-1]
##逻辑回归
glm=glm(train_y~., data =train_data,family=binomial(link=logit))
glm_ptrain=predict(glm,train_data, type ="response")
glm_ptest=predict(glm,test_data, type = "response")
#绘制ROC曲线
library(ROCR)
pred_glm<- prediction(glm_ptrain,train_y)
perf_glm<- performance(pred_glm,"tpr","fpr")
pred_glm_test <- prediction(glm_ptest,test_y)
perf_glm_test <- performance(pred_glm_test ,"tpr","fpr") 
plot(perf_glm,col='red',main="ROC of Glm")
plot(perf_glm_test, col='green',lty=2,add=TRUE);
abline(0,1,lty=2,col='blue')
glm_m_auc<-round(as.numeric(performance(pred_glm,'auc')@y.values),3)
glm_m_str<-paste("Mode_train-AUC:",glm_m_auc,sep="")
glm_m_auc_test<-round(as.numeric(performance(pred_glm_test,'auc')@y.values),3)
glm_m_str_test<-paste("Mode_test-AUC:",glm_m_auc_test,sep="")
legend("bottomright",0.2,c(glm_m_str,glm_m_str_test),2:4)