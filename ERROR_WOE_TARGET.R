#clear the board 
rm(list=ls())
#Ctrl+L 
#read data
w=read.csv("C:/Users/hp/Desktop/R_DATA/bcard_woe.csv");
w$TARGET=as.factor(w$TARGET)
#get the valiable data
w=w[,c(3,4:29)]
rows <- nrow (w)   
indexes <- sample (rows,20000,replace =TRUE)   
w<- w[indexes, ] 
#import the package,use random forest to fill the missing data
library(missForest);
w=missForest(w)$ximp
w[,1]=factor(w[,1])
#set up a function
Fold=function(Z=5,w,D,seed=7777){ 
  n=nrow(w);d=1:n;dd=list() 
  e=levels(w[,D]);T=length(e)
  set.seed(seed)
  for(i in 1:T){
    d0=d[w[,D]==e[i]];j=length(d0)
    ZT=rep(1:Z,ceiling(j/Z))[1:j]
    id=cbind(sample(ZT,length(ZT)),d0);dd[[i]]=id}
  mm=list()
  for(i in 1:Z){u=NULL;
  for(j in 1:T)u=c(u,dd[[j]][dd[[j]][,1]==i,2])
  mm[[i]]=u} 
  return(mm)}
library(MASS);
library(rpart.plot);
library(adabag)
library(randomForest);
library(e1071);
library(kernlab)
library(mda);
library(kknn);
library(nnet);
library(ipred)

D=1;Z=10;n=nrow(w)
ff=paste(names(w)[D],"~.");ff=as.formula(ff)
mm=Fold(Z,w,D,8888)

#set up a matrix
ERROR=matrix(1,Z,11);
J=1
for(i in 1:Z){
  m=mm[[i]]
  a=rpart(ff,w[-m,]) #
  ERROR[i,J]=sum(w[m,D]!=predict(a,w[m,],type="class"))/length(m)
}
J=J+1;
set.seed(1010);
for(i in 1:Z){
  m=mm[[i]]
  a=boosting(ff,w[-m,])#boosting
  ERROR[i,J]=sum(w[m,D]!=predict(a,w[m,])$class)/length(m)}
J=J+1;
set.seed(1010);
for(i in 1:Z){
  m=mm[[i]]
  a=bagging(ff,data =w[-m,])#bagging
  ERROR[i,J]=sum(w[m,D]!=predict(a,w[m,]))/length(m)
}
J=J+1;
set.seed(1010);
for(i in 1:Z){
  m=mm[[i]]
  a=randomForest(ff,data=w[-m,])#random forest
  ERROR[i,J]=sum(w[m,D]!=predict(a,w[m,]))/length(m)
}
J=J+1;
set.seed(1010);
for(i in 1:Z){
  m=mm[[i]]
  a=svm(ff, data = w[-m,],kernal="sigmoid")#֧svm
  ERROR[i,J]=sum(w[m,10]!=predict(a,w[m,]))/length(m)
}
J=J+1;
for(i in 1:Z){
  m=mm[[i]]
  a=ksvm(ff,w[-m,])#kernel svm
  ERROR[i,J]=sum(w[m,D]!=predict(a,w[m,]))/length(m)
}
J=J+1;
for(i in 1:Z){
  m=mm[[i]]
  a=kknn(ff, train= w[-m,],test=w[m,])#K-means
  ERROR[i,J]=sum(w[m,D]!=a$fit)/length(m)
}
J=J+1;for(i in 1:Z){
  m=mm[[i]]
  a=lda(ff,w[-m,])#IDA
  ERROR[i,J]=sum(w[m,D]!=predict(a,w[m,])$class)/length(m)
}
J=J+1;
for(i in 1:Z){
  m=mm[[i]]
  a=mda(ff,w[-m,])#Mix Discriminant Analysis
  ERROR[i,J]=sum(w[m,D]!=predict(a,w[m,]))/length(m)
}
J=J+1;
for(i in 1:Z){
  m=mm[[i]]
  a=fda(ff,w[-m,])#Flexible Discriminant Analysis
  ERROR[i,J]=sum(w[m,D]!=predict(a,w[m,]))/length(m)
}
J=J+1;
for(i in 1:Z){
  m=mm[[i]];
  mc=setdiff(1:n,m)#DIFF
  a=nnet(ff, data = w, subset = mc, size = 5,rang = 0.1,decay = 5e-4, maxit = 200)
  ERROR[i,J]=sum(w[m,D]!=predict(a, w[m,], type = "class"))/length(m)
}
options(digits=3)
ERROR=data.frame(ERROR)
names(ERROR)=c("tree","boost","bag","Rforest",
               "svm-1","svm-2","kknn","lda","mda","fda","nnet")
(ME=apply(ERROR,2,mean));ERROR
ERROR[11,]=ME
#Put the matrix into excel
write.csv(ERROR,file="C:/Users/hp/Desktop/ERROR_WOE_TARGET_20000.csv")