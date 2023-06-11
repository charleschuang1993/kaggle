library(tidyverse)
library(randomForest)
#library(ggplot2)
library(pheatmap)
library(pROC)
#library(MLmetrics)
library(xgboost)
library(ggcorrplot)
library(e1071)
"-------------------------------------------------------------------"
library(MLtool)
install.packages("caret")
"20230607 提交後 分數為0.58，因此改用xgboost取代random forest看看"
"遺失值填補利用資料集本身的其他數據來預測"
"==================================================================="
"雖然屬於二分類問題，但實際隱含了三種病症狀況在裡面"
"==================================================================="
# 將類別型資料轉成出現頻率，藉此數字化

na2median <- function(data, col_with_miss){
   tryCatch({
            sapply(col_with_miss, function(miss_col){
            median_ <- median(na.omit(data[,miss_col]))
            data[,miss_col][is.na(data[,miss_col])] <<- ifelse(is.na(median_),0,median_)
            })
            return(data)
        },
        error=function(e) {
            message('An Error Occurred')
            print(e)
        })
}

transformer <- function(data){
  tryCatch({
    count_table_ <- table(data)
    freq_table_ <- count_table_/sum(count_table_ )
    freq_ <- freq_table_[match(data, names(freq_table_))]
    return(freq_)
  },error = function(e){
       message('An Error Occurred')
            print(e)
  })
}

transformer_v2 <- function(data, target){
  tryCatch({
    data_factor <- as.factor(data)
    freq_table <- table(data_factor[target==1])/sum(target==1)
    freq_  <- freq_table[match(data_factor , names(freq_table))]
    return(freq_)
  }, error = function(e){
     message("error")
     print(e)
  })
}

data_for_validation <- function(train_size, data){
  tryCatch({
    sample_index <- sample(1:nrow(data), size= nrow(data) * train_size)
    train_set_ <- data[sample_index ,]
    validat_set_ <- data[-sample_index ,]
    return(list("trainset" = train_set_, "validatset" = validat_set_ , "index" = sample_index  ))
  },error=function(e){
      message("error")
      print(e)
  })
}


balan_logloss <- function(pred_value, true_value){
  tryCatch({
      pred_value<- sapply(pred_value,function(x) max(min(x, 1-10^-15),10^-15)) #avoid Inf value
      balan_logloss <- (mean(-log(pred_value)*as.numeric(is.element(true_value, 1)))+
      mean(-log(1-pred_value)*as.numeric(!is.element(true_value, 1))))/2
      return(balan_logloss )
  },error=function(e){
      message("error")
      print(e)
  })
 }


string2freq <- function(data, target, mode ){
  tryCatch({
    if(mode==1){
        renew_variable <- sapply(which(sapply(data, is.character)),function(x) data[,x] <<- as.numeric(transformer(data[,x])))
    }else if(mode==2){
        renew_variable <- sapply(which(sapply(data, is.character)),function(x) data[,x] <<- as.numeric(transformer_v2(data[,x],data[,target])))
    }
  return(data)

  },error=function(e){
    message("error")
    print(e)
  }

  )
 }

greek2freq <- function(data, greeks, mode){
  tryCatch({
      greeks_matched <- greeks[match(rownames(data), greeks[,"Id"]),]
      if(mode==1){
        freq_greek <- sapply(c("Alpha", "Beta", "Gamma", "Delta"), function(x) transformer(greeks_matched[,x]))
      }else if(mode==2){
        freq_greek <- sapply(c("Alpha", "Beta", "Gamma", "Delta"), function(x) transformer_v2(greeks_matched[,x], data[,"Class"]))
      }
      freq_greek <- as.data.frame(freq_greek)
      rownames(freq_greek) <- greeks_matched[,"Id"]
      return(freq_greek )

  },error=function(e){
    message("error")
    print(e)
  })
}

add_new_feature <- function(data, variables, target, test_data){
  tryCatch({
      formula_ <- as.formula(paste0(target, " ~ ", paste0(variables, collapse="+")))
      model_for_new_feature <- randomForest(formula_, data)
      pred <- predict(model_for_new_feature, test_data) 
      return(pred)
  },error=function(e){
    message("error")
    print(e)
  })
}

seperated_data_with_miss_value <- function(data, col_with_miss, target_col){
  tryCatch({
      data_clean <- na.omit(data) #擷取沒有NA值的部分
      data_without_miss <- data_clean[,!is.element(colnames(data_clean ),c(setdiff(col_with_miss,target_col),"Class"))] #用剩餘的來預測
      target_with_miss <- data[is.na(data[,target_col]), -which(colnames(data)=="Class")]
      return(list("data_without_miss"= data_without_miss, "target_with_miss"= target_with_miss))
  },error=function(e){
    message("error")
    print(e)
  })

}

predict_missing_value <- function(target_col, col_with_miss, train_data, test_data=NULL){
  tryCatch({
    print(paste0("Current: ",target_col))
    for_missing_value_pred <- seperated_data_with_miss_value(train_data, col_with_miss, target_col)
    data_without_miss <- for_missing_value_pred$"data_without_miss"
    target_with_miss <- for_missing_value_pred$"target_with_miss"
    model_for_target <- randomForest(as.formula(paste0(target_col,"~.")), data_without_miss)
    print("Modeling Finished")
    for_predicted <- if(length(test_data)==0){
                                target_with_miss
                            }else{
                                test_data #如果為test data,則無需切割
                            }
    for_predicted_with_median <- na2median(for_predicted, col_with_miss) #若為na 自動輸出0
    pred_value <- predict(model_for_target, for_predicted_with_median)
    print("Predicting finished")
    #names <- rownames(target_with_miss)
    if(length(test_data)==0){
      train_data[is.na(train_data[,target_col]), target_col] <- pred_value  #回頭找最初有na的樣本替換新值
      return(train_data)
    }else{
      test_data[,target_col] <- pred_value
      return(test_data)
    }
  },error=function(e){
    message("error")
    print(e)
  })
}


miss_value_recovery<- function(train_data = train_set, test_data=NULL){
  tryCatch({
      if(length(test_data)==0){
        print("Train dataset recovery")
        na_col <- which(apply(train_data, 2, function(x) sum(is.na(x)))>0)
        col_with_miss <- names(na_col) #other columns with missing values
        print(paste0("Col with miss: ",col_with_miss))
        for_notprint <- sapply(col_with_miss, function(x)  train_data <<- predict_missing_value(x, col_with_miss, train_data) )
        return(train_data)
      }else{
        print("Test dataset recovery")
        na_col <- which(apply(test_data, 2, function(x) sum(is.na(x)))>0)
        col_with_miss <- names(na_col) #other columns with missing values
        print(paste0("Col with miss: ",col_with_miss))

        for_notprint <- sapply(col_with_miss, function(x)  test_data <<- predict_missing_value(x, col_with_miss, train_data, test_data) )
        return(test_data)
      }
  },error=function(e){
    message("error")
    print(e)
  })
}

scaler <-function(x){
  if(is.numeric(x)&max(x)!=0){
    max_ <- max(x) 
    min_ <- min(x)
    return((x-min_)/max_)
  }else{
    return(x)
  }
 }

data_preprepare <- function(train_set, test_set, y_col_name = "Class", sample_id_col="Id"){
  variables <- setdiff(intersect(colnames(train_set),colnames(test_set)),sample_id_col)
  #' 先將test set的ID獨立提取出來免得到最後遺失
  test_set_id <- test_set[, sample_id_col]
  # remain the ID of training set
  row.names(train_set) <- train_set[, sample_id_col] 
  #重新提取與test set重複的欄位
  train_set <- train_set[,c(variables, y_col_name)]
  #重新提取與training set 重複的欄位
  test_set <- test_set[ , variables]
  # 先將測試資料集可能無法運算的值都先轉NA，後面一併處理
  test_set[sapply(test_set,function(x) is.infinite(x)|is.nan(x)|is.null(x)|is.na(x))] <- NA
  return(list("train_set"= train_set, "test_set" = test_set, "variables" = variables, "test_set_id" = test_set_id))
 }

"=========================================================================="
"data prepare"
"=========================================================================="

"尚待建立利用所有資料預測Gamma之模型，用來讓test資料預測完作為變數之一，再來預測Class"

train_set_ <- read.csv("D:/7. Kaggle/ICR - Identifying Age-Related Conditions/train.csv")
test_set_ <- read.csv("D:/7. Kaggle/ICR - Identifying Age-Related Conditions/test.csv")
greeks <- read.csv("D:/7. Kaggle/ICR - Identifying Age-Related Conditions/greeks.csv")
submission <- read.csv("D:/7. Kaggle/ICR - Identifying Age-Related Conditions/sample_submission.csv")
#' 先將trainin_set和test set共有的欄位提取出來，免得發生模型變數不匹配等問題，並且排除ID


data_preprepare_ <- data_preprepare(train_set_,test_set_)
train_set <- data_preprepare_$train_set
test_set <- data_preprepare_$test_set
variables <- data_preprepare_$variables
test_set_id <- data_preprepare_$test_set_id

greek_param <- c("greek_alpha", "greek_gamma","greek_beta","greek_delta")


"=============================================================================="
"Data Preprocessing"
"=============================================================================="
# categorial type variables need transfering into numerical freqency
#data_with_repair$Class <- as.factor(data_with_repair$Class)
train_set$Class <- as.factor(train_set$Class )
data_for_CV <- data_for_validation(0.6, train_set)

train_set <- data_for_CV$trainset 
test_set <- data_for_CV$validatset 

train_set_v2 <- string2freq(train_set, "Class", mode=1)
test_set_v2 <- string2freq(test_set, "Class", mode=1)
 
#' test data set如果有欄位資料為字串則轉成freq

freq_greek_train <- greek2freq(train_set_v2, greeks, mode=2)
#' 有要用greek來train model才有用
colnames(freq_greek_train) <- c("greek_alpha", "greek_beta","greek_gamma","greek_delta")


"=============================================================================="
"Using greek data to predict missing value"
"=============================================================================="
#要設計一個可以代入test data


"=============================================================================="
"Using whole data to predict greek gamma classification"
"=============================================================================="
"暫時先別轉factor，觀察是否保留numeric type建立回歸模型有助於預測率"

#char_col <- sapply(1:ncol(train_set),  function(x) is.character(train_set[,x]))
#data.frame(sapply(train_set, class))

"2023.05.31 將greeks參數改成每種類別有多少比例的positive，再加上維持numeric type來訓練RF regression模型後，
開始產生預期的效果，也就是balanced log loss 開始下降，從0.12之間往0.10-0.11之間移動。如果改成factor type則又會回升，
或者把greeks的類別改成單純的各自出現頻率，也會讓balanced log loss 回升。這是有趣的觀察與測試，或許可以發表論文，並探討原因
"

#data_with_greek$greek_alpha  <- add_new_feature(data_with_greek, variables, "Alpha", data_with_greek  )
#data_with_greek$greek_beta  <- add_new_feature(data_with_greek, variables, "Beta", data_with_greek  )
#data_with_greek$greek_gamma  <- add_new_feature(data_with_greek, variables, "Gamma", data_with_greek  )
#data_with_greek$greek_delta  <- add_new_feature(data_with_greek, variables, "Delta", data_with_greek  )

test_set_v2_with_repair$greek_alpha  <- add_new_feature(data_with_greek, variables, "greek_alpha" , test_set_v2_with_repair)
test_set_v2_with_repair$greek_beta    <- add_new_feature(data_with_greek, variables, "greek_beta" , test_set_v2_with_repair)
test_set_v2_with_repair$greek_gamma  <- add_new_feature(data_with_greek, variables, "greek_gamma", test_set_v2_with_repair)
test_set_v2_with_repair$greek_delta  <- add_new_feature(data_with_greek, variables, "greek_delta", test_set_v2_with_repair)


"=============================================================================="
"Exploratory Data Analysis (EDA)"
"=============================================================================="
#train_with_greek$Class <- as.numeric(as.character(train_with_greek$Class))
#sapply(c(variables,greek_param), function(x) train_with_greek[,x] <<- as.numeric(train_with_greek[,x]))

corr_mat <- round(cor(train_with_greek , method="spearman"), 1)
p_mat <- cor_pmat(train_with_greek , method="spearman" ) #cor pvalue
pheatmap(corr_mat)

View(corr_mat)

#ggcorrplot(corr_mat )
#ggcorrplot(p_mat)
#train_set_wo_na <- train_set_wo_na[,-which(colnames(train_set_wo_na_char)=="Class")]
#target_col <- "BQ" #target column with missing value

#df_with_predicted_miss <- predict_missing_value(target_col, col_with_miss, data_for_cor )    


"==============================================================================="
"Building Model"
"==============================================================================="
"--------------------------------------------------------------------------------"
"Training model"
"--------------------------------------------------------------------------------"
# 經測試，Class轉成numeric後效果並不好
# Class 轉成 factor type 並且predict用 prob type 效果較好

data_with_greek$Class <- as.factor(data_with_greek$Class )

#data_with_greek[,variables] <- as.data.frame(sapply(data_with_greek[,variables], scaler, simplify = FALSE))
#test_set_v2_with_repair[,variables] <- as.data.frame(sapply(test_set_v2_with_repair[,variables] , scaler, simplify = FALSE))

zero <- table(data_with_greek$Class)[1]
one <- table(data_with_greek$Class)[2]
total <- zero + one
weight_for_0 <- (1 / zero) * (total )
weight_for_1 <- (1 / one) * (total )

greek_param <- c("greek_gamma","greek_alpha","greek_beta", "greek_delta")
#增加weight有幫助
model_RF <- randomForest(as.formula(paste0("Class~", paste0(c(variables, greek_param), collapse="+"))), data_with_greek)

#model_RF <- randomForest(as.formula(paste0("Class~", paste0(c(variables, greek_param), collapse="+"))), 
#                         data_with_greek, 
#                         classwt = c("1"=weight_for_1[[1]],"0"=weight_for_0[[1]]) )

"取前幾名"
impVars <- varImpPlot(model_RF,30)
model_RF <- randomForest(as.formula(paste0("Class~", paste0(c(names(impVars[order(impVars,decreasing=TRUE),][1:20])), collapse="+"))), 
                         data_with_greek)





pred_RF_train <- predict(model_RF , data_with_greek, "prob")
balan_logloss(as.numeric(pred_RF_train[,2]), as.numeric(as.character(data_with_greek$Class))) #pred_RF[,2]

pred_RF_test  <- predict(model_RF , test_set_v2_with_repair, type="prob")
balan_logloss(as.numeric(pred_RF_test[,2]), as.numeric(as.character(test_set_v2_with_repair$Class)))


roc_ <- roc(data_with_greek$Class, as.numeric(pred_RF_train[,2]));roc_ 


roc_plot <- ggroc(roc_ , colour = 'steelblue', size = 2) +
ggtitle(paste0('ROC Curve ', '(AUC = ', round(roc_$auc,4), ')')) +
theme_minimal()+geom_segment(aes(y = 0, yend = 1, x = 1, xend = 0), color = "grey50")
plot(sort(pred_RF_train))

#plot(model_RF)
#pred_RF_ <- as.data.frame(pred_RF)
#pred_RF_$Class <- data_with_repair$Class[data_for_validation_$index]
#tuneRF_ <-tuneRF(x = CV_train_with_repair[,-57], y=CV_train_with_repair[,57], ntreeTry=50, mtryStart = 1)
#mtry_ <- tuneRF_[which.min(tuneRF_[,"OOBError"]),"mtry"]



plot(pred_RF_test[,2], as.numeric(as.character(test_set_v2_with_repair$Class)))
pred_RF_test <- as.data.frame(pred_RF_test, row.names=NULL)
submission <- data.frame(cbind(as.character(test_set_id), pred_RF_test),row.names=NULL)
colnames(submission ) <- c("Id", "class_0", "class_1")
print(submission)

roc_ <- roc(CV_validation_with_repair$Class, as.numeric(pred_RF_validation[,2]))
roc_plot <- ggroc(roc_ , colour = 'steelblue', size = 2) +
ggtitle(paste0('ROC Curve ', '(AUC = ', round(roc_$auc,4), ')')) +
theme_minimal()+geom_segment(aes(y = 0, yend = 1, x = 1, xend = 0), color = "grey50")



plot(pred_RF_validation[,2], CV_validation_with_repair$Class)

"--------------------------------------------------------------------------------"
"Cross-Validation"
"--------------------------------------------------------------------------------"

#MLmetrics::LogLoss(as.numeric(pred_RF[,2]), as.numeric(as.character(data_with_repair$Class[-data_for_validation_$index])))/2
#https://rdrr.io/github/chuvanan/metrics/man/logloss.html


#all_roc <- NULL
success_ <- all_logloss
all_logloss_v1 <- all_logloss 
all_logloss_v2 <- all_logloss 
all_logloss_v3 <- all_logloss 
all_logloss_v4 <- all_logloss  #不同權重 svm 2 gr 1 xg 1
all_logloss_v5 <- all_logloss  #不同權重 svm 2 gr 3 xg 1
all_logloss_v6 <- all_logloss  #不同權重 svm 2 gr 3 xg 1

all_logloss <-NULL
明天試試看 adjusted 再不行就來練 python

for(i in 1:20){
    print("=====================")
    print(paste0("Run ",i))
    print("=====================")
    data_preprepare_ <- data_preprepare(train_set_,test_set_)
    train_set <- data_preprepare_$train_set
    test_set <- data_preprepare_$test_set
    variables <- data_preprepare_$variables
    test_set_id <- data_preprepare_$test_set_id

    greek_param <- c("greek_alpha", "greek_gamma","greek_beta","greek_delta")
    train_set$Class <- as.factor(train_set$Class )
    data_for_CV <- data_for_validation(0.6, train_set)
    train_set <- data_for_CV$trainset 
    test_set <- data_for_CV$validatset 

    train_set_v2 <- string2freq(train_set, "Class", mode=1)
    test_set_v2 <- string2freq(test_set, "Class", mode=1)
    freq_greek_train <- greek2freq(train_set_v2, greeks, mode=1) 
    #colnames(freq_greek_train) <- c("greek_alpha", "greek_beta","greek_gamma","greek_delta")
        
    train_set_v2_with_repair <- miss_value_recovery(train_set_v2)
    #得用test set自己來預測missing value，否則xgboost會不穩定
    test_set_v2_with_repair <- miss_value_recovery(test_set_v2, test_set_v2) 
    train_with_greek <- as.data.frame(cbind(train_set_v2_with_repair, freq_greek_train  )) 
    #xgboost 需要先預測train set的param 再來預測test set的，否則logloss會變大
    train_with_greek$greek_alpha  <- add_new_feature(train_with_greek, variables, "Alpha", train_with_greek)
    train_with_greek$greek_beta  <- add_new_feature(train_with_greek, variables, "Beta", train_with_greek)
    train_with_greek$greek_gamma  <- add_new_feature(train_with_greek, variables, "Gamma", train_with_greek )
    train_with_greek$greek_delta  <- add_new_feature(train_with_greek, variables, "Delta", train_with_greek  )

    test_set_v2_with_repair$greek_alpha  <- add_new_feature(train_with_greek, variables, "greek_alpha" , test_set_v2_with_repair )
    test_set_v2_with_repair$greek_beta    <- add_new_feature(train_with_greek, variables, "greek_beta" , test_set_v2_with_repair)
    test_set_v2_with_repair$greek_gamma  <- add_new_feature(train_with_greek, variables, "greek_gamma", test_set_v2_with_repair)
    test_set_v2_with_repair$greek_delta  <- add_new_feature(train_with_greek, variables, "greek_delta", test_set_v2_with_repair)
    #train_with_greek[,variables] <- as.data.frame(sapply(train_with_greek[,variables], scaler, simplify = FALSE))
    #CV_validation_with_repair[,variables] <- as.data.frame(sapply(CV_validation_with_repair[,variables] , scaler, simplify = FALSE))

    zero <- table(train_with_greek$Class)[1]
    one <- table(train_with_greek$Class)[2]
    total <- zero + one
    # Calculate the weight for each label.
    weight_for_0 <- (1 / zero) * (total / 2.0)
    weight_for_1 <- (1 / one) * (total / 2.0)
    greek_param <- c("greek_gamma","greek_alpha","greek_beta", "greek_delta")
    #增加weight有幫助

    model_RF <- randomForest(as.formula(paste0("Class~", paste0(c(variables,greek_param), collapse="+"))), train_with_greek, classwt = c("1"=weight_for_1[[1]],"0"=weight_for_0[[1]]))
    #model_RF <- randomForest(as.formula(paste0("Class~", paste0(c(variables, greek_param), collapse="+"))), train_with_greek)
    dtrain <- xgb.DMatrix(data = as.matrix(train_with_greek[,c(variables,greek_param)]), label = as.numeric(as.character(train_with_greek$Class)))
    dtest <- xgb.DMatrix(data = as.matrix(test_set_v2_with_repair[,c(variables,greek_param)]), label = as.numeric(as.character(test_set_v2_with_repair$Class)))
    #watchlist <- list(train=dtrain , test=dtest)
    bstSparse <- xgboost(data = as.matrix(train_with_greek[ ,c(variables,greek_param)]), label = as.numeric(as.character(train_with_greek$Class)), 
                         objective = "binary:logistic", 
                         verbose=0, 
                         nrounds=50, 
                         max.depth=50, 
                         eta=0.1, 
                         nthread = 4,
                         eval_metric = "logloss")

    #bstSparse <- xgb.train(data=dtrain , max.depth=100, eta=0.05, nthread = 4,nrounds=100, watchlist=watchlist,objective="binary:logistic",eval_metric = "logloss",verbose=0)

    train_with_greek_svm <- train_with_greek
    train_with_greek_svm$Class <- as.numeric(as.character(train_with_greek_svm$Class))
    model_SVM <- svm(Class~. ,train_with_greek_svm[,c(variables, greek_param,"Class")], probability=TRUE)
    
    pred_SVM_test <- predict(model_SVM, test_set_v2_with_repair) #,probability=TRUE
    pred_RF_test <- predict(model_RF , test_set_v2_with_repair, type="prob")
    pred_xgboost_test <- predict(bstSparse , dtest)

    #prob_1 <- ((as.numeric(scaler(pred_SVM_test))+pred_RF_test[,2])/2)
    #prob_0 <- 1-((as.numeric(scaler(pred_SVM_test))+pred_RF_test[,2])/2)
    #average <- ((as.numeric(scaler(pred_SVM_test))*2)+(pred_RF_test[,2]*3) + pred_xgboost_test)/6
    average <- ((adjusted(as.numeric(scaler(pred_SVM_test)))*3)+(adjusted(pred_RF_test[,2])*5)+adjusted(pred_xgboost_test*2))/10
    
    logloss <- balan_logloss(average , as.numeric(as.character(test_set_v2_with_repair$Class)))
    #logloss <- balan_logloss((attr(pred_SVM_test,"probabilities")[,2]+pred_RF_test[,2])/2, as.numeric(as.character(test_set_v2_with_repair$Class)))
    xgboost_LL <- balan_logloss(pred_xgboost_test, as.numeric(as.character(test_set_v2_with_repair$Class)))
    svm_LL <-balan_logloss(as.numeric(scaler(pred_SVM_test)), as.numeric(as.character(test_set_v2_with_repair$Class)))
    rf_LL <- balan_logloss(as.numeric(pred_RF_test[,2]), as.numeric(as.character(test_set_v2_with_repair$Class)))
    all_logloss  <- rbind(all_logloss, list("rf"=rf_LL,"svm"=svm_LL,"xg"=xgboost_LL ,"average" = logloss))
    print(all_logloss  ) 
  }


plot(ifelse(average>0.5,test(average),average-test_2(average)),average)
test <- function(x){ x*(2-x) }
test(0.7)
test_2 <- function(x){x*(1-x)}
test_2(0.5)

x <- seq(0.0,1,0.1)
plot(x, ifelse(x>0.6,test(x),x-test_2(x)))
plot(c(test_2(seq(0.1,0.5,0.1)),test(seq(0.6,1,0.1))),c(seq(0.1,0.5,0.1),seq(0.6,1,0.1)),cex=5)

adjusted <- function(x,index=0.5){ ifelse(x>index,test(x),x-test_2(x))}
adjusted(x)

entropy_ <- function(x){-(x*log2(x))-((1-x)*log2((1-x)))}

plot(c(seq(0.1,0.5,0.1)*(entropy_(seq(0.1,0.5,0.1))-0.5),test(seq(0.6,1,0.1))),c(seq(0.1,0.5,0.1),seq(0.6,1,0.1)),cex=5)
plot(entropy_(seq(0.0,1,0.1)),cex=5)


0.4*0.8
0.6*1.6666
1/0.7
View(as.data.frame(all_logloss ))

boxplot(all_logloss )
write.csv(all_logloss, "all_logloss.csv")
#tuneRF_ <-tuneRF(x = CV_train_with_repair[,-57], y = CV_train_with_repair[,57], ntreeTry=50, mtryStart = 1)
#mtry_ <- tuneRF_[which.min(tuneRF_[,"OOBError"]),"mtry"]
#model_RF <- randomForest(as.formula(paste0("Class~", paste0(c(variables,greek_param), collapse="+"))), train_with_greek, ntree = 50, mtry = mtry_)
    #tuneRF_ <-tuneRF(x = CV_train_with_repair[,-57], y = CV_train_with_repair[,57], ntreeTry=300, mtryStart = 1)
    #mtry_ <- tuneRF_[which.min(tuneRF_[,"OOBError"]),"mtry"]
    #model_RF <- randomForest(as.formula(paste0("Class~", paste0(c(variables,greek_param), collapse="+"))), train_with_greek, ntree = 300, mtry = mtry_)

"--------------------------------------------------------------------------------"
"Final test"
"--------------------------------------------------------------------------------"

# 缺一個function 把前面的流程和模型包起來，套用在test
#pred_RF_final_test <- as.data.frame(cbind(row.names(test_set), pred_RF_final_test ))
#colnames(pred_RF_final_test ) <- c("Id","class_0","class_1")
#data.frame(pred_RF_final_test, row.names=NULL)

greek_param <- c("greek_alpha","greek_beta", "greek_gamma","greek_delta")
CV_train_v2 <- string2freq(train_set, "Class",mode = 1)
CV_validation_v2 <- string2freq(test_set, "Class", mode = 1)

freq_greek_train <- greek2freq(CV_train_v2, greeks, mode = 1)
CV_train_with_repair <- miss_value_recovery(CV_train_v2)
CV_validation_with_repair <- miss_value_recovery(CV_train_with_repair, CV_validation_v2)
data_with_greek <- as.data.frame(cbind(CV_train_with_repair, freq_greek_train )) 
data_with_greek$greek_alpha  <- add_new_feature(data_with_greek, variables, "Alpha", data_with_greek )
data_with_greek$greek_beta  <- add_new_feature(data_with_greek, variables, "Beta", data_with_greek )
data_with_greek$greek_gamma  <- add_new_feature(data_with_greek, variables, "Gamma", data_with_greek )
data_with_greek$greek_delta  <- add_new_feature(data_with_greek, variables, "Delta", data_with_greek )

CV_validation_with_repair$greek_alpha  <- add_new_feature(data_with_greek, variables, "greek_alpha" , CV_validation_with_repair )
CV_validation_with_repair$greek_beta    <- add_new_feature(data_with_greek, variables, "greek_beta" , CV_validation_with_repair)
CV_validation_with_repair$greek_gamma  <- add_new_feature(data_with_greek, variables, "greek_gamma", CV_validation_with_repair)
CV_validation_with_repair$greek_delta  <- add_new_feature(data_with_greek, variables, "greek_delta", CV_validation_with_repair)

data_with_greek$Class <- as.factor(data_with_greek$Class )
CV_validation_with_repair <- na.omit(CV_validation_with_repair)


model_RF <- randomForest(as.formula(paste0("Class ~", paste0(c(variables, greek_param), collapse="+"))), data_with_greek )

pred_RF_train <- predict(model_RF , data_with_greek, type="prob")
#roc_ <- roc(data_with_greek$Class, as.numeric(pred_RF_train[,2]),quiet=TRUE)
print(balan_logloss(as.numeric(pred_RF_train[,2]), as.numeric(as.character(data_with_greek$Class))))
pred_RF_test <- predict(model_RF , CV_validation_with_repair, type="prob")
pred_RF_test <- as.data.frame(pred_RF_test, row.names=NULL)
submission <- data.frame(cbind(as.character(test_set_id), pred_RF_test),row.names=NULL)
colnames(submission ) <- c("Id", "class_0", "class_1")
print(submission)
write.csv(submission, file="D:/7. Kaggle/ICR - Identifying Age-Related Conditions/submission_v5.csv", row.names=FALSE)






#balan_logloss(as.numeric(submission[,"class_1"]), as.numeric(as.character(CV_validation_with_repair$Class)))


"--------------------------------------------------------------------------------"
"XGboost"
"--------------------------------------------------------------------------------"
#data(agaricus.train, package='xgboost')
#data(agaricus.test, package='xgboost')
#train <- agaricus.train
#test <- agaricus.test

 
dtrain <- xgb.DMatrix(data = as.matrix(data_with_greek[,c(variables,greek_param)]), label = as.numeric(as.character(data_with_greek$Class)))
dtest <- xgb.DMatrix(data = as.matrix(test_set_v2_with_repair[,c(variables,greek_param)]), label = as.numeric(as.character(test_set_v2_with_repair$Class)))

#bstSparse <- xgboost(data = as.matrix(data_with_greek[ ,c(variables,greek_param)]), label = as.numeric(as.character(data_with_greek$Class)), 
#              max.depth = 10, eta = 1, nthread = 4, nrounds = 500, objective = "binary:logistic")

watchlist <- list(train=dtrain , test=dtest)
bstSparse <- xgb.train(data=dtrain , max.depth=200, eta=0.01, nthread = 4,nrounds=100, watchlist=watchlist,objective="binary:logistic",eval_metric = "logloss")


#dtrain <- xgb.DMatrix(data = as.matrix(train_with_greek[,variables_v2]), label = as.numeric(as.character(train_with_greek$Class)))
#dtest <- xgb.DMatrix(data = as.matrix(CV_validation_with_repair[,variables_v2]), label = as.numeric(as.character(CV_validation_with_repair$Class)))

#bst <- xgb.train(data=dtrain , max.depth=6, eta=1, nthread = 4, nrounds=250, watchlist=watchlist, objective = "binary:logistic",eval_metric = "logloss")

pred_xgboost_train <- predict(bstSparse , dtrain)
balan_logloss(pred_xgboost_train, as.numeric(as.character(data_with_greek$Class)))

pred_xgboost_test <- predict(bstSparse , dtest)
balan_logloss(pred_xgboost_test, as.numeric(as.character(test_set_v2_with_repair$Class)))

roc(data_with_greek$Class, pred)
roc(test_set_v2_with_repair$Class, pred)

roc(CV_validation_with_repair$Class, pred)
plot(pred, as.numeric(as.character(train_with_greek$Class)))
plot(pred, as.numeric(as.character(CV_validation_with_repair$Class)))
"--------------------------------------------------------------------------------"
"SVM"
"--------------------------------------------------------------------------------"

data_with_greek_svm <- data_with_greek
data_with_greek_svm$Class <- as.numeric(as.character(data_with_greek_svm$Class))
model_SVM <- svm(Class~. ,data_with_greek[,c(variables,greek_param,"Class")],probability=TRUE)

pred_SVM_train <- predict(model_SVM, data_with_greek_svm,probability=TRUE)
pred_SVM_test <- predict(model_SVM, test_set_v2_with_repair,probability=TRUE)




balan_logloss(attr(pred_SVM_train,"probabilities")[,2], as.numeric(as.character(data_with_greek_svm$Class)))
balan_logloss(attr(pred_SVM_test,"probabilities")[,2], as.numeric(as.character(test_set_v2_with_repair$Class)))
balan_logloss((attr(pred_SVM_test,"probabilities")[,2]+pred_RF_test[,2]+pred_xgboost_test)/3, as.numeric(as.character(test_set_v2_with_repair$Class)))
"--------------------------------------------------------------------------------"
"Multiple Classifier"
"--------------------------------------------------------------------------------"

dtrain <- xgb.DMatrix(data = as.matrix(data_with_greek[,c(variables,greek_param)]), label = as.numeric(as.character(data_with_greek$Class)))
dtest <- xgb.DMatrix(data = as.matrix(test_set_v2_with_repair[,c(variables,greek_param)]), label = as.numeric(as.character(test_set_v2_with_repair$Class)))
watchlist <- list(train=dtrain , test=dtest)
bstSparse <- xgb.train(data=dtrain , max.depth=200, eta=0.01, nthread = 4,nrounds=100, watchlist=watchlist,objective="binary:logistic",eval_metric = "logloss")
pred_xgboost_test <- predict(bstSparse , dtest)
xgboost_LL <- balan_logloss(pred_xgboost_test, as.numeric(as.character(test_set_v2_with_repair$Class)))


"--------------------------------------------------------------------------------"
"Adjusted" (不建議用這個)
"--------------------------------------------------------------------------------"


ddd <- data.frame("svm"= attr(pred_SVM_train,"probabilities")[,2], "rf"=pred_RF_train[,2], "xg"=pred_xgboost_train, "Class" = as.factor(data_with_greek$Class))
ddd_test <- data.frame("svm"= attr(pred_SVM_test,"probabilities")[,2], "rf"=pred_RF_test[,2], "xg"=pred_xgboost_test, "Class" = as.factor(test_set_v2_with_repair$Class))



ddd_model <- svm(Class~., ddd , probability=TRUE)
pred_ddd_train <- predict(ddd_model, ddd ,probability=TRUE)
pred_ddd_test <- predict(ddd_model, ddd_test  ,probability=TRUE)
balan_logloss(attr(pred_ddd_train,"probabilities")[,2], as.numeric(as.character(ddd$Class)))
balan_logloss(attr(pred_ddd_test,"probabilities")[,2], as.numeric(as.character(ddd_test $Class)))


ddd_model <- randomForest(Class~., ddd )
pred_ddd_train <- predict(ddd_model, ddd ,type="prob")
pred_ddd_test <- predict(ddd_model, ddd_test  ,type="prob")
balan_logloss(pred_ddd_train[,2], as.numeric(as.character(ddd$Class)))
balan_logloss(pred_ddd_test[,2], as.numeric(as.character(ddd_test$Class)))


dtrain <- xgb.DMatrix(data = as.matrix(ddd[,c("rf","svm","xg")]), label = as.numeric(as.character(ddd$Class)))
dtest <- xgb.DMatrix(data = as.matrix(ddd_test[,c("rf","svm","xg")]), label = as.numeric(as.character(ddd_test$Class)))

bstSparse <- xgboost(data = as.matrix(ddd[,c("rf","svm","xg")]), label = as.numeric(as.character(ddd$Class)), 
              max.depth = 100, eta = 0.01, nthread = 4, nrounds = 500, objective = "binary:logistic")
pred_ddd_test <- predict(bstSparse , dtest)
balan_logloss(pred_ddd_test, as.numeric(as.character(ddd_test$Class)))



View(ddd )
plot(pred_ddd_test, as.numeric(as.character(test_set_v2_with_repair$Class)), col =  test_set_v2_with_repair$Class, cex=5)


plot(pred_RF_test[,2], pred_xgboost_test, col =  test_set_v2_with_repair$Class, cex=5)
plot(attr(pred_SVM_test,"probabilities")[,2], pred_xgboost_test, col =  test_set_v2_with_repair$Class, cex=5)
plot(attr(pred_SVM_test,"probabilities")[,2], pred_RF_test[,2], col =  test_set_v2_with_repair$Class, cex=5)

plot((attr(pred_SVM_test,"probabilities")[,2]+pred_RF_test[,2]+pred_xgboost_test)/3, pred_RF_test[,2], col =  test_set_v2_with_repair$Class, cex=5)
plot(sqrt(pred_RF_test[,2]*pred_xgboost_test), pred_xgboost_test, col =  test_set_v2_with_repair$Class, cex=5)
plot(as.numeric(pred_RF_test[,2]>0.5&pred_xgboost_test>0.9), pred_xgboost_test, col =  test_set_v2_with_repair$Class, cex=5)
table(test_set_v2_with_repair$Class)
ddd[]

View(ddd[order(ddd$rf,decreasing=TRUE),c("rf","Class")])
View(ddd[order(ddd$svm,decreasing=TRUE),c("svm","Class")])
View(ddd[order(ddd$xg,decreasing=TRUE),c("xg","Class")])

View(data.frame("t"=as.numeric(ddd$svm>0.03) + as.numeric(ddd$rf>0.3)+as.numeric(ddd$xg>0.81),"c"=ddd$Class))


ttt <- data.frame("t"=as.numeric(ddd$svm>0.03) + as.numeric(ddd$rf>0.3)+as.numeric(ddd$xg>0.81),"c"=ddd$Class)

View(data.frame(ifelse(ttt$t==3,1,0), ttt$c))

balan_logloss(ifelse(ttt$t==3,1,0), ttt$c)

"===================================================================="

library(caret)
 
# Load Dataset

control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7; set.seed(seed)
metric <- "Accuracy"
mtry <- sqrt(ncol(CV_train_v2))
tunegrid <- expand.grid(.mtry=mtry)
rf_default <- train(Class~., data = CV_train_with_repair, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_default)

first_ll <- collection_ll # test set 的 miss用cv train v2 (尚未recovery)來預測
second <- collection_ll
third <- collection_ll # 把character的欄位 轉成與class發生率有關
fourth <- collection_ll #把class轉成numeric
greek_and_fourth <- collection_ll #greek to class percentage and Class is type numeric
greek_and_third <- collection_ll #轉回factor 預測 機率
greek_and_freq <-  collection_ll
boxplot(first_ll, second, third, fourth,greek_and_fourth, greek_and_third,greek_and_freq)
View(cbind(first_ll, second, third, fourth,greek_and_fourth, greek_and_third,greek_and_freq))
#如果把greek轉成差異較大的數值呢? 全部乘以10?
#預測greek的部分不要分兩層 直接保留原本預測 alpha beta gamm delta的模型來預測test data set的

R20230607_v1 <- collection_ll
R20230607_v2 <- collection_ll
R20230607_v3 <- collection_ll
R20230607_v4 <- collection_ll
R20230607_v5 <- collection_ll
R20230607_v6 <- collection_ll
R20230607_v7 <- collection_ll #看起來把delta和 beta拿掉效果會好點? #1,1 ,1 only alpha and gamma
R20230607_v8 <- collection_ll #看起來把delta和 beta拿掉效果會好點? 把freq改成2 #1,2 ,1 only alpha and gamma
R20230607_v9 <- collection_ll 
v9看起來最好 保留alpha gamma 被預測 且作為參數再預測 如果再把最後模型改為xgboost看看 
R20230608_v1 <- collection_ll 
R20230608_v1 <- collection_ll 

boxplot(R20230608_v1 )
"加上scaler後 很穩定的在0.3上下"

jpeg("boxplot_logloss_v2.jpg")
boxplot(R20230608_v1)
dev.off()
getwd()

"記得壓縮數值到scale (0,1 )"
"如果有用class weight 去調整 會很穩定在0.13-0.14之間"


collection_ll <- NULL
 
for(i in 1:20){
    train_set$Class <- as.factor(train_set$Class )
    data_for_validation_ <- data_for_validation(0.7, train_set)
    CV_train <- data_for_validation_$trainset 
    CV_validation <- data_for_validation_$validatset 
    CV_train_v2 <- string2freq(CV_train, "Class", mode=1) #改變這個
    freq_greek_train <- (greek2freq(CV_train_v2, greeks, mode=1))
    #colnames(freq_greek_train) <- c("greek_alpha", "greek_beta","greek_gamma","greek_delta")

    CV_validation_v2 <- string2freq(CV_validation, "Class", mode=1)
    CV_train_with_repair <- miss_value_recovery(CV_train_v2)
    CV_validation_with_repair <- miss_value_recovery(CV_validation_v2, CV_validation_v2)

    CV_train_with_repair <- as.data.frame(sapply(CV_train_with_repair, scaler, simplify = FALSE))
    CV_validation_with_repair <- as.data.frame(sapply(CV_validation_with_repair , scaler, simplify = FALSE))

    train_with_greek <- as.data.frame(cbind(CV_train_with_repair, freq_greek_train  )) 
    train_with_greek$greek_alpha  <- add_new_feature(train_with_greek, variables, "Alpha", train_with_greek  )
    train_with_greek$greek_beta  <- add_new_feature(train_with_greek, variables, "Beta", train_with_greek  )
    train_with_greek$greek_gamma  <- add_new_feature(train_with_greek, variables, "Gamma", train_with_greek  )
    train_with_greek$greek_delta  <- add_new_feature(train_with_greek, variables, "Delta", train_with_greek  )

    CV_validation_with_repair$greek_alpha  <- add_new_feature(train_with_greek , variables, "greek_alpha" , CV_validation_with_repair )
    CV_validation_with_repair$greek_beta    <- add_new_feature(train_with_greek  , variables, "greek_beta" , CV_validation_with_repair)
    CV_validation_with_repair$greek_gamma  <- add_new_feature(train_with_greek  , variables, "greek_gamma", CV_validation_with_repair)
    CV_validation_with_repair$greek_delta  <- add_new_feature(train_with_greek  , variables, "greek_delta", CV_validation_with_repair)

    zero <- table(CV_train_with_repair$Class)[1]
    one <- table(CV_train_with_repair$Class)[2]
    total <- zero + one
    # Calculate the weight for each label.
    weight_for_0 <- (1 / zero) * (total / 2.0)
    weight_for_1 <- (1 / one) * (total / 2.0)

    model_RF <- randomForest(as.formula(paste0("Class~", paste0(c(variables, greek_param), collapse="+"))), train_with_greek, classwt = c("1"=weight_for_1[[1]],"0"=weight_for_0[[1]]) )
    #pred_RF_train <- predict(model_RF , CV_train_with_repair)
    #plot(model_RF)
    #varImpPlot(model_RF,30)
    #roc_ <- roc(train_with_greek$Class, as.numeric(pred_RF_train[,2])) #pred_RF[,1]
    #balan_logloss(as.numeric(pred_RF_train[,2]), as.numeric(as.character(CV_train_with_repair$Class))) #pred_RF[,2]
    pred_RF_validation <- predict(model_RF , CV_validation_with_repair,type="prob")
    collection_ll <- c(collection_ll, balan_logloss(as.numeric(pred_RF_validation[,2]), as.numeric(as.character(CV_validation_with_repair$Class))))
    print(collection_ll)

}

balan_logloss(c(1,0,1,0,1,0),c(0.99,0.01,0.99,0.01,0.99,0.01))





library(caret)
 
# Load Dataset

control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7; set.seed(seed)
metric <- "Accuracy"
mtry <- sqrt(ncol(CV_train_v2))
tunegrid <- expand.grid(.mtry=mtry)
rf_default <- train(Class~., data = CV_train_with_repair, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_default)

first_ll <- collection_ll # test set 的 miss用cv train v2 (尚未recovery)來預測
second <- collection_ll
third <- collection_ll # 把character的欄位 轉成與class發生率有關
fourth <- collection_ll #把class轉成numeric
greek_and_fourth <- collection_ll #greek to class percentage and Class is type numeric
greek_and_third <- collection_ll #轉回factor 預測 機率
greek_and_freq <-  collection_ll
boxplot(first_ll, second, third, fourth,greek_and_fourth, greek_and_third,greek_and_freq)
View(cbind(first_ll, second, third, fourth,greek_and_fourth, greek_and_third,greek_and_freq))
#如果把greek轉成差異較大的數值呢? 全部乘以10?
#預測greek的部分不要分兩層 直接保留原本預測 alpha beta gamm delta的模型來預測test data set的

R20230607_v1 <- collection_ll
R20230607_v2 <- collection_ll
R20230607_v3 <- collection_ll
R20230607_v4 <- collection_ll
R20230607_v5 <- collection_ll
R20230607_v6 <- collection_ll
R20230607_v7 <- collection_ll #看起來把delta和 beta拿掉效果會好點? #1,1 ,1 only alpha and gamma
R20230607_v8 <- collection_ll #看起來把delta和 beta拿掉效果會好點? 把freq改成2 #1,2 ,1 only alpha and gamma
R20230607_v9 <- collection_ll 
v9看起來最好 保留alpha gamma 被預測 且作為參數再預測 如果再把最後模型改為xgboost看看 

jpeg("boxplot_logloss.jpg")
boxplot(R20230607_v1, R20230607_v2, R20230607_v3,R20230607_v4,R20230607_v5,R20230607_v6,R20230607_v7,R20230607_v8,R20230607_v9)
dev.off()
getwd()
collection_ll <- NULL
train_with_greek$greek_alpha

for(i in 1:20){
  train_set$Class <- as.factor(train_set$Class )
  data_for_validation_ <- data_for_validation(0.7, train_set)
  CV_train <- data_for_validation_$trainset 
  CV_validation <- data_for_validation_$validatset 
  CV_train_v2 <- string2freq(CV_train, "Class", mode=1) #改變這個
  freq_greek_train <- (greek2freq(CV_train_v2, greeks, mode=1))
  #colnames(freq_greek_train) <- c("greek_alpha", "greek_beta","greek_gamma","greek_delta")
  CV_validation_v2 <- string2freq(CV_validation, "Class", mode=1)
  CV_train_with_repair <- miss_value_recovery(CV_train_v2)
  CV_validation_with_repair <- miss_value_recovery(CV_train_with_repair, CV_validation_v2)
  train_with_greek <- as.data.frame(cbind(CV_train_with_repair, freq_greek_train  )) 
  train_with_greek$greek_alpha  <- add_new_feature(train_with_greek, variables, "Alpha", train_with_greek  )
  #train_with_greek$greek_beta  <- add_new_feature(train_with_greek, variables, "Beta", train_with_greek  )
  train_with_greek$greek_gamma  <- add_new_feature(train_with_greek, variables, "Gamma", train_with_greek  )
  #train_with_greek$greek_delta  <- add_new_feature(train_with_greek, variables, "Delta", train_with_greek  )
  CV_validation_with_repair$greek_alpha  <- add_new_feature(train_with_greek , variables, "greek_alpha" , CV_validation_with_repair )
  #CV_validation_with_repair$greek_beta    <- add_new_feature(train_with_greek  , variables, "greek_beta" , CV_validation_with_repair)
  CV_validation_with_repair$greek_gamma  <- add_new_feature(train_with_greek  , variables, "greek_gamma", CV_validation_with_repair)
  #CV_validation_with_repair$greek_delta  <- add_new_feature(train_with_greek  , variables, "greek_delta", CV_validation_with_repair)

  model_RF <- randomForest(as.formula(paste0("Class~", paste0(c(variables,greek_param), collapse="+"))), train_with_greek)
  #pred_RF_train <- predict(model_RF , CV_train_with_repair)
  #plot(model_RF)
  #varImpPlot(model_RF,30)
  #roc_ <- roc(train_with_greek$Class, as.numeric(pred_RF_train[,2])) #pred_RF[,1]
  #balan_logloss(as.numeric(pred_RF_train[,2]), as.numeric(as.character(CV_train_with_repair$Class))) #pred_RF[,2]
  pred_RF_validation <- predict(model_RF , CV_validation_with_repair,type="prob")
  collection_ll <- c(collection_ll, balan_logloss(as.numeric(pred_RF_validation[,2]), as.numeric(as.character(CV_validation_with_repair$Class))))
  print(collection_ll)

}
