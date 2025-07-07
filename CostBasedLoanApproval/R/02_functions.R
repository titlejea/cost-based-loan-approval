# R/02_functions.R

## 1.1 function split_data
split_data <- function(data){
  set.seed(42)
  n <- nrow(data)
  id <- sample(1:n, size = n*0.7)
  train_df <- data[id, ]
  test_df <- data[-id, ]
  return(list(train = train_df,
              test = test_df))
}

## 1.2 function train_model
train_model <- function(method, train_data, sampling = NULL, tuneGrid = NULL) {
  ctrl <- trainControl(
    method = "cv",
    number = 5,
    sampling = sampling,
    classProbs = TRUE,
    summaryFunction = twoClassSummary,
    verboseIter = TRUE
  )
  
  model <- train(
    loan_status ~ ., 
    data = train_data,
    method = method,
    trControl = ctrl,
    metric = "ROC",              
    tuneGrid = tuneGrid,
    preProcess = c("center", "scale", "zv")
  )
  return(model)
}

## 1.3 function evaluate_model
evaluate_model <- function(model, test_data) {
  pred <- predict(model, newdata = test_data)
  prob <- predict(model, newdata = test_data, type = "prob")
  
  cm <- confusionMatrix(pred, test_data$loan_status)
  roc_obj <- roc(test_data$loan_status, prob[,"bad_loan"],
                 levels = c("good_loan", "bad_loan"), direction = "<")
  auc_val <- auc(roc_obj)
  return(list(confusion = cm, auc = auc_val, roc = roc_obj))
}

## 1.4 function group_varimp
group_varimp <- function(model) {
  imp <- varImp(model)$importance
  imp$feature <- rownames(imp)
  
  imp_grouped <- imp %>%
    mutate(group = case_when(
      str_detect(feature, "grade") ~ "grade",
      str_detect(feature, "sub_grade") ~ "sub_grade",
      str_detect(feature, "term") ~ "term",
      str_detect(feature, "home_ownership") ~ "home_ownership",
      str_detect(feature, "verification_status") ~ "verification_status",
      str_detect(feature, "application_type") ~ "application_type",
      str_detect(feature, "purpose") ~ "purpose",
      str_detect(feature, "emp_length") ~ "emp_length",
      TRUE ~ feature
    )) %>%
    group_by(group) %>%
    summarise(importance = sum(Overall)) %>%
    arrange(desc(importance))
  
  return(imp_grouped)
}

## 1.5 function cost calculation
calculate_cost <- function(pred_prob, actual, threshold) {
  pred_class <- ifelse(pred_prob >= threshold, "good_loan", "bad_loan")
  pred_class <- factor(pred_class, levels = c("good_loan", "bad_loan"))
  actual <- factor(actual, levels = c("good_loan", "bad_loan"))
  
  cm <- table(
    Predicted = factor(pred_class, levels = c("good_loan", "bad_loan")),
    Actual = factor(actual, levels = c("good_loan", "bad_loan"))
  )
  
  TP <- cm["good_loan", "good_loan"]
  FP <- cm["good_loan", "bad_loan"]
  FN <- cm["bad_loan", "good_loan"]
  TN <- cm["bad_loan", "bad_loan"]
  
  usd_to_thb <- 36.5
  
  profit_per_tp <- loan_data %>%
    filter(loan_status == "good_loan") %>%
    summarise(val = mean(loan_amnt * 0.15)) %>%
    pull(val) * usd_to_thb
  
  loss_per_fp <- loan_data %>%
    filter(loan_status == "bad_loan") %>%
    summarise(val = mean(loan_amnt * 0.85)) %>%
    pull(val) * usd_to_thb
  
  loss_per_fn <- loan_data %>%
    filter(loan_status == "good_loan") %>%
    summarise(val = mean(loan_amnt * 0.03)) %>%
    pull(val) * usd_to_thb
  
  total_cost <- (TP * profit_per_tp +
                   FP * -loss_per_fp +
                   FN * -loss_per_fn) / 1e6  # แปลงเป็นล้านบาท
  
  return(data.frame(
    Threshold = threshold,
    TP = TP,
    FP = FP,
    FN = FN,
    TN = TN,
    Specificity = round(TN / (TN + FP), 4),
    Sensitivity = round(TP / (TP + FN), 4),
    Mean_Profit_TP = round(profit_per_tp, 2),
    Mean_Loss_FP = round(loss_per_fp, 2),
    Mean_Loss_FN = round(loss_per_fn, 2),
    Total_Cost = round(total_cost, 4)
  ))
}