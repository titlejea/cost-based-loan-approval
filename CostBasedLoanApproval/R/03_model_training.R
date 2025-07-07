# R/03_model_training.R

# Load libraries
library(tidyverse)
library(caret)
library(pROC)

# Load function scripts
source("R/02_functions.R")

# Load preprocessed data
train <- readRDS("data/train.rds")
test <- readRDS("data/test.rds")

# 1. Train Logistic Regression
model_glm <- train_model("glm", train, sampling = "smote")
result_glm <- evaluate_model(model_glm, test)
imp_grouped_glm <- group_varimp(model_glm)

# 2. Train XGBoost
xgb_grid <- expand.grid(
  nrounds = 100,
  max_depth = 3,
  eta = 0.1,
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)
model_xgb <- train_model("xgbTree", train, sampling = "smote", tuneGrid = xgb_grid)
result_xgb <- evaluate_model(model_xgb, test)

# 3. Print Evaluation Results
cat("\nLogistic Regression Confusion Matrix:\n")
print(result_glm$confusion)
cat("AUC:", round(result_glm$auc, 4), "\n")

cat("\nXGBoost Confusion Matrix:\n")
print(result_xgb$confusion)
cat("AUC:", round(result_xgb$auc, 4), "\n")

# 4. Metrics Comparison Table
auc_table <- data.frame(
  Model = c("Logistic Regression", "XGBoost"),
  Accuracy = c(
    result_glm$confusion$overall["Accuracy"],
    result_xgb$confusion$overall["Accuracy"]
  ),
  Sensitivity = c(
    result_glm$confusion$byClass["Sensitivity"],
    result_xgb$confusion$byClass["Sensitivity"]
  ),
  Specificity = c(
    result_glm$confusion$byClass["Specificity"],
    result_xgb$confusion$byClass["Specificity"]
  ),
  Precision = c(
    result_glm$confusion$byClass["Pos Pred Value"],
    result_xgb$confusion$byClass["Pos Pred Value"]
  ),
  AUC = c(
    round(result_glm$auc, 4),
    round(result_xgb$auc, 4)
  )
)

print(auc_table)
knitr::kable(auc_table, caption = "Model Comparison Metrics")

# 5. Save logistic regression model and importance
if (!dir.exists("models")) dir.create("models")
saveRDS(model_glm, "models/model_glm.rds")
saveRDS(model_xgb, "models/model_xgb.rds")

