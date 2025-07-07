# R/04_threshold_tuning.R

# Load model and test data
model_glm <- readRDS("models/model_glm.rds")
test <- readRDS("data/test.rds")
dti_median <- readRDS("data/dti_median.rds")
revol_median <- readRDS("data/revol_median.rds")

# Re-impute test (safety)
test$emp_length <- addNA(test$emp_length)
levels(test$emp_length)[is.na(levels(test$emp_length))] <- "unknown"
test$dti[is.na(test$dti)] <- dti_median
test$revol_util[is.na(test$revol_util)] <- revol_median

# Predict prob from GLM model
glm_probs <- predict(model_glm, newdata = test, type = "prob")[, "good_loan"]

# Run cost optimization
thresholds <- seq(0.3, 0.9, by = 0.05)
cost_results <- map_df(thresholds, function(thresh) {
  calculate_cost(glm_probs, test$loan_status, thresh)
})

# Save cost table
saveRDS(cost_results, file = "data/cost_results.rds")

# Find best threshold
best_threshold <- cost_results %>%
  arrange(desc(Total_Cost)) %>%
  slice(1)

saveRDS(best_threshold, file = "data/best_threshold.rds")

# Print summary
print(cost_results)
print(best_threshold)

# Recalculate confusion matrix at best threshold
best_thresh_val <- best_threshold$Threshold[1]
pred_class_new <- ifelse(glm_probs >= best_thresh_val, "good_loan", "bad_loan")
pred_class_new <- factor(pred_class_new, levels = c("good_loan", "bad_loan"))

conf <- caret::confusionMatrix(pred_class_new, test$loan_status)
print(conf)
