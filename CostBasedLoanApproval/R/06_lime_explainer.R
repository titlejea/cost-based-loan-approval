# 06_lime_explainer.R

library(lime)
library(ggplot2)
library(readr)
library(caret)

# Load data and model
train <- readRDS("data/train.rds")
test <- readRDS("data/test.rds")
model_glm <- readRDS("models/model_glm.rds")

# Ensure target variable has correct levels
train$loan_status <- factor(train$loan_status, levels = c("good_loan", "bad_loan"))
test$loan_status <- factor(test$loan_status, levels = c("good_loan", "bad_loan"))

# Get predicted probabilities
probs <- predict(model_glm, newdata = test, type = "prob")
threshold <- 0.60

# Select borderline case for bad_loan
borderline_idx <- which(probs$bad_loan > threshold - 0.05 & probs$bad_loan < threshold + 0.05)[1]
borderline_data <- test[borderline_idx, , drop = FALSE]

# Create explainer
set.seed(42)
explainer <- lime(train[1:1000, ], model = model_glm, bin_continuous = TRUE)

# Run explanation for bad_loan
explanation <- explain(
  x = borderline_data,
  explainer = explainer,
  n_features = 10,
  labels = "bad_loan"
)

# Plot and save as PNG
(p <- plot_features(explanation))
ggsave(filename = "outputs/lime_explanation.png", plot = p, width = 8, height = 5)
