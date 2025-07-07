# R/01_data_cleaning.R

# Load packages
library(readr)
library(tidyverse)
library(caret)
library(lubridate)
library(dplyr)
library(pROC)
library(lime)
library(iml)
library(stringr)

# Load function scripts
source("R/02_functions.R")

# Load and sample data
loan_raw <- read_csv("accepted_2007_to_2018Q4.csv", n_max = 1000000)
set.seed(42)
loan_sample <- loan_raw %>%
  sample_n(200000)

# Clean and select columns
loan_clean <- loan_sample %>%
  select(where(~ !all(is.na(.))))

selected_vars <- c(
  "loan_amnt", "term", "grade", "sub_grade",
  "emp_length", "home_ownership", "annual_inc", "verification_status",
  "purpose", "application_type",
  "dti", "revol_util", "revol_bal", 
  "delinq_2yrs", "open_acc", "total_acc", "pub_rec", 
  "pub_rec_bankruptcies", "fico_range_low", "fico_range_high",
  "issue_d", "loan_status"
)

loan_data <- loan_clean %>%
  select(all_of(selected_vars)) %>%
  mutate(
    loan_status = case_when(
      loan_status %in% c("Fully Paid") ~ "good_loan",
      loan_status %in% c("Charged Off", "Default") ~ "bad_loan",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(loan_status)) %>%
  mutate(loan_status = factor(loan_status, levels = c("good_loan", "bad_loan"))) %>%
  mutate(
    term = as.factor(term),
    grade = as.factor(grade),
    sub_grade = as.factor(sub_grade),
    emp_length = as.factor(emp_length),
    home_ownership = as.factor(home_ownership),
    verification_status = as.factor(verification_status),
    purpose = as.factor(purpose),
    application_type = as.factor(application_type),
    issue_month = month(mdy(issue_d))
  ) %>%
  select(-issue_d)

# Split data
split <- split_data(loan_data)
train <- split$train
test <- split$test

# Save dti/revol medians for later
saveRDS(median(train$dti, na.rm = TRUE), "data/dti_median.rds")
saveRDS(median(train$revol_util, na.rm = TRUE), "data/revol_median.rds")

# Impute missing values
train$emp_length <- addNA(train$emp_length)
levels(train$emp_length)[is.na(levels(train$emp_length))] <- "unknown"
train$dti[is.na(train$dti)] <- readRDS("data/dti_median.rds")
train$revol_util[is.na(train$revol_util)] <- readRDS("data/revol_median.rds")

test$emp_length <- addNA(test$emp_length)
levels(test$emp_length)[is.na(levels(test$emp_length))] <- "unknown"
test$dti[is.na(test$dti)] <- readRDS("data/dti_median.rds")
test$revol_util[is.na(test$revol_util)] <- readRDS("data/revol_median.rds")

# Save clean train/test sets
saveRDS(train, file = "data/train.rds")
saveRDS(test, file = "data/test.rds")
saveRDS(loan_data, file = "data/loan_data.rds")
