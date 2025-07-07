# Create folders if not exist
if (!dir.exists("data")) dir.create("data")
if (!dir.exists("outputs")) dir.create("outputs")

# Run each step 
source("R/01_data_cleaning.R")
source("R/03_model_training.R")
source("R/04_threshold_tuning.R")
source("R/05_visualization.R")

cat("✅ All steps completed!\n")
