# model_glm.rds

This file contains the trained logistic regression model saved using R's saveRDS() function.

Due to GitHub's 100MB file size limitation, the original model file is not included in this repository.

To regenerate the model:
- Run the script: R/03_model_training.R
- The output will be saved to: models/model_glm.rds

Note:
If using run_all.R, the model will be trained automatically if not already present.