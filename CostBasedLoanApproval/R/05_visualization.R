# R/05_visualization.R

# Load packages
library(ggplot2)
library(scales)

# Load data
cost_results <- readRDS("data/cost_results.rds")
best_threshold <- readRDS("data/best_threshold.rds")
imp_grouped_glm <- readRDS("data/imp_grouped_glm.rds")

# Plot Cost vs Threshold
ggplot(cost_results, aes(x = Threshold, y = Total_Cost)) +
  geom_line(color = "darkgreen", size = 1.2) +
  geom_point(color = "darkgreen", size = 2) +
  ggtitle("Total Cost by Threshold") +
  ylab("Total Cost (mmTHB)") +
  xlab("Threshold") +
  scale_x_continuous(breaks = seq(0.3, 0.9, by = 0.05)) +  
  scale_y_continuous(
    breaks = scales::pretty_breaks(n = 10),               
    labels = scales::comma_format(accuracy = 0.01)        
  ) +
  theme_minimal(base_size = 14)

ggsave("outputs/plot_threshold_cost.png", width = 8, height = 5)

# Plot Grouped Variable Importance
ggplot(imp_grouped_glm, aes(x = reorder(group, importance), y = importance)) +
  geom_col(fill = "darkblue") +
  coord_flip() +
  labs(title = "Grouped Variable Importance", x = "Feature Group", y = "Importance") +
  theme_minimal(base_size = 14)

ggsave("outputs/plot_grouped_varimp.png", width = 8, height = 5)
