
# Script to Visualise Random Cross Validation of RF Models by EO Source

# Import libraries
library(ggplot2)
library(dplyr)

# Read data
df <- read.csv('/home/s1949330/data/diss_data/model/yes_geo/model_group_stats.csv', header = TRUE)
summary(df)

# Plot Model R2
plot_r2 <- ggplot(df, aes(x = as.factor(Group), y = R2, fill = Group)) +
  geom_boxplot() +
  labs(y = "Coefficient of Determination (R2)",
       fill = "Group") +
  theme_classic() +
  ylim(0.1, 0.35) + 
  scale_fill_manual(values = c(All = "#5A4A6F", Palsar = "#9D5A6C",  Landsat = "#EBB261", Sentinel = "#E47250")) +
  theme(panel.grid.major = element_line(color = "grey95"),
        panel.grid.minor = element_line(color = "grey95")) +

# Plot Model Bias  
plot_bias <- ggplot(df, aes(x = as.factor(Group), y = Bias, fill = Group)) +
  geom_boxplot() +
  geom_hline(yintercept = 0, color = "black", linetype = "dashed", size = 0.5) +
  labs(y = "Bias (Mg/ha)",
       fill = "Group") +
  theme_classic() +
  ylim(-2.5, 5) + 
  scale_fill_manual(values = c(All = "#5A4A6F", Palsar = "#9D5A6C",  Landsat = "#EBB261", Sentinel = "#E47250")) +
  theme(panel.grid.major = element_line(color = "grey95"),
        panel.grid.minor = element_line(color = "grey95"))

# Plot Model RMSE
plot_rmse <- ggplot(df, aes(x = as.factor(Group), y = RMSE, fill = Group)) +
  geom_boxplot() +
  labs(y = "RMSE (Mg/ha)",
       fill = "Group") +
  theme_classic() +
  ylim(20, 30) + 
  scale_fill_manual(values = c(All = "#5A4A6F", Palsar = "#9D5A6C",  Landsat = "#EBB261", Sentinel = "#E47250")) +
  theme(panel.grid.major = element_line(color = "grey95"),
        panel.grid.minor = element_line(color = "grey95"))

# Visualise plots
print(plot_r2)
print(plot_bias)
print(plot_rmse)

# Save plots as figures
ggsave(plot = plot_r2, filename = paste0("/home/s1949330/data/diss_data/part1_figures/model_source_r2.png"), dpi = 300, width = 6, height = 4)
ggsave(plot = plot_bias, filename = paste0("/home/s1949330/data/diss_data/part2_figures/model_source_bias.png"), dpi = 300, width = 6, height = 4)
ggsave(plot = plot_rmse, filename = paste0("/home/s1949330/data/diss_data/part2_figures/model_source_rmse.png"), dpi = 300, width = 6, height = 4)

