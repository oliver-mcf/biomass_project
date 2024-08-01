
# Script to Visualise Random Cross Validation of RF Models by EO Source

# Import libraries
library(ggplot2)
library(dplyr)

# Read data of grouped model stats
df <- read.csv('/home/s1949330/data/diss_data/model/yes_geo/model_group_stats.csv', header = TRUE)
summary(df)
# Read data of best performing model
data <- read.csv('/home/s1949330/scratch/diss_data/TEST_All_FOLD1_PRED-TEST.csv', header = TRUE)
summary(data)

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

# Plot scatter graph of predicted-observed values
plot_point <- ggplot(data, aes(x = Observed, y = Predicted)) +
  geom_point(color = 'lightsteelblue3', size = 0.75) +
  geom_abline(intercept = 0, slope = 1, color = "black", linetype = "solid", size = 0.75) +
  geom_smooth(method = "lm", se = FALSE, color = "red3", linetype = "solid", size = 0.75) +
  geom_smooth(method = "loess", se = FALSE, color = "blue4", linetype = "solid", size = 0.75) +
  scale_y_continuous(limits = c(0, 120), breaks = seq(0, 120, by = 10)) +
  scale_x_continuous(limits = c(0, 120), breaks = seq(0, 120, by = 10)) +
  labs(x = "Observed GEDI AGB Estimate (Mg/ha)",
       y = "Predicted GEDI AGB Estimate (Mg/ha)") +
  coord_fixed(ratio = 1) +
  theme_classic() +
  theme(panel.grid.major = element_line(color = "grey95"))

# Plot histogram of predicted-observed values
plot_hist <- ggplot(data, aes(x = Observed, y = Predicted)) +
  geom_bin2d(binwidth = c(4,4)) +
  scale_fill_gradient(low = "grey98", high = "steelblue4") +
  geom_abline(intercept = 0, slope = 1, color = "black", linetype = "solid", size = 0.75) +
  geom_smooth(method = "lm", se = FALSE, color = "red3", linetype = "solid", size = 0.75) +
  geom_smooth(method = "loess", se = FALSE, color = "blue4", linetype = "solid", size = 0.75) +
  scale_y_continuous(limits = c(0, 120), breaks = seq(0, 120, by = 10)) +
  scale_x_continuous(limits = c(0, 120), breaks = seq(0, 120, by = 10)) +
  labs(x = "Observed GEDI AGB Estimate (Mg/ha)",
       y = "Predicted GEDI AGB Estimate (Mg/ha)") +
  coord_fixed(ratio = 1) +
  theme_classic() +
  theme(panel.grid.major = element_line(color = "grey95"))

# Visualise plots
print(plot_hist)
print(plot_point)
print(plot_r2)
print(plot_bias)
print(plot_rmse)

# Save plots as figures
ggsave(plot = plot_hist, filename = paste0("/home/s1949330/scratch/diss_data/part2_figures/model_hist.png"), dpi = 300, width = 6, height = 4)
ggsave(plot = plot_point, filename = paste0("/home/s1949330/scratch/diss_data/part2_figures/model_scatter.png"), dpi = 300, width = 4, height = 4)
ggsave(plot = plot_r2, filename = paste0("/home/s1949330/scratch/diss_data/part1_figures/model_source_r2.png"), dpi = 300, width = 6, height = 4)
ggsave(plot = plot_bias, filename = paste0("/home/s1949330/scratch/diss_data/part2_figures/model_source_bias.png"), dpi = 300, width = 6, height = 4)
ggsave(plot = plot_rmse, filename = paste0("/home/s1949330/scratch/diss_data/part2_figures/model_source_rmse.png"), dpi = 300, width = 6, height = 4)

