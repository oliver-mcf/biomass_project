
# Script to Visualise Random Cross Validation of RF Models by EO Source

# Import libraries
library(ggplot2)
library(dplyr)

# Read data of grouped model stats
#df <- read.csv('/home/s1949330/scratch/diss_data/model/GROUP_MODEL_STATS.csv', header = TRUE)
#summary(df)
# Read data of best performing model
data <- read.csv('/home/s1949330/Documents/scratch/diss_data/model/TKW-MGR/All_PRED_TEST_FOLD1.csv', header = TRUE)
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
        panel.grid.minor = element_line(color = "grey95"))

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
  geom_segment(aes(x = 0, y = 0, xend = 120, yend = 120), color = "black", linetype = "solid", size = 0.5) +
  geom_smooth(method = "lm", se = FALSE, color = "red3", linetype = "solid", size = 0.75) +
  geom_smooth(method = "loess", se = FALSE, color = "blue4", linetype = "solid", size = 0.75) +
  scale_y_continuous(limits = c(0, 120), breaks = seq(0, 120, by = 10)) +
  scale_x_continuous(limits = c(0, 120), breaks = seq(0, 120, by = 10)) +
  labs(x = "Observed GEDI AGB Estimate (Mg/ha)",
       y = "Predicted GEDI AGB Estimate (Mg/ha)",
       fill = "Frequency") +
  coord_fixed(ratio = 1) +
  theme_classic() +
  theme(panel.grid.major = element_line(color = "grey95"))

# Plot histogram of predicted-observed values
plot_hist <- ggplot(data, aes(x = Observed, y = Predicted)) +
  geom_bin2d(binwidth = c(4,4)) + 
  scale_fill_gradient(low = "grey98", high = "steelblue4") +
  scale_fill_gradientn(colors = c("grey98", "steelblue4"), breaks = scales::breaks_extended(n = 6)) +
  geom_segment(aes(x = 0, y = 0, xend = 120, yend = 120), color = "black", linetype = "solid", size = 0.5) +
  geom_smooth(method = "lm", se = FALSE, color = "red3", linetype = "solid", size = 0.75) +
  geom_smooth(method = "loess", se = FALSE, color = "blue4", linetype = "solid", size = 0.75) +
  scale_y_continuous(limits = c(0, 120), breaks = seq(0, 120, by = 10)) +
  scale_x_continuous(limits = c(0, 120), breaks = seq(0, 120, by = 10)) +
  labs(x = "Observed GEDI AGB Estimate (Mg/ha)",
       y = "Predicted GEDI AGB Estimate (Mg/ha)",
       fill = "Count (n)") +
  coord_fixed(ratio = 1) +
  theme_classic() +
  theme(panel.grid.major = element_line(color = "grey95"))

# Visualise plots
print(plot_hist)
print(plot_point)
#print(plot_r2)
#print(plot_bias)
#print(plot_rmse)

# Save plots as figures
#ggsave(plot = plot_hist, filename = paste0("/home/s1949330/scratch/diss_data/figures/part2_figures/model_TKW-MGR_hist.png"), dpi = 300, width = 6, height = 4)
#ggsave(plot = plot_point, filename = paste0("/home/s1949330/scratch/diss_data/figures/part2_figures/model_TKW-MGR_scatter.png"), dpi = 300, width = 4, height = 4)
#ggsave(plot = plot_r2, filename = paste0("/home/s1949330/scratch/diss_data/figures/part1_figures/model_group_r2.png"), dpi = 300, width = 6, height = 4)
#ggsave(plot = plot_bias, filename = paste0("/home/s1949330/scratch/diss_data/figures/part2_figures/model_group_bias.png"), dpi = 300, width = 6, height = 4)
#ggsave(plot = plot_rmse, filename = paste0("/home/s1949330/scratch/diss_data/figures/part2_figures/model_group_rmse.png"), dpi = 300, width = 6, height = 4)

