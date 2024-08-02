
# Script to Visualise GEDI and Field AGB Estimates

# Import libraries
library(ggplot2)
library(dplyr)
library(tidyr)

# Read data
df <- read.csv('/home/s1949330/scratch/diss_data/model/All/predict/validate/BOTH_ALL_VALIDATION.csv', header = TRUE)
df_long <- df %>%
  pivot_longer(cols = c(GEDI_AGB, Field_AGB), 
               names_to = "Group", 
               values_to = "AGB")

# Visualise aligned data
box <- ggplot(df_long, aes(x = Group, y = AGB, fill = Group)) +
  geom_boxplot() +
  labs(y = "AGB (Mg/ha))",
       fill = "Group") +
  theme_classic() +
  scale_y_continuous(limits = c(0, 120), breaks = seq(0, 120, by = 20)) +
  scale_fill_manual(values = c(GEDI_AGB = "lightsteelblue3", Field_AGB = "pink3")) +
  theme(panel.grid.major = element_line(color = "grey95")) +
  facet_grid(Year ~ Site)

# Read data
df <- read.csv('/home/s1949330/scratch/diss_data/model/All/predict/EXTRAPOLATED_AGB_TIME_SERIES.csv', header = TRUE)
df_long <- df %>%
  pivot_longer(cols = c(TKW, MGR), names_to = "Site", values_to = "Biomass")
site_colors <- c("TKW" = "red3", "MGR" = "green4")
error_bar_color <- "black"

# Plot the data
bar <- ggplot(df_long, aes(x = factor(Year), y = Biomass, fill = Site)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.65), width = 0.6, alpha = 0.6) +
  geom_errorbar(aes(ymin = Biomass - RMSE, ymax = Biomass + RMSE), 
                position = position_dodge(width = 0.75), width = 0.2) +
  scale_fill_manual(values = site_colors) +
  scale_color_manual(values = rep(error_bar_color, 2)) +
  labs(x = "", y = "Extrapolated GEDI AGB Estimates (Mg/ha)", fill = "Site") +
  theme_classic() +
  scale_y_continuous(limits = c(0, 120), breaks = seq(0, 120, by = 10)) +
  theme(panel.grid.major = element_line(color = "grey95"))

# Read data
valid_df <- read.csv('/home/s1949330/scratch/diss_data/model/All/predict/validate/BOTH_ALL_VALIDATION.csv')
summary(valid_df)
valid_df$Site_Year <- paste(valid_df$Site, valid_df$Year, sep = "_")
site_colors <- c("TKW" = "indianred", "MGR" = "green4")
year_shapes <- c("17" = 24, "21" = 21)

# Plot scatter of field-gedi estimates
scatter <- ggplot(valid_df, aes(x = Field_AGB, y = GEDI_AGB, fill = Site, shape = as.factor(Year))) +
  geom_segment(aes(x = 0, y = 0, xend = 120, yend = 120), color = "black", linetype = "solid", size = 0.5) +
  scale_fill_manual(values = site_colors) +
  scale_color_manual(values = rep("black", length(site_colors))) +
  scale_shape_manual(values = year_shapes) +
  geom_point(size = 3, color = "black", stroke = 0.25) +
  geom_smooth(method = "lm", se = FALSE, color = "red3", linetype = "solid", size = 0.75, aes(group = 1)) +
  geom_smooth(method = "loess", se = FALSE, color = "blue4", linetype = "solid", size = 0.75, aes(group = 1)) +
  labs(x = "Field AGB Estimates (Mg/ha)", y = "Extrapolated GEDI AGB Estimates (Mg/ha)", fill = "Site", shape = "Year") +
  theme_classic() + 
  scale_y_continuous(limits = c(0, 120), breaks = seq(0, 120, by = 10)) +
  scale_x_continuous(limits = c(0, 120), breaks = seq(0, 120, by = 10)) +
  theme(panel.grid.major = element_line(color = "grey95")) +
  coord_fixed(ratio = 1)


# Visualise plots
print(bar)
print(box)
print(scatter)

# Save plots as figures
#ggsave(plot = scatter, filename = paste0("/home/s1949330/scratch/diss_data/figures/part1_figures/gedi_field_AGB_scatter.png"), dpi = 300, width = 6, height = 5)
#ggsave(plot = bar, filename = paste0("/home/s1949330/scratch/diss_data/figures/part1_figures/gedi_site_AGB_estimates.png"), dpi = 300, width = 6, height = 4)
#ggsave(plot = box, filename = paste0("/home/s1949330/scratch/diss_data/figures/part2_figures/gedi_field_AGB_estimates.png"), dpi = 300, width = 6, height = 4)


