# Load libraries
library(ggplot2)
library(dplyr)

print('hello')

# Load your dataset
data <- read.csv("forest_plot_inventory.csv")

# Print the shape of the initial dataset
dataset_shape <- dim(data)
print(dataset_shape)

# Show summary of the initial dataset
overall_summary <- summary(data)
print(overall_summary)

# Quality Control Checks
# Check for missing values
missing_values <- colSums(is.na(data))
print("Missing values in each column:")
print(missing_values)

# Check for negative values in height and agb
negative_height <- sum(data$height < 0, na.rm = TRUE)
negative_agb <- sum(data$agb < 0, na.rm = TRUE)
print(paste("Number of negative values in height:", negative_height))
print(paste("Number of negative values in agb:", negative_agb))

# Filter out negative values for height and agb
data <- data %>% filter(height >= 0, agb >= 0)

# Remove rows with NA in agb column
cleaned_data <- data %>% filter(!is.na(agb))

# Show summary for each site
site_summaries <- cleaned_data %>%
  group_by(site) %>%
  summarize(
    count = n(),
    mean_height = mean(height, na.rm = TRUE),
    sd_height = sd(height, na.rm = TRUE),
    mean_agb = mean(agb, na.rm = TRUE),
    sd_agb = sd(agb, na.rm = TRUE),
    min_height = min(height, na.rm = TRUE),
    max_height = max(height, na.rm = TRUE),
    min_agb = min(agb, na.rm = TRUE),
    max_agb = max(agb, na.rm = TRUE)
  )

print(site_summaries)

# Plot-level calculations
plot_summary <- cleaned_data %>%
  group_by(site) %>%
  summarize(
    total_agb = sum(agb, na.rm = TRUE),
    mean_height = mean(height, na.rm = TRUE)
  )

print("Plot-level summary:")
print(plot_summary)

# Create the scatter plot with linear regressions
ggplot(cleaned_data, aes(x = height, y = agb, color = site)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  geom_smooth(aes(group = 1), method = "lm", se = FALSE, linetype = "dashed", color = "black") +
  labs(title = "Scatter Plot of Height and AGB by Site",
       x = "Height",
       y = "AGB (Biomass)",
       color = "Site") +
  theme_classic()

# Overall linear regression for height and agb
overall_lm <- lm(agb ~ height, data = cleaned_data)
print(summary(overall_lm))

