
# Script to Examine Seasonal Artefacts in GEDI L4A Product

# Load necessary libraries
library(lubridate)
library(ggplot2)
library(dplyr)
library(RColorBrewer)

# Define file paths and sites to plot
data_path <- "/home/s1949330/Documents/scratch/diss_data/gedi/gedi_data.csv"
output_directory <- "/home/s1949330/data/diss_data/figures/part2_figures/"
sites_to_plot <- c("TKW", "MGR")

# Read data
site_data <- read.csv(data_path)
site_data$Date <- as.Date(site_data$Date, format = "%d/%m/%Y")
filtered_site_data <- site_data[site_data$Site %in% sites_to_plot, ]

# Function to create ACF data frame
create_acf_df <- function(df_site) {
  biomass_ts <- ts(df_site$AGB_Mean_filt, start = c(year(min(df_site$Date)), month(min(df_site$Date))), frequency = 12)
  acf_values <- acf(biomass_ts, na.action = na.pass, lag.max = 36, plot = FALSE)
  acf_df <- data.frame(Lag = 1:length(acf_values$acf[-1]), ACF = acf_values$acf[-1])
  return(acf_df)
}

# Function to calculate significance threshold
significance_threshold <- function(n) {
  return(1.96 / sqrt(n))
}

# Function to plot and save ACF
plot_acf <- function(df_site, site_name) {
  acf_df <- create_acf_df(df_site)
  n <- nrow(df_site)
  threshold <- significance_threshold(n)
  plot <- ggplot(acf_df, aes(x = Lag, y = ACF)) +
    geom_hline(yintercept = 0, color = "black", linetype = "solid", size = 0.5) +
    geom_hline(yintercept = c(-threshold, threshold), col = "red3", linetype = "dashed") +
    geom_bar(stat = "identity", width = 0.2, fill = "steelblue", alpha = 0.5) +
    geom_line(color = "black", size = 0.5) +
    geom_point(color = "black", size = 1) +
    labs(title = paste(site_name), x = "Lag (Months)", y = "Autocorrelation") +
    theme_classic() +
    scale_x_continuous(breaks = 1:36) +
    theme(panel.grid.major = element_line(color = "grey95"),
          panel.grid.minor.y = element_line(color = "gray95"))
  ggsave(filename = paste0(output_directory, "ACF_plot_", site_name, ".png"), plot = plot, dpi = 300, width = 6, height = 4)
}

# Generate ACF plots for each site
for (site in sites_to_plot) {
  df_site <- subset(site_data, Site == site)
  plot_acf(df_site, site)
}

# Function to plot austral AGB
plot_austral_agb <- function(data, site_name) {
  data <- data %>%
    mutate(Date = ymd(Date),
           Month = month(Date),
           Year = ifelse(Month >= 8, year(Date) + 1, year(Date)),
           Month = factor(Month, levels = c(8:12, 1:7), labels = c(8:12, 1:7))) %>%
    mutate(YearMonth = paste(Year, Month, sep = "-"))
  
  plot <- ggplot(data, aes(x = Month, y = AGB_Mean_filt, fill = factor(Year))) +
    geom_bar(stat = "identity", position = "dodge", width = 0.7) +
    scale_x_discrete(labels = month.abb[c(8:12, 1:7)]) +
    ylim(0, 150) + 
    labs(title = paste(site_name),
         x = "Month",
         y = "AGB (Mg/ha)",
         fill = "Year") +
    theme_classic() +
    theme(legend.position = "bottom") +
    scale_fill_brewer(palette = "Set2") +
    theme(panel.grid.major = element_line(color = "grey95"),
          panel.grid.minor.y = element_line(color = "gray95"))
  
  ggsave(filename = paste0(output_directory, "AGB_plot_", site_name, ".png"), plot = plot, dpi = 300, width = 6, height = 4)
}

# Plot yearly AGB for each site
for (site in sites_to_plot) {
  site_data_subset <- filtered_site_data[filtered_site_data$Site == site, ]
  plot_austral_agb(site_data_subset, site)
}

# Function to calculate significance threshold for residuals
calculate_threshold <- function(residuals) {
  sd_residuals <- sd(residuals)
  n <- length(residuals)
  threshold <- 1.96 * sd_residuals / sqrt(n)
  return(threshold)
}

# Function to fit AR model and plot residuals
plot_residuals <- function(data, site_name) {
  AR_model <- arima(data, order = c(1, 0, 0))
  residuals_data <- residuals(AR_model)
  start_date <- as.Date("2019-08-01")
  date_sequence <- seq.Date(start_date, by = "month", length.out = length(residuals_data))
  residuals_df <- data.frame(Date = date_sequence, Residuals = residuals_data)
  residuals_numeric <- as.numeric(residuals_data)
  valid_residuals <- na.omit(residuals_numeric)
  threshold <- calculate_threshold(valid_residuals)
  vertical_dates <- date_sequence[c(1, 12, 24, 36)]
  
  plot <- ggplot(residuals_df, aes(x = Date, y = Residuals)) +
    geom_vline(xintercept = as.numeric(vertical_dates), color = "lightblue", linetype = "solid") +
    geom_point(color = "black", size = 2) +
    geom_segment(aes(x = Date, xend = Date, y = 0, yend = Residuals), color = "black") +
    geom_hline(yintercept = 0, color = "black", linetype = "solid") +
    geom_hline(yintercept = c(-threshold, threshold), color = "red3", linetype = "dashed") +
    ylim(-40, 40) +
    labs(title = paste(site_name), x = "", y = "Residuals (Mg/ha)") +
    theme_classic() + 
    scale_x_date(date_breaks = "1 month", date_labels = "%m/%y") +
    theme(panel.grid.major = element_line(color = "grey95"),
          panel.grid.minor.y = element_line(color = "gray95"),
          axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave(filename = paste0(output_directory, "Residuals_plot_", site_name, ".png"), plot = plot, dpi = 300, width = 6, height = 4)
}

# Plot residuals for each site
for (site in sites_to_plot) {
  site_data_subset <- filtered_site_data[filtered_site_data$Site == site, "AGB_Mean_filt"]
  plot_residuals(site_data_subset, site)
}

