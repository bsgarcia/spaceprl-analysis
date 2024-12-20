library(brms) 
library(TEfits)
library(dplyr)
library(ggplot2)

# Read the CSV file
df <- read.csv("../data/raw/fullpilot13_2.csv")

# Filter rows where expName is 'FullPilot12'
df <- subset(df, expName %in% c("FullPilot12", 'FullPilot13'))

# Group by prolificID and filter groups with at least 480 rows
df <- df %>% 
  group_by(prolificID) %>% 
  filter(n() >= 480) %>% 
  ungroup()

# Keep prolificIDs that are more than 10 characters long
df <- df[nchar(df$prolificID) > 10, ]

# Show the number of unique prolificIDs
length(unique(df$prolificID))

# ----------------------------
# Compute 'opti_ss'
df$opti_ss <- ((df$m1 > df$m2) & (df$choice == 1)) | ((df$m1 < df$m2) & (df$choice == 2))

# Compute 'opti_ff'
df$opti_ff <- ((df$p1 > df$p2) & (df$choice == 1)) | ((df$p1 < df$p2) & (df$choice == 2))

# Compute 'opti_ev'
df$opti_ev <- ((df$ev1 > df$ev2) & (df$choice == 1)) | ((df$ev1 < df$ev2) & (df$choice == 2))

df <- df[df$session==2,]
df <- df[df$prolificID=='610a1655e385fd7341e12778',]
# fit a `TEfit` model

# Ensure the data is numeric
df$opti_ss <- as.numeric(df$opti_ss)
df$t <- as.numeric(df$t)

# Convert to a matrix if required
input_data <- as.matrix(df[, c('opti_ss', 't')])

# Fit the TEfit model
mod_simple <- TEfit(input_data)

 (mod_simple,plot_title='Time-evolving fit of artificial data')