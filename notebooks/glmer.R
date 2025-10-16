# Load required libraries
library(lme4)
library(dplyr)

# Set working directory to the notebook directory
setwd("p:/CodeProjects/Current/spaceprl-analysis/notebooks")

# Load the data
# df <- read.csv("../data/interim/df_for_r.csv")
# print the df
# print(head(df))

# Helper functions
minmax_scale <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# Prepare data for shield training (session 1)
exps <- c('FullPilot12', 'FullPilot12_2')
df_shield <- df %>%
  filter(session == 1, expName %in% exps, group != 'random') 

# Fit model for shield training (forcefield/perceptual)
formula_shield <- 'choice ~ delta_p + (1 + delta_p | prolificID)'
model_shield <- glmer(
  formula_shield,
  data = df_shield,
  family = binomial(link = "probit"),
  control = glmerControl(
    optimizer = "bobyqa",
    optCtrl = list(maxfun = 100000),
    calc.derivs = FALSE
  )
)

# # Print summary
# print("Shield Training Model Summary:")
# print(summary(model_shield))

# Extract random effects for shield training
ranef_shield <- ranef(model_shield)$prolificID
ranef_shield_df <- as.data.frame(ranef_shield)
ranef_shield_df$prolificID <- rownames(ranef_shield_df)
names(ranef_shield_df) <- c('intercept_ff', 'noise_ff', 'prolificID')

# Prepare data for spaceship training (session 2)
df_spaceship <- df %>%
  filter(session == 2, expName %in% exps, group != 'random') 
 
# Fit model for spaceship training (value)
formula_spaceship <- 'choice ~ delta_m + (1 + delta_m | prolificID)'
model_spaceship <- glmer(
  formula_spaceship,
  data = df_spaceship,
  family = binomial(link = "probit"),
  control = glmerControl(
    optimizer = "bobyqa",
    optCtrl = list(maxfun = 100000),
    calc.derivs = FALSE
  )
)

# # Print summary
# print("\nSpaceship Training Model Summary:")
# print(summary(model_spaceship))

# Extract random effects for spaceship training
ranef_spaceship <- ranef(model_spaceship)$prolificID
ranef_spaceship_df <- as.data.frame(ranef_spaceship)
ranef_spaceship_df$prolificID <- rownames(ranef_spaceship_df)
names(ranef_spaceship_df) <- c('intercept_ss', 'noise_ss', 'prolificID')

# Merge the two dataframes
random_effects_df <- merge(
  ranef_shield_df[, c('prolificID', 'noise_ff')],
  ranef_spaceship_df[, c('prolificID', 'noise_ss')],
  by = 'prolificID',
  all = TRUE
)

# Save to CSV
# write.csv(random_effects_df, "../data/interim/re_df.csv", row.names = FALSE)

# print("\nRandom effects extracted and saved to re_df.csv")
# print(head(random_effects_df))