"""
This script performs generalized linear mixed models (GLMM) analysis on memory performance data
to investigate the effects of experimental conditions (Rest vs. Distraction) and condition order.
It includes model fitting, comparison, diagnostics, and post-hoc analyses using emmeans.

Key Features:
- Fits binomial GLMMs with subject-level random effects
- Compares nested models to select the best-fitting model
- Performs model diagnostics (convergence, quality checks, predictions)
- Computes marginal means and contrasts for condition and order effects
- Handles interactions between condition and order

Dependencies: dplyr, tidyr, glmmTMB, emmeans, performance, DHARMa, modelbased, visualize
"""

# Load required packages
if (!requireNamespace("pacman")) install.packages("pacman")
pacman::p_load(
  dplyr, tidyr, here, stringr,      # Data manipulation
  emmeans, glmmTMB,                  # Statistical modeling
  modelbased, performance, visualize, # Model diagnostics
  knitr, DHARMa                     # Additional utilities
)

# Load preprocessed performance data
load(here("data/data_raw_df_performance.rda"))

#################################################################################
# EFFECT OF CONDITION
#################################################################################

# Define a helper function to fit GLMMs with binomial family
fit_glmm <- function(formula, data) {
  model <- glmmTMB(
    formula = formula,
    data = data,
    family = binomial(),  # Binary outcome (performance: 0 or 1)
  )
  return(model)
}

# Define and fit two candidate models:
# m1: Random intercept for subject
# m2: Random intercept and slope for condition by subject
formula_1 <- performance ~ condition + (1 | subject)
formula_2 <- performance ~ condition + (condition | subject)

m1 <- fit_glmm(formula_1, df_performance)
m2 <- fit_glmm(formula_2, df_performance)

# Compare model performance metrics
compare_performance(m1, m2, metrics = c('R2', 'RMSE', 'AICc', 'BIC'), rank = TRUE)

# Likelihood ratio test to compare nested models
anova(m1, m2)

# Select the best model (here m1)
model <- m1

# View model summary and coefficients
summary(model)
coef(model)

# --- MODEL DIAGNOSTICS ---
# Check for convergence issues
check_convergence(model)

# Overall model quality checks
check_model(model)

# Visual inspection of predictions vs. observed
check_predictions(model) |>
  plot()

# --- POST-HOC ANALYSES ---
# Marginal means and pairwise contrasts for condition effect
model |>
  emmeans(~ condition, type = "response") |>
  contrast(method = "pairwise")  # Compare Rest vs. Distraction

#################################################################################
# EFFECT OF ORDER
#################################################################################

# Redefine fit_glmm with prior for random effects to improve convergence
fit_glmm <- function(formula, data, gamma_mean = 1e8) {
  model <- glmmTMB(
    formula = formula,
    data = data,
    family = binomial(),
    prior = data.frame(
      prior = glue::glue("gamma({gamma_mean}, 2.5)"),
      class = "ranef"  # Prior for random effects
    )
  )
  return(model)
}

# Define and fit two candidate models with condition * order interaction:
# m1: Random intercept for subject
# m2: Random intercept and slope for condition by subject
formula_1 <- performance ~ condition * order + (1 | subject)
formula_2 <- performance ~ condition * order + (condition | subject)

m1 <- fit_glmm(formula_1, df_performance)
m2 <- fit_glmm(formula_2, df_performance)

# Compare model performance metrics
compare_performance(m1, m2, metrics = c('R2', 'RMSE', 'AICc', 'BIC'), rank = TRUE)

# Likelihood ratio test to compare nested models
anova(m1, m2)

# Select the best model (here m1)
model <- m1

# View model summary and coefficients
summary(model)
coef(model)

# --- MODEL DIAGNOSTICS ---
# Check for convergence issues
check_convergence(model)

# Overall model quality checks
check_model(model)

# Visual inspection of predictions vs. observed
check_predictions(model) |>
  plot()

# --- POST-HOC ANALYSES ---
# Marginal means and pairwise contrasts for condition effect
model |>
  emmeans(~ condition, type = "response") |>
  contrast(method = "pairwise")  # Compare Rest vs. Distraction

# Marginal means and pairwise contrasts for order effect
model |>
  emmeans(~ order, type = "response") |>
  contrast(method = "pairwise")  # Compare order levels

# Interaction effects: condition by order
model |>
  emmeans(~ condition | order, type = "response") |>
  contrast(method = "pairwise")  # Compare conditions within each order level

model |>
  emmeans(~ order | condition, type = "response") |>
  contrast(method = "pairwise")  # Compare order within each condition level

# Get estimated marginal means for the condition * order interaction
emm_interaction <- model |>
  emmeans(~ condition * order, type = "response")
emm_interaction

# Example output:
# condition   order  prob     SE  df asymp.LCL asymp.UCL
# Rest        D-R   0.980 0.0098 Inf     0.948     0.992
# Distraction D-R   0.948 0.0215 Inf     0.886     0.977
# Rest        R-D   0.910 0.0325 Inf     0.823     0.957
# Distraction R-D   0.971 0.0128 Inf     0.932     0.988

# Custom contrasts for specific comparisons
# Compare Distraction in D-R vs Rest in R-D
contrast(
  emm_interaction,
  method = list("Distraction:D-R vs Rest:R-D" = c(0, 1, -1, 0)),  # Coefficients for the 4 groups
  type = "response"
)

# Compare Rest in D-R vs Distraction in R-D
contrast(
  emm_interaction,
  method = list("Rest:D-R vs Distraction:R-D" = c(1, 0, 0, -1)),
  type = "response"
)