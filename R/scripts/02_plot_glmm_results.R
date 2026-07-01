"""
This script creates plots to visualize the results of generalized linear
mixed models (GLMM) analyzing memory performance under different conditions (Rest vs. Distraction)
and condition orders (D-R vs. R-D). It includes:

1. A violin plot with individual data points showing performance distribution by condition
2. A faceted plot showing the interaction between condition and order
3. Statistical annotations (odds ratios, p-values, confidence intervals)
4. Custom formatting for publication (colors, labels, themes)

The plots are saved as PDF and PNG files for inclusion in manuscripts or presentations.

Dependencies: ggplot2, glmmTMB, emmeans, dplyr, cowplot, here, scales
Custom functions: ggplot_utils.R (for theme_pdf, save_ggplot, geom_point2)
"""

# Load required packages
if (!requireNamespace("pacman")) install.packages("pacman")
pacman::p_load(
  crayon,     # For colored console output
  dplyr,      # Data manipulation
  emmeans,    # Estimated marginal means
  glue,       # String interpolation
  ggplot2,    # Plotting
  glmmTMB,    # Generalized linear mixed models
  here,       # File path handling
  modelbased, # Model diagnostics
  scales,     # Scale functions for ggplot2
  see        # Additional visualization tools
)
source(here("R/ggplot_utils.R"))  # Load custom ggplot utilities

# Load data
load(here("data/data_raw.rda"))          # Raw data
load(here("data/data_raw_df_performance.rda"))  # Processed performance data

# Define a helper function to fit GLMMs with binomial family
fit_glmm <- function(formula, data) {
  model <- glmmTMB(
    formula = formula,
    data = data,
    family = binomial(),  # Binary outcome (performance: 0 or 1)
  )
  return(model)
}

#################################################################################
# PLOT 1: EFFECT OF CONDITION (Rest vs. Distraction)
#################################################################################

# Fit model
formula <- performance ~ condition + (1 | subject)
m1 <- fit_glmm(formula, df_performance)

# Perform pairwise contrasts for condition effect
m1 |>
  emmeans(~ condition, type = "response") |>
  contrast(method = "pairwise")

# Extract estimated marginal means for plotting
df_emm <- m1 |>
  emmeans(~ condition, type = "response") |>
  as.data.frame()

# Prepare data for statistical annotations (stars for significance)
effects <- tibble::tibble(
  condition = c("Rest", "Distraction"),
  x_star = 1.5,        # x-position for stars
  y_star = 1.05,       # y-position for stars
  stars = c("ns"),      # Significance stars ("ns" = not significant)
  x_line = x_star - 0.5,  # Start of bracket line
  x_line_end = x_star + 0.5,  # End of bracket line
  y_line = 1.02        # y-position for bracket line
)

# Calculate per-subject performance (proportion of memorised items)
df_per_subject <- df_performance |>
  group_by(subject, condition) |>
  reframe(
    prob = (sum(performance) / number_of_trials[1])  # Mean performance per subject/condition
  ) |>
  ungroup()

# Set factor levels for consistent ordering in plots
desired_order <- c("Rest", "Distraction")
df_per_subject <- df_per_subject |>
  mutate(condition = factor(condition, levels = desired_order))
df_emm <- df_emm |>
  mutate(condition = factor(condition, levels = desired_order))
effects <- effects |>
  mutate(condition = factor(condition, levels = desired_order))

# Create the plot
plot <- ggplot(mapping = aes(x = condition, y = prob, color = condition, fill = condition)) +
  
  # Custom x-axis labels with statistics
  scale_x_discrete(limits = c(
    "Rest\n0.95 (0.01)\n[0.91; 0.97]",
    "Distraction\n0.96 (0.01)\n[0.93; 0.98]"
  )) +
  
  # Violin plot showing distribution of per-subject performance
  geom_violin(
    data = df_per_subject |>
      mutate(condition = case_when(
        condition == "Rest" ~ "Rest\n0.95 (0.01)\n[0.91; 0.97]",
        condition == "Distraction" ~ "Distraction\n0.96 (0.01)\n[0.93; 0.98]",
        TRUE ~ as.character(condition)
      )),
    color = "grey82",
    flip = 1,          # Horizontal violins
    alpha = 0.2,       # Transparency
    scale = "width",   # Scale violins to same width
    linewidth = 0.1,
  ) +
  
  # Connect individual subject points across conditions
  geom_line(
    data = df_per_subject |>
      mutate(condition = case_when(
        condition == "Rest" ~ "Rest\n0.95 (0.01)\n[0.91; 0.97]",
        condition == "Distraction" ~ "Distraction\n0.96 (0.01)\n[0.93; 0.98]",
        TRUE ~ as.character(condition)
      )),
    aes(x = condition, y = prob, group = subject),
    color = "grey82",
    show.legend = FALSE,
    inherit.aes = FALSE
  ) +
  
  # Individual subject points
  geom_point(
    data = df_per_subject |>
      mutate(condition = case_when(
        condition == "Rest" ~ "Rest\n0.95 (0.01)\n[0.91; 0.97]",
        condition == "Distraction" ~ "Distraction\n0.96 (0.01)\n[0.93; 0.98]",
        TRUE ~ as.character(condition)
      )),
    color = "black",
    alpha = 0.5,
    size = 1.5,
    show.legend = FALSE
  ) +
  
  # Lines connecting marginal means
  geom_line(
    data = df_emm |>
      mutate(condition = case_when(
        condition == "Rest" ~ "Rest\n0.95 (0.01)\n[0.91; 0.97]",
        condition == "Distraction" ~ "Distraction\n0.96 (0.01)\n[0.93; 0.98]",
        TRUE ~ as.character(condition)
      )),
    color = "black",
    aes(group = 1, color = condition),
    linewidth = 1,
    show.legend = FALSE
  ) +
  
  # Error bars for marginal means (95% CI)
  geom_errorbar(
    data = df_emm |>
      mutate(condition = case_when(
        condition == "Rest" ~ "Rest\n0.95 (0.01)\n[0.91; 0.97]",
        condition == "Distraction" ~ "Distraction\n0.96 (0.01)\n[0.93; 0.98]",
        TRUE ~ as.character(condition)
      )),
    aes(ymin = asymp.LCL, ymax = asymp.UCL, color = condition),
    width = 0,
    linewidth = 0.75,
    color = "black",
    show.legend = FALSE
  ) +
  
  # Points for marginal means
  geom_point2(
    data = df_emm |>
      mutate(condition = case_when(
        condition == "Rest" ~ "Rest\n0.95 (0.01)\n[0.91; 0.97]",
        condition == "Distraction" ~ "Distraction\n0.96 (0.01)\n[0.93; 0.98]",
        TRUE ~ as.character(condition)
      )),
    aes(color = condition),
    size = 2,
    shape = 21,
    stroke = 1,
    color = "black"
  ) +
  
  # Statistical significance stars
  geom_text(
    data = effects,
    aes(x = .data$x_star, y = .data$y_star, label = .data$stars),
    inherit.aes = FALSE,
    color = "black",
    size = 4
  ) +
  
  # Bracket for significance annotation
  geom_segment(
    data = effects,
    aes(x = .data$x_line, xend = .data$x_line_end, y = .data$y_line, yend = .data$y_line),
    inherit.aes = FALSE,
    color = "black",
    linewidth = 0.5,
    linetype = "dashed"
  ) +
  
  # Plot labels and title
  labs(
    title = "N = 30\nOR : 0.77, p = 0.28",  # Overall odds ratio and p-value
    colour = NULL,
    fill = NULL
  ) +
  
  # Color scheme for conditions
  scale_discrete_manual(
    aesthetics = c("color", "fill"),
    values = c(
      "Rest\n0.95 (0.01)\n[0.91; 0.97]" = "#0173b2",    # Blue for Rest
      "Distraction\n0.96 (0.01)\n[0.93; 0.98]" = "#de8f05"  # Orange for Distraction
    )
  ) +
  
  # Adjust y-axis limits
  scale_y_continuous(expand = expansion(mult = c(0.04, 0.04))) +
  
  # Axis labels
  labs(
    x = NULL,
    y = "Proportion of memorised items",
  ) +
  
  # Custom theme for publication
  theme_pdf(
    base_theme = theme_minimal,
    base_size = 9,
    axis_relative_x = 1.2,
    strip.text = element_text(face = "plain", family = "sans-serif", color = 'black'),
    axis.text = element_text(color = "black"),
  ) +
  
  # Additional theme adjustments
  theme(
    text = element_text(family = "sans-serif"),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
    legend.position = "none"
  )

# Display the plot
plot

# Save the plot as PDF
save_ggplot(
  plot = plot,
  path = here("figures/plot_glmm_condition_emmeans.pdf"),
  ncol = 2,
  height = 120,
  width = 105
)

#################################################################################
# PLOT 2: INTERACTION EFFECT (Condition × Order)
#################################################################################

# Fit model
formula <- performance ~ condition * order + (1 | subject)

m1 <- fit_glmm(formula, df_performance)

# Perform pairwise contrasts for condition × order interaction
m1 |>
  emmeans(~ condition * order, type = "response") |>
  contrast(method = "pairwise")

df_emm <- m1 |>
  emmeans(~ condition * order, type = "response") |>
  as.data.frame()

# Calculate per-subject performance for each condition × order combination
df_per_subject <- df_performance |>
  group_by(subject, condition) |>
  reframe(
    order = unique(order),
    prob = (sum(performance) / number_of_trials[1])
  ) |>
  ungroup()

# Custom contrast information for cross-facet annotation
cross_facet_contrasts <- tibble::tibble(
  condition = "Rest",
  contrast = "(D-R) / (R-D)",
  odds_ratio = 4.830,
  SE = 3.01,
  df = Inf,
  z_ratio = 2.530,
  p_value = 0.0114,
  stars = "*"  # Significance star
)

# Prepare data for statistical annotations (stars for significance)
effects <- tibble::tibble(
  order = c("D-R", "R-D"),
  x_star = 1.5,        # x-position for stars
  y_star = 1.03,       # y-position for stars
  stars = c("*", "***"), # Significance stars
  x_line = x_star - 0.5,  # Start of bracket line
  x_line_end = x_star + 0.5,  # End of bracket line
  y_line = 1.02        # y-position for bracket line
)

# Create the faceted plot
plot <- ggplot(mapping = aes(x = condition, y = prob, fill = condition)) +
  
  # Violin plot for D-R order
  geom_violinhalf(
    data = df_per_subject |>
      filter(order == "D-R") |>
      mutate(condition = factor(
        condition,
        levels = c("Distraction", "Rest"),
        labels = c(
          "Distraction\n0.95 (0.02)\n[0.89; 0.98]",
          "Rest\n0.98 (0.01)\n[0.95; 0.99]"
        )
      )),
    color = "grey82",
    flip = 1,
    alpha = 0.2,
    scale = "width",
    linewidth = 0.1,
  ) +
  
  # Violin plot for R-D order
  geom_violinhalf(
    data = df_per_subject |>
      filter(order == "R-D") |>
      mutate(condition = factor(
        condition,
        levels = c("Rest", "Distraction"),
        labels = c(
          "Rest\n0.91 (0.03)\n[0.82; 0.96]",
          "Distraction\n0.97 (0.01)\n[0.93; 0.99]"
        )
      )),
    color = "grey82",
    flip = 1,
    alpha = 0.2,
    scale = "width",
    linewidth = 0.1,
  ) +
  
  # Connect individual subject points for D-R order
  geom_line(
    data = df_per_subject |>
      filter(order == "D-R") |>
      mutate(condition = factor(
        condition,
        levels = c("Distraction", "Rest"),
        labels = c(
          "Distraction\n0.95 (0.02)\n[0.89; 0.98]",
          "Rest\n0.98 (0.01)\n[0.95; 0.99]"
        )
      )),
    aes(x = condition, y = prob, group = subject),
    color = "grey82",
    show.legend = FALSE,
    inherit.aes = FALSE
  ) +
  
  # Connect individual subject points for R-D order
  geom_line(
    data = df_per_subject |>
      filter(order == "R-D") |>
      mutate(condition = factor(
        condition,
        levels = c("Rest", "Distraction"),
        labels = c(
          "Rest\n0.91 (0.03)\n[0.82; 0.96]",
          "Distraction\n0.97 (0.01)\n[0.93; 0.99]"
        )
      )),
    aes(x = condition, y = prob, group = subject),
    color = "grey82",
    show.legend = FALSE,
    inherit.aes = FALSE
  ) +
  
  # Individual subject points for D-R order
  geom_point(
    data = df_per_subject |>
      filter(order == "D-R") |>
      mutate(condition = factor(
        condition,
        levels = c("Distraction", "Rest"),
        labels = c(
          "Distraction\n0.95 (0.02)\n[0.89; 0.98]",
          "Rest\n0.98 (0.01)\n[0.95; 0.99]"
        )
      )),
    alpha = 0.5,
    size = 1.5,
    show.legend = FALSE
  ) +
  
  # Individual subject points for R-D order
  geom_point(
    data = df_per_subject |>
      filter(order == "R-D") |>
      mutate(condition = factor(
        condition,
        levels = c("Rest", "Distraction"),
        labels = c(
          "Rest\n0.91 (0.03)\n[0.82; 0.96]",
          "Distraction\n0.97 (0.01)\n[0.93; 0.99]"
        )
      )),
    alpha = 0.5,
    size = 1.5,
    show.legend = FALSE
  ) +
  
  # Lines connecting marginal means for D-R order
  geom_line(
    data = df_emm |>
      filter(order == "D-R") |>
      mutate(condition = factor(
        condition,
        levels = c("Distraction", "Rest"),
        labels = c(
          "Distraction\n0.95 (0.02)\n[0.89; 0.98]",
          "Rest\n0.98 (0.01)\n[0.95; 0.99]"
        )
      )),
    aes(group = 1),
    linewidth = 1,
    show.legend = FALSE
  ) +
  
  # Lines connecting marginal means for R-D order
  geom_line(
    data = df_emm |>
      filter(order == "R-D") |>
      mutate(condition = factor(
        condition,
        levels = c("Rest", "Distraction"),
        labels = c(
          "Rest\n0.91 (0.03)\n[0.82; 0.96]",
          "Distraction\n0.97 (0.01)\n[0.93; 0.99]"
        )
      )),
    aes(group = 1),
    linewidth = 1,
    show.legend = FALSE
  ) +
  
  # Error bars for marginal means (D-R order)
  geom_errorbar(
    data = df_emm |>
      filter(order == "D-R") |>
      mutate(condition = factor(
        condition,
        levels = c("Distraction", "Rest"),
        labels = c(
          "Distraction\n0.95 (0.02)\n[0.89; 0.98]",
          "Rest\n0.98 (0.01)\n[0.95; 0.99]"
        )
      )),
    aes(ymin = asymp.LCL, ymax = asymp.UCL),
    width = 0,
    linewidth = 0.75,
    color = "black",
    show.legend = FALSE
  ) +
  
  # Error bars for marginal means (R-D order)
  geom_errorbar(
    data = df_emm |>
      filter(order == "R-D") |>
      mutate(condition = factor(
        condition,
        levels = c("Rest", "Distraction"),
        labels = c(
          "Rest\n0.91 (0.03)\n[0.82; 0.96]",
          "Distraction\n0.97 (0.01)\n[0.93; 0.99]"
        )
      )),
    aes(ymin = asymp.LCL, ymax = asymp.UCL),
    width = 0,
    linewidth = 0.75,
    color = "black",
    show.legend = FALSE
  ) +
  
  # Points for marginal means (D-R order)
  geom_point2(
    data = df_emm |>
      filter(order == "D-R") |>
      mutate(condition = factor(
        condition,
        levels = c("Distraction", "Rest"),
        labels = c(
          "Distraction\n0.95 (0.02)\n[0.89; 0.98]",
          "Rest\n0.98 (0.01)\n[0.95; 0.99]"
        )
      )),
    size = 2,
    shape = 21,
    stroke = 1,
    color = "black"
  ) +
  
  # Points for marginal means (R-D order)
  geom_point2(
    data = df_emm |>
      filter(order == "R-D") |>
      mutate(condition = factor(
        condition,
        levels = c("Rest", "Distraction"),
        labels = c(
          "Rest\n0.91 (0.03)\n[0.82; 0.96]",
          "Distraction\n0.97 (0.01)\n[0.93; 0.99]"
        )
      )),
    size = 2,
    shape = 21,
    stroke = 1,
    color = "black"
  ) +
  
  # Statistical significance stars
  geom_text(
    data = effects,
    aes(x = .data$x_star, y = .data$y_star, label = .data$stars),
    inherit.aes = FALSE,
    color = "black",
    size = 4
  ) +
  
  # Bracket for significance annotation
  geom_segment(
    data = effects,
    aes(x = .data$x_line, xend = .data$x_line_end, y = .data$y_line, yend = .data$y_line),
    inherit.aes = FALSE,
    color = "black",
    linewidth = 0.5
  ) +
  
  # Faceting by order with custom labels
  facet_wrap(
    ~ order,
    scales = "free_x",
    labeller = labeller(
      order = c(
        "D-R" = "Distraction-Rest\nn = 15\nOR : 2.71, p = 0.02",
        "R-D" = "Rest-Distraction\nn = 15\nOR : 0.30, p < 0.001"
      )
    )
  ) +
  
  # Color scheme for conditions
  scale_discrete_manual(
    aesthetics = c("colour", "fill"),
    values = c(
      "Rest\n0.91 (0.03)\n[0.82; 0.96]" = "#0173b2",    # Blue for Rest
      "Distraction\n0.97 (0.01)\n[0.93; 0.99]" = "#de8f05",  # Orange for Distraction
      "Rest\n0.98 (0.01)\n[0.95; 0.99]" = "#0173b2",    # Blue for Rest
      "Distraction\n0.95 (0.02)\n[0.89; 0.98]" = "#de8f05"  # Orange for Distraction
    )
  ) +
  
  # Adjust y-axis limits with extra space for bracket
  scale_y_continuous(expand = expansion(mult = c(0.04, 0.08))) +
  
  # Axis labels
  labs(
    x = NULL,
    y = "Proportion of memorised items",
  ) +
  
  # Custom theme for publication
  theme_pdf(
    base_theme = theme_minimal,
    base_size = 9,
    axis_relative_x = 1.2,
    strip.text = element_text(face = "plain", family = "sans-serif", color = 'black'),
    axis.text = element_text(color = "black"),
  ) +
  
  # Additional theme adjustments
  theme(
    text = element_text(family = "sans-serif"),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
    legend.position = "none",
    plot.margin = margin(t = 20, r = 5, b = 5, l = 5)  # Extra top margin for bracket
  )

# Add cross-facet bracket with cowplot
library(cowplot)

# Draw the plot then overlay the annotation
final_plot <- ggdraw(plot) +
  # Left end of bracket (above "Rest" in facet D-R)
  draw_line(x = c(0.4, 0.655), y = c(0.81, 0.81), color = "black", size = 0.5) +
  # Right end of bracket (above "Rest" in facet R-D)
  draw_label(
    cross_facet_contrasts$stars,
    x = 0.52,
    y = 0.825,
    size = 10,
    color = "black"
  )

# Display the final plot
final_plot

# Save the final plot as PDF and PNG
save_ggplot(
  plot = final_plot,
  path = here("figures/plot_glmm_emmeans.pdf"),
  ncol = 2,
  height = 120
)

save_ggplot(
  plot = final_plot,
  path = here("figures/plot_glmm_condition_x_order_emmeans.pdf"),
  ncol = 2,
  height = 120
)