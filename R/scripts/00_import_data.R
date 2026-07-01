"""
This script processes raw behavioral, categorizing trials
based on recall performance and preparing datasets for statistical 
analysis and visualization.
It handles data cleaning, reshaping, and aggregation to compute memory metrics 
(e.g., learned, forgotten, memorised) for each subject, trial, and condition 
(Rest vs. Distraction).

Key Outputs:
- df_diff: Trial-level data with initial/final scores and memory categories.
- df_counts: Counts of memory categories per trial.
- df_performance: Binary performance metrics (1 = memorised, 0 = forgotten).
- Saved files: .rda and .csv files for further analysis.

Dependencies: dplyr, tidyr, here, stringr, report
"""

# Load required packages
if (!requireNamespace("pacman")) install.packages("pacman")
pacman::p_load(dplyr, tidyr, here, stringr)  # Data manipulation and file path handling

# --- DATA LOADING AND CLEANING ---

df <- 
  read.csv(here("data/data_source.csv"), sep = ";") |> 
  select(-c(image_object_id, image_landscape_id)) |> 
  mutate(
    across(c(where(is.character), "order"), as.factor),
    condition = factor(condition, levels = c("Rest", "Distraction")),
    phase = factor(phase, levels = c("initial", "final"))
  ) 

# Load raw data and convert categorical variables to factors with explicit levels
df_diff <- read.csv(here("data/data_source.csv"), sep = ";") |>
  mutate(
    across(c(where(is.character), "order"), as.factor),  # Convert character columns to factors
    condition = factor(condition, levels = c("Rest", "Distraction")),  # Ensure Rest comes before Distraction
    phase = factor(phase, levels = c("initial", "final"))  # Ensure initial comes before final
  ) |>
  # Reshape data from long to wide format: one column per phase (initial/final)
  pivot_wider(
    id_cols = c(subject, trial_number, condition, order, encoding_phase, rating),
    names_from = phase,
    values_from = score,
    names_glue = "score_{phase}"  # Creates columns: score_initial, score_final
  ) |>
  # Clean trial_number to extract numeric part (e.g., "1bis" -> "1")
  mutate(trial_base = gsub("[^0-9]", "", trial_number)) |>
  # Group by trial and compute mean scores (handles duplicates)
  group_by(subject, trial_base, condition, order, encoding_phase, rating) |>
  summarise(
    score_initial = mean(score_initial, na.rm = TRUE),
    score_final = mean(score_final, na.rm = TRUE),
    .groups = "drop"  # Remove grouping after summarization
  ) |>
  rename(trial_number = trial_base)  # Rename for consistency

# --- MEMORY CATEGORY CLASSIFICATION ---
# Classify trials into memory categories based on initial/final recall
df_diff$score_diff <- with(
  df_diff,
  factor(
    ifelse(score_initial == 0 & score_final == 0, "never_learned",  # Not recalled in either phase
           ifelse(score_initial == 0 & score_final == 1, "learned",   # Recalled only in final phase
                  ifelse(score_initial == 1 & score_final == 0, "forgotten",  # Recalled only in initial phase
                         ifelse(score_initial == 1 & score_final == 1, "memorised", NA)))),  # Recalled in both phases
    levels = c("never_learned", "learned", "forgotten", "memorised")  # Ordered factor levels
  )
)

# --- COUNT MEMORY CATEGORIES PER TRIAL ---
# Count occurrences of each memory category (score_diff) per trial
df_counts <- df_diff %>%
  group_by(subject, order, condition, trial_number, rating, encoding_phase) %>%
  count(score_diff) %>%        # Count occurrences of each category
  pivot_wider(
    names_from = score_diff,   # Spread into one column per category
    values_from = n,           # Fill with counts
    values_fill = 0            # Replace NA with 0
  ) %>%
  ungroup()  # Remove grouping

# --- COMPUTE PERFORMANCE METRIC ---
# Binary performance: 1 = memorised, 0 = forgotten, NA = never learned or newly learned
df_counts$performance <- with(df_counts,
                              ifelse(memorised == 1 & forgotten == 0, 1,  # Memorised: recalled in both phases
                                     ifelse(memorised == 0 & forgotten == 1, 0,  # Forgotten: recalled initially but not finally
                                            ifelse(memorised == 0 & forgotten == 0, NA, NA)))  # Exclude never learned or newly learned
)

# Add trial counts per subject/condition and remove NA rows
df_performance <- na.omit(df_counts)
df_performance <- df_performance %>%
  group_by(subject, condition) %>%
  mutate(number_of_trials = n()) %>%  # Count trials per subject/condition
  ungroup()

# --- SAVE PROCESSED DATA ---
# Save datasets for further analysis
save(df, file = here("data/data_raw.rda"))
save(df_performance, file = here("data/data_raw_df_performance.rda"))  # Performance metrics
write.csv(df_performance, here("data/data_memory_learned_trials.csv"))  # CSV for sharing