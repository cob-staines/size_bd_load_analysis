# formula and prior builders for the GAM (sourced by 06_gam_modeling.qmd; copied verbatim into
# r-brms-contained/size_bd_contrast_gam_modeling.R). keep the two in sync. model description:
# README.md ("Downstream models" -> "GAM").
#
# the builders take one species row of 06_gam_species.csv (or a list) with the stage 1 structure
# flags: hu_has_pop_year, hu_has_event, mu_has_pop_year, mu_has_event, sigma_by_population.
#
# requires: tidyverse, brms, and response_col (contrast_data_functions.R)

# ---- formula ---------------------------------------------------------------------------------
# per part (mu = log load among positives, hu = non-detection): one SVL smooth + the stage 1
# nuisance intercepts of that part.
#
# size: metamorph baseline (the intercept) + a thin plate smooth over subadults and adults.
# a numeric-by smooth is not centered, so it carries the post-metamorph offset itself; adding
# is_metamorph as well is rank deficient. species without metamorphs get a plain (centered)
# s(svl_mm), and the intercept is then the mean level.
#
# nuisance: (1 | population), plus (1 | pop_year) and (1 | pop_year_event) where stage 1 kept
# them for that part (R2 flags); sigma varies by population where stage 1's does (R3).
gam_formula = function(sp_row, model_data, response = response_col) {
  svl_term = if (any(model_data$life_stage_simple == "metamorph")) {
    "s(svl_mm, by = is_not_metamorph)"
  } else {
    "s(svl_mm)"
  }
  part_rhs = function(has_pop_year, has_event) {
    groups = c("population", if (has_pop_year) "pop_year", if (has_event) "pop_year_event")
    paste(c("1", svl_term, paste0("(1 | ", groups, ")")), collapse = " + ")
  }
  mu_rhs = part_rhs(sp_row$mu_has_pop_year, sp_row$mu_has_event)
  hu_rhs = part_rhs(sp_row$hu_has_pop_year, sp_row$hu_has_event)
  sigma_rhs = if (sp_row$sigma_by_population) "1 + (1 | population)" else "1"

  bf(as.formula(paste(response, "~", mu_rhs)),
     as.formula(paste("hu ~", hu_rhs)),
     as.formula(paste("sigma ~", sigma_rhs)))
}

# ---- priors ----------------------------------------------------------------------------------
# the stage 1 nuisance priors (contrast_priors() in contrast_model_functions.R, without the stage
# and contrast-sd terms): intercepts, intercept sds, sigma. the smooth terms keep brms' defaults
# (flat on its linear part `b` (class bs in the Stan code), student_t(3, 0, 2.5) on its
# wiggliness `sds`), as in the earlier GAM fits.
# note: with a metamorph baseline, the Intercept is the metamorph level; without metamorphs it
# is the mean level. normal(5, 4) / normal(0, 2) are wide for either.
gam_priors = function(sp_row) {
  p = c(
    # mu: log load density among positives
    prior(normal(5, 4), class = "Intercept"),
    prior(exponential(1), class = "sd"),
    # hu: logit of non-detection
    prior(normal(0, 2), class = "Intercept", dpar = "hu"),
    prior(exponential(1), class = "sd", dpar = "hu"),
    # sigma: log residual sd of log load
    prior(normal(0, 1), class = "Intercept", dpar = "sigma")
  )
  if (sp_row$sigma_by_population) p = c(p, prior(exponential(1), class = "sd", dpar = "sigma"))
  p
}

# ---- run settings ----------------------------------------------------------------------------
# shared by the local fits (06) and the cluster script, so both give the same model and draws:
# 4 chains x (2000 warmup + 2000 sampling) = 8000 draws
gam_n_chains = 4
gam_n_iter = 4000
gam_n_warmup = 2000
gam_adapt_delta = 0.95
gam_max_treedepth = 12
gam_seed = 124

# model file name (without .rds) for one species: brm(file = ) adds .rds
gam_model_name = function(sp) paste0("gam_m4_", sp)
