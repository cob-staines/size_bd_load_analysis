# formula, prior and data-prep builders for the stage 1 contrast models (sourced by
# 02_contrast_modeling.qmd; copied verbatim into r-brms-contained/size_bd_contrast_modeling.R).
# keep the two in sync. model description: README.md ("Stage 1 model").
#
# requires: tidyverse, brms, and contrast_data_functions.R (add_contrast_groups, response_col)

# ---- data ------------------------------------------------------------------------------------
# one species' swabs with stage_c (-0.5 adult / +0.5 subadult), pop_year and pop_year_event
# (14-day events). droplevels so random-effect terms have no empty levels.
prep_contrast_data = function(brms_data, species_name) {
  brms_data %>%
    filter(taxon_capture == species_name) %>%
    add_contrast_groups() %>%
    droplevels()
}

# ---- formula ---------------------------------------------------------------------------------
# random-effect terms by tier. intercepts vary at every level in every tier (they absorb site,
# year and event-wide swings in Bd, e.g. outbreaks, whether or not both stages were swabbed);
# contrast slopes (stage_c) only at the levels the data support:
#   A: population + site-year + event      B: population + site-year
#   C: population                          none (mu only): no stage term
contrast_re_terms = function(tier) {
  switch(tier,
         A = "(1 + stage_c | population) + (1 + stage_c | pop_year) + (1 + stage_c | pop_year_event)",
         B = "(1 + stage_c | population) + (1 + stage_c | pop_year) + (1 | pop_year_event)",
         C = "(1 + stage_c | population) + (1 | pop_year) + (1 | pop_year_event)",
         none = "(1 | population) + (1 | pop_year) + (1 | pop_year_event)",
         stop("unknown tier: ", tier))
}

contrast_fixed_terms = function(tier) if (tier == "none") "1" else "1 + stage_c"

# hurdle lognormal: mu = mean log load among positives (load part), hu = probability of a zero
# (prevalence part, logit scale), sigma = residual sd of log load, varying by population.
# tiers are set separately for mu (tier_mu, load counts) and hu (tier_hu, prevalence counts).
contrast_formula = function(tier_hu, tier_mu, response = response_col) {
  bf(as.formula(paste(response, "~", contrast_fixed_terms(tier_mu), "+", contrast_re_terms(tier_mu))),
     as.formula(paste("hu ~", contrast_fixed_terms(tier_hu), "+", contrast_re_terms(tier_hu))),
     sigma ~ 1 + (1 | population))
}

# ---- priors ----------------------------------------------------------------------------------
# groups whose contrast (stage_c) varies, by tier (matches contrast_re_terms)
contrast_slope_groups = function(tier) {
  switch(tier,
         A = c("population", "pop_year", "pop_year_event"),
         B = c("population", "pop_year"),
         C = "population",
         none = character(0),
         stop("unknown tier: ", tier))
}

# contrast sd priors, one per group with a stage_c slope (brms needs the group named for a
# coefficient-level sd prior). dpar = "" is the mu part.
contrast_sd_priors = function(tier, dpar = "") {
  groups = contrast_slope_groups(tier)
  if (length(groups) == 0) return(NULL)
  do.call(c, lapply(groups, function(g) {
    set_prior("exponential(2)", class = "sd", coef = "stage_c", group = g, dpar = dpar)
  }))
}

# fixed effects and intercept sds as in the unified / seasonal models. the contrast sds get
# exponential(2): tighter than the intercept sds (exponential(1)), so a consistent contrast is the
# default and variation in it must be earned from the data. prior median 0.35, 95% below 1.5
# (log / logit units). brms errors on priors that match no parameter, so stage terms only get
# priors when the part has them (tier != "none").
contrast_priors = function(tier_hu, tier_mu) {
  p = c(
    # mu: log load density among positives
    prior(normal(5, 4), class = "Intercept"),
    prior(exponential(1), class = "sd"),
    # hu: logit of non-detection
    prior(normal(0, 2), class = "Intercept", dpar = "hu"),
    prior(normal(0, 2), class = "b", dpar = "hu"),
    prior(exponential(1), class = "sd", dpar = "hu"),
    contrast_sd_priors(tier_hu, dpar = "hu"),
    # sigma: log residual sd of log load
    prior(normal(0, 1), class = "Intercept", dpar = "sigma"),
    prior(exponential(1), class = "sd", dpar = "sigma"),
    # intercept-contrast correlations (every tier of hu has at least one)
    prior(lkj(2), class = "cor")
  )
  if (tier_mu != "none") {
    p = c(p,
          prior(normal(0, 3), class = "b"),
          contrast_sd_priors(tier_mu))
  }
  p
}

# contrast sd prior quantiles, for the prior vs posterior check
contrast_sd_prior_quantile = function(p) qexp(p, rate = 2)
