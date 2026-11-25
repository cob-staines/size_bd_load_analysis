# formula, prior and data-prep builders for the stage 1 contrast models (sourced by
# 02_contrast_modeling.qmd and 03_contrast_analysis.Rmd; copied verbatim into
# r-brms-contained/size_bd_contrast_modeling.R). keep the two in sync. model description:
# README.md ("Stage 1 model", "Model structure rules").
#
# the builders take one job: a row of 01_contrast_jobs.csv (or a list) with
#   tier_hu, tier_mu        "A" / "B" / "C" / "fixed" (+ "none" for mu)
#   hu_has_pop_year, hu_has_event, mu_has_pop_year, mu_has_event
#                           whether that part has the site-year / event level (R2)
#   sigma_by_population     sigma ~ (1 | population) if TRUE, sigma ~ 1 if FALSE (R3)
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

# ---- structure of one part -------------------------------------------------------------------
# grouping levels present in a part: population always; site-year and event unless dropped (R2)
contrast_levels = function(has_pop_year = TRUE, has_event = TRUE) {
  c("population", if (has_pop_year) "pop_year", if (has_event) "pop_year_event")
}

# levels whose contrast (stage_c) varies, by tier, among the levels present:
#   A: population + site-year + event      B: population + site-year
#   C: population                          fixed / none: no contrast sd
contrast_slope_groups = function(tier, has_pop_year = TRUE, has_event = TRUE) {
  by_tier = switch(tier,
                   A = c("population", "pop_year", "pop_year_event"),
                   B = c("population", "pop_year"),
                   C = "population",
                   fixed = character(0),
                   none = character(0),
                   stop("unknown tier: ", tier))
  intersect(by_tier, contrast_levels(has_pop_year, has_event))
}

# random-effect terms of a part. intercepts vary at every level present (they absorb site, year
# and event-wide swings in Bd, e.g. outbreaks, whether or not both stages were swabbed);
# contrast slopes only at the levels the tier allows.
contrast_re_terms = function(tier, has_pop_year = TRUE, has_event = TRUE) {
  slopes = contrast_slope_groups(tier, has_pop_year, has_event)
  terms = map_chr(contrast_levels(has_pop_year, has_event), function(g) {
    if (g %in% slopes) paste0("(1 + stage_c | ", g, ")") else paste0("(1 | ", g, ")")
  })
  paste(terms, collapse = " + ")
}

# fixed part: a mean contrast unless the part has no stage term ("none")
contrast_fixed_terms = function(tier) if (tier == "none") "1" else "1 + stage_c"

# ---- formula ---------------------------------------------------------------------------------
# hurdle lognormal: mu = mean log load among positives (load part), hu = probability of a zero
# (prevalence part, logit scale), sigma = residual sd of log load among positives (varying by
# population where the data support it, R3). tiers and levels are set separately for mu (load
# counts, positive swabs) and hu (prevalence counts, all swabs).
contrast_formula = function(job, response = response_col) {
  mu_rhs = paste(contrast_fixed_terms(job$tier_mu), "+",
                 contrast_re_terms(job$tier_mu, job$mu_has_pop_year, job$mu_has_event))
  hu_rhs = paste(contrast_fixed_terms(job$tier_hu), "+",
                 contrast_re_terms(job$tier_hu, job$hu_has_pop_year, job$hu_has_event))
  sigma_rhs = if (job$sigma_by_population) "1 + (1 | population)" else "1"

  bf(as.formula(paste(response, "~", mu_rhs)),
     as.formula(paste("hu ~", hu_rhs)),
     as.formula(paste("sigma ~", sigma_rhs)))
}

# ---- priors ----------------------------------------------------------------------------------
# contrast sd priors, one per group with a stage_c slope (brms needs the group named for a
# coefficient-level sd prior). dpar = "" is the mu part.
contrast_sd_priors = function(groups, dpar = "") {
  if (length(groups) == 0) return(NULL)
  do.call(c, lapply(groups, function(g) {
    set_prior("exponential(2)", class = "sd", coef = "stage_c", group = g, dpar = dpar)
  }))
}

# fixed effects and intercept sds as in the unified / seasonal models. the contrast sds get
# exponential(2): tighter than the intercept sds (exponential(1)), so a consistent contrast is the
# default and variation in it must be earned from the data. prior median 0.35, 95% below 1.5
# (log / logit units). brms errors on priors that match no parameter, so each prior is only
# added when its parameter exists (stage terms, contrast sds, correlations, sigma sd).
contrast_priors = function(job) {
  hu_slopes = contrast_slope_groups(job$tier_hu, job$hu_has_pop_year, job$hu_has_event)
  mu_slopes = contrast_slope_groups(job$tier_mu, job$mu_has_pop_year, job$mu_has_event)

  p = c(
    # mu: log load density among positives
    prior(normal(5, 4), class = "Intercept"),
    prior(exponential(1), class = "sd"),
    # hu: logit of non-detection (hu always has a stage term)
    prior(normal(0, 2), class = "Intercept", dpar = "hu"),
    prior(normal(0, 2), class = "b", dpar = "hu"),
    prior(exponential(1), class = "sd", dpar = "hu"),
    contrast_sd_priors(hu_slopes, dpar = "hu"),
    # sigma: log residual sd of log load
    prior(normal(0, 1), class = "Intercept", dpar = "sigma")
  )
  if (job$tier_mu != "none") p = c(p, prior(normal(0, 3), class = "b"))
  p = c(p, contrast_sd_priors(mu_slopes))
  if (job$sigma_by_population) p = c(p, prior(exponential(1), class = "sd", dpar = "sigma"))
  # intercept-contrast correlations exist only where a contrast slope does
  if (length(hu_slopes) + length(mu_slopes) > 0) p = c(p, prior(lkj(2), class = "cor"))
  p
}

# contrast sd prior quantiles, for the prior vs posterior check
contrast_sd_prior_quantile = function(p) qexp(p, rate = 2)
