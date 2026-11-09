# shared species contrast code for 03_size_bd_unified_analysis.qmd, 05_species_regression.qmd
# and th_sensitivity_analysis.Rmd. requires tidyverse, brms and tidybayes to be loaded.
#
# species x life stage linear predictors (hu, mu) at each species' median subadult weather,
# including taxon-specific weather slopes (fixed + r_taxon_capture) and excluding population,
# year and climate zone effects. subadult - adult contrasts do not depend on the weather values
# chosen, since weather slopes do not vary by life stage.

# most common value
mmode = function(x) {
  ux = unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# posterior summary with median and equal-tailed (quantile) intervals.
# pd: probability of direction. support: "strong" if 0 is outside the 95% interval
# (pd > 0.975), "weak" if 0 is outside the 50% but inside the 95% interval
# (0.75 < pd <= 0.975), otherwise "none". net_effect = 2 * P(> 0) - 1.
f_draw_agg = function(draws, group_vars, agg_var) {

  draws |>
    group_by(across(all_of(group_vars))) |>
    reframe(
      "{agg_var}_lower_50" := tidybayes::median_qi(.data[[agg_var]], .width = 0.50)[, 2],
      "{agg_var}_upper_50" := tidybayes::median_qi(.data[[agg_var]], .width = 0.50)[, 3],
      "{agg_var}_lower_95" := tidybayes::median_qi(.data[[agg_var]], .width = 0.95)[, 2],
      "{agg_var}_upper_95" := tidybayes::median_qi(.data[[agg_var]], .width = 0.95)[, 3],
      "{agg_var}_median"   := median(.data[[agg_var]]),
      "{agg_var}_net_effect" := (sum(.data[[agg_var]] > 0) / n()) * 2 - 1,
      "{agg_var}_pd" := max(mean(.data[[agg_var]] > 0), mean(.data[[agg_var]] < 0))
    ) |>
    mutate("{agg_var}_support" := case_when(
      .data[[paste0(agg_var, "_lower_95")]] > 0 | .data[[paste0(agg_var, "_upper_95")]] < 0 ~ "strong",
      .data[[paste0(agg_var, "_lower_50")]] > 0 | .data[[paste0(agg_var, "_upper_50")]] < 0 ~ "weak",
      .default = "none"))
}

# taxon-specific weather slopes, so each species' linpred at its median weather uses its own
# slopes rather than the across-species slopes
re_spp_env = ~ (0 + precip_mm_10_z + temp_c_30_z + I(temp_c_30_z^2) | taxon_capture)

# model response column (e.g. bd_load_co100_density)
model_response = function(model) {
  as.character(model$formula$formula[[2]])
}

# newdata: each species x life stage at the species' median subadult weather
spp_lss_newdata = function(model) {
  env_subadult = model$data |>
    filter(life_stage_simple == "subadult") |>
    group_by(taxon_capture) |>
    summarize(precip_mm_10_z = median(precip_mm_10_z),
              temp_c_30_z = median(temp_c_30_z),
              kg_code_3 = mmode(kg_code_3),
              .groups = "drop")

  model$data |>
    distinct(taxon_capture, life_stage_simple) |>
    left_join(env_subadult, by = "taxon_capture")
}

# mu contrast is only estimable where both life stages have Bd-positive loads; otherwise one
# life stage's mu comes from the prior and model structure alone
mu_contrast_taxa = function(model) {
  resp = model_response(model)

  model$data |>
    group_by(taxon_capture, life_stage_simple) |>
    summarise(n_pos = sum(.data[[resp]] > 0), .groups = "drop") |>
    group_by(taxon_capture) |>
    filter(n_distinct(life_stage_simple) == 2, all(n_pos > 0)) |>
    distinct(taxon_capture) |>
    pull(taxon_capture)
}

# hu and mu linear predictor draws per species x life stage (absolute levels: within-species
# use only), with prevalence = 1 - inv_logit(hu)
spp_lss_linpred_draws = function(model, newdata = spp_lss_newdata(model)) {
  f_dpar = function(dp) {
    model |>
      linpred_draws(newdata = newdata, dpar = dp, re_formula = re_spp_env) |>
      ungroup() |>
      select(taxon_capture, life_stage_simple, .draw, all_of(dp))
  }

  f_dpar("hu") |>
    left_join(f_dpar("mu"), by = c("taxon_capture", "life_stage_simple", ".draw")) |>
    mutate(prevalence = 1 - plogis(hu), .after = hu)
}

# subadult - adult contrast draws per species and draw. mu_contrast is NA for taxa without an
# estimable mu contrast (taxa_mu)
spp_contrast_draws = function(spp_lss_draws, taxa_mu) {
  spp_lss_draws |>
    pivot_wider(id_cols = c(taxon_capture, .draw),
                names_from = life_stage_simple,
                values_from = c(hu, mu)) |>
    transmute(taxon_capture,
              .draw,
              contrast = "subadult - adult",
              hu_contrast = hu_subadult - hu_adult,
              mu_contrast = if_else(taxon_capture %in% taxa_mu, mu_subadult - mu_adult, NA_real_))
}

# contrast summaries per species (one row per species); NA mu columns where not estimable
spp_contrast_agg = function(contrast_draws) {
  hu_agg = f_draw_agg(contrast_draws, c("taxon_capture", "contrast"), "hu_contrast")
  mu_agg = f_draw_agg(contrast_draws |> filter(!is.na(mu_contrast)),
                      c("taxon_capture", "contrast"), "mu_contrast")

  hu_agg |>
    left_join(mu_agg, by = c("taxon_capture", "contrast"))
}

# all of the above for one model: draws, contrast draws and summary
spp_contrasts = function(model) {
  taxa_mu = mu_contrast_taxa(model)
  lss_draws = spp_lss_linpred_draws(model)
  contrast_draws = spp_contrast_draws(lss_draws, taxa_mu)

  list(mu_resp = model_response(model),
       taxa_mu_contrast = taxa_mu,
       spp_lss_draws = lss_draws,
       contrast_draws = contrast_draws,
       combined_agg = spp_contrast_agg(contrast_draws))
}
