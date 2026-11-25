# model check functions for the seasonal brms models (sourced by 02_season_all_species.qmd).
# these generalize the checks written inline in 02_season_modeling.qmd: the season axis is
# "months since each species' seasonal origin" (season_m), so periods are counted from the
# origin rather than from day of year. see 02_season_modeling.qmd for the longer explanations
# of what each check means.
#
# requires: tidyverse, brms, patchwork loaded by the calling script

# ---- convergence ---------------------------------------------------------------------------
# one row per fit. targets: max_rhat < 1.01; min_ess_bulk / tail > ~400; n_divergent 0 (a few
# tolerable); n_max_treedepth is an efficiency warning only; min_ebfmi > 0.3
conv_check = function(fit, max_treedepth = 12) {
  np = nuts_params(fit)
  dr = posterior::summarise_draws(posterior::as_draws_df(fit),
                                  posterior::default_convergence_measures())
  tibble(n_obs = nrow(fit$data),
         max_rhat = max(dr$rhat, na.rm = TRUE),
         worst_rhat_par = dr$variable[which.max(dr$rhat)],
         min_ess_bulk = min(dr$ess_bulk, na.rm = TRUE),
         min_ess_tail = min(dr$ess_tail, na.rm = TRUE),
         n_divergent = sum(np$Value[np$Parameter == "divergent__"]),
         n_max_treedepth = sum(np$Value[np$Parameter == "treedepth__"] >= max_treedepth),
         # brms converts cmdstanr output to a stanfit, so rstan can read the sampler energy
         min_ebfmi = min(rstan::get_bfmi(fit$fit)))
}

# ---- loo / pareto k --------------------------------------------------------------------------
# k > 0.7: leave-one-out approximation unreliable for that (influential) observation
loo_check = function(fit) {
  if (is.null(fit$criteria$loo)) {
    return(tibble(n_obs = nrow(fit$data), n_k_gt_0.7 = NA_integer_,
                  pct_k_gt_0.7 = NA_real_, max_k = NA_real_))
  }
  k = loo::pareto_k_values(fit$criteria$loo)
  tibble(n_obs = length(k),
         n_k_gt_0.7 = sum(k > 0.7),
         pct_k_gt_0.7 = round(100 * mean(k > 0.7), 2),
         max_k = max(k))
}

# ---- posterior predictive checks by stage ----------------------------------------------------
# prop_zero -> hurdle (hu) part; mean / sd of log positive load -> mu / sigma
prop_zero = function(y) mean(y == 0)
mean_log_pos = function(y) mean(log(y[y > 0]))
sd_log_pos = function(y) sd(log(y[y > 0]))

# group: the stage column stored in the fit's data ("life_stage_simple" in the seasonal models;
# "stage_c" (-0.5 adult / 0.5 subadult) in the contrast models, whose data hold only model terms)
ppc_stage = function(fit, spp, ndraws = 500, group = "life_stage_simple") {
  stats = list(prop_zero = prop_zero, mean_log_pos = mean_log_pos, sd_log_pos = sd_log_pos)
  plots = imap(stats, function(f, nm) {
    pp_check(fit, type = "stat_grouped", stat = f, group = group, ndraws = ndraws) +
      labs(title = if (nm == "prop_zero") str_to_sentence(gsub("_", " ", spp)) else NULL,
           subtitle = nm) +
      theme(legend.position = "none")
  })
  wrap_plots(plots, nrow = 1)
}

# ---- seasonal posterior predictive check ------------------------------------------------------
# observed prevalence and mean log load (positives) per period and stage, against the same
# summaries from replicated datasets. periods are period_weeks long, counted from the species'
# seasonal origin (season_m = months since origin). periods kept with >= min_n swabs
# (prevalence) or >= min_pos positives (load).
season_ppc = function(fit, ndraws = 200, period_weeks = 2, min_n = 5, min_pos = 3) {
  d = fit$data %>%
    mutate(weeks_since_origin = season_m * 30.44 / 7,
           period = floor(weeks_since_origin / period_weeks),
           grp = paste(life_stage_simple, period, sep = "|"),
           # one factor fixes the group order for rowsum(), table() and arrange() alike
           grp = factor(grp, levels = sort(unique(grp))))

  yrep = posterior_predict(fit, ndraws = ndraws)   # draws x observations
  pos = t(yrep > 0)                                # observations x draws
  logy = t(ifelse(yrep > 0, log(yrep), 0))

  # rowsum() sums rows within each group -> (groups x draws)
  n_grp = as.vector(table(d$grp))
  n_pos = rowsum(pos * 1, d$grp)
  sum_log = rowsum(logy, d$grp)

  prev_rep = n_pos / n_grp
  load_rep = sum_log / n_pos       # NaN where a replicate has no positives in that group

  q = function(m) {
    t(apply(m, 1, quantile, probs = c(0.025, 0.25, 0.5, 0.75, 0.975), na.rm = TRUE))
  }
  q_names = ~ c("l95", "l50", "med", "u50", "u95")

  obs = d %>%
    group_by(grp, life_stage_simple, period) %>%
    summarise(n = n(),
              n_pos_obs = sum(bd_load_co100_density > 0),
              prev_obs = mean(bd_load_co100_density > 0),
              load_obs = if (n_pos_obs > 0) mean(log(bd_load_co100_density[bd_load_co100_density > 0])) else NA_real_,
              .groups = "drop") %>%
    arrange(grp)

  bind_rows(
    bind_cols(obs, as_tibble(q(prev_rep), .name_repair = q_names)) %>%
      filter(n >= min_n) %>%
      mutate(metric = "prevalence", observed = prev_obs),
    bind_cols(obs, as_tibble(q(load_rep), .name_repair = q_names)) %>%
      filter(n_pos_obs >= min_pos) %>%
      mutate(metric = "log load (positives)", observed = load_obs)
  ) %>%
    mutate(period_mid_week = (period + 0.5) * period_weeks)
}

plot_season_ppc = function(ppc, metric_name, ncol = 6) {
  ppc %>%
    filter(metric == metric_name) %>%
    mutate(species = str_to_sentence(gsub("_", " ", taxon_capture))) %>%
    ggplot(aes(x = period_mid_week, color = life_stage_simple, fill = life_stage_simple)) +
    geom_linerange(aes(ymin = l95, ymax = u95), linewidth = 0.5, alpha = 0.5,
                   position = position_dodge(width = 0.8)) +
    geom_linerange(aes(ymin = l50, ymax = u50), linewidth = 1.5, alpha = 0.6,
                   position = position_dodge(width = 0.8)) +
    geom_point(aes(y = observed), shape = 21, color = "black", size = 1.5,
               position = position_dodge(width = 0.8)) +
    facet_wrap(~ species, scales = "free", ncol = ncol) +
    labs(x = "Weeks since seasonal origin", y = metric_name, color = "stage", fill = "stage",
         title = paste("Seasonal predictive check:", metric_name),
         subtitle = "points = observed per period; bars = 50% / 95% predictive intervals") +
    theme_classic() +
    theme(strip.text = element_text(face = "italic"))
}

# share of observed period summaries inside their 95% predictive interval (expect ~0.95)
season_ppc_coverage = function(ppc) {
  ppc %>%
    group_by(taxon_capture, metric) %>%
    summarise(n_periods = n(),
              coverage_95 = mean(observed >= l95 & observed <= u95),
              .groups = "drop")
}

# ---- smooth sds: posterior vs prior ----------------------------------------------------------
# one row per smooth sd. "shared" = the seasonal curve common to both stages; "difference" =
# the subadult - adult difference curve (the ordered-factor smooth, named with "stage_o")
sds_summary = function(fit) {
  posterior_summary(fit, variable = "^sds_", regex = TRUE, probs = c(0.025, 0.5, 0.975)) %>%
    as_tibble(rownames = "variable") %>%
    mutate(part = if_else(str_detect(variable, "^sds_hu_"), "hu", "mu"),
           smooth = if_else(str_detect(variable, "stage_o"), "difference", "shared"))
}

# quantiles of a half student-t(df, 0, scale) prior: quantile p of |t| = scale * qt((1 + p) / 2, df)
half_t_quantile = function(p, scale, df = 3) scale * qt((1 + p) / 2, df)
