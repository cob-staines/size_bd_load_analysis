# shared constants and data helpers for size_bd_contrast (sourced by 01_contrast_tiers.Rmd and
# 02_contrast_modeling.qmd; mirrored in r-brms-contained/size_bd_contrast_modeling.R). these
# values are the source of truth; keep the README front matter in sync.
#
# requires: tidyverse (dplyr, tidyr, lubridate) loaded by the calling script

# ---- constants -----------------------------------------------------------------------------
response_col = "bd_load_co100_density"

# event = a run of survey dates (of the species) at one site with no gap > event_max_gap days;
# runs longer than event_max_span days are split at their longest gap until each piece fits
event_max_gap = 5
event_max_span = 14
min_swabs_unit = 5         # co-sampled unit: >= 5 swabs of each stage ...
min_pos_unit = 1           # ... and, for load, >= 1 positive swab of each stage
min_n_stage = 20           # inclusion: >= 20 swabs of each stage ...
min_co_pops = 1            # ... and >= 1 co-sampled population
# contrast levels rule (README "Contrast levels"): the contrast varies at a level with at least
# this many co-sampled units there, and only if it also varies at the level above
min_co_pops_slope = 3        # site-level contrast sd: >= 3 co-sampled populations
min_co_popyears_slope = 5    # site-year-level contrast sd: >= 5 co-sampled site-years
min_co_events_slope = 10     # event-level contrast sd: >= 10 co-sampled events
min_pct_swabs_co_pop = 25    # no contrast sd at any level if < 25% of swabs are in co-sampled
                             # populations (between-population: tier D)
# time-segregated flag (caveat only): fewer co-sampled events than half the number of
# co-sampled site-years (so >= 1 co-sampled site-year). no minimum number of site-years
# (2026-11-26; was >= 3): the minimum left out the most segregated species (1-2 co-sampled
# site-years, 0 co-sampled events, e.g. l. blairi)
time_seg_max_event_ratio = 0.5
# nuisance structure (2026-11-25, after the first cluster fits: divergences and poor mixing
# traced to terms the data could not identify; see README "Nuisance structure")
min_children_per_parent = 1.1  # R2: keep the site-year (event) intercept only if there are >= 1.1
                               #     site-years per site (events per site-year)
min_pos_sigma_pop = 20     # R3: sigma varies by population only with >= 3 populations ...
min_pops_sigma = 3         #     ... that have >= 20 positive swabs each (10 until round 3)

# ---- grouping columns ------------------------------------------------------------------------
# stage_c: -0.5 adult, +0.5 subadult. the intercept is then the midpoint of the two stages and
# the stage_c coefficient the full subadult - adult contrast; a group's stage_c sd is how much
# the contrast varies at that level.
# pop_year / pop_year_event: explicit ids (rather than population:year:... in the formula) so
# parameter names stay readable.
#
# events follow the survey design rather than the calendar: consecutive-day visits stay together,
# weekly revisits (gaps of 7 days) are separate events, and an intensive resurvey period longer
# than 2 weeks is split where the surveys paused longest. built per species x population from
# the species' own swab dates (each model sees one species). the year is part of the id, so an
# event stays inside its site-year (none crossed the new year in the 2026-11 data).

# split one run of sorted dates (numeric) at its longest gap, recursively, until every piece
# spans <= max_span days. returns piece numbers 1, 2, ...
split_long_run = function(d, max_span) {
  if (length(d) < 2 || max(d) - min(d) + 1 <= max_span) return(rep(1L, length(d)))
  j = which.max(diff(d))
  a = split_long_run(d[1:j], max_span)
  b = split_long_run(d[(j + 1):length(d)], max_span)
  c(a, b + max(a))
}

# event number of each date in a sorted vector of distinct dates (numeric) at one site: a new run
# starts after a gap > max_gap days; long runs are then split by split_long_run
event_ids = function(d, max_gap = event_max_gap, max_span = event_max_span) {
  run = cumsum(c(TRUE, diff(d) > max_gap))
  out = integer(length(d))
  k = 0L
  for (r in unique(run)) {
    i = which(run == r)
    piece = split_long_run(d[i], max_span)
    out[i] = piece + k
    k = k + max(piece)
  }
  out
}

add_contrast_groups = function(df) {
  events = df %>%
    distinct(taxon_capture, population, date) %>%
    arrange(taxon_capture, population, date) %>%
    group_by(taxon_capture, population) %>%
    mutate(event_k = event_ids(as.numeric(date))) %>%
    ungroup()

  df %>%
    left_join(events, by = c("taxon_capture", "population", "date")) %>%
    mutate(life_stage_simple = factor(life_stage_simple, levels = c("adult", "subadult")),
           stage_c = if_else(life_stage_simple == "subadult", 0.5, -0.5),
           pop_year = paste(population, year, sep = "_"),
           pop_year_event = paste(population, year, event_k, sep = "_"))
}

# ---- co-sampling counts ----------------------------------------------------------------------
# per unit (of the column `unit`, e.g. pop_year_event): swabs and positives per stage, and whether
# the unit is co-sampled for prevalence (co) and load (co_load). the one definition of a
# co-sampled unit, used for the contrast rule below and to mark group contrasts in
# 03_contrast_analysis.Rmd. expects both stages to occur somewhere in df.
co_unit_flags = function(df, unit, response = response_col,
                         min_swabs = min_swabs_unit, min_pos = min_pos_unit) {
  df %>%
    group_by(taxon_capture, unit_id = .data[[unit]], life_stage_simple) %>%
    summarise(n = n(), n_pos = sum(.data[[response]] > 0), .groups = "drop") %>%
    pivot_wider(names_from = life_stage_simple, values_from = c(n, n_pos), values_fill = 0) %>%
    mutate(co = n_adult >= min_swabs & n_subadult >= min_swabs,
           co_load = co & n_pos_adult >= min_pos & n_pos_subadult >= min_pos)
}

# per species: number of units (of the column `unit`), co-sampled units for prevalence and load,
# and the share of swabs that sit in co-sampled units
count_co_units = function(df, unit, response = response_col,
                          min_swabs = min_swabs_unit, min_pos = min_pos_unit) {
  co_unit_flags(df, unit, response, min_swabs, min_pos) %>%
    mutate(n_unit = n_adult + n_subadult) %>%
    group_by(taxon_capture) %>%
    summarise(n_units = n(),
              n_co = sum(co),
              n_co_load = sum(co_load),
              pct_swabs_in_co = round(100 * sum(n_unit[co]) / sum(n_unit), 1),
              # the same share with the load definition (>= 1 positive of each stage as well);
              # used for the load part of the predictive model's check fit (2026-11-26)
              pct_swabs_in_co_load = round(100 * sum(n_unit[co_load]) / sum(n_unit), 1),
              .groups = "drop")
}

# ---- contrast levels rule --------------------------------------------------------------------
# tier of one part (prevalence or load), from its co-sampled counts. the contrast varies at a
# level when that level has enough co-sampled units AND the contrast also varies at the level
# above (nested), so the tier is just the number of levels it varies at:
#   A: site + site-year + event      B: site + site-year      C: site      D: none (mean only)
# a slope also needs its intercept level to be in the model (R2: has_pop_year / has_event).
# D when the species is between-population (< min_pct_swabs_co_pop % of swabs in co-sampled
# populations: hundreds of single-stage sites, where intercept and contrast deviations cannot be
# told apart) or has < min_co_pops_slope co-sampled populations.
# "none" (load part with no positives in a stage) passes through.
contrast_tier = function(n_co_pops, n_co_popyears, n_co_events, pct_swabs_in_co_pop,
                         has_pop_year, has_event, none = FALSE) {
  site = n_co_pops >= min_co_pops_slope & pct_swabs_in_co_pop >= min_pct_swabs_co_pop
  site_year = site & has_pop_year & n_co_popyears >= min_co_popyears_slope
  event = site_year & has_event & n_co_events >= min_co_events_slope
  case_when(none ~ "none",
            event ~ "A",
            site_year ~ "B",
            site ~ "C",
            .default = "D")
}

# why a part is tier D (NA otherwise), for tables and labels
d_reason = function(tier, n_co_pops, pct_swabs_in_co_pop) {
  case_when(tier != "D" ~ NA_character_,
            pct_swabs_in_co_pop < min_pct_swabs_co_pop ~ "between-population",
            n_co_pops < min_co_pops_slope ~ "few co-sampled sites")
}

# ---- nuisance structure (R2, R3) -------------------------------------------------------------
# per species: which grouping levels each part can separate, and whether sigma can vary by
# population. computed per part from the swabs that inform it: all swabs for prevalence (hu),
# positive swabs for load (mu).
#   R2: a level is kept only if it splits its parent: >= min_children_per_parent site-years per
#       site (events per site-year). otherwise the two levels are (nearly) the same groups, only
#       their summed variance is identified, and the sampler wanders along the split (e.g.
#       n. viridescens: 1.02 site-years per site, 1.02 events per site-year).
#   R3: sigma ~ (1 | population) only with >= min_pops_sigma populations that have
#       >= min_pos_sigma_pop positive swabs; otherwise sigma ~ 1. a site's sigma (spread of log
#       load among positives) is learned only from that site's positive swabs. set at 10 from the
#       first fits; raised to 20 after round 3, where all four species with >= 3 sites at 10 but
#       not at 20 positives had an uninformed site-level sd of sigma (5% quantile ~0.01) and
#       s. flotator had 25 divergences in its funnel. every species whose sd was clearly
#       informed has >= 3 sites with >= 20 positives.
level_structure = function(df, response = response_col) {
  separable = function(x, prefix) {
    x %>%
      group_by(taxon_capture) %>%
      summarise(!!paste0(prefix, "_py_per_pop") := n_distinct(pop_year) / n_distinct(population),
                !!paste0(prefix, "_ev_per_py") := n_distinct(pop_year_event) / n_distinct(pop_year),
                .groups = "drop")
  }
  positives = df %>% filter(.data[[response]] > 0)

  separable(df, "hu") %>%
    left_join(separable(positives, "mu"), by = "taxon_capture") %>%
    left_join(positives %>%
                count(taxon_capture, population) %>%
                group_by(taxon_capture) %>%
                summarise(n_pops_pos_sigma = sum(n >= min_pos_sigma_pop), .groups = "drop"),
              by = "taxon_capture") %>%
    mutate(n_pops_pos_sigma = coalesce(n_pops_pos_sigma, 0L),
           hu_has_pop_year = hu_py_per_pop >= min_children_per_parent,
           hu_has_event = hu_ev_per_py >= min_children_per_parent,
           mu_has_pop_year = coalesce(mu_py_per_pop >= min_children_per_parent, FALSE),
           mu_has_event = coalesce(mu_ev_per_py >= min_children_per_parent, FALSE),
           sigma_by_population = n_pops_pos_sigma >= min_pops_sigma)
}

# ---- stage sampling conditions (descriptor, 2026-11-26) -------------------------------------
# not part of the block copied into r-brms-contained (the cluster does not use it).
# are subadults and adults swabbed under different weather? event intercepts absorb weather
# shared by both stages within an event, but where the stages are sampled under different
# conditions, single-stage units bring those differences into the mean contrast: it is then
# partly a "realized" contrast (each stage in the conditions it is found in) rather than a
# "same-condition" one (README "How to read the results").
# measure: standardized mean difference (smd), subadult - adult, in units of the pooled sd of the
# two stages (sqrt((var_subadult + var_adult) / 2), the covariate-balance convention). computed
# overall, within population (the stage difference among swabs of the same site: what the site
# intercepts do not remove; the main column) and within site-year (within-season timing only).
# within-group differences are fixed-effects estimates (the variable and the stage indicator both
# centered within group), so only groups holding both stages contribute.
# a descriptor, not a flag: first drafted as a flag at |smd| >= 0.25 (the usual balance cutoff),
# it marked 22 of 31 species (19 with the pooled sd used here). imbalance is a feature of these
# data (life-stage timing, habitat use), concentrated in the sparse species; the data-rich rana
# are balanced (|smd| <= 0.15).
stage_climate_vars = c(temp = "temp_c_30_day_mean", precip = "precip_mm_10_day_sum")

# within-group difference between stages (s = 1 subadult, 0 adult); NA if no group holds both
within_diff = function(x, s, g) {
  s_c = s - ave(s, g)
  if (sum(s_c^2) == 0) return(NA_real_)
  sum((x - ave(x, g)) * s_c) / sum(s_c^2)
}

stage_climate_balance = function(df, vars = stage_climate_vars) {
  df %>%
    filter(if_all(all_of(unname(vars)), ~ !is.na(.x))) %>%
    group_by(taxon_capture) %>%
    group_modify(function(d, key) {
      s = as.numeric(d$life_stage_simple == "subadult")
      imap_dfc(vars, function(col, name) {
        # precipitation is right-skewed (many dry spells): compared on log1p
        x = if (name == "precip") log1p(d[[col]]) else d[[col]]
        sd_pooled = sqrt((var(x[s == 1]) + var(x[s == 0])) / 2)
        smd = function(diff) if (is.na(sd_pooled) || sd_pooled == 0) NA_real_ else diff / sd_pooled
        tibble(!!paste0("smd_", name, "_overall") := smd(mean(x[s == 1]) - mean(x[s == 0])),
               !!paste0("smd_", name, "_pop") := smd(within_diff(x, s, d$population)),
               !!paste0("smd_", name, "_popyear") := smd(within_diff(x, s, d$pop_year)))
      })
    }) %>%
    ungroup()
}
