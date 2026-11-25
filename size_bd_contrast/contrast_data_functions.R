# shared constants and data helpers for size_bd_contrast (sourced by 01_contrast_tiers.Rmd and
# 02_contrast_modeling.qmd; mirrored in r-brms-contained/size_bd_contrast_modeling.R). these
# values are the source of truth; keep the README front matter in sync.
#
# requires: tidyverse (dplyr, tidyr, lubridate) loaded by the calling script

# ---- constants -----------------------------------------------------------------------------
response_col = "bd_load_co100_density"

event_days = 14            # event = population x year x 14-day bin of day of year
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
# time-segregated flag (caveat only): >= 3 co-sampled site-years, and fewer co-sampled events
# than half the number of co-sampled site-years
time_seg_min_popyears = 3
time_seg_max_event_ratio = 0.5
# nuisance structure (2026-11-25, after the first cluster fits: divergences and poor mixing
# traced to terms the data could not identify; see README "Nuisance structure")
min_children_per_parent = 1.1  # R2: keep the site-year (event) intercept only if there are >= 1.1
                               #     site-years per site (events per site-year)
min_pos_sigma_pop = 10     # R3: sigma varies by population only with >= 3 populations ...
min_pops_sigma = 3         #     ... that have >= 10 positive swabs each

# ---- grouping columns ------------------------------------------------------------------------
# stage_c: -0.5 adult, +0.5 subadult. the intercept is then the midpoint of the two stages and
# the stage_c coefficient the full subadult - adult contrast; a group's stage_c sd is how much
# the contrast varies at that level.
# pop_year / pop_year_event: explicit ids (rather than population:year:... in the formula) so
# parameter names stay readable. 14-day bins are cut at the new year.
add_contrast_groups = function(df) {
  df %>%
    mutate(life_stage_simple = factor(life_stage_simple, levels = c("adult", "subadult")),
           stage_c = if_else(life_stage_simple == "subadult", 0.5, -0.5),
           event_bin = (yday(date) - 1) %/% event_days,
           pop_year = paste(population, year, sep = "_"),
           pop_year_event = paste(population, year, event_bin, sep = "_"))
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
#       >= min_pos_sigma_pop positive swabs; otherwise sigma ~ 1. in the first fits, the site-level
#       sd of sigma was clearly informed above this threshold (small, 0.1-0.3 on the log scale)
#       and mostly prior below it.
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
