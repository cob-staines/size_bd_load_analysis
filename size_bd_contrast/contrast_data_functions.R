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
tier_a_min_events = 10     # tier A: >= 10 co-sampled events
tier_b_min_popyears = 5    # tier B: >= 5 co-sampled site-years
# model structure rules (2026-11-25, after the first cluster fits: divergences and poor mixing
# traced to terms the data could not identify; see README "Model structure rules")
min_co_pops_slope = 3      # R1: a site-level contrast sd needs >= 3 co-sampled populations;
                           #     tier C parts with fewer get tier "fixed" (one mean contrast)
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
# co-sampled unit, used for the tier counts below and to mark group contrasts in
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

# ---- tiers -----------------------------------------------------------------------------------
# A: contrast varies by population, site-year and event; B: population and site-year;
# C: population only. applied separately to prevalence (hu) and load (mu) counts.
tier_from_counts = function(n_co_events, n_co_popyears) {
  case_when(n_co_events >= tier_a_min_events ~ "A",
            n_co_popyears >= tier_b_min_popyears ~ "B",
            .default = "C")
}

# ---- model structure rules (R1-R3) ---------------------------------------------------------
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

# R2 then R1: final tier of a part, from its counts-based tier (A / B / C):
#   R2: reduced to the finest contrast-slope level the part still has (a slope cannot sit on a
#       dropped level): A needs the event level, B the site-year level, otherwise C.
#   R1: a (resulting) tier C part with fewer than min_co_pops_slope co-sampled populations ->
#       "fixed": one mean contrast, no contrast sd (one or two sites cannot say how much the
#       contrast varies between sites). the mean contrast then mostly reflects those sites.
# "none" (load part with no positives in a stage) passes through.
final_tier = function(tier_counts, n_co_pops, has_pop_year, has_event) {
  reduced = case_when(tier_counts == "none" ~ "none",
                      tier_counts == "A" & has_event ~ "A",
                      tier_counts %in% c("A", "B") & has_pop_year ~ "B",
                      .default = "C")
  if_else(reduced == "C" & n_co_pops < min_co_pops_slope, "fixed", reduced)
}
