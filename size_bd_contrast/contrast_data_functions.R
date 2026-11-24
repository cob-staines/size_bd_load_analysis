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
# per species: number of units (of the column `unit`), co-sampled units for prevalence and load,
# and the share of swabs that sit in co-sampled units
count_co_units = function(df, unit, response = response_col,
                          min_swabs = min_swabs_unit, min_pos = min_pos_unit) {
  df %>%
    group_by(taxon_capture, unit_id = .data[[unit]], life_stage_simple) %>%
    summarise(n = n(), n_pos = sum(.data[[response]] > 0), .groups = "drop") %>%
    pivot_wider(names_from = life_stage_simple, values_from = c(n, n_pos), values_fill = 0) %>%
    mutate(co = n_adult >= min_swabs & n_subadult >= min_swabs,
           co_load = co & n_pos_adult >= min_pos & n_pos_subadult >= min_pos,
           n_unit = n_adult + n_subadult) %>%
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
