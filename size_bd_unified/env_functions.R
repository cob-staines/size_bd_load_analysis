# Climate covariates from the grid-cell cache (python/environmental_timeseries_pull/update_env_cache.py).
# Shared by size_bd_unified/01_model_data.Rmd and size_bd_gam/01_GAM_model_data.Rmd.

# site -> 0.05 deg grid cell lookup written by update_env_cache.py next to each registry
read_site_cells = function(cells_file, sites) {
  site_cells = read_csv(cells_file, col_types = cols(site = "c", cell_id = "c"))
  missing_cells = setdiff(unique(sites), site_cells$site)
  if (length(missing_cells) > 0) {
    stop("sites without a grid cell (rerun update_env_cache.py): ", paste(missing_cells, collapse = ", "))
  }
  site_cells
}

# daily CHIRPS precipitation and CHIRTS tmax per cell, with rolling windows and past-year weekly SDs
build_env_agg = function(cell_ids,
                         cache_dir = here("python", "environmental_timeseries_pull", "env_cache")) {
  read_env = function(product) {
    read_parquet(file.path(cache_dir, paste0(product, ".parquet"))) %>%
      filter(cell_id %in% cell_ids) %>%
      transmute(cell_id, date = as.Date(date), value)
  }

  # CHIRTS tmax from GEE through 2016, from CHIRTS-ERA5 from 2017
  env_daily = full_join(
    read_env("chirps") %>% dplyr::rename(precipitation_mm = value),
    bind_rows(read_env("chirts_tmax"), read_env("chirts_era5_tmax")) %>%
      dplyr::rename(temperature_max_c = value),
    by = c("cell_id", "date")) %>%
    # one row per calendar day, so the rolling windows below span calendar days
    group_by(cell_id) %>%
    complete(date = seq(min(date), max(date), by = "day")) %>%
    ungroup() %>%
    arrange(cell_id, date)

  env_agg = env_daily %>%
    group_by(cell_id) %>%
    mutate(precip_mm_10_day_sum = zoo::rollapply(precipitation_mm, 10, sum, fill = NA, align = "right"),
           precip_mm_30_day_sum = zoo::rollapply(precipitation_mm, 30, sum, fill = NA, align = "right"),
           temp_c_10_day_mean = zoo::rollapply(temperature_max_c, 10, mean, fill = NA, align = "right"),
           temp_c_30_day_mean = zoo::rollapply(temperature_max_c, 30, mean, fill = NA, align = "right")
           ) %>%
    ungroup()

  # continuous 7-day blocks (52 * year + week(date) merged Dec 31 into the next year's week 1)
  env_var_index = env_daily %>%
    mutate(week_number = as.integer(date) %/% 7)

  env_var_week = env_var_index %>%
    group_by(cell_id, week_number) %>%
    # na.rm: a block can straddle the edge of a site's date window (days outside it are not pulled)
    summarize(weekly_precipitation_mm = if (all(is.na(precipitation_mm))) NA_real_ else mean(precipitation_mm, na.rm = TRUE) * 7,
              weekly_temperature_max_c = if (all(is.na(temperature_max_c))) NA_real_ else max(temperature_max_c, na.rm = TRUE),
              .groups = "drop") %>%
    group_by(cell_id) %>%
    mutate(past_year_weekly_precipitation_mm_sd = zoo::rollapply(weekly_precipitation_mm, 52, sd, fill = NA, align = "right"),
           past_year_weekly_tempurature_max_c_sd = zoo::rollapply(weekly_temperature_max_c, 52, sd, fill = NA, align = "right")) %>%
    ungroup()

  env_agg %>%
    left_join(env_var_index %>%
                select(cell_id, date, week_number), by = c("cell_id", "date")) %>%
    left_join(env_var_week %>%
                select(cell_id,
                       week_number,
                       past_year_weekly_precipitation_mm_sd,
                       past_year_weekly_tempurature_max_c_sd), by = c("cell_id", "week_number")) %>%
    select(-week_number)
}

# join climate to samples by site -> cell and date; every sample needs values,
# missing ones mean the cache lacks those cell-dates
join_env = function(samples, site_cells, env_agg) {
  out = samples %>%
    left_join(site_cells %>%
                select(site, cell_id), by = "site") %>%
    left_join(env_agg, by = c("cell_id", "date"))

  env_missing = out %>%
    filter(is.na(precip_mm_10_day_sum) | is.na(temp_c_30_day_mean)) %>%
    count(taxon_capture, site, cell_id, name = "n_samples")

  if (nrow(env_missing) > 0) {
    print(env_missing, n = 50)
    stop(sum(env_missing$n_samples), " samples without climate values (rerun update_env_cache.py)")
  }
  out
}
