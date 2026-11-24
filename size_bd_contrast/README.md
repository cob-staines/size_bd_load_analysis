---
# machine-readable summary. contrast_data_functions.R is the source of truth for the values
# used in code; keep this block in sync with it.
analysis: size_bd_contrast
status: stage 1 in development (cluster scripts written, fits next); stage 2 planned
sibling_of: size_bd_unified (m11g)
data: size_bd_unified/intermediates/bd_model_env_unified_2026-11-12.RData
response: bd_load_co100_density
family: hurdle_lognormal
stage_coding: {variable: stage_c, adult: -0.5, subadult: 0.5}
groups:
  population: population
  pop_year: population x year
  pop_year_event: population x year x 14-day bin of day of year
event_days: 14
co_sampled_unit:
  prevalence: ">= 5 swabs of each stage"
  load: ">= 5 swabs of each stage and >= 1 positive swab of each stage"
inclusion: ">= 20 swabs of each stage and >= 1 co-sampled population (prevalence definition)"
tiers:  # assigned separately for hu (prevalence counts) and mu (load counts)
  A: ">= 10 co-sampled pop_year_event units"
  B: ">= 5 co-sampled pop_year units (not A)"
  C: "otherwise"
  none: "mu only: a stage has no positive swabs; no stage term in mu"
sensitivity_fits:
  - {taxon_capture: anaxyrus_canorus, variant: sens_tierA, tier_hu: A, tier_mu: A}
priors:
  mu_intercept: normal(5, 4)
  mu_stage_c: normal(0, 3)
  hu_intercept: normal(0, 2)
  hu_stage_c: normal(0, 2)
  sd_intercepts: exponential(1)
  sd_stage_c: exponential(2)
  cor: lkj(2)
  sigma_intercept: normal(0, 1)
  sigma_sd: exponential(1)
---

# size_bd_contrast

Per-species models of the subadult - adult difference in *Bd* infection (the **stage
contrast**) and how consistent it is across sites, years and sampling events within a season,
followed by cross-species models of those contrasts. A sibling of the unified model pipeline
(`size_bd_unified/`, m11g), not a replacement: m11g remains the reference model and a
consistency check.

## Why this analysis

- m11g describes temporal variation with species-level lagged temperature and precipitation
  terms (plus pooled Köppen-Geiger zones). Those terms are shared by both life stages, are
  symmetric across rising and falling conditions, and are largely confounded with season, which
  also carries behavior (breeding, hibernation, migration, habitat use), cohort growth and
  mortality. Reporting them as "temperature" and "precipitation" effects overstates what is known.
- Pooled terms also force one combined model of all species.
- The all-species seasonal smooth models (`size_bd_season/02_season_all_species.qmd`) mostly
  failed convergence, plausibly because most species are sampled over a narrow part of the year.

This analysis drops weather and climate from the per-species models and instead lets the stage
contrast itself vary at each level, so the data say how consistent it is. Environmental
conditions return in stage 2, as moderators of the contrast.

The aim is comparable per-species profiles that can inform what drives susceptibility or
resilience (e.g. *R. muscosa* vs *R. sierrae*; *A. canorus* vs *A. americanus* /
*A. fowleri*), not a general cross-species claim.

## Questions

Stage 1 (per species, separately for prevalence and load):

- **Q1.** Is infection higher in subadults than adults? (mean contrast)
- **Q2.** Is the contrast consistent across sites? (population-level contrast SD)
- **Q3.** Is it consistent across years within sites? (site-year-level contrast SD)
- **Q4.** Is it consistent within a season? (event-level contrast SD; tier A only)
- **Q5.** At which level does the contrast vary most? (variance shares; a per-species profile)

Stage 2 (across species, planned):

- **Q6.** Is the contrast phylogenetically structured, or related to life-history traits or
  species-typical climate? (species level)
- **Q7.** Do sites in different climates show different contrasts? (population level)
- **Q8.** Does the contrast move with conditions through time, and is that seasonal
  (climatology) or a response to unusual conditions (anomaly)? Does it differ between rising and
  falling temperatures? (event level)

## Definitions

- **Stage contrast**: subadult - adult. Prevalence part on the logit-prevalence scale
  (= -logit(hu)); load part as a difference in mu (mean log load density among positives).
  Positive = subadults higher. Contrasts are compared within species only (labs differ).
- **Event**: population x year x 14-day bin of day of year (bins cut at the new year). 3- and
  4-week bins were checked and change counts very little; 2 weeks keeps finer resolution.
- **Co-sampled unit** (event, site-year or population): both stages swabbed, >= 5 swabs of each
  stage. For load, also >= 1 positive swab of each stage.
- **Inclusion**: >= 20 swabs of each stage and >= 1 co-sampled population. Species without a
  co-sampled population have no within-site comparison; their contrast would rest entirely on
  comparing different populations.
- **Tiers** (assigned separately for prevalence and load, from the co-sampled counts):
  - **A**: >= 10 co-sampled events -> contrast varies by population, site-year and event
  - **B**: >= 5 co-sampled site-years -> contrast varies by population and site-year
  - **C**: otherwise -> contrast varies by population only; flagged as weakly identified
  - **none** (load only): one stage has no positive swabs; the load part has no stage term
- **Flags** (interpretation caveats, from `01_contrast_tiers.Rmd`):
  - *between-population*: few swabs fall in co-sampled populations (e.g. *N. viridescens*), so
    the contrast mostly compares different populations
  - *time-segregated*: stages are found in the same site-years but rarely the same events (e.g.
    *P. crucifer*), so the contrast partly compares different times of year

## Stage 1 model

One model per species (and per sensitivity variant), hurdle lognormal, no weather or climate:

```r
# tier A (hu or mu)
~ 1 + stage_c + (1 + stage_c | population) + (1 + stage_c | pop_year) + (1 + stage_c | pop_year_event)
# tier B
~ 1 + stage_c + (1 + stage_c | population) + (1 + stage_c | pop_year) + (1 | pop_year_event)
# tier C
~ 1 + stage_c + (1 + stage_c | population) + (1 | pop_year) + (1 | pop_year_event)
# none (mu only)
~ 1 + (1 | population) + (1 | pop_year) + (1 | pop_year_event)

sigma ~ 1 + (1 | population)
```

- `stage_c` = -0.5 adult, +0.5 subadult. The intercept is the midpoint of the two stages and the
  `stage_c` coefficient is the full contrast, so each group's `stage_c` SD is directly "how much
  the contrast varies at this level", separate from how much overall *Bd* levels vary (the
  intercept SD). A factor coding would make the contrast SD a derived quantity with no direct
  prior.
- Intercepts vary at every level in every tier: they absorb site, year and event-wide swings
  (outbreaks, seasonal peaks) whether or not both stages were swabbed, which also keeps
  differences in sampling timing between stages from being read as a stage contrast. Contrast
  slopes are only added at levels the data can support (the tiers).
- All data are used. Contrast SDs are informed mainly by co-sampled units; single-stage units
  inform the intercepts. The >= 5-swab rule chooses model terms, it does not filter data.
- Priors: see the front matter. The contrast SD prior, exponential(2), is tighter than the
  intercept SDs so a consistent contrast is the default and variation must be earned from the
  data (revisit after the prior vs posterior checks).
- Sensitivity: *A. canorus* (tier C by the rules, but a focal species with 13 co-sampled events
  at >= 2 swabs) is also fit with the tier A structure.

## Stage 1 analyses (`03_contrast_analysis.Rmd`, planned)

- Per species and part: mean contrast (median, 50 / 95% intervals, pd, support via
  `f_draw_agg`), contrast SD per level against its prior, variance shares, tier and flags.
- Tier A: event contrasts against day of year (calendar axis from each species' seasonal
  origin), optionally with life-history windows.
- Consistency: mean contrast vs m11g; *A. canorus* sensitivity vs main fit; optionally vs the
  seasonal models (`size_bd_season/`).
- Exports for stage 2: species, population, site-year and event contrasts (fixed + random parts)
  with posterior mean and sd, a thinned draw set, and the identifiers needed to join climate.

## Stage 2 (planned; separate plan)

Contrast-level (meta-analytic) models, with stage 1 uncertainty carried through
(`contrast | se(sd)` or repeated fits over posterior draws), phylogeny from
`size_bd_unified/intermediates/phylo_tree.nwk`:

| Level | Response | Covariates |
|---|---|---|
| species | species mean contrast | traits, species-typical climate, phylogeny |
| population | population contrasts | site climate (mean, seasonality), Köppen-Geiger zone, species slopes, phylogeny |
| event | event contrasts | site climatology for that time of year, anomaly from it, temperature trend, species slopes, phylogeny |

Climatology / anomaly decomposition needs multi-year site x day-of-year climate normals (new
work in `size_bd_unified/env_functions.R`). Seasonal (climatology) slopes are reported as
"seasonal-environmental"; only anomaly slopes support a direct weather interpretation.

## Files and run order

| File | Role | Status |
|---|---|---|
| `contrast_data_functions.R` | shared constants, grouping columns, co-sampling counts, tier rules | written |
| `01_contrast_tiers.Rmd` | event definition screen (unit definitions x swab thresholds), counts, inclusion, tiers, flags; writes the event screen, species table and cluster job table | written |
| `contrast_model_functions.R` | data prep, formula and prior builders by tier (copied into the cluster script) | written |
| `02_contrast_modeling.qmd` | formula and priors by tier, prior validation, local timing, fit / load, model checks | written |
| `r-brms-contained/size_bd_contrast_modeling.R` + `hpc/` | cluster fits, one job per array index (`hpc/submit_size_bd_contrast.sh` submits the four resource groups) | written |
| `03_contrast_analysis.Rmd` | stage 1 results, consistency checks, stage 2 exports | planned |
| `04_contrast_stage2.qmd` | cross-species contrast models | planned |

Outputs: `intermediates/` (tables), `models/` (fits, not tracked), `figures/`.

Upstream: `size_bd_unified/01_model_data.Rmd` (model data).

## Open questions

- Load co-sampling uses >= 1 positive of each stage; a stricter positive count may better match
  how much a unit informs the load contrast.
- Contrast SD prior scale (exponential(2)) to be checked against posteriors.
- Stage 2 climate summaries and trait set to be defined.
