---
# machine-readable summary. contrast_data_functions.R is the source of truth for the values
# used in code; keep this block in sync with it.
analysis: size_bd_contrast
status: stage 1 round 3 fits pending (consolidated contrast rule: n. viridescens, l. chiricahuensis); analysis drafted; stage 2 planned
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
contrast_rule:  # per part (hu: prevalence counts, mu: load counts); README "Contrast levels"
  # the contrast varies at a level with at least this many co-sampled units there, and only if
  # it also varies at the level above (nested)
  min_co_units: {population: 3, pop_year: 5, pop_year_event: 10}
  # no contrast sd at any level when fewer swabs than this sit in co-sampled populations
  # (species-level, prevalence definition)
  min_pct_swabs_in_co_pop: 25
tiers:  # names for the result: number of levels at which the contrast varies
  A: "population + pop_year + pop_year_event"
  B: "population + pop_year"
  C: "population"
  D: "none: mean contrast only (reason: < 3 co-sampled populations, or between-population)"
  none: "mu only, off the scale: a stage has no positive swabs; no stage term in mu"
flags:  # interpretation caveat only; prevalence co-sampling counts
  time_segregated: ">= 3 co-sampled pop_year units and n co-sampled events < n co-sampled pop_years / 2"
nuisance_structure:  # intercept levels and sigma; README "Nuisance structure"
  R2_min_children_per_parent: 1.1   # site-years per site, events per site-year; per part
  R3_sigma_by_population: ">= 3 populations with >= 10 positive swabs"
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
  sigma_sd: exponential(1)   # only where sigma varies by population (R3)
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
- **Contrast levels** (one rule, applied separately to prevalence and load from their
  co-sampled counts): the contrast varies at a level when there are at least **3 co-sampled
  sites / 5 co-sampled site-years / 10 co-sampled events**, and only if it also varies at the
  level above (an event-level contrast needs a site-year-level one). No level gets a contrast SD
  when **fewer than 25% of the species' swabs fall in co-sampled sites**: the contrast then
  mostly compares different populations.
- **Tiers** name the result, by the number of levels at which the contrast varies:
  - **A** (3): site, site-year and event
  - **B** (2): site and site-year
  - **C** (1): site
  - **D** (0): one mean contrast, no contrast SD. Reported with its reason: *few co-sampled
    sites* (< 3; the mean mostly reflects those sites) or *between-population* (e.g.
    *N. viridescens*: 2.6% of swabs in co-sampled sites). Its interval covers uncertainty about
    the sampled sites only, so it is narrower than it would be for a species-wide mean.
  - **none** (load only, off the scale): one stage has no positive swabs, so there is no load
    contrast and no stage term
- **Flag** (an interpretation caveat; it does not change the model): *time-segregated* when
  there are **>= 3 co-sampled site-years and fewer co-sampled events than half the number of
  co-sampled site-years** (prevalence definition; e.g. *P. crucifer*). Stages are found in the
  same site-years but rarely in the same 14-day event, so the contrast partly compares different
  times of year.

## Stage 1 model

One model per species (and per sensitivity variant), hurdle lognormal, no weather or climate:

```r
# tier A (hu or mu)
~ 1 + stage_c + (1 + stage_c | population) + (1 + stage_c | pop_year) + (1 + stage_c | pop_year_event)
# tier B
~ 1 + stage_c + (1 + stage_c | population) + (1 + stage_c | pop_year) + (1 | pop_year_event)
# tier C
~ 1 + stage_c + (1 + stage_c | population) + (1 | pop_year) + (1 | pop_year_event)
# tier D
~ 1 + stage_c + (1 | population) + (1 | pop_year) + (1 | pop_year_event)
# none (mu only)
~ 1 + (1 | population) + (1 | pop_year) + (1 | pop_year_event)

sigma ~ 1 + (1 | population)   # or sigma ~ 1 (R3)
# (1 | pop_year) and (1 | pop_year_event) are left out where a part cannot separate them (R2)
```

- `stage_c` = -0.5 adult, +0.5 subadult. The intercept is the midpoint of the two stages and the
  `stage_c` coefficient is the full contrast, so each group's `stage_c` SD is directly "how much
  the contrast varies at this level", separate from how much overall *Bd* levels vary (the
  intercept SD). A factor coding would make the contrast SD a derived quantity with no direct
  prior.
- Intercepts vary at every level in every tier: they absorb site, year and event-wide swings
  (outbreaks, seasonal peaks) whether or not both stages were swabbed, which also keeps
  differences in sampling timing between stages from being read as a stage contrast. Contrast
  slopes are only added at levels the data can support (the contrast levels rule).
- All data are used. Contrast SDs are informed mainly by co-sampled units; single-stage units
  inform the intercepts. The >= 5-swab rule chooses model terms, it does not filter data.
- Priors: see the front matter. The contrast SD prior, exponential(2), is tighter than the
  intercept SDs so a consistent contrast is the default and variation must be earned from the
  data (revisit after the prior vs posterior checks).
- Sensitivity: *A. canorus* (tier C prevalence / D load by the rule, but a focal species with
  13 co-sampled events at >= 2 swabs) is also fit with the tier A structure.

## Contrast levels: why this rule

The rule decides which contrast SDs are estimated; everything readers interpret follows from
it. It replaced, in round 3 (2026-11-25), a counts-based tier table plus two exceptions that
answered the same question ("can the data say how much the contrast varies at this level?"):

- Round 2's **R1** (a tier C part with < 3 co-sampled sites gets no contrast SD) is the site
  threshold. In the first fits a site-level contrast SD over one or two sites produced funnels
  (divergences where the SD is near 0).
- The **between-population** condition was a flag; it now removes the contrast SDs. In
  *N. viridescens* (round 2: Rhat 1.03, 34 divergences on the site-level contrast SD) about 276
  of 281 sites have mostly one stage. At such a site only `r_intercept - 0.5 r_contrast` is
  identified, so the site-level contrast SD and its correlation with the intercept sit on
  hundreds of ridges, informed by only 4-5 co-sampled sites.
- **Nesting** (a level's contrast needs the level above) replaced "A: >= 10 events" regardless of
  site-years. It changed one part: *L. chiricahuensis* prevalence (10 events, 4 site-years)
  from A to C. Its round 2 site-year and event contrast SDs were both 0.33 [0.01, 1.59], the
  prior.

Every other species and part kept its round 2 tier. Code: `contrast_tier()` in
`contrast_data_functions.R`.

**Threshold sensitivity** (thresholds fixed a priori, not tuned): the 25% cutoff is robust
(*N. viridescens* 2.6%; the next species not in D, *P. crucifer*, 58%). Species exactly at a
threshold: 3 co-sampled sites (*H. phyllodes*, *A. canorus*, *B. faber*, prevalence), 5
site-years (*L. warszewitschii*, *R. pipiens*, *A. fowleri*), 10 events (*C. panamansis*,
*R. catesbeiana*).

**What the data inform**: in round 2, contrast SDs clearly moved away from the prior only in the
data-rich *Rana* (*R. cascadae*, *R. clamitans*, *R. muscosa*, *R. sierrae*; perhaps the
*R. catesbeiana* event level). Tier B and C SDs mostly reflect the prior. They are kept because
they widen the mean contrast's interval for between-site variation, which keeps tiers A-C
comparable; tier D intervals lack this (see Definitions).

## Nuisance structure

Rules for terms that absorb variation but are not interpreted: intercept levels and sigma.
Added after the first cluster fits (2026-11-24), where 13 of 32 jobs failed the convergence
flag. The failures traced to terms the data could not identify, not to the sampler (small
species already ran at adapt_delta 0.99). Details and evidence: `02_contrast_modeling.qmd`
("Fit rounds"). Applied per part, from the swabs that inform it: all swabs for prevalence,
positive swabs for load. Code: `level_structure()` in `contrast_data_functions.R`. (Labels R2
and R3 are kept from round 2, where R1 was the site threshold, now part of the contrast rule.)

- **R2, separable levels**: the site-year (event) intercept is kept only if there are at least
  1.1 site-years per site (events per site-year). Otherwise the two levels are nearly the same
  groups and only their summed variance is identified (e.g. *N. viridescens*: 1.02 and 1.02,
  Rhat 1.12). A contrast slope cannot sit on a dropped level; under the current rule this
  changes no tier.
- **R3, sigma**: sigma varies by population only with at least 3 populations that have at
  least 10 positive swabs; otherwise `sigma ~ 1`. Above that threshold the site-level SD of
  log sigma was clearly informed in the first fits (small, 0.13-0.33); below it the posterior
  was mostly the prior. `sigma ~ 1` is an assumption (same spread of log load at all sites), not
  a finding. Sigma does not enter the load contrast (a difference in mean log load), but where
  sites differ in spread (labs, protocols, outbreak phase), modeling it keeps that noise out of
  the contrast SDs, which is why it is kept where the data support it.

## Stage 1 analyses (`03_contrast_analysis.Rmd`)

- Per species and part: mean contrast (median, 50 / 95% intervals, pd, support via
  `f_draw_agg`), contrast SD per level against its prior, variance shares, tier (with the reason
  for D) and the time-segregated flag.
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
| `contrast_data_functions.R` | shared constants, grouping columns, co-sampling counts, contrast rule, nuisance structure | written |
| `01_contrast_tiers.Rmd` | event definition screen (unit definitions x swab thresholds), counts, inclusion, tiers, flag; writes the event screen, species table and cluster job table | written |
| `contrast_model_functions.R` | data prep, formula and prior builders by tier (copied into the cluster script) | written |
| `02_contrast_modeling.qmd` | formula and priors by tier, prior validation, local timing, fit / load, model checks | written |
| `r-brms-contained/size_bd_contrast_modeling.R` + `hpc/` | cluster fits, one job per array index (`hpc/submit_size_bd_contrast.sh` submits the four resource groups) | written |
| `03_contrast_analysis.Rmd` | stage 1 results (mean contrasts, contrast sds, sign consistency, variance shares, group contrasts), consistency checks, stage 2 exports | written |
| `04_contrast_stage2.qmd` | cross-species contrast models | planned |

Outputs: `intermediates/` (tables), `models/` (fits, not tracked), `figures/`.

Upstream: `size_bd_unified/01_model_data.Rmd` (model data).

## Open questions

- Load co-sampling uses >= 1 positive of each stage; a stricter positive count may better match
  how much a unit informs the load contrast.
- Contrast SD prior scale (exponential(2)) to be checked against posteriors.
- R3: whether species with a constant sigma combine data sources with different measurement
  precision (`project`) has not been checked.
- Stage 2 climate summaries and trait set to be defined.
