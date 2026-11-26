---
# machine-readable summary. contrast_data_functions.R is the source of truth for the values
# used in code; keep this block in sync with it.
analysis: size_bd_contrast
status: stage 1 round 3 fits pending (all jobs: survey-run events and consolidated contrast rule); analysis drafted; stage 2 planned
replaces: size_bd_unified m11g (02/03/05; decided 2026-11-25, README "Relation to the unified model")
uses_from_size_bd_unified: 01_model_data.Rmd, env_functions.R, contrast_functions.R, phylo_tree.nwk, naive cutoff bootstrap
data: size_bd_unified/intermediates/bd_model_env_unified_2026-11-12.RData
response: bd_load_co100_density
family: hurdle_lognormal
stage_coding: {variable: stage_c, adult: -0.5, subadult: 0.5}
groups:
  population: population
  pop_year: population x year
  pop_year_event: population x year x survey run
event:  # survey run, per species x population (round 3; rounds 1-2: 14-day bins of day of year)
  max_gap_days: 5     # a new event starts after a gap of more than 5 days between survey dates
  max_span_days: 14   # longer runs are split at their longest gap until each piece fits
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
  time_segregated: "n co-sampled events < n co-sampled pop_years / 2 (so >= 1 co-sampled pop_year)"   # >= 3 pop_years until 2026-11-26
nuisance_structure:  # intercept levels and sigma; README "Nuisance structure"
  R2_min_children_per_parent: 1.1   # site-years per site, events per site-year; per part
  R3_sigma_by_population: ">= 3 populations with >= 20 positive swabs"   # 10 until round 3
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
followed by cross-species models of those contrasts. This is the primary analysis of the stage
contrast; it replaces the unified model (`size_bd_unified/`, m11g). See "Relation to the unified
model" for what moves, what is lost and what is still to do.

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
- **Q6.** Does the contrast change with the overall level of infection? (correlation between a
  unit's intercept and its contrast, by level: e.g. are subadults hit disproportionately in
  high-Bd sites, years or events, or does the gap close when everyone is infected?)

Stage 2 (across species, planned):

- **Q7.** Is the contrast phylogenetically structured, or related to life-history traits or
  species-typical climate? (species level)
- **Q8.** Do sites in different climates show different contrasts? (population level)
- **Q9.** Does the contrast move with conditions through time, and is that seasonal
  (climatology) or a response to unusual conditions (anomaly)? Does it differ between rising and
  falling temperatures? (event level)

## Definitions

- **Stage contrast**: subadult - adult. Prevalence part on the logit-prevalence scale
  (= -logit(hu)); load part as a difference in mu (mean log load density among positives).
  Positive = subadults higher. Contrasts are compared within species only (labs differ).
- **Event**: a survey run at one site: consecutive survey dates (of the species) with no gap
  over **5 days**. Runs longer than **14 days** are split at their longest gap until each piece
  is 14 days or less. Events follow the survey design rather than the calendar:
  - gaps between survey dates peak at 1 day (consecutive-day designs) and again at 7 and 14
    days (weekly and fortnightly programs). A 5-day gap keeps consecutive days together without
    chaining weekly revisits (a 7-day gap made a 76-day *A. canorus* "event")
  - the 14-day cap splits intensive resurvey periods (visits every 2-5 days for up to 24 days),
    keeping the 2-week resolution of the earlier definition. Within such runs the stage mix can
    shift (*A. canorus* 2021: adults first, then emerging metamorphs), which is why they are split
  - compared with the earlier 14-day bins of day of year (rounds 1-2): 4167 vs 4155 events, 442
    vs 447 co-sampled, no tier changes; the bins had cut 138 visit runs at an arbitrary edge.
    79% of events are a single day. Evidence: `01_contrast_tiers.Rmd` (event screen)
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
  there are **fewer co-sampled events than half the number of co-sampled site-years**
  (prevalence definition; e.g. *P. crucifer*, 9 site-years, 1 event). Stages are found in the
  same site-years but rarely in the same survey event, so the contrast partly compares different
  times of year.
  - No minimum number of site-years (until 2026-11-26: >= 3). With 1-2 site-years the ratio is
    lumpy (one event moves it across 0.5), but the minimum left out the most segregated species:
    *L. blairi* (2 site-years, 0 events), *R. sylvatica*, *S. albomaculata*, *S. holbrookii*
    (1, 0). These are tier D (*few co-sampled sites*), a caveat about how many sites, not about
    timing. Threshold sensitivity: *R. luteiventris* sits exactly at the cutoff (2 site-years,
    1 event, ratio 0.5; not flagged).
  - A co-sampled event needs >= 5 swabs of each stage, so a low ratio can also mean both stages
    were swabbed in the same runs but in small numbers (e.g. weekly visits with 2-3 swabs per
    stage), not at different times. Check the species' event counts before reading the flag as
    seasonal separation.

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
  site-years. It changed one part: *L. chiricahuensis* prevalence (10 events as 14-day bins, 9 as survey
  runs; 4 site-years)
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
  least 20 positive swabs; otherwise `sigma ~ 1`. A site's sigma (spread of log load among its
  positives) is learned only from that site's positive swabs, so the count of positives per
  site, not the size of the loads, is what informs it. The threshold was 10 from the first fits
  and was raised to 20 after round 3: the four species with at least 3 sites at 10 but not at
  20 positives (*P. regilla*, *S. albomaculata*, *S. flotator*, *S. sila*) all had a site-level
  SD of log sigma consistent with zero (5% quantile 0.01-0.02), and *S. flotator* had 25
  divergences in that SD's funnel. Every species whose SD was clearly informed (*R. cascadae*,
  *R. catesbeiana*, *R. clamitans*, *R. muscosa*, *R. sierrae*, *N. viridescens*, *R. pipiens*,
  *L. sphenocephalus*; 0.13-0.34) has at least 3 sites with at least 20 positives. The change
  is calibrated on fits (as the first threshold was), applied to every species alike;
  *R. luteiventris* keeps an uninformed SD at exactly 3 such sites and samples cleanly. `sigma ~ 1` is an assumption (same spread of log load at all sites), not
  a finding. Sigma does not enter the load contrast (a difference in mean log load), but where
  sites differ in spread (labs, protocols, outbreak phase), modeling it keeps that noise out of
  the contrast SDs, which is why it is kept where the data support it.

## Stage 1 analyses (`03_contrast_analysis.Rmd`)

- Per species and part: mean contrast (median, 50 / 95% intervals, pd, support via
  `f_draw_agg`) with each stage's level at a typical site (prevalence %, median load among
  positives; within-species context), contrast SD per level against its prior, variance shares,
  tier (with the reason for D) and the time-segregated flag.
- Q6: intercept-contrast correlation by level (forest plot; unit scatter as illustration).
- Contrast SD / intercept SD ratio per level (does the gap vary less than overall infection?),
  and intercept vs contrast variance shares over the same levels.
- Per-draw season and year summaries (within-site-year / within-site slopes, late - early,
  seasonal curves) for the contrast (co-sampled units) and, as a supplement, for overall
  infection (intercepts of observed units, every tier). Overall infection is shown relative to
  each species' average sampled event, never as absolute prevalence or load (labs differ):
  prevalence as an odds ratio of a Bd-positive swab (y axis per species, read within species),
  load as a ratio of median load among positives (one y axis for all species; lab offsets
  cancel in a ratio).
- Tier A: event contrasts against day of year (calendar axis from each species' seasonal
  origin), optionally with life-history windows.
- Consistency: mean contrast vs m11g (agreed in sign and rough size; kept until m11g is
  retired from the manuscript inputs); *A. canorus* sensitivity vs main fit; optionally vs the
  seasonal models (`size_bd_season/`).
- Exports for stage 2: species, population, site-year and event contrasts (fixed + random parts)
  with posterior mean and sd, a thinned draw set, and the identifiers needed to join climate.

## How to read the results

Examples are from the round 2 fits; numbers will shift with round 3.

**Mean contrast (Q1).** Report the posterior median with its 50% and 95% intervals and the
support label from `f_draw_agg` (pd: the probability the contrast has the sign of its median).
Tier D intervals describe the sampled sites only (Definitions).

**Contrast SDs (Q2-Q4)**, read against the prior (exponential(2): median ~0.35, 95% range
0-1.5). Each SD is variation *of the subadult - adult difference* between units at that level,
beyond the level above (event SD = between events within a site-year). It is not variation in
prevalence or load themselves; that is the intercept SD. Three patterns:

1. **Looks like the prior** (median near 0.35, interval filling the prior range): the data cannot
   answer at this level. Report "not informed", not "consistent". E.g. *R. boylii* prevalence
   site-year, 0.43 [0.01, 1.62]. Expected where few units are co-sampled.
2. **Upper bound well inside the prior range**: the data rule out large variation, i.e. evidence
   the contrast is *consistent* at that level. E.g. *R. cascadae* prevalence site,
   0.18 [0.01, 0.61].
3. **Lower bound clearly above 0**: the contrast *varies* at that level. E.g. *R. sierrae* load
   event, 1.93 [1.64, 2.21].

The lower bound of a weakly informed SD always sits near 0 (an SD cannot be negative), so "does
the interval include 0" is not a test. Look at where the upper bound sits and whether the lower
bound is clearly above 0. "Informed" (fading in plots, contraction >= 0.5) is a reading aid; the
power-scaling check in `02` ("Prior sensitivity") says whether a posterior follows its prior.

**Size.** An SD is on the scale of the contrast: about 95% of units fall within +/- 2 SD of the
mean contrast. Prevalence (logit): SD 0.5 = within +/- 1 logit (odds ratio x/÷ 2.7). Load (log):
SD 1.9 = within +/- 3.8 log units. Sign consistency combines the SD with the mean: SD 0.5 around a
mean of 2 rarely flips sign, around 0.2 it often does.

**Wording**, e.g. "*R. sierrae* shows clear event-level variation in the load contrast: the
subadult - adult difference in log load among positives varies between survey events within the
same site-year (event SD 1.93, 95% interval 1.64-2.21)." Avoid "significant"; use the posterior
interval and support labels.

**Model changes vs lack of information.** A term is removed only for computational failure
(divergences, poor mixing, terms the data cannot identify: the contrast levels rule, R2, R3). A
prior-dominated SD is reported as not informed, not removed: removing it would assume zero
variation and narrow the mean contrast's interval without new information.

## Relation to the unified model (m11g)

Decided 2026-11-25: m11g is retired; the contrast models are the primary analysis. In m11g the
weather and climate terms were there to control for distractions from the stage contrast, not
as part of the question, and they are shared by both stages and confounded with season (Why
this analysis). The contrast models drop them and let the contrast vary by site, year and
event instead; environment returns only as a moderator of the contrast (stage 2).

What replaces what:

| m11g output | Replacement |
|---|---|
| species stage contrasts (`unified_03_c`) | per-species mean contrasts (`03`, `contrast_03_a`) |
| between-population spread, variance components (`unified_03_j/k`) | contrast sds by level, variance shares (`03`) |
| species meta-model with phylogeny and traits (`05`, `unified_05_*`) | stage 2a (species level, below), on `03` outputs |
| environment -> infection (`unified_03_d/e/h/i`) | dropped. Stage 2 asks environment -> contrast instead |
| cutoff sensitivity (m11g at co100 / co300 / co1000) | naive bootstrap cutoff check only (`size_bd_unified/th_sensitivity_analysis.Rmd`, naive part; independent of m11g) |
| environment priors for the GAM (`size_bd_gam/02`, m11g new-taxon draws) | open: alternative to brainstorm (Open questions) |

What is lost, accepted:

- pooling of sparse species toward the across-species distribution (m11g taxon effects).
  Stage 2a restores pooling as a second step (species random effect + phylogeny), with stage 1
  uncertainty carried through.
- six species in m11g without a within-site comparison or with < 20 swabs of a stage
  (*C. crassidigitus*, *C. fitzingeri*, *R. marina*: no co-sampled population; *E. prosoblepon*,
  *H. cinerea*, *L. palustris*: < 20 swabs of a stage). Stated as a limitation: their m11g
  contrast rested on comparing different populations.
- environment effects on overall infection levels; not a target question.

Kept from `size_bd_unified/`: `01_model_data.Rmd` (model data), `env_functions.R`,
`contrast_functions.R` (`f_draw_agg`), `svl_stage_cutoffs.R`, the phylogenetic tree, data
summaries (species / lab stats, sex composition) and the naive cutoff bootstrap. To do when
retiring: re-point manuscript inputs from `unified_03_*` / `unified_05_*` to the contrast
exports, and drop the m11g consistency check from `03` (or keep it as a historical note).

## Stage 2 (planned; separate plan)

Contrast-level (meta-analytic) models, with stage 1 uncertainty carried through
(`contrast | se(sd)` or repeated fits over posterior draws), phylogeny from
`size_bd_unified/intermediates/phylo_tree.nwk`:

| Level | Response | Covariates |
|---|---|---|
| species (2a; replaces unified `05`) | species mean contrast | traits, species-typical climate, phylogeny |
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

## Glossary

- **Stage contrast**: subadult - adult difference; prevalence part on the logit scale, load part
  in mean log load among positives. Positive = subadults higher.
- **Mean contrast**: the `stage_c` coefficient; the contrast at a typical site, year and event.
- **Contrast SD**: SD of the contrast's deviations at one level (site, site-year, event). How much
  the contrast varies between units there. Not the same as the intercept SD (how much Bd varies).
- **Intercept SD**: how much overall Bd levels (prevalence or load, both stages) vary between
  units at a level.
- **Co-sampled unit**: a site, site-year or event with >= 5 swabs of each stage (load: also >= 1
  positive of each stage). Contrast SDs are informed mainly by these.
- **Survey run (event)**: survey dates at a site with no gap over 5 days, split at 14 days.
- **Tier**: number of levels at which the contrast varies (A 3, B 2, C 1, D 0), from the contrast
  levels rule.
- **Partial pooling / shrinkage**: group estimates are pulled toward their parent in proportion
  to how little data they have. Non-co-sampled units sit near the mean by construction.
- **Prior-dominated vs informed**: whether a posterior mostly reflects the prior or the data.
  Judged from the posterior vs prior plot and power-scaling, not from whether 0 is in the
  interval.
- **Contraction**: 1 - posterior variance / prior variance. A rough reading aid; trimming the
  prior's long tail alone can give ~0.5.
- **Sign consistency**: probability a new unit's contrast has the sign of the mean,
  Phi(|mean| / SD).
- **Variance share**: SD^2 of a level / sum over modeled levels (Q5).
- **pd / support**: probability of direction (share of draws with the sign of the median), and
  the labels `f_draw_agg` derives from it.
- **R-hat**: agreement between chains (rank-normalized bulk and folded tail versions; the larger
  is reported). > 1.01 = chains disagree. Fixed by longer runs when chains explore slowly, by
  more chains when one is stuck, and by changing the model when it is not identified.
- **ESS (bulk / tail)**: effective number of independent draws for the center / tails of a
  posterior. > 400 wanted.
- **Divergence**: the sampler failed to follow the posterior's curvature (often a funnel where an
  SD nears 0). Divergent regions are under-explored, which biases estimates; a few are tolerated.
- **adapt_delta**: sampler target acceptance rate. Higher = smaller steps = fewer divergences,
  slower; it changes how the posterior is explored, not the model.
- **Power-scaling** (priorsense): reweight the draws as if the prior (or likelihood) were raised
  to a power alpha, to see whether a posterior depends on it, without refitting. For the
  exponential(2) contrast-SD prior, alpha 0.5 / 2 = exponential(1) / exponential(4).
- **Funnel**: the geometry where a group SD near 0 squeezes its group deviations together; hard
  to sample, the usual source of divergences in hierarchical models.

## Decision log

Dated one-liners; the reasoning is in the linked sections.

- 2026-11-24: event = 14-day bins; co-sampled unit >= 5 swabs of each stage (`01` event
  screen). Tiers A/B/C from co-sampled counts. Round 1: all 32 jobs; 13 failed the convergence
  flag (`02` "Fit rounds").
- 2026-11-25: round 2 structure rules R1 (< 3 co-sampled sites: no site contrast SD), R2
  (separable levels), R3 (sigma by site); 19 jobs refit (Nuisance structure).
- 2026-11-25: contrast levels rule replaces tiers + R1 + between-population flag; tier D
  replaces "fixed"; *N. viridescens* C -> D, *L. chiricahuensis* prevalence A -> C (Contrast
  levels).
- 2026-11-25: event = survey run (gap <= 5 days, split at 14 days) (Definitions, `01` event
  screen).
- 2026-11-25: every job 2000 warmup + 4000 sampling (16000 draws); *R. sierrae* stays at
  adapt_delta 0.95; *R. muscosa*'s borderline R-hat read as slow tail mixing, accepted if it
  stays ~1.01 with tail ESS > 1000 (`02` "Fit rounds").
- 2026-11-25: LOO off by default (not used); memory and time requests from sacct
  (`r-brms-contained/hpc/run_size_bd_contrast_modeling.sbatch`).
- 2026-11-25: contrast-SD prior sensitivity checked alone by power-scaling (`02` "Prior
  sensitivity"), with the prior's log density rebuilt from the SD draws: works on any fit without
  a refit or the log-likelihood (round 3 was submitted before the `lprior_sdcontrast` tag; the
  tag stays, unused). First result, *H. phyllodes* (round 2, tier C): mean contrasts 0.02-0.03
  (robust), site SDs 0.22-0.27 (prior-dominated). Uninformed SDs are kept, not downgraded (How
  to read the results).
- 2026-11-25: round 3 (all 32 jobs) prepared: `hpc/submit_size_bd_contrast_round3.sh`.
- 2026-11-25: intercepts analysed alongside the contrast in `03` (stage levels at a typical
  site, intercept-contrast correlation as new question Q6 with stage 2 renumbered Q7-Q9, SD
  ratio and variance shares, absolute season / year patterns in the supplement).
- 2026-11-25: m11g retired; contrast models primary (Relation to the unified model). Stage 2a
  replaces unified `05`; cutoff sensitivity from the naive bootstrap only; environment enters as
  a moderator of the contrast (stage 2), not of overall infection; GAM environment terms to be
  re-planned.
- 2026-11-26: round 3 results: 31 of 32 pass; *S. flotator* fails with 25 divergences on
  `sd_population__sigma_Intercept` (5 at the tolerance in round 2). R3 raised from >= 10 to
  >= 20 positive swabs in >= 3 sites, which drops `sigma ~ (1 | population)` from *P. regilla*,
  *S. albomaculata*, *S. flotator* and *S. sila* (none informed); no tier changes. Round 4 refits
  those four (Nuisance structure, `02` "Fit rounds").
- 2026-11-26: time-segregated flag drops its >= 3 co-sampled site-years minimum; the minimum
  had excluded the most segregated species. Flagged: 7 -> 11 species (adds *L. blairi*,
  *R. sylvatica*, *S. albomaculata*, *S. holbrookii*; all tier D, few co-sampled sites). Caveat
  only, no refit (Definitions).
- 2026-11-26: full power-scaling check in `02` ("Prior sensitivity") computes the
  log-likelihood summed per draw in blocks of draws instead of calling priorsense on the
  brmsfit, which built the full draws x swabs matrices and was OOM-killed at 30 GB (*R. sierrae*).
  Same result on *R. clamitans* by both routes.
- 2026-11-26: overall-infection season patterns in `03` shown relative to each species' average
  sampled event (per draw), not as absolute prevalence / load, to keep levels from being
  compared between species. Prevalence as odds ratios (free y per species), load as ratios of
  median load among positives (shared y); season / year tables gain ratio columns. Figures and
  tables renamed `03_absolute_*` -> `03_relative_*` (Stage 1 analyses).

## Open questions

- Load co-sampling uses >= 1 positive of each stage; a stricter positive count may better match
  how much a unit informs the load contrast.
- Contrast SD prior scale (exponential(2)): check with the contrast-SD power-scaling in `02`
  after round 3; refit a few species with exponential(1) / exponential(4) if it flags mean
  contrasts.
- R3: whether species with a constant sigma combine data sources with different measurement
  precision (`project`) has not been checked.
- Stage 2 climate summaries and trait set to be defined.
- GAM (`size_bd_gam/`) environment terms without m11g priors. Options to brainstorm: drop the
  environment terms (as here: control for site / year / event with random effects instead);
  keep them with weakly informative priors; or replace weather with a seasonal term. The GAM's
  purpose (finer age resolution for data-rich species) should decide which.
- Retirement to-dos: manuscript inputs (`unified_03_*`, `unified_05_*`) re-pointed to contrast
  exports; m11g consistency check in `03`; species-level mean contrasts for the six m11g-only
  species reported as excluded.
