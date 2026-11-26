---
# machine-readable summary. contrast_data_functions.R is the source of truth for the values
# used in code; keep this block in sync with it.
analysis: size_bd_contrast
status: stage 1 fits (round 4 refits for the R3 change); analysis drafted; environment screen run; predictive model being planned
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
descriptors:  # species-level, reported with results; not flags, change no model
  stage_sampling_conditions: "smd of 30-day temperature and log1p 10-day precipitation, subadult - adult, within population, in pooled sds of the two stages"
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
conditions return in the downstream models, as possible moderators of the contrast (Downstream
models).

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

Downstream (Downstream models):

- **Q7.** Is the species mean contrast phylogenetically structured, or related to habitat, life
  history, body size or species-typical climate? (predictive model, across species)
- **Q8.** Do sites in different climates show different contrasts? (within species; needs
  long-term site climate)
- **Q9.** Does the contrast move with conditions through time (within season, between years)?
  Seasonal-environmental with raw lags; a weather interpretation needs a climatology / anomaly
  split. (environment screen, then per-species moderation fits if warranted)

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
- **Stage sampling conditions** (a descriptor, not a flag; 2026-11-26): standardized mean
  difference (SMD) in 30-day temperature and log1p 10-day precipitation between subadult and
  adult swabs, subadult - adult, in pooled SDs of the two stages. Computed overall, within
  population (the main column: the difference among swabs of the same site) and within site-year
  (within-season timing). Positive = subadults swabbed in warmer / wetter conditions.
  - Drafted as a flag at |SMD| >= 0.25 (the usual covariate-balance cutoff), it would mark 19 of
    31 species (22 with the first, within-population SD): imbalance is a feature of these data,
    tied to life-stage timing (e.g. metamorph emergence) and habitat use, not an exception. The
    data-rich *Rana* are balanced (*R. sierrae*, *R. muscosa*, *R. cascadae*, *R. clamitans*,
    *R. catesbeiana*: |SMD| <= 0.15); several sparse species have subadults swabbed in much warmer
    conditions (*P. crucifer* 2.8, *R. sylvatica* 1.9, *L. blairi* 1.6, *R. pipiens* 1.5,
    *H. versicolor* 1.4, *A. americanus* 1.4). Reading: How to read the results ("As-sampled
    contrasts").

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
  tier (with the reason for D), the time-segregated flag, and the stage sampling conditions
  (SMDs, share of swabs in co-sampled events).
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
- Exports for the downstream models: species, population, site-year and event contrasts (fixed + random parts)
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

**As-sampled contrasts.** A mean contrast is the subadult - adult difference as the two stages
were sampled. It blends two things:

- a *same-condition* contrast, from co-sampled events, where the event intercepts remove weather
  and anything else shared by both stages. Relevant to resistance and tolerance mechanisms.
- a *realized* contrast, from single-stage units: each stage in the conditions it is found in
  (season, weather, habitat use). Relevant to exposure and population dynamics. Being found at a
  different time is part of being a subadult (metamorph emergence), so this is not simply a bias.

The mix depends on how much of a species' information comes from co-sampled events and how
differently the stages are sampled (stage sampling conditions, Definitions). Balanced, data-rich
species (*R. sierrae*, *R. muscosa*, *R. cascadae*, *R. clamitans*, *R. catesbeiana*) are close
to same-condition contrasts; imbalanced, sparse species (*P. crucifer*, *R. sylvatica*,
*L. blairi*, *A. americanus*, *R. pipiens*, ...) lean toward realized ones. Neither is wrong, but
they are not quite the same quantity: say which when comparing species, and carry the
descriptor into any cross-species model. A contrast is also specific to the conditions a species
was sampled in: the environment screen suggests the load gap widens in warmer years, at least in
*R. sierrae* (Downstream models). Event-level contrast SDs, event-level correlations (Q6) and
group contrasts of co-sampled units are within-event comparisons and not affected.

**Model changes vs lack of information.** A term is removed only for computational failure
(divergences, poor mixing, terms the data cannot identify: the contrast levels rule, R2, R3). A
prior-dominated SD is reported as not informed, not removed: removing it would assume zero
variation and narrow the mean contrast's interval without new information.

## Relation to the unified model (m11g)

Decided 2026-11-25: m11g is retired; the contrast models are the primary analysis. In m11g the
weather and climate terms were there to control for distractions from the stage contrast, not
as part of the question, and they are shared by both stages and confounded with season (Why
this analysis). The contrast models drop them and let the contrast vary by site, year and
event instead; environment returns only as a moderator of the contrast (Downstream models).

What replaces what:

| m11g output | Replacement |
|---|---|
| species stage contrasts (`unified_03_c`) | per-species mean contrasts (`03`, `contrast_03_a`) |
| between-population spread, variance components (`unified_03_j/k`) | contrast sds by level, variance shares (`03`) |
| species meta-model with phylogeny and traits (`05`, `unified_05_*`) | predictive model (Downstream models), on `03` outputs |
| environment -> infection (`unified_03_d/e/h/i`) | dropped. The downstream models ask environment -> contrast instead |
| cutoff sensitivity (m11g at co100 / co300 / co1000) | naive bootstrap cutoff check only (`size_bd_unified/th_sensitivity_analysis.Rmd`, naive part; independent of m11g) |
| environment priors for the GAM (`size_bd_gam/02`, m11g new-taxon draws) | not needed: the GAM takes stage 1's nuisance structure (event intercepts) and drops climate terms (Downstream models) |

What is lost, accepted:

- pooling of sparse species toward the across-species distribution (m11g taxon effects).
  The predictive model restores pooling as a second step (species random effect + phylogeny),
  with stage 1 uncertainty carried through.
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

## Downstream models

Decided 2026-11-26. Stage 1 feeds two downstream analyses; the GAM runs separately:

```
stage 1 contrast models (per species)
   |-> predictive model           species mean contrasts ~ phylogeny + traits (+ species-typical climate)
   |-> environment and contrast   within species: does the contrast move with conditions?
                                  environment screen (04) -> per-species moderation fits if warranted
GAM (size_bd_gam/)                finer age resolution; stage 1 nuisance structure, no climate terms
```

**Predictive model** (replaces `size_bd_unified/05_species_regression.qmd`). Response: species
mean contrasts (prevalence and load), stage 1 uncertainty carried through (`contrast | se(sd)` or
fits over posterior draws), phylogenetic covariance from
`size_bd_unified/intermediates/phylo_tree.nwk`. Candidate predictors: habitat breadth, aquatic
index, body size (adult median SVL) and stage gap (subadult / adult median SVL), species-typical
climate (pending long-term site climate). Development mode dropped: only 2 of 31 species are
coded direct (*I. henselii*, and *C. panamansis*, which is probably larval: males carry tadpoles
to streams), so its effect would rest on one species. With 31 species, predictors are few per
model (small hypothesis-specific models with regularizing priors).

The mix of same-condition and realized contrasts across species (How to read the results) is
handled in three parts (decided 2026-11-26):

- *main*: the stage 1 mean contrasts of all species (31 for prevalence, 30 for load), with the
  stage sampling conditions (SMDs, share of swabs in co-sampled events) reported next to every
  species-level result.
- *sensitivity, same-condition response*: the same model on species mean contrasts from
  co-sampled events only, from the environment screen's null model (22 species for prevalence,
  15 for load). Every species then contributes the same kind of contrast. Agreement with the
  main fit = trait conclusions do not rest on the mix; disagreement is itself a result about
  timing.
- *check, descriptor as covariate*: one fit adding a stage sampling conditions term; its slope
  near 0 = the mix does not shift contrasts systematically. Read cautiously: it uses a predictor
  slot, may track data sparsity or lineage, and may absorb real biology (timing is part of being
  a subadult).

Not pursued: a balanced-species subset (about 12 species, mostly *Rana*: too few, and
phylogenetically clustered); a potential-bias adjustment (brings back a weather model with m11g's
assumptions); a "timing component" (realized - same-condition) as a stated question: the species
with both estimates are not a representative set, so it is discussed species by species rather
than tested as a general result.

How m11g handled the same issue (for comparison): its stage-shared weather slopes made every
contrast a weather-adjusted one by assumption; timing not captured by weather stayed in the
contrast unreported (no event level, stage-specific site-year intercepts); and where stage and
temperature were nearly collinear within a species (e.g. *P. crucifer*), partial pooling settled
the trade-off by pulling that species' weather slope toward the across-species mean, dominated
by *R. sierrae*. The unified `05` model then treated all contrasts as the same quantity.

**Environment and the contrast** (Q8, Q9). No separate pooled climate model: the event intercepts
already absorb weather shared by both stages (weather is near-constant within an event), the GAM
no longer needs climate priors, and environment effects on overall infection are not a target
question. What remains is whether the *contrast* changes with conditions, asked within species:

- *Environment screen* (`04_contrast_environment_screen.qmd`, run 2026-11-26): empirical
  contrasts of co-sampled events (442 events in 22 species for prevalence; 229 in 15 for load,
  >= 2 positives per stage) with their sampling se, meta-regressed on within-season and
  between-year deviations of 30-day temperature and log2 10-day precipitation (species-varying
  within-season slopes, site random effect). Uses no stage 1 output. Results:
  - load: larger subadult - adult gap in warmer site-years, +0.28 log units per degree C
    [0.08, 0.48], pd 0.998; within-season warmth the same direction, weaker (+0.14 [-0.06,
    0.35], pd 0.93). Environment lowers the event-level SD from 1.74 to 1.63 (~12% of the
    variance). The year slopes are shared across species, so in effect event-weighted:
    *R. sierrae* holds 134 of 229 load events.
  - prevalence: smaller gap in wetter site-years, -0.12 per doubling of precipitation
    [-0.24, 0.00], pd 0.97; other terms near 0. The event-level SD is barely identified (the
    sampling se dominates), so variance explained cannot be read.
  - all seasonal-environmental (raw lags; site-year means also shift with when in the season a
    year was sampled), not weather effects.
- *Per-species moderation fits* (option B), if pursued: stage 1 refits adding `stage_c x`
  within-site-year weather deviations (main effects absorbed by the event intercepts), only for
  parts whose contrast varies at that level (tier A for weather; tiers A-C for site climate), so
  the moderator does not take credit for all unit-to-unit variation. "How much" = the reduction
  in the event (site) contrast SD. Kept as new job variants; the main fits stay the source for
  Q1-Q6.
- *Site climate* (Q8) needs long-term normals (e.g. WorldClim / CHELSA bioclim at site
  coordinates): the site mean of swab-date weather is "climate at sampling time". A weather
  interpretation (Q9) needs a climatology / anomaly split (multi-year site x day-of-year normals).

**GAM** (`size_bd_gam/`, option 1): add the event intercept `(1 | pop_year_event)` and drop the
climate terms and their m11g prior. The SVL curve is then estimated from size differences within
events, with nothing used twice. Cost: the between-event size comparisons, which are exactly the
ones confounded with timing. Share of SVL variance within events vs within site-years (the
current structure): little change for the data-rich *Rana* (*R. sierrae* 0.56 vs 0.59,
*R. muscosa* 0.58 vs 0.61), large where size and timing go together (*P. crucifer* 0.16 vs 0.60,
*R. pipiens* 0.27 vs 0.57, *R. catesbeiana* 0.42 vs 0.75). If climate terms are ever needed:
per species, not one global slope (*Bd* has a thermal optimum, so a global slope can have the
wrong sign for a species).

## Files and run order

| File | Role | Status |
|---|---|---|
| `contrast_data_functions.R` | shared constants, grouping columns, co-sampling counts, contrast rule, nuisance structure | written |
| `01_contrast_tiers.Rmd` | event definition screen (unit definitions x swab thresholds), counts, inclusion, tiers, time-segregated flag, stage sampling conditions; writes the event screen, species table and cluster job table | written |
| `contrast_model_functions.R` | data prep, formula and prior builders by tier (copied into the cluster script) | written |
| `02_contrast_modeling.qmd` | formula and priors by tier, prior validation, local timing, fit / load, model checks | written |
| `r-brms-contained/size_bd_contrast_modeling.R` + `hpc/` | cluster fits, one job per array index (`hpc/submit_size_bd_contrast.sh` submits the four resource groups) | written |
| `03_contrast_analysis.Rmd` | stage 1 results (mean contrasts, contrast sds, sign consistency, variance shares, group contrasts), consistency checks, exports for the downstream models | written |
| `04_contrast_environment_screen.qmd` | environment screen: empirical contrasts of co-sampled events vs within-season and between-year weather (Downstream models) | written, run |
| predictive model (file to be named) | species mean contrasts ~ phylogeny and traits | planned |

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
- **Same-condition vs realized contrast**: the subadult - adult difference under the same
  conditions (co-sampled events) vs as each stage experiences them (timing, weather, habitat
  use). A mean contrast blends the two (How to read the results).
- **SMD (standardized mean difference)**: a difference between two groups in units of their
  pooled SD; here, how differently the stages are sampled (stage sampling conditions).
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
- 2026-11-26: downstream structure (Downstream models): stage 1 feeds a predictive model
  (replaces unified `05`) and within-species environment analyses; no separate pooled climate
  model; GAM option 1 (event intercepts, no climate terms; no m11g prior). Predictive model:
  development mode dropped (2 direct developers), body size and stage gap added.
- 2026-11-26: stage sampling conditions (SMD of weather between stages) added to `01` and the
  job table as a descriptor. Drafted as a flag at |SMD| >= 0.25, it marked most species (19-22
  of 31): imbalance is a feature of the data, so it is reported, not flagged. Mean contrasts read
  as "as sampled" (same-condition vs realized; How to read the results). No refit.
- 2026-11-26: environment screen (option D, `04`): load gap larger in warmer site-years (+0.28
  per degree C, pd 0.998; mostly *R. sierrae*), prevalence gap smaller in wetter site-years
  (-0.12 per doubling, pd 0.97); seasonal-environmental, not weather.
- 2026-11-26: predictive model handles the same-condition / realized mix as main (stage 1 mean
  contrasts, all species, descriptor reported) + sensitivity (same-condition species contrasts
  from the environment screen's null model, 22 / 15 species) + one check fit with the descriptor
  as a covariate. Timing component (realized - same-condition) not a stated question: discussed
  per species (Downstream models).

## Open questions

- Load co-sampling uses >= 1 positive of each stage; a stricter positive count may better match
  how much a unit informs the load contrast.
- Contrast SD prior scale (exponential(2)): check with the contrast-SD power-scaling in `02`
  after round 3; refit a few species with exponential(1) / exponential(4) if it flags mean
  contrasts.
- R3: whether species with a constant sigma combine data sources with different measurement
  precision (`project`) has not been checked.
- Predictive model: final predictor set and model grouping; which descriptor term for the check
  fit (abs SMD of temperature, or share of swabs in co-sampled events); export of the
  same-condition species contrasts from `04` (null models); species-typical climate.
- Long-term site climate (bioclim at site coordinates) for Q8 and species-typical climate; then
  the between-site terms in `04`.
- Per-species moderation fits (option B): worth the refits given the screen's year-level load
  signal? Which parts (tier A weather only, or also site climate)?
- Potential-bias measure (imbalance x the species' climate slope for overall infection), if any
  imbalanced species becomes central to the paper.
- Climatology / anomaly split, needed only for a weather interpretation.
- GAM option 1 to implement in `size_bd_gam/02_size_bd_GAM_modeling.qmd`.
- Check *C. panamansis* development mode in `iucn_habitat_metrics.csv` (probably larval; moot
  for the predictive model, which drops development mode).
- Retirement to-dos: manuscript inputs (`unified_03_*`, `unified_05_*`) re-pointed to contrast
  exports; m11g consistency check in `03`; species-level mean contrasts for the six m11g-only
  species reported as excluded.
