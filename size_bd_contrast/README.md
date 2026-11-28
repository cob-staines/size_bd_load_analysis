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
  history (extended larval period)? (predictive model, across species; climate and body size
  left out, see Downstream models)
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
By that check (2026-11-27), contrast SDs are mostly prior-shaped in every tier: tiers B / C
strongly (sensitivity 0.17-0.42), tier A less (mostly 0.03-0.19). Only *R. sierrae* (load event
and site, prevalence event) and *R. muscosa* (load site-year) are data-dominated (< 0.05). Read
tier A SDs as "better informed", not as fixed by the data; their upper bounds still carry the
reading above.

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
summaries (species / lab stats, sex composition) and the naive cutoff bootstrap. Manuscript
figures and tables re-pointed to the contrast exports 2026-11-27 (decision log). The m11g
consistency check stays in the `03` code, not reported (decided 2026-11-27).

## Downstream models

Decided 2026-11-26. Stage 1 feeds two downstream analyses; the GAM runs separately:

```
stage 1 contrast models (per species)
   |-> predictive model           species mean contrasts ~ phylogeny + traits (no climate)
   |-> environment and contrast   within species: does the contrast move with conditions?
                                  environment screen (04) -> per-species moderation fits if warranted
GAM (06, 07)                      finer age resolution; stage 1 nuisance structure, no climate terms
```

**Predictive model** (replaces `size_bd_unified/05_species_regression.qmd`; decided 2026-11-26).

- *Response*: species mean contrasts, prevalence and load in one multivariate model
  (`mvbf`, `subset()` for load: *B. faber* has no load contrast), so the phylogenetic
  correlation between the two parts is estimated. 31 species for prevalence, 30 for load.
- *Uncertainty*: each species' posterior mean and sd of its mean contrast in
  `se(sd, sigma = TRUE)`. Assumes roughly normal posteriors: checked with half-eye plots of the
  draws (`03`, and `04` for the same-condition contrasts); refits over posterior draws only if
  some are clearly skewed.
- *Tier D*: a tier D interval covers the sampled sites only, so its se is inflated with the
  missing between-site part of a species-wide mean: se_inflated^2 = se^2 + tau^2 / n_co_pops,
  tau = median site-level contrast SD over the informed species of that part, n_co_pops = the
  species' co-sampled populations (that part's definition). Fixed before fitting.
- *Predictors* (standardized): log habitat breadth, aquatic index, extended larval period
  (binary: age at metamorphosis > 12 months, i.e. overwintering larvae; 8 species: *A. mavortium*,
  *A. canorus*, *R. cascadae*, *R. catesbeiana*, *R. clamitans*, *R. luteiventris*, *R. muscosa*,
  *R. sierrae*; from `data/species_notes_2026-11-26.csv`, coding fixed for *R. catesbeiana*,
  *R. clamitans* and *H. phyllodes* on 2026-11-26), plus a phylogenetic random effect
  (`(1 | p | gr(taxon_capture, cov = A))`, correlation matrix from
  `size_bd_unified/intermediates/phylo_tree.nwk`). The extended group overlaps partly with
  *Rana* (the phylogenetic effect competes) and with the Sierra Nevada epizootics; the two
  eastern bullfrogs weaken the overlap with region.
- *Left out*:
  - species-typical climate: across species it is almost a region label (the Panama species all
    at ~23 degrees C and ~3500 mm, the Sierra Nevada species at 0-1 degrees C, the Brazilian
    species at ~19 degrees C), entangled with lab and lineage, and within species long-term site
    climate explains little of the contrast (environment screen, below). Available in
    `size_bd_unified/intermediates/species_bioclim.csv`.
  - body size: correlated with habitat breadth (log adult SVL vs log habitat breadth, r = 0.71).
  - stage gap (subadult / adult SVL): partly reflects sampling (metamorphs vs older juveniles)
    and the SVL cutoff, and is mechanically tied to the load response: load density = copies /
    SVL, so the density contrast = the per-swab contrast - log(gap), and a gap slope would be
    fixed at -1 in part by definition. Size questions belong in the GAM. (*R. luteiventris* has
    no SVL: a constant placeholder of 25 mm, since its swabs used standardized stroke lengths;
    life stage was assigned from morphology. Fine for stage 1; it would have been dropped had
    SVL-based predictors been used.)
  - development mode: 1 direct developer after the coding fixes (*I. henselii*).
  - age at maturity: mostly inferred from congeners at low confidence.
  - candidates for later models: disease history (Bd-associated decline status, e.g.
    Scheele et al. 2019; or epizootic vs enzootic phase), larval habitat (stream vs pond),
    breeding phenology.
- *Variants*: an ordinal version of age at metamorphosis (`mo()`, D < R < M < E) as a
  sensitivity fit; only its total effect is readable (D and R hold 1 and 2 species).

The mix of same-condition and realized contrasts across species (How to read the results) is
handled in three parts (decided 2026-11-26):

- *main*: the stage 1 mean contrasts of all species (31 for prevalence, 30 for load), with the
  stage sampling conditions (SMDs, share of swabs in co-sampled events) reported next to every
  species-level result.
- *sensitivity, same-condition response*: the same model on species mean contrasts from
  co-sampled events only (22 species for prevalence, 15 for load). Every species then
  contributes the same kind of contrast. Agreement with the main fit = trait conclusions do not
  rest on the mix; disagreement is itself a result about timing. Prevalence comes from a
  binomial model on the co-sampled events' counts, load from the empirical event contrasts
  (environment screen, fixed-species null model). The empirical prevalence logits were replaced
  on 2026-11-27 (Same-condition prevalence estimators, below); they stay as a labelled
  comparison fit.
- *check, descriptor as covariate*: one fit adding the share of swabs in co-sampled events
  (prevalence definition for the prevalence part, load definition for the load part; chosen over
  |SMD| of temperature, which would miss timing differences in aseasonal Panama); its slope
  near 0 = the mix does not shift contrasts systematically. Read cautiously: it uses a predictor
  slot, may track data sparsity or lineage, and may absorb real biology (timing is part of being
  a subadult).

Results so far (`05`, 2026-11-27; medians, 95% intervals):

| term | main | same-condition (binomial prevalence) | main on same-condition species |
|---|---|---|---|
| prevalence, extended larval period | +1.43 [0.72, 2.12], pd 1.00 | +0.55 [-0.30, 1.37], pd 0.91 | +1.40 |
| load, extended larval period | +1.34 [0.34, 2.24], pd 0.99 | +0.72 [-0.81, 2.16], pd 0.83 | +0.79 |
| prevalence, log habitat breadth | -0.54 [-0.90, -0.16], pd 1.00 | -0.20 [-0.65, 0.28], pd 0.81 | -0.50 |
| load, log habitat breadth | -0.66 [-1.14, -0.16], pd 0.99 | -0.49 [-1.24, 0.35], pd 0.88 | -0.56 |
| aquatic index | ~0 | ~0 | ~0 |

- *robust*: to any single species (leave-one-out; *A. canorus* moves it most, to +1.03
  prevalence and +0.97 load, interval just touching 0 for load; *A. mavortium* is not the driver,
  +1.22 load without it), to the check fit (co-sampling share slope ~0: -0.01 prevalence, -0.12
  load, pd 0.70; trait slopes unchanged), to an ordinal coding of age at metamorphosis, and to
  dropping the 11 time-segregated species, where it gets larger (prevalence +1.70 [0.89, 2.41],
  load +1.64 [0.51, 2.65]; habitat breadth unchanged).
- *weaker on same-condition contrasts*: the extended-larval effect keeps its direction at
  ~40-55% of its main size. For load the gap is the species set (the main model on the
  same-condition species already drops to +0.79); for prevalence it is the response (the main
  model on those species keeps +1.40), concentrated in species whose same-condition estimate is
  realized-dominated in stage 1 (*R. muscosa*, *R. pipiens*) or mostly prior (1-2 co-sampled
  events: *P. crucifer*, *L. sphenocephalus*, *H. versicolor*). The time-segregated species are not
  what carries the main effect (their contrasts are mixed in sign and noisy; dropping them
  sharpens it). For load, the species without co-sampled load events do carry part of it.
- *phylogenetic structure (Q7)*: informed without traits. In an intercept-only phylogenetic
  model, H2 is 0.99 [0.69, 1.00] (prevalence) and 0.96 [0.56, 1.00] (load), against a roughly
  uniform prior (median 0.52, 95% 0.00-1.00): related species have similar contrasts. With the
  traits in the model, H2 falls back to the prior's range (0.87 [0.02, 1.00], 0.76 [0.01, 1.00]):
  the extended larval period (6 of 8 species *Rana*) takes up most of the clade structure, and
  loo shows no clear gain from the phylogeny once traits are in (elpd difference -2.6, se 2.7,
  prevalence; -1.5, se 1.8, load). The extended effect stays positive with the phylogenetic
  effect included, so there is trait signal beyond relatedness, resting on the species that
  break the overlap: non-*Rana* extended species (*A. canorus*, the most influential;
  *A. mavortium*) and non-extended *Rana* (*R. boylii*, *R. pipiens*, *R. sylvatica*).
- *prevalence - load correlation*: the total cross-species correlation (phylogenetic +
  non-phylogenetic species effects, traits in the model) is 0.51 [-0.02, 0.88], pd 0.97:
  species with larger prevalence gaps tend to have larger load gaps. The phylogenetic-only split
  is not informed (main 0.55 [-0.48, 0.94]); without traits it is 0.87 [0.44, 0.98], partly
  shared trait structure. Within-species estimation errors of the two parts are uncorrelated
  (|r| <= 0.013 for every species), so this is not an estimation artefact.
- *asynchrony, narrowed*: some asynchronous, sparsely co-sampled species have strongly negative
  realized contrasts (*P. crucifer* -2.17, *L. sphenocephalus* -1.40; and, not time-segregated,
  *R. pipiens* -2.05, *P. regilla* -1.76). A plausible mechanism: their subadults are swabbed in
  warmer conditions (*P. crucifer* SMD 2.8, *R. pipiens* 1.5), so if Bd is lower then, realized ~
  same-condition + (change in overall infection with conditions) x (difference in stage
  sampling conditions) comes out negative. But as a group the time-segregated species are mixed
  in sign and do not carry the extended-larval effect, and there is no linear trend in the
  co-sampling share (check fit). Discussed per species, as a hypothesis about realized
  contrasts, not as a general pattern. (The time-segregated flag cannot mark species with no
  co-sampled site-years, e.g. *P. regilla*, *A. varius*, *R. haematiticus*, *S. sila*; the main
  model on the same-condition species covers those.)
- *synchrony is not a main covariate* (decided 2026-11-27): the check fit already tests it (no
  shift); it is partly caused by life history (overwintering larvae make stages co-occur), so
  adjusting for it would remove part of the trait effect; it reflects survey design as much as
  biology; and the pattern is concentrated in a few species, not linear. Instead: the main fit
  without the 11 time-segregated species (flag set a priori) as a targeted sensitivity fit.
- provisional wording: the subadult - adult contrast is phylogenetically structured, and
  species with extended (overwintering) larval periods show larger contrasts in both prevalence
  and load. The trait effect holds whichever single species is dropped and without the
  time-segregated species; it overlaps with phylogeny (most extended species are *Rana*), so its
  signal beyond relatedness rests on a few species; and on within-event (same-condition)
  comparisons it keeps its direction at about 40-55% of its size with weaker support (prevalence
  +0.55, pd 0.91). Species with larger prevalence gaps tend to have larger load gaps (total
  correlation 0.51, pd 0.97). Wider-ranging species show smaller contrasts in the stage 1-based
  fits, weakly or not at all on same-condition contrasts for prevalence. Aquatic index: no
  association.

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
- *Per-species moderation fits* (option B; on hold since 2026-11-26), if pursued: stage 1 refits adding `stage_c x`
  within-site-year weather deviations (main effects absorbed by the event intercepts), only for
  parts whose contrast varies at that level (tier A for weather; tiers A-C for site climate), so
  the moderator does not take credit for all unit-to-unit variation. "How much" = the reduction
  in the event (site) contrast SD. Kept as new job variants; the main fits stay the source for
  Q1-Q6. Given the screen, the justified scope would be weather only, for the tier A parts:
  site-climate moderators are low priority.
- *Site climate* (Q8): long-term normals (CHELSA bioclim 1981-2010 at site coordinates,
  `size_bd_unified/site_bioclim.Rmd`), not the site mean of swab-date weather ("climate at
  sampling time"). Added to the screen as a third model (`site`: weather terms + annual mean
  temperature, temperature seasonality, log2 annual precipitation, precipitation seasonality,
  each population's climate centered within species, shared slopes). The between-site slopes rest
  on species with co-sampled events at several sites spanning a climate range: *R. sierrae* (42
  populations, 6.1 degrees C range), *R. muscosa* (16), *R. clamitans* (11, 13 degrees C),
  *R. cascadae*, *N. viridescens*, *R. catesbeiana* (5-7). Mean temperature and temperature
  seasonality are correlated (r = -0.73 among co-sampled events), so their separate slopes are
  less certain than their joint effect. Results (run 2026-11-26):
  - the weather slopes do not move when site climate is added (load temp_year +0.30 [0.10, 0.50],
    pd 0.998; prevalence precip_year -0.11 [-0.23, 0.02], pd 0.96): the between-year
    associations are not explained by which climates the sites sit in.
  - site climate shows weak evidence at most. Load: smaller gap at sites with more seasonal
    rainfall, -0.46 per 10 points of precipitation seasonality [-1.08, 0.15], pd 0.93 (1 of 8
    site-climate slopes; mostly the Sierra Nevada *Rana*); mean temperature, temperature
    seasonality and annual precipitation near 0 (pd 0.63-0.77). Prevalence: all four near 0
    (pd 0.54-0.86).
  - load site SD 0.77 -> 0.66 (taken at face value ~27% of the between-site variance in the load
    contrast, but the intervals overlap widely: 0.27-1.23 vs 0.11-1.15); prevalence site SD
    unchanged (0.24 vs 0.23).
  - reading for Q8: within species, little sign that long-term site climate shapes the contrast;
    precipitation seasonality for load is a weak, mostly Sierra Nevada candidate.

  A weather interpretation (Q9) needs a climatology / anomaly split (multi-year site x
  day-of-year normals).

**Same-condition prevalence estimators** (`04`, 2026-11-26/27). Three ways to estimate a
species' prevalence contrast from its co-sampled events, compared with stage 1:

| estimator | treats event data as | limitation | *R. sierrae* | *R. muscosa* |
|---|---|---|---|---|
| empirical logits | one Haldane-corrected logit difference per event, normal with known se | capped near +/- 2.4 in small all-or-nothing cells (60% of *R. sierrae*'s subadult cells) | 0.39 | 0.48 |
| Mantel-Haenszel | pooled 2 x 2 tables | assumes one common odds ratio across events | 0.61 | 0.42 |
| binomial model | counts per event and stage, event intercepts and contrasts | one event-contrast SD shared across species | 0.81 | 0.30 |
| (stage 1) | | | 0.99 | 1.29 |

- The binomial model is the stage 1 prevalence part restricted to co-sampled events: a Bernoulli
  model on swabs and a binomial on counts per event and stage give the same likelihood, and the
  event term is the same as stage 1's tier A `(1 + stage_c | pop_year_event)`. Simplifications:
  one model with species as fixed effects (no pooling before the predictive model), one
  event-level contrast SD shared across species, no site-year level. What separates it from
  stage 1 is the data (co-sampled events only), not the structure: the same-condition vs
  realized distinction.
- Result: close to stage 1 for most species (*R. sierrae*, *R. cascadae*, *R. catesbeiana*,
  *L. chiricahuensis*, *A. canorus*), so the earlier same-condition gap was mostly the
  estimators. Real exceptions: *R. muscosa* (0.30 vs 1.29, all three estimators agree) and
  probably *R. pipiens* (-0.84 vs -2.05, 5 events). Deviations go both ways (*C. panamansis*,
  *S. flotator* larger), so stage 1 is not systematically inflated. Species with 1-2 co-sampled
  events are mostly prior (sd 1.3-1.8 vs prior sd 2) and get little weight downstream.
- Stage 1 is not restricted to co-sampled units: that would drop 9 species, cut others to a
  fraction of their swabs, lose the intercept information from single-stage events, and answer
  only the same-condition question. Stage 1 stays the main (as-sampled) estimate; the binomial
  contrasts are the same-condition sensitivity.

**GAM** (`06_gam_modeling.qmd`, `07_gam_analysis.Rmd`; option 1, decided 2026-11-26,
written 2026-11-27; replaces `size_bd_gam/02` and `03`, kept until retired). A secondary,
qualitative model for species in the discussion, kept as close as possible to each species'
stage 1 model:

- same as stage 1: response and family, the nuisance intercepts `(1 | population)`,
  `(1 | pop_year)`, `(1 | pop_year_event)`, which of them each part keeps (the R2 / R3 flags of
  `01_contrast_species.csv`, reused, not recomputed), the nuisance priors, and the event
  definition (`event_ids()`)
- different by design: metamorphs included (an intercept for the metamorph baseline), and a thin
  plate smooth `s(svl_mm, by = is_not_metamorph)` (`s(svl_mm)` without metamorphs) in place of
  `stage_c`. Smooth terms keep brms' default priors
- dropped: the climate terms and their m11g prior (nothing used twice), and the stage-specific
  site and site-year intercepts of the earlier GAM (`m3`, inherited from m11g). Those let the
  stage gap vary by site as a step at the SVL cutoff, an artifact of the cutoff for SVL-staged
  species, and asked stage 1's Q2-Q3 again. No varying size effect either
- species: the 14 with >= 100 swabs of each stage in the stage 1 data. A post-hoc description of
  the existing GAM data (`size_bd_gam/00`, chosen by hand for SVL coverage): all 14 have >= 130,
  every other stage 1 species <= 97 (*A. americanus*, 97 adults). Others can be added by name
  (data prep in `size_bd_gam/00`), labeled as below the cutoff
- events are built from all the species' swab dates, metamorphs included, so event counts can
  differ slightly from stage 1

How informed a curve is: the event intercepts are random effects, so comparisons between events
are down-weighted (by how much events vary anyway), not discarded. The curve rests on the same
comparisons as the species' stage 1 contrast and is about as informed, with the same caveats.
E.g. *P. crucifer*: 1 co-sampled event of 191, time-segregated, yet a stage 1 prevalence contrast
of -2.16 [95%: -3.24, -1.20], pd 1, from comparisons between events within its 9 co-sampled
site-years. Co-sampled event counts therefore describe a curve but are not used to include or
exclude species. `06_gam_species.csv` carries each species' stage 1 tier, counts, co-sampled
events, time-segregated flag and contrast support, for choosing which curves to show. Share of
SVL variance within events vs within site-years: *R. sierrae* 0.56 vs 0.59, *R. muscosa* 0.58 vs
0.61, *P. crucifer* 0.16 vs 0.60, *R. pipiens* 0.27 vs 0.57, *R. catesbeiana* 0.42 vs 0.75 (low
within-event shares: across the stage boundary the curve leans on between-event comparisons,
where size and timing go together). If climate terms are ever needed: per species, not one
global slope (*Bd* has a thermal optimum, so a global slope can have the wrong sign for a
species).

## Files and run order

| File | Role | Status |
|---|---|---|
| `contrast_data_functions.R` | shared constants, grouping columns, co-sampling counts, contrast rule, nuisance structure | written |
| `01_contrast_tiers.Rmd` | event definition screen (unit definitions x swab thresholds), counts, inclusion, tiers, time-segregated flag, stage sampling conditions; writes the event screen, species table and cluster job table | written |
| `contrast_model_functions.R` | data prep, formula and prior builders by tier (copied into the cluster script) | written |
| `02_contrast_modeling.qmd` | formula and priors by tier, prior validation, local timing, fit / load, model checks | written |
| `r-brms-contained/size_bd_contrast_modeling.R` + `hpc/` | cluster fits, one job per array index (`hpc/submit_size_bd_contrast.sh` submits the four resource groups) | written |
| `03_contrast_analysis.Rmd` | stage 1 results (mean contrasts, contrast sds, sign consistency, variance shares, group contrasts), consistency checks, exports for the downstream models | written |
| `04_contrast_environment_screen.qmd` | environment screen: empirical contrasts of co-sampled events vs within-season and between-year weather, and long-term site climate (Downstream models) | written, run (weather and site models) |
| `05_contrast_predictive_model.qmd` | predictive model: species mean contrasts ~ log habitat breadth + aquatic index + extended larval period + phylogeny (main, same-condition, check, ordinal fits) | written |
| `gam_model_functions.R` | GAM formula and prior builders, shared run settings (copied verbatim into the cluster script) | written |
| `06_gam_modeling.qmd` | GAM: per-species prevalence and load by SVL (thin plate smooth + metamorph baseline) with the stage 1 nuisance intercepts; species table, structure check, timing test, fits (local or cluster), fit health, prior sensitivity (Downstream models) | written, fits pending |
| `r-brms-contained/size_bd_contrast_gam_modeling.R` + `hpc/` | GAM cluster fits, one species per array task (`hpc/submit_size_bd_contrast_gam.sh` submits the four resource groups; provisional until the timing test) | written |
| `07_gam_analysis.Rmd` | GAM curves (average site / site-year / event), figures, manuscript exports (off until the fits are reviewed) | written |

Outputs: `intermediates/` (tables), `models/` (fits, not tracked), `figures/`.

Upstream: `size_bd_unified/01_model_data.Rmd` (model data); GAM data from `size_bd_gam/00`-`01`.

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
- **CJS distance** (power-scaling sensitivity): how far the posterior moves when the prior (or
  likelihood) is scaled. It compares the posterior as fitted with the posterior under the scaled
  prior, not prior with posterior. ~0 = the posterior does not follow the prior (data dominate,
  informed); > 0.05 = it does (priorsense's threshold; the prior does part of the work). A
  parameter whose posterior equals its prior has a high value: any prior change passes straight
  through. "Prior-likelihood conflict" = both prior and likelihood values > 0.05; for group SDs
  this is common and mostly means both matter.
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
- 2026-11-26: long-term site climate from CHELSA v2.1 bioclim, 1981-2010, 30 arc-seconds (period
  close to the KG raster's; better mountain precipitation than WorldClim): bio1 annual mean
  temperature, bio4 temperature seasonality, bio12 annual precipitation, bio15 precipitation
  seasonality (`size_bd_unified/site_bioclim.Rmd`). Species-typical climate = mean over the
  species' sites, each site weighted equally.
- 2026-11-26: between-site climate added to the environment screen (`04`, `site` model) from the
  CHELSA site climate; species-typical climate (site-weighted means and ranges) written by
  `site_bioclim.Rmd` for the predictive model.
- 2026-11-26: site-climate screen results (`04`, `site` model): weather slopes unchanged; site
  climate weak at most (load: precipitation seasonality -0.46 per 10 points, pd 0.93; others
  near 0). Q8 read as little within-species evidence. Option B on hold (if revisited: weather
  only, tier A). Species-typical climate left out of the predictive model (regional, entangled
  with lab and lineage; little within-species signal); file kept.
- 2026-11-26: predictive model specified (Downstream models): one multivariate model for both
  parts; predictors log habitat breadth, aquatic index, extended larval period (binary, 8 of 31
  species) + phylogeny; mean / sd in se(); tier D se inflated (se^2 + tau^2 / n_co_pops); body
  size, stage gap, climate, development mode and age at maturity left out; check fit uses the
  share of swabs in co-sampled events (load definition for load); same-condition contrasts from
  a fixed-species null model in `04`.
- 2026-11-27: same-condition prevalence response = binomial model on co-sampled events' counts
  (`04`), replacing the empirical logits, which the Haldane ceiling compressed (Mantel-Haenszel
  in between). Binomial close to stage 1 for most species; *R. muscosa* (0.30 vs 1.29) a real
  exception. Stage 1 stays on all units (Same-condition prevalence estimators). The empirical
  version stays in `05` as a labelled comparison fit.
- 2026-11-27: predictive model robustness (Downstream models): the extended-larval effect holds
  in every leave-one-out fit (not driven by *A. mavortium*; *A. canorus* most influential) and
  keeps its direction, at ~40-55% size, on same-condition contrasts (binomial prevalence +0.55,
  pd 0.91). Synchrony stays a check, not a main covariate (partly caused by life history);
  sensitivity fit without the 11 time-segregated species added.
- 2026-11-27: predictive model checks run (Downstream models): without the time-segregated
  species the extended-larval effect gets larger (prevalence +1.70, load +1.64), so it does not
  rest on their realized contrasts; raw phylogenetic signal of the contrast informed (H2 0.96-0.99
  without traits) and largely taken up by the traits; total prevalence - load correlation 0.51
  (pd 0.97); within-species estimation errors uncorrelated; phylogeny adds no clear predictive
  value once traits are in.
- 2026-11-27: stage 1 power-scaling reviewed (`02`, exports `02_powerscale_all_priors.csv`,
  `02_powerscale_contrast_sd_prior.csv`). Contrast-SD prior alone: 32 of 39 mean contrasts
  < 0.05; the 7 above are mild (0.05-0.10; largest *R. muscosa* prevalence 0.10, the contrast
  with large between-unit variation). So the exponential(2) scale is kept
  without refits. Contrast SDs: 69 of 73 prior-sensitive, tier A less so (How to read the
  results). All priors together: load mean contrasts clean in 26 of 31 (flags = wide tier D
  posteriors and *R. pipiens*); prevalence flagged in 18 of 32 (0.05-0.21; largest
  *R. luteiventris* 3.20 +/- 1.15), not from the contrast-SD prior; split by prior part added
  (`02` "Which prior moves the mean contrast?").
- 2026-11-27: prior split run (`02_powerscale_prior_split.csv`; all pareto k <= 0.65). The
  prevalence flags come mainly from the contrast prior itself (normal(0, 2)); intercept and
  nuisance priors move almost nothing (1 of 63 contrasts > 0.05). 49 of 63 mean contrasts shift
  < 0.2 posterior SD under every prior part (a 95% interval is ~4 SD). Larger shifts are all
  shrinkage toward 0: *R. luteiventris* prevalence (tier D, contrast prior, +/- 0.66 SD;
  prior-shaped, reported as such), *A. mavortium* load (0.43), *R. muscosa* prevalence
  (contrast-SD prior, 0.45), and the large negative prevalence contrasts of *B. faber*,
  *P. regilla*, *R. pipiens*, *P. crucifer* (0.25-0.39). Shrinkage at both ends narrows the
  extended vs other gap, so the extended-larval slope in `05` is conservative with respect to
  the stage 1 priors. No refits: the reweighted posteriors already show the looser-prior answer
  and no fit failed computationally.

- 2026-11-27: manuscript figures and tables re-pointed from m11g to the contrast exports
  (`size-bd-dynamics-manuscript/build_intermediates.Rmd`, captions in `main_text.qmd`; text not
  revised). Species contrasts (cascade, scatters, tree maps, combined rank, contrast table) from
  `contrast_03_a`, tier D drawn as open symbols; phylogenetic signal table and trait figure from
  `contrast_05_b` / `contrast_05_a` (main fit, 50 / 95% intervals); population figures replaced
  by per-population contrasts (`contrast_03_c`) and contrast SDs by level (`contrast_03_b`);
  weather and variance-component figures removed. Data summary table, sex check and seasonal
  size structure restricted to the 31 contrast species (the six m11g-only species dropped).
  Main-text contrast figure = contrasts on the phylogeny (posterior medians, 50 / 95%, support,
  tier D open, extended larval period marked), replacing the cascade (moved to the supplement,
  ranked) and the contMap trees (ancestral states reconstructed from medians, ignoring
  uncertainty). Same-condition estimates left out of the manuscript figures for now. m11g
  consistency check kept in the `03` code, not reported.

- 2026-11-27: GAM option 1 written as `06_gam_modeling.qmd` / `07_gam_analysis.Rmd` in this
  folder (shared event definition, flags and priors; `size_bd_gam/02`-`03` kept until retired).
  Matches each species' stage 1 model except for metamorphs and the SVL smooth: nuisance
  intercepts and their R2 / R3 flags from `01_contrast_species.csv`, stage 1 nuisance priors
  (smooth terms at brms defaults), no climate terms or m11g prior. The `m3` stage-specific site and
  site-year intercepts dropped (a step at the SVL cutoff; stage 1's Q2-Q3). Species: >= 100
  swabs of each stage in the stage 1 data, set post hoc to describe the existing 14 (>= 130 vs
  <= 97 for the rest). A minimum of co-sampled events was considered and rejected: with random
  event intercepts, between-event comparisons are down-weighted, not discarded, so a species'
  curve is about as informed as its stage 1 contrast (*P. crucifer*: 1 co-sampled event, strong
  stage 1 prevalence contrast); tier and flags are carried in `06_gam_species.csv` instead.
  Fits saved as `models/gam_m4_<species>.rds`, without LOO (no model comparison). Builders and
  run settings in `gam_model_functions.R`, copied into `r-brms-contained/size_bd_contrast_gam_modeling.R`
  for cluster fits (one species per array task; resource groups provisional, from the stage 1
  fits of the same species, until the timing test in `06`).

## Open questions

- Load co-sampling uses >= 1 positive of each stage; a stricter positive count may better match
  how much a unit informs the load contrast.
- R3: whether species with a constant sigma combine data sources with different measurement
  precision (`project`) has not been checked.
- Predictive model inputs written (to run: `01`, `03`, `04`): load-definition co-sampling share
  (`count_co_units()`, `01`); `03_predictive_responses.csv` (species mean / sd of the mean
  contrast, tier D-inflated se, skewness, half-eye plot); `04_same_condition_species_contrasts.csv`
  (fixed-species null model; the random-effect version would pool twice), with a half-eye
  check. Inputs run 2026-11-26: no skewed posteriors (|skew| <= 0.42), so mean / sd suffice;
  tier D inflation small (<= 9%). Three load tier D species have 0 co-sampled load populations
  (*S. sila*, *A. mavortium*, *S. holbrookii*): their load contrasts rest entirely on
  between-population comparisons. Model script: `05_contrast_predictive_model.qmd` (written).
- *H. phyllodes* larval habitat coded D in `data/species_notes_2026-11-26.csv` (stream tadpoles:
  probably A); not used in the model.
- Predictive model: settle the wording of the results (provisional reading in Downstream
  models). Manuscript exports (`contrast_05_*`) and figures enabled 2026-11-27. Optional: prior sensitivity of
  the extended-larval slope (`powerscale_sequence`), since it showed a mild prior-likelihood
  conflict (the prior pulls it in); a simulation of the achievable precision of the residual
  phylogenetic signal (31 species, these se's, this tree).
- *R. muscosa*: its stage 1 prevalence contrast (1.29) is mostly realized (same-condition 0.30 by
  the binomial model, 0.42 Mantel-Haenszel, 0.48 empirical), although its weather is balanced.
  A species-level look at which single-stage events, sites or years carry its subadult signal
  would help its interpretation (*R. muscosa* vs *R. sierrae* is a focal comparison).
- Per-species moderation fits (option B): on hold. If revisited: weather only, tier A parts.
- Potential-bias measure (imbalance x the species' climate slope for overall infection), if any
  imbalanced species becomes central to the paper.
- Climatology / anomaly split, needed only for a weather interpretation.
- GAM (`06`, `07`): fits pending (14 species; timing test in `06`, then cluster or local). Then fit health, prior sensitivity of the
  smooth wiggliness, choice of species for the figures, and re-pointing the manuscript from
  `GAM_03_*` to `contrast_07_*`. *A. americanus* (97 adults) can be added if the discussion needs
  more *Anaxyrus*. Retire `size_bd_gam/02`-`03` once `07` is in use.
- Check *C. panamansis* development mode in `iucn_habitat_metrics.csv` (probably larval; moot
  for the predictive model, which drops development mode).
- Manuscript methods / results / discussion text still describes m11g (figures and captions
  re-pointed 2026-11-27; text to be revised by the authors once the analysis is final).
