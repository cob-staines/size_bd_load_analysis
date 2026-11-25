# CLAUDE.md - size-bd analysis

## General guidance
- Always review CLAUDE.md and git status when starting a new session
- Never commit sensitive information (logins, et.c) to git. Throw loud warnings if the user asks for this.
- In response to user prompts, always summarize proposed actions for user review before implementing.
- Never commit without user permission.
- Prompt user to commit coherent changes to git on a regular basis.

## Project-specific guidance

Purpose: In this study we compared Bd infection intensities between recently metamorphosed amphibians and mature adults, for a geographically and phylogenetically diverse set of species, to provide insights into disease and population dynamics and resistance/tolerance mechanisms, and consider implications for conservation. Using an extensive collection of in situ amphibian survey and swab data from across the Americas, we modeled within-species differences in infection with body size as a proxy for age, to explore how common the pattern of greater infection in subadults is across species. We examined how infection dynamics relate with environmental and life-history factors, and across phylogeny, to provide context for disease and population dynamics and inform future investigations of resistance and tolerance mechanisms. Finally, we characterized this dynamic with finer age resolution for a subset of data-rich species, to support our interpretation of contrasting disease dynamics between age classes.

To achieve this purpose:
- All analysis must be repeatable with scripts
- The human researcher(s) are using this project both to advance understandings of Bd dynamics, and to learn how to run complex bayesian analysis in R. Responses should work toward supporting understanding, not just a final project. Comments in any code help with this.
- Ask for permission any time you are seeking to compile or model as an agent. By default the script should be run by the humans, and you will be asked to assist in brainstorming and script writing.

### Learning mode
- Explain with this project's own fits and numbers (e.g. a species' posterior), not generic examples.
- Treat restatements ("is this accurate?") as understanding checks: say what is right, correct the specific slip.
- Use Bayesian reporting language: posterior median, 50 / 95% intervals, pd / support labels (`f_draw_agg`); "informed" vs "prior-dominated". Avoid "significant" and "confidence interval".
- Each analysis folder's README defines its terms (e.g. `size_bd_contrast/README.md`: Definitions, How to read the results, Glossary, Decision log). Keep these current when decisions change, with a dated decision-log entry.

### Modeling conventions
- Change a model for computational failure (divergences, poor mixing, terms the data cannot identify), not because a parameter is uninformed; report uninformed parameters as such.
- Thresholds and rules are fixed a priori from the sampling design, not tuned on results; report sensitivity at the thresholds instead.
- Ask before any brms refit; prefer checks on existing fits (power-scaling, cached draws).

### Cluster workflow (r-brms-contained)
- Cluster scripts carry verbatim copies of local helpers (e.g. `size_bd_contrast/contrast_data_functions.R` and `contrast_model_functions.R` in `r-brms-contained/size_bd_contrast_modeling.R`). Change both copies together and diff them.
- Before a submission: pull `r-brms-contained` on the cluster and copy the current jobs file into `data/`. Commit the two repos separately.
- Size memory and time from `sacct` (MaxRSS, Elapsed) of earlier runs, not estimates.