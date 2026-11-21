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