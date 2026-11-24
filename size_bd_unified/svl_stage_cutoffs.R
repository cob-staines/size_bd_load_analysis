# subadult / adult stage rules per species and data source: the single definition used by
# 00_survey_data.Rmd (stage assignment) and 01_model_data.Rmd (svl histograms with cutoff lines).
# a tracked .R file rather than a csv (csvs are git-ignored in this repo).
#
# columns:
#   taxon_capture  species
#   dataset        data source the rule applies to: "ribbitr_db" = the RIBBiTR database (all of
#                  its projects); otherwise the `project` value of an external dataset in
#                  00_survey_data.Rmd (kupferberg2022, daversa2026, RACA, BOFA, RALU, usgs)
#   stage_rule     "svl_cutoff": svl_mm >= svl_cutoff_mm -> adult, below -> subadult
#                  "field_label": stage taken from the source's own life stage labels
#   svl_cutoff_mm  cutoff (NA for field labels)
#   min_svl_mm     optional lower bound: subadults must have svl_mm > min_svl_mm (smaller
#                  animals are dropped); NA = no bound
#   reference      source of the cutoff (merged from "data/Species notes - size cutoffs.csv",
#                  which this table replaces; 03_size_bd_unified_analysis.qmd exports it to the
#                  manuscript species table)
#
# requires: tidyverse (tibble, dplyr, purrr) loaded by the calling script

svl_stage_cutoffs = tribble(
  ~taxon_capture,               ~dataset,          ~stage_rule,    ~svl_cutoff_mm, ~min_svl_mm, ~reference,
  # ---- RIBBiTR database: usa ----
  "acris_blanchardi",           "ribbitr_db",      "svl_cutoff",   16,             NA,          "animaldiversity.org/accounts/Acris_blanchardi",
  "anaxyrus_americanus",        "ribbitr_db",      "svl_cutoff",   54,             NA,          "Wright and Wright 1949",
  "anaxyrus_canorus",           "ribbitr_db",      "svl_cutoff",   40,             NA,          "Lannoo 2005",
  "anaxyrus_debilis",           "ribbitr_db",      "svl_cutoff",   37,             NA,          "Lannoo 2005",
  "anaxyrus_fowleri",           "ribbitr_db",      "svl_cutoff",   51,             NA,          "Conant and Collins 1991",
  "hyla_chrysoscelis",          "ribbitr_db",      "svl_cutoff",   32,             NA,          "Lannoo 2005",
  "hyla_cinerea",               "ribbitr_db",      "svl_cutoff",   32,             NA,          "Conant and Collins 1998",
  "hyla_versicolor",            "ribbitr_db",      "svl_cutoff",   32,             NA,          "Johnson 1987",
  "lithobates_blairi",          "ribbitr_db",      "svl_cutoff",   51,             NA,          "Minton 2001; Lannoo 2005",
  "lithobates_chiricahuensis",  "ribbitr_db",      "svl_cutoff",   57,             NA,          "Platz and Mecham 1963",
  "lithobates_palustris",       "ribbitr_db",      "svl_cutoff",   43,             NA,          "Minton 2001",
  "lithobates_sphenocephalus",  "ribbitr_db",      "svl_cutoff",   51,             NA,          "Conant and Collins 1991",
  "pseudacris_crucifer",        "ribbitr_db",      "svl_cutoff",   20,             NA,          "Lannoo 2005",
  "pseudacris_feriarum",        "ribbitr_db",      "svl_cutoff",   19,             NA,          "Hulse, McCoy and Censky 1998",
  "pseudacris_regilla",         "ribbitr_db",      "svl_cutoff",   25,             NA,          "Lannoo 2005",
  "rana_catesbeiana",           "ribbitr_db",      "svl_cutoff",   95,             NA,          "Howard 1981",
  "rana_clamitans",             "ribbitr_db",      "svl_cutoff",   60,             NA,          "Lannoo 2005",
  "rana_muscosa",               "ribbitr_db",      "svl_cutoff",   40,             8,           "Knapp (pers. comm.)",
  "rana_pipiens",               "ribbitr_db",      "svl_cutoff",   51,             NA,          "nrc.gov/docs/ML1428/ML14282A848.pdf",
  "rana_sierrae",               "ribbitr_db",      "svl_cutoff",   40,             8,           "Knapp (pers. comm.)",
  "rana_sylvatica",             "ribbitr_db",      "svl_cutoff",   37,             NA,          "Martof 1963",
  "scaphiopus_holbrookii",      "ribbitr_db",      "svl_cutoff",   44,             NA,          "Conant and Collins 1991",
  "spea_multiplicata",          "ribbitr_db",      "svl_cutoff",   37,             NA,          "Stebbins 1985",
  # ---- RIBBiTR database: brazil ----
  "hylodes_phyllodes",          "ribbitr_db",      "svl_cutoff",   27.5,           NA,          "Heyer and Cocroft 1986",
  # reference differed between sources: Kwet and Sole 2005 (code), Gui Becker pers. comm. (species notes)
  "ischnocnema_henselii",       "ribbitr_db",      "svl_cutoff",   23,             NA,          "Kwet and Sole 2005; Gui Becker (pers. comm.)",
  "brachycephalus_pitanga",     "ribbitr_db",      "svl_cutoff",   10.8,           NA,          "Kwet and Sole 2005",
  # 63 mm everywhere (was 70 in the RIBBiTR database rule; BOFA data and species notes used 63)
  "boana_faber",                "ribbitr_db",      "svl_cutoff",   63,             NA,          "Gui Becker (pers. comm.)",
  # ---- RIBBiTR database: panama ----
  "atelopus_varius",            "ribbitr_db",      "svl_cutoff",   25,             NA,          "Savage 1972; Crump 1988",
  "craugastor_crassidigitus",   "ribbitr_db",      "svl_cutoff",   20.2,           NA,          "Lynch and Myers 1983",
  "craugastor_fitzingeri",      "ribbitr_db",      "svl_cutoff",   23.5,           NA,          "Lynch and Myers 1983",
  "colostethus_panamansis",     "ribbitr_db",      "svl_cutoff",   18.8,           NA,          "AmphibiaWeb",
  "dendrobates_auratus",        "ribbitr_db",      "svl_cutoff",   25,             NA,          "Leenders 2001",
  "espadarana_prosoblepon",     "ribbitr_db",      "svl_cutoff",   21,             NA,          "AmphibiaWeb",
  "lithobates_warszewitschii",  "ribbitr_db",      "svl_cutoff",   35,             NA,          "Basante 2025",
  "pristimantis_ridens",        "ribbitr_db",      "svl_cutoff",   16,             NA,          "Wang 2008",
  "rhaebo_haematiticus",        "ribbitr_db",      "svl_cutoff",   42,             NA,          "Savage 2002",
  "rhinella_marina",            "ribbitr_db",      "svl_cutoff",   85,             NA,          "Easteal 1963",
  "sachatamia_albomaculata",    "ribbitr_db",      "svl_cutoff",   20.5,           NA,          "Savage 2002",
  "silverstoneia_flotator",     "ribbitr_db",      "svl_cutoff",   15.1,           NA,          "Ibanez and Smith 1995",
  "silverstoneia_nubicola",     "ribbitr_db",      "svl_cutoff",   15.4,           NA,          "Grant and Myers 2013",
  "smilisca_sila",              "ribbitr_db",      "svl_cutoff",   31.6,           NA,          "Duellman 2001",
  # ---- external datasets ----
  "rana_boylii",                "kupferberg2022",  "svl_cutoff",   38,             NA,          "californiaspecies.com/animals/rana-boylii-pop-4",
  # daversa2026: svl >= 40 -> adult; smaller animals labeled subadult or adult -> subadult
  # (metamorphs / larvae dropped), see 00_survey_data.Rmd
  "anaxyrus_canorus",           "daversa2026",     "svl_cutoff",   40,             NA,          "Cutoff derived from source dataset (Daversa et al. 2026)",
  "rana_cascadae",              "RACA",            "svl_cutoff",   50,             NA,          "Ronald and Dumas 1971",
  "boana_faber",                "BOFA",            "svl_cutoff",   63,             NA,          "Gui Becker (pers. comm.)",
  # RALU: age class labels; svl_mm is a constant placeholder (25), see 00_survey_data.Rmd
  "rana_luteiventris",          "RALU",            "field_label",  NA,             NA,          "Using life stage from Russell et al. 2010; 25 mm used for linear density for all individuals",
  # usgs: age labels (juvenile -> subadult) for all usgs species
  "ambystoma_mavortium",        "usgs",            "field_label",  NA,             NA,          "Using life stage from dataset (Waddle et al. 2019)",
  "notophthalmus_viridescens",  "usgs",            "field_label",  NA,             NA,          "Using life stage from dataset (Waddle et al. 2019)"
)

# cutoff for one species in one dataset (errors unless exactly one svl_cutoff rule matches)
svl_cutoff_for = function(taxon, dataset_name) {
  x = svl_stage_cutoffs %>%
    filter(taxon_capture == taxon, dataset == dataset_name, stage_rule == "svl_cutoff") %>%
    pull(svl_cutoff_mm)
  if (length(x) != 1) stop("expected one svl cutoff for ", taxon, " in ", dataset_name, ", found ", length(x))
  x
}

# one row per species: its cutoff and reference(s), for species-level tables and plots (GAM
# pipeline, manuscript species table). a species' cutoff is the same in every data source (the
# function errors otherwise, which keeps the table consistent); NA if it is staged from field
# labels only. references from all of the species' rules, separated by "; ".
svl_cutoff_species = function(taxa = unique(svl_stage_cutoffs$taxon_capture)) {
  out = svl_stage_cutoffs %>%
    filter(taxon_capture %in% taxa) %>%
    group_by(taxon_capture) %>%
    summarise(n_cutoffs = n_distinct(svl_cutoff_mm[stage_rule == "svl_cutoff"]),
              svl_mm_cutoff = if (n_cutoffs == 1) unique(svl_cutoff_mm[stage_rule == "svl_cutoff"]) else NA_real_,
              reference = paste(unique(na.omit(reference)), collapse = "; "),
              .groups = "drop")
  conflicting = out$taxon_capture[out$n_cutoffs > 1]
  if (length(conflicting) > 0) stop("different svl cutoffs across data sources for: ", paste(conflicting, collapse = ", "))
  missing = setdiff(taxa, out$taxon_capture)
  if (length(missing) > 0) stop("no stage rule in svl_stage_cutoffs for: ", paste(missing, collapse = ", "))
  out %>%
    select(-n_cutoffs) %>%
    slice(match(taxa, taxon_capture))   # same order as taxa
}

# case_when() expression assigning life_stage_simple from a set of svl_cutoff rules, for use
# inside mutate(): mutate(life_stage_simple = !!svl_stage_case(rules)). built as an expression
# (rather than a join) so it also runs inside a database query (dbplyr) without copying the
# table to the database. per species:
#   svl_mm >= cutoff                         -> "adult"
#   svl_mm <  cutoff (and > min_svl_mm)      -> "subadult"
#   anything else (other species, NA svl)    -> NA
svl_stage_case = function(rules) {
  rules = filter(rules, stage_rule == "svl_cutoff")
  conditions = pmap(rules, function(taxon_capture, svl_cutoff_mm, min_svl_mm, ...) {
    # !! inserts the value from the table; bare taxon_capture / svl_mm stay column names
    subadult_condition = if (is.na(min_svl_mm)) {
      rlang::expr((taxon_capture == !!taxon_capture) & (svl_mm < !!svl_cutoff_mm))
    } else {
      rlang::expr((taxon_capture == !!taxon_capture) & (svl_mm > !!min_svl_mm) & (svl_mm < !!svl_cutoff_mm))
    }
    list(rlang::expr((taxon_capture == !!taxon_capture) & (svl_mm >= !!svl_cutoff_mm) ~ "adult"),
         rlang::expr(!!subadult_condition ~ "subadult"))
  }) %>%
    list_flatten()
  rlang::expr(case_when(!!!conditions, .default = NA_character_))
}
