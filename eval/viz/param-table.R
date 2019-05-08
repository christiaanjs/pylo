config <- file.path('..', 'config.yaml') %>% yaml::read_yaml()

simDf <- tribble(
  ~Parameter, ~Value,
  "Number of replicates", config$n_runs,
  "Number of taxa", config$n_taxa,
  "Population size interval", sprintf("(%d, %d)", config$min_pop_size, config$max_pop_size),
  "Sampling window", config$sampling_window,
  "Mutation rate", config$mutation_rate,
  "Sequence length", config$sequence_length,
  "Base nucleotide frequencies", sprintf("(%s)", paste(config$frequencies, collapse=", ")),
  "Transition-tranversion ratio", config$kappa
)

simLatex <- knitr::kable(simDf, format='latex', caption='Simulation parameters', label='simparams')


analysisDf <- tribble(
  ~Parameter, ~Value,
  "Base frequencies prior", "Dirichlet(1.0, 1.0, 1.0, 1.0)",
  "Transition-tranversion ratio prior", sprintf("LogNormal(%f, %f)", config$prior_params$kappa$m, config$prior_params$kappa$s),
  "Population size prior", sprintf("LogNormal(%f, %f)", config$prior_params$pop_size$m, config$prior_params$pop_size$s),
  "MCMC chain length", config$chain_length,
  "Variational iterations", config$n_iter
)

analysisLatex <- knitr::kable(analysisDf, format='latex', caption = 'Analysis parameters', label='analysisparams')
