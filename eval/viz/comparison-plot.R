library(ggplot2)
library(dplyr)

outDir <- file.path('..', 'out')
runOutDirs <- file.path(outDir, dir(outDir))

summaries <- file.path(runOutDirs, 'run_summary.pickle') %>% lapply(yaml::read_yaml)
seeds <- summaries %>% lapply(function(x) x$config$seed)

results <- file.path(runOutDirs, 'results.csv') %>% lapply(readr::read_csv)
df <- results %>% setNames(seeds) %>% bind_rows(.id = 'seed')


withTime <- df %>%
  group_by(method) %>% 
  mutate(elapsed_time = date_time - min(date_time)) %>% 
  ungroup()

ggplot(withTime, aes(x = elapsed_time, y = error)) +
  geom_point() + 
  facet_grid(seed ~ method, scales = 'free')
