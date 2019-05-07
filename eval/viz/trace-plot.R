library(ggplot2)
library(dplyr)
library(tidyr)

outDir <- file.path('..', 'out')
runOutDirs <- file.path(outDir, dir(outDir))

summaries <- file.path(runOutDirs, 'run_summary.yaml') %>% lapply(yaml::read_yaml)
config <- file.path('..', 'config.yaml') %>% yaml::read_yaml()
seeds <- summaries %>% lapply(function(x) x$config$seed)

results <- file.path(runOutDirs, 'trace.csv') %>% lapply(readr::read_csv)
df <- results %>% setNames(seeds) %>% bind_rows(.id = 'seed')

usedSeed <- 4

df %>% 
  filter(seed == usedSeed) %>% 
  gather(variable, value, tree_height, pop_size, clock_rate) %>% 
  ggplot(aes(x=value, fill=method)) + 
  geom_histogram( alpha=0.5, bins=30) +
  facet_wrap(~ variable , scales='free')

ggsave('out/marginal-comparison.png')
  
df %>%
  filter(seed == usedSeed) %>% 
  GGally::ggpairs(columns = c('clock_rate', 'pop_size', 'tree_height'), aes(colour = method))
