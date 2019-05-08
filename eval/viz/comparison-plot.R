library(ggplot2)
library(dplyr)

outDir <- file.path('..', 'out')
runOutDirs <- file.path(outDir, 0:2)

summaries <- file.path(runOutDirs, 'run_summary.yaml') %>% lapply(yaml::read_yaml)
config <- file.path('..', 'config.yaml') %>% yaml::read_yaml()
seeds <- summaries %>% lapply(function(x) x$config$seed)

results <- file.path(runOutDirs, 'results.csv') %>% lapply(readr::read_csv)
df <- results %>% setNames(seeds) %>% bind_rows(.id = 'seed') %>% select(X1, date_time, error, method, seed)

nSamples <- 1500

withTime <- df %>%
  group_by(method, seed) %>% 
  arrange(date_time) %>% 
  mutate(elapsed_time = date_time - min(date_time)) %>% 
  ungroup()

withTime %>%
  filter(elapsed_time < 500)
  ggplot(aes(x = elapsed_time, y = error)) +
  geom_smooth() + 
  facet_grid(method ~ seed, scales='free') #+ 
  #theme(
  #  strip.background = element_blank(),
  #  strip.text.y = element_blank()
  #)
ggsave('out/comparison-plot.png')
