library(ggplot2)
library(dplyr)
library(tidyr)
library(phytools)

outDir <- file.path('..', 'out')
runOutDirs <- file.path(outDir, 0:2)

summaries <- file.path(runOutDirs, 'run_summary.yaml') %>% lapply(yaml::read_yaml)
config <- file.path('..', 'config.yaml') %>% yaml::read_yaml()
seeds <- summaries %>% lapply(function(x) x$config$seed)

results <- file.path(runOutDirs, 'trace.csv') %>% lapply(readr::read_csv)
df <- results %>% setNames(seeds) %>% bind_rows(.id = 'seed')

getTreeHeight <- function(newick){
  tree <- read.newick(text=newick)
}

trueValues <- summaries %>%
  sapply(function(x) c(seed = x$config$seed, pop_size = x$pop_size, tree_height = read.newick(text=x$newick_string) %>% nodeHeights %>% max())) %>% 
  t() %>% 
  as.data.frame() %>% 
  gather(variable, value, tree_height, pop_size)

df %>% 
  gather(variable, value, tree_height, pop_size) %>% 
  ggplot(aes(x=value, fill=method)) + 
  geom_histogram( alpha=0.5, bins=30) +
  geom_vline(aes(xintercept = value, lty='True value'), data = trueValues, color='darkgreen') +
  facet_wrap(variable ~ seed, scales='free') + scale_linetype_manual(values = c('True value' = 'dashed') ,name = NULL)

ggsave('out/marginal-comparison.png')
  
df %>%
  filter(seed == usedSeed) %>% 
  GGally::ggpairs(columns = c('clock_rate', 'pop_size', 'tree_height'), aes(colour = method))

