library(ggplot2)
library(dplyr)
library(readr)

df <- read_csv('../dengue/dengue-scores.csv')

maxKappa <- quantile((df %>% filter(method == 'nuts'))$kappa, probs = 0.9)

df %>%
  filter(kappa < maxKappa) %>% 
  ggplot(aes(x=kappa, fill=method)) + 
  geom_density( alpha=0.4)

ggsave('out/dengue-kappa-plot.png')


df %>% 
  ggplot(aes(x=tree_height, fill=method)) + 
  expand_limits(x = 0) +
  geom_density(alpha=0.4)


ggsave('out/dengue-tree-plot.png')

df %>% 
  ggplot(aes(x=pop_size, fill=method)) + 
  expand_limits(x = 0) +
  geom_density(alpha=0.4)

ggsave('out/dengue-pop-plot.png')

df %>%
  ggplot(aes(x=tree_height, y=pop_size)) +
  expand_limits(x = 0, y = 0) + 
  geom_density2d() + 
  facet_wrap(~ method)

