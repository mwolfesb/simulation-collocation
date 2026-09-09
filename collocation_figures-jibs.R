##### Analysis of Simulation Data for Competitive Collocation Paper (JIBS) ###################
# Note: Updates to certain packages (e.g., ggplot2) can slightly alter the produced figures.

#load packages
pcks <- c('tidyverse', 'ggplot2', 'ggpattern', 'writexl', 'patchwork')

for (pck in pcks) {
  if(!require(pck, character.only = TRUE)) install.packages(pck); library(pck, character.only = TRUE)
}

# set path to working directory
setwd('simulation') # if necessary

################## Load Data ###############
sim_dat <- read_csv('./results/merged_data.csv', name_repair = 'universal')
sim_dat <- sim_dat %>% 
  mutate(Home_country.firm_0 = Home_country.firm_0 + 1,
        year = year + 1) # Correct python index start at 0

# DF with final outcome
colloc <- filter(sim_dat, year == 50)

# Sample 20 random seeds for illustrative purposes (Appendix example)
set.seed(10)
colloc_extract <- colloc %>% 
  select(Seed, Home_country.firm_0, Market.size.variation, alpha, beta,
         FMA, LoF.Strength, LoF.discount, Entry.barriers, Competitive.Intensity) %>% 
  slice_sample(n=20)
write_xlsx(colloc_extract, './results/sample_extract_20.xlsx')

################## Create Plots ###############
# set quantile-thresholds
low_quant <- .25
high_quant <- .75

# Simulation variables for FOR - loops
sim_vars <- c('Competitive.Intensity', 'Entry.barriers', 'FMA', 'LoF.Strength', 
              'LoF.discount', 'Market.size.variation', 'Home.market.size.ratio')
sim_label <- c('a) ', 'b) ', 'c) ', 'd) ', 'e) ', 'f) ', 'g) ')

### A: Analysis - Collocation Frequency ------------------
# Histograms for collocation frequency
cl <- ggplot(colloc, aes(x = colloc.0))+
  xlab('Collocation Index (firm A)') + ylab('Scenarios') + 
  scale_y_continuous(expand = c(0,0)) +
  scale_x_continuous(expand = c(0,0)) +
  theme_classic() +
  theme(axis.text = element_text(size = 25),
        axis.title = element_text(size = 25))

# Full sample
cl + geom_histogram(bins = 35, color = 'black', fill = 'grey', alpha = 0.8, boundary = 0)
ggsave('paper_figures/collocation_fullsample.png', create.dir = TRUE, width = 40, height = 40, units = "cm")


# Performance graph for different levels of collocation
clpe <- ggplot(colloc, aes(x = colloc.0, y = Cumu_profit.firm_0))+
  xlab('Collocation Index (firm A)') + ylab('Cumulative Cash Flows') + 
  scale_y_continuous(expand = c(0,0)) +
  scale_x_continuous(expand = c(0,0)) +
  theme_classic() +
  theme(axis.text = element_text(size = 25),
        axis.title = element_text(size = 25),
        legend.text = element_text(size = 25),
        legend.title = element_text(face = 'italic', size = 25, hjust = 0.5, vjust = 0.5))

# Full sample
clpe + geom_smooth(aes(linetype = 'local parametric fit'),method = 'loess', color = 'black') +
  geom_smooth(aes(linetype = 'linear trend'),method = 'lm', color = 'green') +
  scale_y_continuous(labels = function(x) format(x, big.mark = ",", scientific = FALSE)) +
  stat_summary(fun = mean, geom = 'line', color = 'black', alpha = 0.4) +
  scale_linetype_manual(name = 'Trendline',values = c('local parametric fit'  = "solid", 'linear trend' = "dashed")) +
  ggtitle('Full sample')+
  theme(plot.title = element_text(face = 'bold', size = 25, hjust = 0.5))
ggsave(paste0('paper_figures/cumcash_fullsample.png'),create.dir = TRUE, width = 40, height = 40, units = "cm")


# Simple area, performance (cumulative cash flows), and overlapping area & histogram
for (simvar in sim_vars) {
  print(simvar)
  sl <- sim_label[which(sim_vars == simvar)] #get numbering for paper
  
  # filter upper and lower quantiles for area and performance graphs
  dat_low <- filter(colloc, .data[[simvar]] < quantile(.data[[simvar]], low_quant))
  dat_high <- filter(colloc, .data[[simvar]] > quantile(.data[[simvar]], high_quant))

  # Comparison of overall histogramm (full sample) and area charts for quantiles
  cl + 
    geom_histogram(bins = 35, color = 'black', fill = 'grey', alpha = 0.8, boundary = 0)+
    geom_area(data = dat_low, aes(fill = 'y1', y = after_stat(count)*2), stat = 'bin', color = 'black', alpha = 0.5) +
    geom_area(data = dat_high, aes(fill = 'y2', y = after_stat(count) *2), stat = 'bin', color = 'black', alpha = 0.5)+
    scale_y_continuous(expand = c(0,0), sec.axis = sec_axis(~ . / 2, name = "Density (for Area)"))+
    scale_fill_manual(values = c('y1' =  '#2ca02c', 'y2' = '#ff7f0e'),
                      labels = c('y1' = 'Low (<0.25)', 'y2' = 'High (>0.75)'))+
    labs(fill = 'Quantile')+
    ggtitle(paste0(sl,simvar))+
    theme(plot.title = element_text(face = 'bold', size = 25, hjust = 0.5))
  ggsave(paste0('paper_figures/',simvar,'area_and_hist.png'), create.dir = TRUE, width = 40, height = 40, units = "cm")

  # Line plot with cumulated cash flows comparing full, lower and upper quantiles of simulation variable
  clpe + 
    geom_smooth(data = dat_low, aes(linetype = 'Low'), method = 'loess', color = 'gray30') +
    geom_smooth(aes(linetype = 'Full Sample'), method = 'loess', color = 'green')+
    geom_smooth(data = dat_high, aes(linetype =  'High'), method = 'loess', color = 'black') +
    scale_linetype_manual(name = 'Quantile',values = c('Low'  = "solid", 'Full Sample' = "dashed", 'High' = "dotted")) +
    scale_y_continuous(labels = function(x) format(x, big.mark = ",", scientific = FALSE)) +
    ggtitle(paste0(sl,simvar))+
    theme(plot.title = element_text(face = 'bold', size = 25, hjust = 0.5))
  ggsave(paste0('paper_figures/cumcash/', simvar, '.png'),create.dir = TRUE, width = 40, height = 40, units = "cm")

  # Line plot with comparison of cummulated profits (firm A and B) including mean lines for firm A and B (stat_summary)
  # low quantile
  clpe + 
    geom_smooth(data = dat_low, aes(linetype = 'Firm A'), method = 'loess', color = 'black') +
    geom_smooth(data = dat_low ,aes(y=Cumu_profit.firm_1, linetype = 'Firm B'), method = 'loess', color = 'green')+
    stat_summary(data = dat_low, fun = mean, geom = 'line', aes(y = Cumu_profit.firm_0, linetype = 'Firm A'), color = 'black', alpha = 0.4) +
    stat_summary(data = dat_low, fun = mean, geom = 'line', aes(y = Cumu_profit.firm_1), color = 'green', alpha = 0.4) +
    coord_cartesian(ylim = c(0, 2000000))+
    scale_linetype_manual(name = 'Actors',values = c('Firm A'  = "solid", 'Firm B' = "dashed")) +
    scale_y_continuous(labels = function(x) format(x, big.mark = ",", scientific = FALSE)) +
    ggtitle(paste0(sl,simvar, ' - Low'))+
    theme(plot.title = element_text(face = 'bold', size = 25, hjust = 0.5))
  ggsave(paste0('paper_figures/cumcash_comparison_wMean/', simvar, '_low.png'),create.dir = TRUE, width = 40, height = 40, units = "cm")
  
  # upper quantile
  clpe + 
    geom_smooth(data = dat_high, aes(linetype = 'Firm A'), method = 'loess', color = 'black') +
    geom_smooth(data = dat_high ,aes(y=Cumu_profit.firm_1, linetype = 'Firm B'), method = 'loess', color = 'green')+
    stat_summary(data = dat_high,fun = mean, geom = 'line', aes(y = Cumu_profit.firm_0, linetype = 'Firm A'), color = 'black', alpha = 0.4) +
    stat_summary(data = dat_high, fun = mean, geom = 'line', aes(y = Cumu_profit.firm_1), color = 'green', alpha = 0.4) +
    coord_cartesian(ylim = c(0, 2000000))+
    scale_linetype_manual(name = 'Actors',values = c('Firm A'  = "solid", 'Firm B' = "dashed")) +
    scale_y_continuous(labels = function(x) format(x, big.mark = ",", scientific = FALSE)) +
    ggtitle(paste0(sl,simvar, ' - High'))+
    theme(plot.title = element_text(face = 'bold', size = 25, hjust = 0.5))
  ggsave(paste0('paper_figures/cumcash_comparison_wMean/', simvar, '_high.png'),create.dir = TRUE, width = 40, height = 40, units = "cm")
}


##################################### Compare Outliers Samples (all vs. full collocation) ############################
sim_vars_paper <- c('Competitive.Intensity', 'Entry.barriers',  'FMA', 'LoF.Strength', 'LoF.discount', 
                    'Market.size.variation', 'Home.market.size.ratio')

# extract seeds with full collocation and full avoidance
full_colloc <- colloc %>% 
  filter(colloc.0 == 1) %>% 
  select(all_of(sim_vars_paper)) %>% 
  mutate(colloc_group = 'Complete.Collocation')

colloc_high <- colloc %>% 
  filter(colloc.0 > 0.9, colloc.0 < 1) %>% 
  select(all_of(sim_vars_paper)) %>% 
  mutate(colloc_group = 'High.Collocation')

full_avoidance <- colloc %>% 
  filter(colloc.0 == 0) %>% 
  select(all_of(sim_vars_paper)) %>% 
  mutate(colloc_group = 'Complete.Avoidance')

colloc_summary <- colloc %>% 
  select(all_of(sim_vars_paper)) %>% 
  mutate(colloc_group = 'All.Scenarios')

## Combine two dfs for boxplots
# Collocation
comp_sample_colloc <- rbind(colloc_summary, full_colloc)
comp_sample_colloc <- comp_sample_colloc %>% 
  pivot_longer(cols = -colloc_group,
               names_to = 'Variables',
               values_to = 'Value')

# High collocation vs Complete collocation
comp_sample_highcolloc <- rbind(colloc_high, full_colloc)
comp_sample_highcolloc <- comp_sample_highcolloc %>% 
  pivot_longer(cols = -colloc_group,
               names_to = 'Variables',
               values_to = 'Value')

# Avoidance
comp_sample_avoid <- rbind(colloc_summary, full_avoidance)
comp_sample_avoid <- comp_sample_avoid %>% 
  pivot_longer(cols = -colloc_group,
               names_to = 'Variables',
               values_to = 'Value')


## Visualize Boxplots
# Collocation
ggplot(comp_sample_colloc, aes(x = colloc_group, y = Value, fill = colloc_group)) +
  geom_boxplot()+
  scale_fill_manual(values = c('All.Scenarios' =  '#2ca02c', 'Complete.Collocation' = '#ff7f0e'),
                    labels = c('All.Scenarios' = 'All scenarios', 'Complete.Collocation' = 'Complete Collocation (100%)'))+
  facet_wrap(~factor(Variables, 
            levels = c('Competitive.Intensity', 'Entry.barriers', 'Home.market.size.ratio', 'FMA', 
            'LoF.discount', 'LoF.Strength', 'Market.size.variation')), scales = 'free')+
  labs(y='Value') +
  theme_bw() +
  theme(legend.position = 'none',
        strip.text = element_text(size = 15),
        axis.text = element_text(size = 15),
        axis.title = element_text(size = 15),
        title = element_text(size = 15))
ggsave('paper_figures/sample_comparison/boxplots_collocation.png',create.dir = TRUE, width = 40, height = 40, units = "cm")


# High Collocation vs. Complete Collocation
ggplot(comp_sample_highcolloc, aes(x = factor(colloc_group, levels = c('High.Collocation', 'Complete.Collocation')), y = Value, fill = colloc_group)) +
  geom_boxplot()+
  scale_fill_manual(values = c('High.Collocation' =  '#2ca02c',  'Complete.Collocation' = '#ff7f0e'),
                    labels = c('High.Collocation' = 'High Collocation','Complete.Collocation' = 'Complete Collocation (100%)'))+
  facet_wrap(~factor(Variables, 
            levels = c('Competitive.Intensity', 'Entry.barriers', 'Home.market.size.ratio', 'FMA', 
            'LoF.discount', 'LoF.Strength', 'Market.size.variation')), scales = 'free')+
  labs(y='Value') +
  theme_bw() +
  theme(legend.position = 'none',
        strip.text = element_text(size = 15),
        axis.text = element_text(size = 15),
        axis.title = element_text(size = 15),
        title = element_text(size = 15))
ggsave('paper_figures/sample_comparison/boxplots_highcolloc.png',create.dir = TRUE, width = 40, height = 40, units = "cm")


# Avoidance
ggplot(comp_sample_avoid, aes(x = colloc_group, y = Value, fill = colloc_group)) +
  geom_boxplot()+
  scale_fill_manual(values = c('All.Scenarios' =  '#2ca02c', 'Complete.Avoidance' = '#ff7f0e'),
                    labels = c('All.Scenarios' = 'All scenarios', 'Complete.Avoidance' = 'Complete Avoidance'))+
  facet_wrap(~factor(Variables, 
            levels = c('Competitive.Intensity', 'Entry.barriers', 'Home.market.size.ratio', 'FMA', 
            'LoF.discount', 'LoF.Strength', 'Market.size.variation')), scales = 'free') +
  labs(y='Value') +
  theme_bw() +
  theme(legend.position = 'none',
        strip.text = element_text(size = 15),
        axis.text = element_text(size = 15),
        axis.title = element_text(size = 15),
        title = element_text(size = 15))
ggsave('paper_figures/sample_comparison/boxplots_avoidance.png',create.dir = TRUE, width = 40, height = 40, units = "cm")