file <- "~/xperiments/x20190827-ner-tensorflow2.0-ds/aaaaa.csv"

library(tidyverse)

df <- read_delim(file, delim = ',')
#df <- read_csv2(file)

df %>% gather(dsource, developer, developer_label_1, developer_label_alt, developer_label_normal, developer_original_label) %>%
  gather(lsource, label, label_1, label_alt, label_redirect, label_wiki_dis) %>%
  dplyr::select(-X1) %>%
  distinct() ->
  software
  
write_delim(software, delim = ',', path = file)
