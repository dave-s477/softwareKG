# merge two reasoning tables - to be used if a new graph is generated for the integration of the manually annotated information

library(tidyverse)
annotated_software = read_csv("software_reasoning_list_total.csv")

linked_graph_software = read_csv("production_model_linked_names.csv.gz")

linked_graph_software %>%
  dplyr::select(name, count,linked_name) %>%
  dplyr::left_join(annotated_software, by='name') %>%
  dplyr::select(-count.y, -linked_name.y) %>%
  dplyr::rename(count = count.x, linked_name=linked_name.x) %>%
  dplyr::arrange(desc(count)) -> 
  merged

write_csv(merged, path='software_reasoning_list_total_neu.csv')
  
