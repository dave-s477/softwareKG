# do some basic checks on the table of software and licences

library(tidyverse)


software = read_csv("software_reasoning_list_total.csv")

# licence with respect to code: no code -> no licence
software %>%
  dplyr::mutate(Licence = ifelse(`Source Available`=='yes', Licence, NA)) ->
  software

software %>%
  dplyr::filter(`Source Available` == 'yes') %>%
  print(n=40)

software %>%
  dplyr::group_by(Free, `Source Available`, Licence) %>%
  dplyr::summarise(n=n(), sum=sum(count)) %>%
  dplyr::arrange(desc(n))%>%
  print(n=40)


# find entries which are free and have not information about source
software %>%
  dplyr::filter(`Source Available`=='no', is.na(Licence))


# find software which is not available (in executable and source)
software %>%
  dplyr::filter(`Source Available`=='no') %>%
  print(n=40)


df %>%
  dplyr::filter(`Source Available`=='no') %>%
  print(n=40)

#find different licences
software %>%
  dplyr::group_by(Licence) %>%
  dplyr::summarise(n=n()) %>%
  dplyr::arrange(desc(n)) %>%
  print(n=40)
