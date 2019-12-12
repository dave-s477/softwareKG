# Accessing PLoS with the help of rplos
# the package "fulltext" might also be helpful for further processing and retrieval. https://ropensci.github.io/fulltext-book/chunks.html

# LIMITS TO PLOS according to rplos github https://github.com/ropensci/rplos
#Please limit your API requests to 7200 requests a day, 300 per hour, 10 per minute and allow 5 seconds for your search to return results. 
#If you exceed this threshold, we will lock out your IP address. 
#If you're a high-volume user of the PLOS Search API and need more API requests a day, please contact us at api@plos.org to discuss your options. 
#We currently limit API users to no more than five concurrent connections from a single IP address.

# Basic library
library(rplos)
library(fulltext)
library(dplyr)
library(xml2)
library(XML)
library(lubridate)


# Extract all relevant text form the xml and save the textfiles to useful locations
title.parser <- function(xml.title){
  plain.text <- paste0(xml_text(xml.title), ":\n\n")
  return(plain.text)
}
caption.title.parser <- function(xml.title){
  plain.text <- paste0(xml_text(xml.title))
  return(plain.text)
}
label.parser <- function(xml.label){
  plain.text <- paste0(xml_text(xml.label), ":\n")
  return(plain.text)
}
paragraph.parser <- function(xml.paragraph){
  plain.text <- paste0(xml_text(xml.paragraph), "\n")
  return(plain.text)
}
#TODO
supplementary.parser <- function(xml.supplementary){
  plain.text <- ""
  children.xml <- xml_children(xml.supplementary)
  children.names <- xml_name(children.xml)
  if (length(children.xml) > 0){
    for (i in 1:length(children.xml)) {
      if (children.names[i] == "title") {
        plain.text <- paste0(plain.text, title.parser(children.xml[i]))
      } else if (children.names[i] == "p") {
        plain.text <- paste0(plain.text, paragraph.parser(children.xml[i]))
      } else if (children.names[i] == "sec") {
        plain.text <-
          paste0(plain.text,
                 section.parser(children.xml[i], is_in_method = FALSE))
      } else if (children.names[i] == "supplementary-material") {
        plain.text <-
          paste0(plain.text, supplementary.parser(children.xml[i]))
      } else if (children.names[i] == "caption") {
        plain.text <-
          paste0(plain.text,
                 caption.parser(children.xml[i], in_method =  FALSE))
      } else if (children.names[i] == "label") {
        plain.text <- paste0(plain.text, label.parser(children.xml[i]))
      } else if (children.names[i] == "fig") {
        plain.text <- paste0(plain.text, fig.parser(children.xml[i]))
      } else if (children.names[i] == "table-wrap") {
        plain.text <- paste0(plain.text, table.parser(children.xml[i]))
      } else {
        plain.text <- paste0(plain.text, xml_text(children.xml[i]))
      }
    }
  }
  plain.text <- paste0(plain.text, "\n")
  return(plain.text)  
}
caption.parser <- function(xml.caption, in_method){
  plain.text <- ""
  children.xml <- xml_children(xml.caption)
  children.names <- xml_name(children.xml)
  if (length(children.xml) > 0){
    for (i in 1:length(children.xml)) {
      if (children.names[i] == "title") {
        plain.text <-
          paste0(plain.text, caption.title.parser(children.xml[i]))
      } else if (children.names[i] == "p") {
        plain.text <- paste0(plain.text, paragraph.parser(children.xml[i]))
      } else if (children.names[i] == "sec") {
        plain.text <-
          paste0(plain.text,
                 section.parser(children.xml[i], is_in_method = in_method))
      } else if (children.names[i] == "supplementary-material") {
        plain.text <-
          paste0(plain.text, supplementary.parser(children.xml[i]))
      } else if (children.names[i] == "caption") {
        plain.text <-
          paste0(plain.text,
                 caption.parser(children.xml[i], in_method = in_method))
      } else if (children.names[i] == "label") {
        plain.text <- paste0(plain.text, label.parser(children.xml[i]))
      } else if (children.names[i] == "fig") {
        plain.text <- paste0(plain.text, fig.parser(children.xml[i]))
      } else if (children.names[i] == "table-wrap") {
        plain.text <- paste0(plain.text, table.parser(children.xml[i]))
      } else {
        plain.text <- paste0(plain.text, xml_text(children.xml[i]))
      }
    }
  }
  plain.text <- paste0(plain.text, "")
  return(plain.text)
}
section.parser <- function(xml.section, is_in_method){
  plain.text <- "+++Begin Section+++\n"
  children.xml <- xml_children(xml.section)
  children.names <- xml_name(children.xml)
  if (length(children.xml) > 0){
    for (i in 1:length(children.xml)) {
      if (children.names[i] == "title") {
        title = title.parser(children.xml[i])
        plain.text <- paste0(plain.text, title)
        is_in_method = is_in_method |
          any(grepl(x = title, "method", ignore.case = TRUE))
        if (!is_in_method) {
          return("")
        }
      } else if (children.names[i] == "p") {
        plain.text <- paste0(plain.text, paragraph.parser(children.xml[i]))
      } else if (children.names[i] == "sec") {
        plain.text <-
          paste0(plain.text,
                 section.parser(children.xml[i], is_in_method = is_in_method))
      } else if (children.names[i] == "supplementary-material") {
        plain.text <-
          paste0(plain.text, supplementary.parser(children.xml[i]))
      } else if (children.names[i] == "caption") {
        plain.text <-
          paste0(plain.text,
                 caption.parser(children.xml[i], in_method = is_in_method))
      } else if (children.names[i] == "fig") {
        plain.text <- paste0(plain.text, fig.parser(children.xml[i]))
      } else if (children.names[i] == "table-wrap") {
        plain.text <- paste0(plain.text, table.parser(children.xml[i]))
      } else {
        plain.text <- paste0(plain.text, xml_text(children.xml[i]))
      }
    }
  }
  plain.text <- paste0(plain.text, "+++End Section+++\n\n")
  if (is_in_method){
    return(plain.text)  
  }
  return("")
}
fig.parser <- function(xml.figure, in_method){
  plain.text <- "\nFigure data removed from full text. "
  children.xml <- xml_children(xml.figure)
  children.names <- xml_name(children.xml)
  if (length(children.xml) > 0){
    for (i in 1:length(children.xml)) {
      if (children.names[i] == "object-id") {
        plain.text <-
          paste0(plain.text,
                 "Figure identifier and caption: ",
                 xml_text(children.xml[i]),
                 "\n")
      } else if (children.names[i] == "caption") {
        plain.text <-
          paste0(plain.text,
                 caption.parser(children.xml[i], in_method = in_method))
      } else if (children.names[i] == "label") {
        plain.text <- paste0(plain.text, label.parser(children.xml[i]))
      } else {
        
      }
    }
  }
  plain.text <- paste0(plain.text, "\n\n")
  return(plain.text)
}
table.parser <- function(xml.table, in_method){
  plain.text <- "\nTable data removed from full text. "
  children.xml <- xml_children(xml.table)
  children.names <- xml_name(children.xml)
  if (length(children.xml) > 0){
    for (i in 1:length(children.xml)) {
      if (children.names[i] == "object-id") {
        plain.text <-
          paste0(plain.text,
                 "Table identifier and caption: ",
                 xml_text(children.xml[i]),
                 "\n")
      } else if (children.names[i] == "caption") {
        plain.text <-
          paste0(plain.text,
                 caption.parser(children.xml[i], in_method = in_method))
      } else if (children.names[i] == "label") {
        plain.text <- paste0(plain.text, label.parser(children.xml[i]))
      } else if (children.names[i] == "table-wrap-foot") {
        plain.text <- paste0(plain.text, "\n", xml_text(children.xml[i]))
      } else {
        
      }
    }
  }
  plain.text <- paste0(plain.text, "\n\n")
  return(plain.text)
}
body.parser <- function(body, full.paper=FALSE){
  plain.text <- ""
  sections <- xml_children(body)
  section.names <- xml_name(sections)
  if (length(section.names) > 0){
    for (i in 1:length(section.names)) {
      if (section.names[i] == "sec") {
        plain.text <- paste0(plain.text, section.parser(sections[i], full.paper))
      } else if (section.names[i] == "p") {
        plain.text <- paste0(plain.text, paragraph.parser(sections[i]))
      } else {
        plain.text <- paste0(plain.text, xml_text(sections[i]))
      }
    }
  }
  plain.text <- paste0(plain.text, "")
  #cat(plain.text)
  return(plain.text)
}
# TODO
ack.parser <- function(xml.ack){
  plain.text <- "+++Acknowledgement+++\n"
  children.xml <- xml_children(xml.ack)
  children.names <- xml_name(children.xml)
  if (length(children.xml) > 0){
    for (i in 1:length(children.xml)) {
      if (children.names[i] == "title") {
        plain.text <- paste0(plain.text, title.parser(children.xml[i]))
      } else if (children.names[i] == "p") {
        plain.text <- paste0(plain.text, paragraph.parser(children.xml[i]))
      } else if (children.names[i] == "sec") {
        plain.text <-
          paste0(plain.text,
                 section.parser(children.xml[i], is_in_method = FALSE))
      } else if (children.names[i] == "supplementary-material") {
        plain.text <-
          paste0(plain.text, supplementary.parser(children.xml[i]))
      } else if (children.names[i] == "caption") {
        plain.text <-
          paste0(plain.text,
                 caption.parser(children.xml[i], in_method = FALSE))
      } else {
        plain.text <- paste0(plain.text, xml_text(children.xml[i]))
      }
    }
  }
  plain.text <- paste0(plain.text, "+++End Acknowledgement+++\n\n")
  return(plain.text)
}
ref.list.parser <- function(xml.reflist){
  plain.text <- ""
  children.xml <- xml_children(xml.reflist)
  children.names <- xml_name(children.xml)
  if (length(children.xml) > 0){
    for (i in 1:length(children.xml)) {
      if (children.names[i] == "ref") {
        plain.text <- paste0(plain.text, ref.parser(children.xml[i]))
      } else if (children.names[i] == "title") {
        plain.text <- paste0(plain.text, title.parser(children.xml[i]))
      } else {
        plain.text <- paste0(plain.text, xml_text(children.xml[i]))
      }
    }
  }
  plain.text <- paste0(plain.text, "")
  return(plain.text)
}
ref.parser <- function(xml.ref){
  plain.text <- ""
  children.xml <- xml_children(xml.ref)
  children.names <- xml_name(children.xml)
  if (length(children.xml) > 0){
    for (i in 1:length(children.xml)) {
      if (children.names[i] == "label") {
        plain.text <- paste0(plain.text, label.parser(children.xml[i]))
      } else if (grepl("citation", children.names[i])) {
        plain.text <- paste0(plain.text, citation.parser(children.xml[i]))
      } else {
        plain.text <- paste0(plain.text, xml_text(children.xml[i]))
      }
    }
  }
  plain.text <- paste0(plain.text, "\n\n")
  return(plain.text)
}
citation.parser <- function(xml.citation){
  plain.text <- ""
  children.xml <- xml_children(xml.citation)
  children.names <- xml_name(children.xml)
  if (xml_length(children.xml)[1] == 0){
    plain.text <- paste0(plain.text, xml_text(xml.citation))
  } else {
    if (length(children.xml) > 0) {
      for (i in 1:length(children.xml)) {
        plain.text <- paste(plain.text, xml_text(children.xml[i]), ";")
      }
    }
  }
  return(plain.text)
}
back.parser <- function(xml.back){
  plain.text <- ""
  children.xml <- xml_children(xml.back)
  children.names <- xml_name(children.xml)
  if (length(children.names) > 0){
    for (i in 1:length(children.names)){
      if (children.names[i] == "ack") {
        plain.text <- paste0(plain.text, ack.parser(children.xml[i]))
      } else if (children.names[i] == "ref-list") {
        plain.text <- paste0(plain.text, ref.list.parser(children.xml[i]))
      } else if (children.names[i] == "ref"){
        plain.text <- paste0(plain.text, ref.parser(children.xml[i]))
      } else if (children.names[i] == "title"){
        plain.text <- paste0(plain.text, title.parser(children.xml[i]))
      } else if (children.names[i] == "citation"){
        plain.text <- paste0(plain.text, citation.parser(children.xml[i]))
      } else {
        plain.text <- paste0(plain.text, xml_text(children.xml[i]))
      }
    }
  }
  plain.text <- paste0(plain.text, "")
  #cat(plain.text)
  return(plain.text)
}
get.doi <- function(xml){
  doi.candidates <- xml_find_all(xml_child(xml_child(xml, "front"), "article-meta"), "article-id")
  for (id in doi.candidates){
    if (xml_attr(id, "pub-id-type") == "doi"){
      return(xml_text(id))
    } 
  }
}
extract.xml <- function(xml, full.paper=FALSE){
  article.doi <- get.doi(xml)
  title.plain <- xml_text(xml_child(xml_child(xml_child(xml_child(xml, "front"), "article-meta"), "title-group"), "article-title"))
  abstract.plain <- xml_text(xml_child(xml_child(xml_child(xml, "front"), "article-meta"), "abstract"))
  full.text.plain <- body.parser(xml_children(xml)[2], full.paper)
  if (3 <= length(xml_children(xml))){
    references.plain <- back.parser(xml_children(xml)[3])  
  }else{
    warning("Document has no references")
    references.plain <- ""
  }
  
  if (full.paper){
    paper.text <- paste0(article.doi, "\n", title.plain, "\n\n+++Abstract+++\n", abstract.plain, "\n+++End Abstract+++\n\n", full.text.plain, "\n", references.plain)
    return(paper.text)
  }
  return(full.text.plain)
  
}
run <- function(){
  base.dir <- file.path(getwd(), "annotation_papers")
  if (!dir.exists(base.dir)){
    dir.create(base.dir)
  }
  setwd(base.dir)
  for (journal.title in journals.to.search){
    new.dir <- strsplit(journal.title, ":")[[1]][2]
    if (!dir.exists(new.dir)){
      dir.create(new.dir)
    }
    write.csv2(subsample[[journal.title]], file.path(base.dir, new.dir, "overview.csv"))
    for (year in as.character(date(unique(subsample[[journal.title]]$date_year)))){
      print(paste0("At year ", year))
      year.folder <- file.path(base.dir, new.dir, toString(year))
      if (!dir.exists(year.folder)){
        dir.create(year.folder)
      }
      papers.current.year <- subsample[[journal.title]] %>% filter(date_year == date(year))
      write.csv2(papers.current.year, file.path(year.folder, "overview.csv"))
      for (doi in papers.current.year$id){
        print(paste0("At Doi ", doi))
        file.name <- strsplit(doi, "/")[[1]][2]
        xml.text <- full.text.journalwise[[journal.title]][doi] # die Texte 
        xml <- read_xml(xml.text[[doi]])
        # Save original xml
        xml.name <- paste0(file.name, ".xml")
        write_xml(xml, file = file.path(year.folder, xml.name))
        
        # Save plain text
        full.text <- extract.xml(xml)
        file.to.write <- file(file.path(year.folder, file.name))
        writeLines(full.text, file.to.write)
        close(file.to.write)
      }
    }
  }
}
#read_xml() and get the required tags.. xml_children().. xml.children[[2]] ist body ... xml_text(child)