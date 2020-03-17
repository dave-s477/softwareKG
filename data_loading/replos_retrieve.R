require(rplos)
require(here)

dir.create('data/R_loading', showWarnings = FALSE)

# get amount of available papers
p <- rplos::plossubject(q='"Social sciences"',fl='id', limit=0)$meta

# get paper ids
if (file.exists("data/R_loading/paper_ids.RData")){
  papers <- readRDS("data/R_loading/paper_ids.RData")
}else{
  papers <- rplos::plossubject(q='"Social sciences"',fl='id,subject,article_type', limit=p$numFound) 
  saveRDS(papers, file="data/R_loading/paper_ids.RData")
}

#sample papers to be annotated
research_articles <- papers$data$id[papers$data$article_type=='Research Article']
set.seed(12345)
# wisely choosen papers to annotate
article_ids <- sample(x = length(research_articles), size = 500)
ann_articles <- research_articles[article_ids]

# just get me those articles
ann_fn <- paste0("data/R_loading/papers_annotate_",length(article_ids), ".RData")
if (!file.exists(ann_fn)){
  time <- system.time(texts_ann <- rplos::plos_fulltext(ann_articles))
  saveRDS(object = texts_ann, file = ann_fn)
}

# and now get all of them
paper_chunks <- seq(1, length(papers$data$id), by=1000)

lapply(paper_chunks, function(chunk){
  fn <- paste0("data/R_loading/paper_",chunk,"_",chunk+999,".RData")
  
    if (!file.exists(fn)){
      cat("Working on file", fn,"\n")
      p <- papers$data$id[chunk:(min(chunk+999, length(papers$data$id)))]
      p <- p[!grepl(x = p, pattern = "annotation")]
      time <- system.time(texts <- rplos::plos_fulltext(p))
      saveRDS(texts, file=fn)
      cat("Finished on file", fn, "in", time[3], "seconds\n")
  }
})