source("data_loading/plos_one_retriever.R")

papers <- readRDS("data/R_loading/paper_ids.RData")

# if full.paper is false, only the methods section will be extracted
extract.text <- function(fn, outputfolder, full.paper=FALSE){
  xml_folder <- paste0(outputfolder,"XML")
  txt_folder <- paste0(outputfolder,"TEXT")
  
  if (file.exists(fn)){
    texts <- readRDS(fn)
    ids <- names(texts)
    
    for (id in ids){
      xml_fn <- paste0(xml_folder, "/",sub(pattern = "/", replacement="_", x=id),".xml")
      
      cat("Working on file",xml_fn,"\n")
      if (!file.exists(xml_fn)){
        lines <- strsplit(texts[[id]], '\n')[[1]]
        ll <- lapply(lines, trimws)
        xml_text <- paste0(ll, collapse=" ")
        write(xml_text, file = xml_fn)
      }
      
      txt_fn <- paste0(txt_folder, "/",sub(pattern = "/", replacement="_", x=id),".txt")
      if (!file.exists(txt_fn)){
        xml = read_xml(xml_fn)
        txt = extract.xml(xml, full.paper)
        lines <- strsplit(txt, '\n')[[1]]
        lines <- lines[!startsWith(lines,"+++")]
        
        f = file(txt_fn)
        writeLines(text = lines,f)
        close(f)
      }
    }
  }else{
    cat("File",fn,"not available yet\n")
  }
}

 chunks <- seq(1, papers$meta$numFound, by=1000)
 for (chunk in chunks){
   fn <- paste0("data/R_loading/paper_",chunk,"_",chunk+999,".RData")
   dir.create('data/R_loading/XML', showWarnings = FALSE)
   dir.create('data/R_loading/TEXT', showWarnings = FALSE)
 
   extract.text(fn = fn, outputfolder = "data/R_loading/", full.paper = FALSE)
 
 }
 