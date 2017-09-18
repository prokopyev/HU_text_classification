# This script randomises articles loaded from a txt

# (c) Richard Kunert September 2017

wd = 'C:\\Users\\Richard\\Desktop\\R\\HU_text_classification'

setwd(wd)

#articles included in review acc to Insa
incl = read.delim(paste(wd, 'abstracts', "FL Review Library_Einschluss_ALL (128).txt", sep = '\\'),
                       sep = '\t', row.names = NULL, header = F ,na.strings = c("", "  "), quote = NULL)

incl = data.frame(title = incl[1:nrow(incl)-1,1], 
                  abstract = incl[2:nrow(incl),2])

incl = incl[!is.na(incl[,2]),]

write.table(excl, "incl.txt", sep="\t") 

#articles excluded from review acc to Insa
excl = read.delim(paste(wd, 'abstracts', "FL Review Library_Raus_ALL (601).txt", sep = '\\'),
                  sep = '\t', row.names = NULL, header = F ,na.strings = c("", ""), quote = NULL)

excl = data.frame(title = excl[1:nrow(excl)-1,1], 
                  abstract = excl[2:nrow(excl),2])

excl = excl[!is.na(excl[,2]),]

uns = read.delim(paste(wd, 'abstracts', "FL Review Library_Unsicher(17).txt", sep = '\\'),
                  sep = '\t', row.names = NULL, header = F ,na.strings = c("", ""), quote = NULL)

uns = data.frame(title = uns[1:nrow(uns)-1,1], 
                  abstract = uns[2:nrow(uns),2])

uns = uns[!is.na(uns[,2]),]

write.table(uns, "uns.txt", sep="\t") 

#combine and write to file

classified = cbind(rbind(matrix(rep(1,each=nrow(incl))), matrix(rep(0,each=nrow(excl)))), rbind(incl, excl))
classified$title = trimws(classified$title)

colnames(classified) <- c("included", "title", "abstract")

write.table(classified, "classified.txt", sep="\t") 

#get all abstracts into machine readable format
all = read.delim(paste(wd, 'abstracts', "FL Review Library_Gesamt.txt", sep = '\\'),
                 sep = '\t', row.names = NULL, header = F ,na.strings = c("", ""), quote = NULL)
#Problem: abstract sometimes distributed over many columns
abstract = character()
for (r in 2:nrow(all)){#for each row
  if (!is.na(all[r,2])){
    abstract_x = character()
    for(c in 2:ncol(all)){#for each column
      if (!is.na(all[r,c])){
        abstract_x = paste(abstract_x, as.character(all[r,c]))
      }
    }
    if (nchar(abstract_x) < 10){#not a real abstract
      print('excluded:')
      print(r)
      print(abstract_x)
    } else{
      abstract = c(abstract, abstract_x)  
    }
    
  }
}
#no abstract provided for Mazmanian, P. E., et al. (2014): added little blurp
# remove characters before e-mail: pie@ufs.ac.za
# remove double title of Knutson, K., et al. (2010)
# last entry seems to be duplicate, not removed
title = all[!is.na(all[,1]) & all[,1] != '"', 1]
#head(title)

all = data.frame(title = title, 
                 abstract = abstract)
write.table(all, "ALL.txt", sep="\t")

#the go through all classified and unclassified entries and discover that some entries
#which should be classified according to their alphabet position are actually not classified
