#This script classifies article abstracts with LSA

#(c) Richard Kunert

if(!require(textstem)){install.packages('textstem')} #
library(textstem) #lemmatization (rather than stupid stemming)

if(!require(tm)){install.packages('tm')} #text mining
library(tm)

if(!require(lsa)){install.packages('lsa')} #latent semantic analysis
library(lsa)

if(!require(ggplot2)){install.packages('ggplot2')} #plotting
library(ggplot2)


wd = 'C:\\Users\\Richard\\Desktop\\R\\HU_text_classification'

setwd(wd)

abstracts = read.delim(paste(wd, 'abstracts', "ALL.txt", sep = '\\'),
                        sep = '\t', row.names = NULL, header = T, na.strings = c("", ""), quote = NULL)

#Lemmatize the corpus
abstracts$abstract = lemmatize_strings(abstracts$abstract)

#########################################################################
#LSA

#create document term matrix
corp <- Corpus(VectorSource(abstracts$abstract))
trm = TermDocumentMatrix(corp, control = list(weighting = weightTfIdf, removePunctuation = TRUE, 
                                      removeNumbers = TRUE, stopwords = TRUE,
                                      tolower = T,
                                      minWordLength = 3))

#trm = lw_bintf(trm) * gw_idf(trm) # weighting

space = lsa(trm, dims = 100) # create LSA space
#The diagonal matrix Sk contains the singular values of TDM in descending order.
#The ith singular value indicates the amount of variation along ith axis. 
#Tk will consist of terms and values along each dimension. 
#Dk will consist of documents and its values along each dimension.

###################################################################################
#Classification with LSA typicality value

#The typicality value is just a correlation between the inclusion centroid's semantic dimension values 
# and the semantic dimension values of the astract
inclusion_centroid = colMeans(space$dk[abstracts$included == 1,], na.rm = T)
corr_centroid = unlist(lapply(1:nrow(abstracts), function(x) cor(data.frame(space$dk[x,], inclusion_centroid))[1,2]))

#One could now plot the distribution of typicality values, i.e. the Pearson correlation coefficients
#start off with a histogram only made for extracting values. The interest is not in looking at p
p = ggplot(data = data.frame(x = corr_centroid,
                             included = as.factor(abstracts$included)),
           aes(x = x, fill = included, linetype = included)) +
  geom_histogram(position = 'identity', alpha = 0.5, color = 'black')
# extract relevant variables from the plot object to a new data frame
# your grouping variable 'included' is named 'group' in the plot object
df <- ggplot_build(p)$data[[1]][ , c("xmin", "y", "group")]
#get the factor levels interpretable
df$group[df$group == 4] = 'unclassified'
df$group[df$group == 3] = 'included'
df$group[df$group == 2] = 'unsure'
df$group[df$group == 1] = 'excluded'
#plot
ggplot(data = df, aes(x = xmin, y = y, color = factor(group))) +
  geom_step(size = 1.5)

#An ROC curve (true positive rate versus false positive rate) would also be nice
ROC = matrix(NA, nrow= 100, ncol = 3)
counter = 0
for(r in seq(-1, 1, length.out = 100)){#for each possible cut-off value
  counter = counter + 1
  TPR = sum(corr_centroid[abstracts$included == 1] > r, na.rm = T)/sum(abstracts$included == 1, na.rm = T)
  FPR = sum(corr_centroid[abstracts$included == 0] > r, na.rm = T)/sum(abstracts$included == 0, na.rm = T)
  ACC = sum(corr_centroid[abstracts$included == 1] > r, na.rm = T,#true positive
            corr_centroid[abstracts$included == 0] < r, na.rm = T)/#true negative
    sum(abstracts$included == 0, abstracts$included == 1, na.rm = T)
  ROC[counter,] = c(FPR, TPR, ACC)
}
plot(ROC[,1:2], xlab = 'False positive rate', ylab = 'True positive rate')
plot(ROC[,3], xlab = 'idx', ylab = 'Accuracy', ylim = c(0,1))
max(ROC[,3])

#Finally, explore some concrete values
min(corr_centroid[abstracts$included == 1], na.rm = T)
min(corr_centroid[abstracts$included == 0], na.rm = T)
min(corr_centroid[abstracts$included == 0.5], na.rm = T)

max(corr_centroid[abstracts$included == 1], na.rm = T)
max(corr_centroid[abstracts$included == 0], na.rm = T)
max(corr_centroid[abstracts$included == 0.5], na.rm = T)

#proportion of to-be-excluded abstracts which can be excluded without excluding any to-be-included abstracts
sum(corr_centroid[abstracts$included == 0] < 
      min(corr_centroid[abstracts$included == 1], na.rm = T), na.rm = T)/
  sum(classified$included == 0)

rejection_cutoff = min(corr_centroid[abstracts$included == 1], na.rm = T)#no false inclusions
rejection_cutoff = 0.25#only value with accuracy above 90% according to grid search

true_rejections = sum(corr_centroid[abstracts$included == 0] < rejection_cutoff, na.rm = T)
false_inclusions = sum(corr_centroid[abstracts$included == 0] > rejection_cutoff, na.rm = T)
true_inclusions = sum(corr_centroid[abstracts$included == 1] > rejection_cutoff, na.rm = T)
false_rejections = sum(corr_centroid[abstracts$included == 1] < rejection_cutoff, na.rm = T)

accuracy = sum(true_rejections, true_inclusions)/sum(true_rejections, true_inclusions, false_rejections, false_inclusions)
accuracy
###################################################################################
#SVM

if(!require(e1071)){install.packages('e1071')} #SVM
library(e1071)

#if(!require(RTextTools)){install.packages('RTextTools')} ##
#library(RTextTools)

#create a 50/50 corpus
zero_idx = sample(which(abstracts$included == 0),
                  sum(abstracts$included == 1.0, na.rm = T))

#full corpus
zero_idx = which(abstracts$included == 0)

classified_idx = c(which(abstracts$included == 1), zero_idx)
classified =  abstracts[classified_idx,]
classified_lsa = space$dk[classified_idx,]

# SPLIT SAMPLE
testing_size = 100#number of articles in testing set
total_sample = 1:nrow(classified)
testing_sample = sample(total_sample, testing_size)
training_sample = total_sample[!(total_sample %in% testing_sample)]

mean(classified$included[training_sample])
mean(classified$included[testing_sample])

#Train SVM
#somehow this fails
# svm_tune <- tune(svm, train.x=classified_lsa[training_sample,],
#                  train.y=factor(classified$included[training_sample]), 
#                  kernel="radial", ranges=list(cost=10^(-2:4), gamma=c(0.01, 0.25, .5, 1, 2, 4, 8, 16, 32)))
# 
# print(svm_tune)
# #svm_tune$performances$error

svm_m = svm(x = classified_lsa[training_sample,],#training vectors
    y = factor(classified$included[training_sample]),#training classification
    kernel = 'radial', cost = 10, gamma = 1, probability = T, scale = F)
#best performance for 50/50 sample with cost = 10 and gamma = 1

#performance on training sample
pred <- predict(svm_m,classified_lsa[training_sample,])
table(pred,factor(classified$included[training_sample]))
#overall accuracy
sum(unlist(lapply(1:length(pred), function(x) pred[x] == classified$included[training_sample[x]])))/length(pred)

#performance on test sample
pred <- predict(svm_m,classified_lsa[testing_sample,], probability = T)
table(pred,factor(classified$included[testing_sample]))
#overall accuracy
sum(unlist(lapply(1:length(pred), function(x) pred[x] == classified$included[testing_sample[x]])))/length(pred)

#mean probability of belonging to zero case
mean(attr(pred, 'probabilities')[classified$included[testing_sample] == 1, 2])#ideally low
mean(attr(pred, 'probabilities')[classified$included[testing_sample] == 0, 2])#ideally high

#proportion of to-be-excluded abstracts which can be excluded without excluding any to-be-included abstracts
sum(attr(pred, 'probabilities')[classified$included[testing_sample] == 0, 2] > 
      max(attr(pred, 'probabilities')[classified$included[testing_sample] == 1, 2]))/
  sum(classified$included[testing_sample] == 0)

#One could now plot the distribution of typicality values, i.e. the Pearson correlation coefficients
#start off with a histogram only made for extracting values. The interest is not in looking at p
p = ggplot(data = data.frame(x = attr(pred, 'probabilities')[, 2],
                             included = as.factor(classified$included[testing_sample])),
           aes(x = x, fill = included, linetype = included)) +
  geom_histogram(position = 'identity', alpha = 0.5, color = 'black')
# extract relevant variables from the plot object to a new data frame
# your grouping variable 'included' is named 'group' in the plot object
df <- ggplot_build(p)$data[[1]][ , c("xmin", "y", "group")]
#get the factor levels interpretable
df$group[df$group == 2] = 'should be included'
df$group[df$group == 1] = 'should be excluded'
#plot
ggplot(data = df, aes(x = xmin, y = y, color = factor(group))) +
  geom_step(size = 1.5)