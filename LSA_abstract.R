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

##############################################################################################
# GLOBAL VARIABLES
nDim = 170 #how many dimensions to get for the LSA


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

print('Creating LSA space...')
space = lsa(trm, dims = nDim) # create LSA space
#The diagonal matrix Sk contains the singular values of TDM in descending order.
#The ith singular value indicates the amount of variation along ith axis. 
#Tk will consist of terms and values along each dimension. 
#Dk will consist of documents and its values along each dimension.

###################################################################################
#Classification with LSA typicality value

# #The typicality value is just a correlation between the inclusion centroid's semantic dimension values 
# # and the semantic dimension values of the astract
# inclusion_centroid = colMeans(space$dk[abstracts$included == 1,], na.rm = T)
# corr_centroid = unlist(lapply(1:nrow(abstracts), function(x) cor(data.frame(space$dk[x,], inclusion_centroid))[1,2]))
# 
# #One could now plot the distribution of typicality values, i.e. the Pearson correlation coefficients
# #start off with a histogram only made for extracting values. The interest is not in looking at p
# p = ggplot(data = data.frame(x = corr_centroid,
#                              included = as.factor(abstracts$included)),
#            aes(x = x, fill = included, linetype = included)) +
#   geom_histogram(position = 'identity', alpha = 0.5, color = 'black')
# # extract relevant variables from the plot object to a new data frame
# # your grouping variable 'included' is named 'group' in the plot object
# df <- ggplot_build(p)$data[[1]][ , c("xmin", "y", "group")]
# #get the factor levels interpretable
# df$group[df$group == 4] = 'unclassified'
# df$group[df$group == 3] = 'included'
# df$group[df$group == 2] = 'unsure'
# df$group[df$group == 1] = 'excluded'
# #plot
# ggplot(data = df, aes(x = xmin, y = y, color = factor(group))) +
#   geom_step(size = 1.5)
# 
# #An ROC curve (true positive rate versus false positive rate) would also be nice
# ROC = matrix(NA, nrow= 100, ncol = 3)
# counter = 0
# for(r in seq(-1, 1, length.out = 100)){#for each possible cut-off value
#   counter = counter + 1
#   TPR = sum(corr_centroid[abstracts$included == 1] > r, na.rm = T)/sum(abstracts$included == 1, na.rm = T)
#   FPR = sum(corr_centroid[abstracts$included == 0] > r, na.rm = T)/sum(abstracts$included == 0, na.rm = T)
#   ACC = sum(corr_centroid[abstracts$included == 1] > r, na.rm = T,#true positive
#             corr_centroid[abstracts$included == 0] < r, na.rm = T)/#true negative
#     sum(abstracts$included == 0, abstracts$included == 1, na.rm = T)
#   ROC[counter,] = c(FPR, TPR, ACC)
# }
# plot(ROC[,1:2], xlab = 'False positive rate', ylab = 'True positive rate')
# plot(ROC[,3], xlab = 'idx', ylab = 'Accuracy', ylim = c(0,1))
# max(ROC[,3])
# 
# #Finally, explore some concrete values
# # min(corr_centroid[abstracts$included == 1], na.rm = T)
# # min(corr_centroid[abstracts$included == 0], na.rm = T)
# # min(corr_centroid[abstracts$included == 0.5], na.rm = T)
# # 
# # max(corr_centroid[abstracts$included == 1], na.rm = T)
# # max(corr_centroid[abstracts$included == 0], na.rm = T)
# # max(corr_centroid[abstracts$included == 0.5], na.rm = T)
# 
# #proportion of to-be-excluded abstracts which can be excluded without excluding any to-be-included abstracts
# sum(corr_centroid[abstracts$included == 0] < 
#       min(corr_centroid[abstracts$included == 1], na.rm = T), na.rm = T)/
#   sum(abstracts$included == 0, na.rm = T)
# 
# rejection_cutoff = min(corr_centroid[abstracts$included == 1], na.rm = T)#no false inclusions
# rejection_cutoff = 0.25#only value with accuracy above 90% according to grid search
# 
# true_rejections = sum(corr_centroid[abstracts$included == 0] < rejection_cutoff, na.rm = T)
# false_inclusions = sum(corr_centroid[abstracts$included == 0] > rejection_cutoff, na.rm = T)
# true_inclusions = sum(corr_centroid[abstracts$included == 1] > rejection_cutoff, na.rm = T)
# false_rejections = sum(corr_centroid[abstracts$included == 1] < rejection_cutoff, na.rm = T)
# 
# accuracy = sum(true_rejections, true_inclusions)/sum(true_rejections, true_inclusions, false_rejections, false_inclusions)
# accuracy

###################################################################################
#BALANCE DATA SET

#First we need to get the testing data set from the full corpus

#reduce full sample to those cases in which human raters could determine label
classified_idx = c(which(abstracts$included == 1), which(abstracts$included == 0))
#classified =  abstracts[classified_idx,]

# determine testing and training subsamples
testing_size = 100#number of articles in testing set
testing_sample_idx = sample(classified_idx, testing_size)
training_sample_idx = classified_idx[!(classified_idx %in% testing_sample_idx)]

#So now, there are several different options for dealing with the imbalance in the training sample 

data_training_raw = abstracts[training_sample_idx,]
data_testing_raw = abstracts[testing_sample_idx,]

data_testing_lsa = space$dk[testing_sample_idx,]
class_testing = as.factor(abstracts$included[testing_sample_idx])

print('Balancing training data...')
if (FALSE) {#simply ignore the problem and live with the imbalance
  
  data_training_lsa = space$dk[training_sample_idx,] 
  class_training = as.factor(abstracts$included[training_sample_idx])
  
} else if (FALSE) {#randomly undersampling the majority case 
  
  zero_idx = sample(which(data_training_raw$included == 0),
                    sum(data_training_raw$included == 1.0, na.rm = T))
  
  data_training_lsa = space$dk[training_sample_idx,] 
  data_training_lsa = data_training_lsa[c(zero_idx, which(data_training_raw$included == 1.0)),]
  
  class_training = as.factor(data_training_raw$included[c(zero_idx, which(data_training_raw$included == 1.0))])
  
} else if (TRUE) {#SMOTE (Synthetic Minority Oversampling TEchnique)
  
  
  if(!require(smotefamily)){install.packages('smotefamily')} #Synthetic Minority Oversampling TEchnique
  if(!require(FNN)){install.packages('FNN')} #
  
  library(smotefamily)

  data_training_SMOTE = SMOTE(as.data.frame(space$dk[training_sample_idx,]),
                            data_training_raw$included, K = 5, dup_size = 0)
  
  data_training_lsa = data_training_SMOTE$data[,1:nDim]
  class_training = as.factor(data_training_SMOTE$data[,nDim + 1])
  
} else if (FALSE) {#ROSE (Random Over-Sampling Examples)
  
  if(!require(ROSE)){install.packages('ROSE')} #Random Over-Sampling Examples
  library(ROSE)
  
  data_training_rose = as.data.frame(space$dk[training_sample_idx,])
  data_training_rose$included = abstracts$included[training_sample_idx]
  
  data.rose <- ROSE(included~., data=data_training_rose, seed=3)
  
  data_training_lsa = data.rose$data[,1:nDim]
  class_training = as.factor(data.rose$data$included)
  
}


###################################################################################
#SVM

if (TRUE){
  
  if(!require(e1071)){install.packages('e1071')} #SVM
  library(e1071)

  #Determine optimal SVM parameters
  print('Determining optimal SVM parameters...')
  svm_tune <- tune(svm, train.x=data_training_lsa,
                   train.y=class_training,
                   kernel="radial",
                   scale = F,
                   parallel.core = 2,
                   ranges=list(cost=10^(0:4),
                               gamma=c(seq(0.01,0.2, 0.01), 0.25, .5, c(seq(1,15, 2), 30, 50, 100))))

  print(svm_tune)
  #svm_tune$performances$error
  
  #Train SVM
  print('Training SVM classifier...')
  svm_m = svm(x = data_training_lsa,#training vectors
              y = class_training,#training classification
              kernel = 'radial', probability = T, scale = F,
              cost = svm_tune$best.parameters$cost,
              gamme = svm_tune$best.parameters$gamma)
  #best performance for 50/50 sample with cost = 10 and gamma = 1
  
  #performance on training sample
  pred <- predict(svm_m,data_training_lsa)
  table(pred,factor(class_training))
  #overall accuracy (training sample)
  sum(unlist(lapply(1:length(pred), function(x) pred[x] == class_training[x])))/length(pred)
  
  #performance on test sample
  pred <- predict(svm_m,data_testing_lsa, probability = T)
  print('Testing sample confusion matrix:')
  print(table(pred,factor(class_testing)))
  #overall accuracy (testing sample)
  print(sprintf('Testing sample accuracy: %1.2f',sum(unlist(lapply(1:length(pred), function(x) pred[x] == class_testing[x])))/length(pred)))
  
  #mean probability of belonging to zero case
  mean(attr(pred, 'probabilities')[class_testing == 1, 2])#ideally low
  mean(attr(pred, 'probabilities')[class_testing == 0, 2])#ideally high
  
  #proportion of to-be-excluded abstracts which can be excluded without excluding any to-be-included abstracts
  print(sprintf('Proportion of to-be-excluded abstracts in testing sample which can be excluded without excluding any to-be-included abstracts: %1.2f',sum(attr(pred, 'probabilities')[class_testing == 0, 2] > 
        max(attr(pred, 'probabilities')[class_testing == 1, 2]))/
    sum(class_testing == 0)))
  
  #One could now plot the distribution of typicality values, i.e. the Pearson correlation coefficients
  #start off with a histogram only made for extracting values. The interest is not in looking at p
  p = ggplot(data = data.frame(x = attr(pred, 'probabilities')[, 2],
                               included = as.factor(class_testing)),
             aes(x = x, fill = included, linetype = included)) +
    geom_histogram(position = 'identity', alpha = 0.5, color = 'black')
  # extract relevant variables from the plot object to a new data frame
  # your grouping variable 'included' is named 'group' in the plot object
  df <- ggplot_build(p)$data[[1]][ , c("xmin", "y", "group")]
  #get the factor levels interpretable
  df$group[df$group == 2] = 'should be included'
  df$group[df$group == 1] = 'should be excluded'
  #plot
  print(ggplot(data = df, aes(x = xmin, y = y, color = factor(group))) +
    geom_step(size = 1.5))
}