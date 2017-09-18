#This script includes helper functions for classifying article abstracts

#(c) Richard Kunert September 2017

#########################################################################
#Load libraries
if(!require(lsa)){install.packages('lsa')} #latent semantic analysis
library(lsa)

if(!require(tm)){install.packages('tm')} #text mining
library(tm)

if(!require(topicmodels)){install.packages('topicmodels')} #Latent Dirichlet Allocation
library(topicmodels)

if(!require(smotefamily)){install.packages('smotefamily')} #Synthetic Minority Oversampling TEchnique
if(!require(FNN)){install.packages('FNN')} #if not installed we get an error
library(smotefamily)

if(!require(ROSE)){install.packages('ROSE')} #Random Over-Sampling Examples
library(ROSE)

if(!require(e1071)){install.packages('e1071')} #SVM
library(e1071)

if(!require(ggplot2)){install.packages('ggplot2')} #plotting
library(ggplot2)

#########################################################################
#LSA

LSA_fun = function(texts, nDim = 100, verbose = T){ 
  
  #This function turns a list of input strings (texts) into an nDim dimensional semantic space
  #The output is a nrow(texts) x nDim matrix representing each string as a vector in semantic space
  
  if (verbose) print('Creating LSA space...')
  
  #texts should be a list of strings
  
  #create document term matrix
  corp <- Corpus(VectorSource(texts))
  tdm = TermDocumentMatrix(corp, control = list(weighting = weightTfIdf, removePunctuation = TRUE, 
                                                removeNumbers = TRUE, stopwords = TRUE,
                                                tolower = T,
                                                minWordLength = 3))
  
  # create LSA space
  lsa_space = lsa(tdm, dims = nDim)
  #The diagonal matrix Sk contains the singular values of TDM in descending order.
  #The ith singular value indicates the amount of variation along ith axis. 
  #Tk will consist of terms and values along each dimension. 
  #Dk will consist of documents and its values along each dimension.
  
  return(lsa_space$dk)
}

#########################################################################
#LDA (Latent Dirichlet Allocation)

LDA_fun = function(texts, nDim = 100, verbose = T){ 
  
  #This function turns a list of input strings (texts) into an nDim dimensional topic space
  #The output is a nrow(texts) x nDim matrix representing each string as a vector in topic space
  
  if (verbose) print('Creating LDA topics...')
  
  corp <- Corpus(VectorSource(texts))
  dtm_lda = DocumentTermMatrix(corp, control = list(weighting = weightTf,#weightTfIdf does not work
                                                    removePunctuation = TRUE, 
                                                    removeNumbers = TRUE, stopwords = TRUE,
                                                    tolower = T,
                                                    minWordLength = 3))
  
  
  lda_space = LDA(dtm_lda, nDim)
  
  #as.matrix(terms(lda,6))#top 6 terms in each topic
  
  return(lda_space@gamma)#probabilities associated with each topic assignment
}

###################################################################################
#BALANCE DATA SET

balance_fun = function(texts, classes, balance_algo = 'SMOTE'){
  
  #A collection of options for turning an unbalanced data set into a balanced one
  #The data set is defined by texts (a n x m matrix with rows representing texts and columns representing dimensions)
  #and classes (the class membership labels corresponding to rows in texts)
  #values for balance_algo = c('none', 'under', 'SMOTE', 'ROSE')
  
  print('Balancing training data...')
  
  if (balance_algo == 'none') {#simply ignore the problem and live with the imbalance
    
    texts_out = texts 
    classes_out = classes
    
  } else if (balance_algo == 'under') {#randomly undersampling the majority case 
    
    zero_idx = sample(which(classes == 0),
                      sum(classes == 1.0, na.rm = T))
    
    texts_out = texts[c(zero_idx, which(classes == 1.0)),]
    classes_out = classes[c(zero_idx, which(classes == 1.0))]
    
  } else if (balance_algo == 'SMOTE') {#SMOTE (Synthetic Minority Oversampling TEchnique)
    
    texts_SMOTE = SMOTE(as.data.frame(texts), classes, K = 5, dup_size = 0)
    
    texts_out = texts_SMOTE$data[,1:ncol(texts)]
    classes_out = texts_SMOTE$data[,ncol(texts) + 1]
    
  } else if (balance_algo == 'ROSE') {#ROSE (Random Over-Sampling Examples)
    
    data_for_ROSE = as.data.frame(texts)
    data_for_ROSE$classes = classes
    
    data_ROSE <- ROSE(classes~., data=data_for_ROSE)
    
    texts_out = data_ROSE$data[,1:ncol(texts)]
    classes_out = data_ROSE$data$classes
    
  }
  
  return(data.frame(texts = texts_out,
                    classes = classes_out))
  
}

###################################################################################
#SVM

SVM_fun = function (texts_train, texts_test, classes_train, classes_test, verbose = T){
  
  #This function trains a SVM classifier on text data and provides some diagnostics
  #It automatically finds optimal classifier settings
  #It outputs diagnostics
  
  #Determine optimal SVM parameters
  print('Determining optimal SVM parameters...')
  svm_tune <- tune(svm, train.x=texts_train,
                   train.y=classes_training,
                   kernel="radial",
                   scale = F,
                   parallel.core = 2,
                   ranges=list(cost=10^(0:4),
                               gamma=c(seq(0.01,0.2, 0.04), 0.25, .5, c(seq(1,15, 4), 30, 50, 100))))
  
  if(verbose) print(svm_tune)
  #svm_tune$performances$error
  
  #Train SVM
  print('Training SVM classifier...')
  svm_m = svm(x = texts_train,#training vectors
              y = classes_train,#training classification
              kernel = 'radial', probability = T, scale = F,
              cost = svm_tune$best.parameters$cost,
              gamme = svm_tune$best.parameters$gamma)
  
  #performance on training sample
  #pred <- predict(svm_m, texts_train)
  #table(pred,factor(classes_train))
  #overall accuracy (training sample)
  #sum(unlist(lapply(1:length(pred), function(x) pred[x] == classes_train[x])))/length(pred)
  
  #performance on test sample
  pred <- predict(svm_m, texts_test, probability = T)
  if(verbose) print('Testing sample confusion matrix:')
  if(verbose) print(table(pred,factor(classes_test)))
  #overall accuracy (testing sample)
  if(verbose) print(sprintf('Testing sample accuracy: %1.2f',
                            sum(unlist(lapply(1:length(pred), function(x) pred[x] == classes_test[x])))/length(pred)))
  
  #mean probability of belonging to zero case
  #mean(attr(pred, 'probabilities')[classes_test == 1, 2])#ideally low
  #mean(attr(pred, 'probabilities')[classes_test == 0, 2])#ideally high
  
  #proportion of to-be-excluded abstracts which can be excluded without excluding any to-be-included abstracts
  if(verbose) print(sprintf('Proportion of to-be-excluded abstracts in testing sample which can be excluded without excluding any to-be-included abstracts: %1.2f',
                            sum(attr(pred, 'probabilities')[classes_test == 0, 2] > 
                                  max(attr(pred, 'probabilities')[classes_test == 1, 2]))/
                              sum(classes_test == 0)))
  
  #start off with a histogram only made for extracting values. The interest is not in looking at p
  p = ggplot(data = data.frame(x = attr(pred, 'probabilities')[, 2],
                               included = as.factor(classes_test)),
             aes(x = x, fill = included, linetype = included)) +
    geom_histogram(position = 'identity', alpha = 0.5, color = 'black')
  # extract relevant variables from the plot object to a new data frame
  # the grouping variable is named 'group' in the plot object
  df <- ggplot_build(p)$data[[1]][ , c("xmin", "y", "group")]
  #get the factor levels interpretable
  df$group[df$group == 2] = 'should be included'
  df$group[df$group == 1] = 'should be excluded'
  
  p_multihist = ggplot(data = df, aes(x = xmin, y = y, color = factor(group))) +
    geom_step(size = 1.5)
  
  if(verbose) print(p_multihist)
  
  #return all sorts of diagnostics
  return(list(testing_sample_size = length(classes_test),
              training_sample_size = length(classes_train),
              training_prop_positive_cases = sum(classes_train),
              nDim = ncol(texts_train),
              SVM_parameters = svm_tune$best.parameters,
              testing_sample_acc = sum(unlist(lapply(1:length(pred), function(x) pred[x] == classes_testing[x])))/length(pred),
              prop_excl_before_FN = sum(attr(pred, 'probabilities')[classes_test == 0, 2] > 
                                          max(attr(pred, 'probabilities')[classes_test == 1, 2]))/
                sum(classes_test == 0),
              testing_confusion_matrix = table(pred,factor(classes_test)),
              multihist = p_multihist
  ))
}