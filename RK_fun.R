#This script includes helper functions for classifying article abstracts

#(c) Richard Kunert September 2017

#########################################################################
#Load libraries

if(!require(textstem)){install.packages('textstem')} #
library(textstem) #lemmatization (rather than stupid stemming)

if(!require(lsa)){install.packages('lsa')} #latent semantic analysis
library(lsa)

if(!require(tm)){install.packages('tm')} #text mining
library(tm)

if(!require(topicmodels)){install.packages('topicmodels')} #Latent Dirichlet Allocation
library(topicmodels)

if(!require(RTextTools)){install.packages('RTextTools')} #create_matrix function
library(RTextTools)

if(!require(smotefamily)){install.packages('smotefamily')} #Synthetic Minority Oversampling TEchnique
if(!require(FNN)){install.packages('FNN')} #if not installed we get an error
library(smotefamily)

if(!require(data.table)){install.packages('data.table')} #Synthetic Minority Oversampling TEchnique
library(data.table)#rbindlist function

if(!require(ROSE)){install.packages('ROSE')} #Random Over-Sampling Examples
library(ROSE)

if(!require(e1071)){install.packages('e1071')} #SVM
library(e1071)

if(!require(ggplot2)){install.packages('ggplot2')} #plotting
library(ggplot2)

if(!require(caTools)){install.packages('caTools')} #Boosting
library(caTools)

if(!require(randomForest)){install.packages('randomForest')} #Random Forest
library(randomForest)


#########################################################################
#Data Cleaning

Cleaning_fun = function(texts){ 
  
  #Remove non-ASCII characters
  texts = iconv(texts, "latin1", "ASCII", sub=" ")
  
  #Remove non-alphanumeric characters
  texts = gsub("[^[:alnum:] ]", "", texts)
  
  #Remove footers
  texts = removeWords(texts,
                      c('PsycINFO Database Record c \\d+ APA all rights reserved',
                        'C \\d+ Elsevier Ltd All rights reserved',
                        'C \\d+ Elsevier BV All rights reserved',
                        'C \\d+ Elsevier GmbH All rights reserved',
                        'C \\d+ Elsevier Inc All rights reserved',
                        'C \\d+ Elsevier Ireland Ltd All rights reserved',
                        'C \\d+ Published by Elsevier Inc',
                        'c \\d+ Wiley Periodicals Inc',
                        'C \\d+ The Authors Published by Elsevier Ltd',
                        'C \\d+ The Authors Production and hosting by Elsevier BV on behalf of King Saud University',
                        'C \\d+ Published by Elsevier BV on behalf of European Cystic Fibrosis Society',
                        'Copyright c \\d+ John Wiley  Sons Ltd',
                        'C \\d+ Institution of Chemical Engineers Published by Elsevier BV All rights reserved',
                        'Contains \\d+ table',
                        'Contains \\d+ tables',
                        'Contains \\d+ figure',
                        'Contains \\d+ figures',
                        'Contains \\d+ note',
                        'Contains \\d+ notes',
                        'Contains \\d+ footnotes',
                        'Contains \\d+ figure and \\d+ tables',
                        'Contains \\d+ figures and \\d+ tables',
                        'Contains \\d+ figures and \\d+ online resources',
                        'Contains \\d+ figures and \\d+ table',
                        'Contains \\d+ figures and \\d+ footnote',
                        'Contains \\d+ table and \\d+ figures',
                        'Contains \\d+ table and \\d+ note',
                        'Contains \\d+ tables and \\d+ figure',
                        'Contains \\d+ tables and \\d+ resources',
                        'Contains \\d+ tables and \\d+ notes',
                        'Contains \\d+ tables \\d+ figures and \\d+ note',
                        'Contains \\d+ tables and \\d+ figures',
                        'Contains \\d+ notes and 1 table',
                        'Contains \\d+ notes \\d+ figures and \\d+ table',
                        'Contains \\d+ figures \\d+ tables and \\d+ notes',
                        'Contains \\d+ figures 1 table and 1 footnote',
                        'C \\d+ by the American College of Cardiology Foundation',
                        'Anesth Analg \\d+ \\d+',
                        'Global Health Promotion \\d+  \\d+ \\d+',
                        'C \\d+ Acoustical Society of America',
                        'C \\d+ Association of Program Directors in Surgery Published by Elsevier Inc All rights reserved',
                        'There are no conflicts of interest to declare',
                        'C \\d+ by The International Union of Biochemistry and Molecular Biology',
                        'Journal of International Business Studies \\d+ \\d+ \\d+',
                        'C \\d+ Jurusan Fisika FMIPA UNNES Semarang',
                        'Ann Emerg Med \\d+ \\d+',
                        'Clin Trans Sci \\d+  Volume \\d+ \\d+',
                        'C \\d+ Wiley Periodicals Inc J Res Sci Teach \\d+ \\d+ \\d+',
                        'Copyright c \\d+ Strategic Management Society',
                        'c \\d+ by The International Union of Biochemistry and Molecular Biology \\d+ \\d+'
                      ))
  
  #Lemmatize the corpus
  texts = lemmatize_strings(texts)
  
  gc()
  
  return(texts)
  
  }

#########################################################################
#LSA

LSA_fun = function(texts, nDim = 100, ngramLength = 1, verbose = T){ 
  
  #This function turns a list of input strings (texts) into an nDim dimensional semantic space
  #The output is a nrow(texts) x nDim matrix representing each string as a vector in semantic space
  
  if (verbose) print(sprintf('Creating LSA space with nDim %d and ngramLength %d...', nDim, ngramLength))
  
  #texts should be a list of strings
  
  #create document term matrix
  corp <- Corpus(VectorSource(texts))
  
  ngramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = ngramLength, max = ngramLength))
  
  tdm = TermDocumentMatrix(corp, control = list(weighting = weightTfIdf, removePunctuation = TRUE, 
                                                removeNumbers = TRUE, stopwords = TRUE,
                                                tolower = T,
                                                minWordLength = 3,
                                                tokenize=ngramTokenizer))
  
  # create LSA space
  lsa_space = lsa(tdm, dims = nDim)
  #The diagonal matrix Sk contains the singular values of TDM in descending order.
  #The ith singular value indicates the amount of variation along ith axis. 
  #Tk will consist of terms and values along each dimension. 
  #Dk will consist of documents and its values along each dimension.
  
  gc()
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
  
  gc()
  return(lda_space@gamma)#probabilities associated with each topic assignment
}

#########################################################################
#Raw text vectorization

TXT_fun = function(texts, weighting = tm::weightTfIdf, ngramLength = 1, verbose = T){ 
  
  #This function turns a list of input strings (texts) into an n dimensional word space
  #The output is a nrow(texts) x n (words) matrix representing each string as a vector in word space
  
  if (verbose) print(sprintf('Vectorizing text at ngramLength %d...', ngramLength))
  
  corp <- Corpus(VectorSource(texts))
  dtm = create_matrix(texts, language="english", removeNumbers=TRUE,
                      stemWords=F, removeStopwords=TRUE, toLower=TRUE,
                      ngramLength= ngramLength, weighting=weighting)#removeSparseTerms=.95,
  # dtm = DocumentTermMatrix(corp, control = list(weighting = weightTfIdf, removePunctuation = TRUE, 
  #                                               removeNumbers = TRUE, stopwords = TRUE,
  #                                               tolower = T,
  #                                               minWordLength = 3))
  sparse_DTM <- sparseMatrix(i = dtm$i, j = dtm$j, x = dtm$v,
                             dims = dim(dtm),
                             dimnames = list(rownames(dtm), colnames(dtm)))
  
  gc()
  return(as.matrix(sparse_DTM))
  
}

###################################################################################
#BALANCE DATA SET

balance_fun = function(texts, classes, balance_algo = 'SMOTE', verbose = T){
  
  #A collection of options for turning an unbalanced data set into a balanced one
  #The data set is defined by texts (a n x m matrix with rows representing texts and columns representing dimensions)
  #and classes (the class membership labels corresponding to rows in texts)
  #values for balance_algo = c('none', 'under', 'SMOTE', 'ROSE')
  
  if (verbose) print('Balancing training data...')
  
  if (balance_algo == 'none') {#simply ignore the problem and live with the imbalance
    
    texts_out = texts 
    classes_out = classes
    
  } else if (balance_algo == 'under') {#randomly undersampling the majority case 
    
    zero_idx = sample(which(classes == 0),
                      sum(classes == 1.0, na.rm = T))
    
    texts_out = texts[c(zero_idx, which(classes == 1.0)),]
    classes_out = classes[c(zero_idx, which(classes == 1.0))]
    
  } else if (balance_algo == 'SMOTE') {#SMOTE (Synthetic Minority Oversampling TEchnique)
    
    #the following code is a nearly verbatim copy of the SMOTE function.
    #I adjusted it to work with matrices
    
    #otherwise use this, which fails with very large input matrix (raw DTM)
    #texts_SMOTE = SMOTE(as.data.frame(texts), classes, K = 5, dup_size = 0)
    
    X = texts
    target = classes
    K = 5
    dup_size = 0
    
    ncD = ncol(X)
    n_target = table(target)
    classP = names(which.min(n_target))
    P_set = subset(X, target == names(which.min(n_target)))[sample(min(n_target)),]#minority
    N_set = subset(X, target != names(which.min(n_target)))#majority
    P_class = rep(names(which.min(n_target)), nrow(P_set))
    N_class = target[target != names(which.min(n_target))]
    sizeP = nrow(P_set)
    sizeN = nrow(N_set)
    knear = knearest(P_set, P_set, K)
    sum_dup = n_dup_max(sizeP + sizeN, sizeP, sizeN, dup_size)
    syn_dat = NULL
    for (i in 1:sizeP) {
      if (is.matrix(knear)) {
        pair_idx = knear[i, ceiling(runif(sum_dup) * K)]
      }
      else {
        pair_idx = rep(knear[i], sum_dup)
      }
      g = runif(sum_dup)
      P_i = matrix(unlist(P_set[i, ]), sum_dup, ncD, byrow = TRUE)
      Q_i = as.matrix(P_set[pair_idx, ])
      syn_i = P_i + g * (Q_i - P_i)
      syn_dat = rbind(syn_dat, syn_i)
    }
    P_set = cbind(P_set, P_class)
    #P_set[, ncD + 1] = P_class#doesn't work for matrices -RK
    #colnames(P_set) = c(colnames(X), "class")
    N_set = cbind(N_set, N_class)
    #N_set[, ncD + 1] = N_class
    #colnames(N_set) = c(colnames(X), "class")
    rownames(syn_dat) = NULL
    syn_dat = data.frame(syn_dat)
    syn_dat = cbind(syn_dat, rep(names(which.min(n_target)), nrow(syn_dat))) 
    #syn_dat[, ncD + 1] = rep(names(which.min(n_target)), nrow(syn_dat))
    #colnames(syn_dat) = c(colnames(X), "class")
    names(syn_dat) = NULL
    NewD = data.frame(rbindlist(list(as.data.frame(P_set),
                                     as.data.frame(syn_dat), 
                                     as.data.frame(N_set))))
    #NewD = rbind(P_set, syn_dat, N_set)
    rownames(NewD) = NULL
    D_result = list(data = NewD, syn_data = syn_dat, orig_N = N_set, 
                    orig_P = P_set, K = K, K_all = NULL, dup_size = sum_dup, 
                    outcast = NULL, eps = NULL, method = "SMOTE")
    class(D_result) = "gen_data"

    #end of copying SMOTE function
        
    texts_SMOTE = D_result
    
    texts_out = as.matrix(texts_SMOTE$data[,1:ncol(texts)])
    class(texts_out) = 'numeric'
    classes_out = texts_SMOTE$data[,ncol(texts) + 1]
    
  } else if (balance_algo == 'ROSE') {#ROSE (Random Over-Sampling Examples)
    
    data_for_ROSE = as.data.frame(texts)
    data_for_ROSE$classes = classes
    
    data_ROSE <- ROSE(classes~., data=data_for_ROSE)
    
    texts_out = data_ROSE$data[,1:ncol(texts)]
    classes_out = data_ROSE$data$classes
    
  }
  
  gc()
  
  return(list(texts = texts_out,
                    classes = classes_out))
  
}

###################################################################################
#SVM

SVM_fun = function (texts_train, texts_test, classes_train, classes_test,
                    param = data.frame(cost = NA, gamma = NA), verbose = T){
  
  #This function trains a SVM classifier on text data and provides some diagnostics
  #It automatically finds optimal classifier settings
  #It outputs diagnostics
  
  #Determine optimal SVM parameters
  if (is.na(param$cost)){
    if (verbose) print('Determining optimal SVM parameters...')
    svm_tune <- tune(svm, train.x=texts_train,
                     train.y=classes_train,
                     kernel="radial",
                     scale = T,
                     parallel.core = 2,
                     ranges=list(cost=10,#10^(0:3),
                                 gamma=seq(20,100,10)))#3^(0:5)))
    
    if(verbose) print(svm_tune)
    #svm_tune$performances$error
    
    param$cost = svm_tune$best.parameters$cost
    param$gamma = svm_tune$best.parameters$gamma
  }
  
  #Train SVM
  if (dim(texts_train)[2] <= 2000){#relatively few dimensions )(probably LSA data set)
    
    if (verbose) print(sprintf('Training SVM classifier using radial kernel with cost %1.2f and gamma %1.2f...',
                  param$cost, param$gamma))
    
          svm_m = svm(x = texts_train,#training vectors
                      y = classes_train,#training classification
                      kernel = 'radial', probability = T,
                      scale = F,
                      cost = param$cost,
                      gamma = param$gamma)
  } else {#many dimensions (probably raw word count data set)
    
    print(sprintf('Training SVM classifier using linear kernel...'))
    
          svm_m = svm(x = texts_train,#training vectors
                      y = classes_train,#training classification
                      kernel = 'linear', probability = T,
                      scale = T)
  }
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
  
  # p = ggplot(data = data.frame(x = attr(pred, 'probabilities')[, 2],
  #                              included = as.factor(classes_test)),
  #            aes(x = x, fill = included, linetype = included)) +
  #   geom_histogram(position = 'identity', alpha = 0.5, color = 'black')
  # # extract relevant variables from the plot object to a new data frame
  # # the grouping variable is named 'group' in the plot object
  # df <- ggplot_build(p)$data[[1]][ , c("xmin", "y", "group")]
  # #get the factor levels interpretable
  # df$group[df$group == 2] = 'should be included'
  # df$group[df$group == 1] = 'should be excluded'
  # 
  # p_multihist = ggplot(data = df, aes(x = xmin, y = y, color = factor(group))) +
  #   geom_step(size = 1.5)
  # 
  # if(verbose) print(p_multihist)
  
  gc()
  
  #return all sorts of diagnostics
  return(list(pred = pred,
    testing_sample_size = length(classes_test),
              training_sample_size = length(classes_train),
              training_prop_positive_cases = mean(as.numeric(as.character(classes_train))),
              nDim = ncol(texts_train),
              #SVM_parameters = svm_tune$best.parameters,
              testing_sample_acc = sum(unlist(lapply(1:length(pred), function(x) pred[x] == classes_test[x])))/length(pred),
              prop_excl_before_FN = sum(attr(pred, 'probabilities')[classes_test == 0, 2] > 
                                          max(attr(pred, 'probabilities')[classes_test == 1, 2]))/
                sum(classes_test == 0),
              max_rej_conf_of_TP = max(attr(pred, 'probabilities')[classes_test == 1, 2]),#ideally low
              testing_confusion_matrix = table(pred,factor(classes_test))#,
              #multihist = p_multihist
  ))
}

###################################################################################
#Boosting

boost_fun = function (texts_train, texts_test, classes_train, classes_test,
                    nIter = ncol(texts_train), verbose = T){
  
  #This function trains a logitboost classification algorithm using decision stumps on text data and provides some diagnostics
  #It outputs diagnostics
  
  #Train Boosting algo
    if (verbose) print(sprintf('Training Logitboost classificatioon algorithm...'))
    boost_m = LogitBoost(xlearn = as.matrix(texts_train),#training vectors,
                         ylearn = classes_train,#training classification,
                         nIter = nIter)
    
  #performance on training sample
  #pred <- predict(boost_m, texts_train)
  #table(pred,factor(classes_train))
  #overall accuracy (training sample)
  #sum(unlist(lapply(1:length(pred), function(x) pred[x] == classes_train[x])))/length(pred)
  
  #performance on test sample
  pred <- predict(boost_m, texts_test)
  if(verbose) print('Testing sample confusion matrix:')
  if(verbose) print(table(pred,factor(classes_test)))
  #overall accuracy (testing sample)
  if(verbose) print(sprintf('Testing sample accuracy: %1.2f',
                            sum(unlist(lapply(1:length(pred), function(x) pred[x] == classes_test[x])))/length(pred)))
  
  gc()
  
  #return all sorts of diagnostics
  return(list(pred = pred,
              testing_sample_size = length(classes_test),
              training_sample_size = length(classes_train),
              training_prop_positive_cases = mean(as.numeric(as.character(classes_train))),
              nIter = nIter,
              testing_sample_acc = sum(unlist(lapply(1:length(pred), function(x) pred[x] == classes_test[x])))/length(pred),
              testing_confusion_matrix = table(pred,factor(classes_test))#,
  ))
}

###################################################################################
#RF

RF_fun = function (texts_train, texts_test, classes_train, classes_test,
                      ntree = 500, verbose = T){
  
  #This function trains a random forest classification algorithm on text data and provides some diagnostics
  #It outputs diagnostics
  
  #Train RF algo
  if (verbose) print(sprintf('Training Randoom Forest algorithm...'))
  RF_m = randomForest(x = as.matrix(texts_train),#training vectors,
                      y = classes_train,#training classification,
                      xtest = as.matrix(texts_test),
                      ytest = relevel(as.factor(classes_test), '1'),
                      ntree = ntree)
  
  #performance on test sample
  if(verbose) print('Testing sample confusion matrix:')
  if(verbose) print(table(RF_m$test$predicted,factor(classes_test)))
  #overall accuracy (testing sample)
  if(verbose) print(sprintf('Testing sample accuracy: %1.2f',
                            sum(unlist(lapply(1:length(RF_m$test$predicted), function(x) RF_m$test$predicted[x] == classes_test[x])))/length(RF_m$test$predicted)))
  
  gc()
  
  #return all sorts of diagnostics
  return(list(pred = RF_m$test$predicted,
              testing_sample_size = length(classes_test),
              training_sample_size = length(classes_train),
              training_prop_positive_cases = mean(as.numeric(as.character(classes_train))),
              testing_sample_acc = sum(unlist(lapply(1:length(RF_m$test$predicted), function(x) RF_m$test$predicted[x] == classes_test[x])))/length(RF_m$test$predicted),
              testing_confusion_matrix = table(RF_m$test$predicted,factor(classes_test))#,
  ))
}