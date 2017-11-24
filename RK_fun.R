# Copyright (c) 2017 Humboldt-Universität zu Berlin
#   
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#   
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

##############################################################################################

# This script includes helper functions for classification_master_RK.R
# It was written by Richard Kunert for the Humboldt University's bologna.lab led by Wolfgang Deicke.
# The bologna.lab is funded by Bundesministerium für Bildung und Forschung.
# For questions and comments, please get in touch with Richard Kunert: rikunert@gmail.com.
# Berlin, November 2017

#########################################################################
# LOAD LIBRARIES

if(!require(textstem)){install.packages('textstem')}  # lemmatization (rather than just stemming)
library(textstem)

if(!require(lsa)){install.packages('lsa')}  # latent semantic analysis
library(lsa)

if(!require(tm)){install.packages('tm')}  # text mining
library(tm)

if(!require(data.table)){install.packages('data.table')}  # rbindlist function
library(data.table)

if(!require(e1071)){install.packages('e1071')}  # SVM
library(e1071)

if(!require(caTools)){install.packages('caTools')}  # Boosting
library(caTools)

#########################################################################
# DATA CLEANING

cleaning_fun <- function(texts) { 
  # Clean and lemmatize English language texts (academic abstracts)
  #
  # Args:
  #   text: A list of texts.
  #
  # Returns:
  #   A list of cleaned and lemmatized texts
  
  # Remove non-ASCII characters
  texts <- iconv(texts, "latin1", "ASCII", sub = " ")
  
  # Remove non-alphanumeric characters
  texts <- gsub("[^[:alnum:] ]", "", texts)
  
  # Remove footers
  texts <- removeWords(texts,
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
  
  # Lemmatize the corpus
  texts <- lemmatize_strings(texts)
  
  return(texts)
  
}

#########################################################################
# LATENT SEMANTIC ANALYSIS

LSA_fun <- function(texts, nDim = 100, ngramLength = 1, verbose = T) { 
  # Turn a list of input strings (texts) into an nDim dimensional semantic space via LSA
  #
  # Args:
  #   text: list of texts
  #   nDim: the number of dimensions into which texts and words are abstracted during LSA
  #   ngramLength: basic unit for LSA, default (1) is words
  #   verbose: If TRUE, print start of code
  #
  # Returns:
  #   nrow(texts) x nDim matrix representing each string as a vector in semantic space
  
  if (verbose) 
    print(sprintf('Creating LSA space with nDim %d and ngramLength %d...', nDim, ngramLength))
  
  corp <- Corpus(VectorSource(texts))  # create text corpus
  
  ngramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = ngramLength, max = ngramLength))
  
  tdm <- TermDocumentMatrix(corp, control = list(weighting = weightTfIdf, removePunctuation = TRUE, 
                                                removeNumbers = TRUE, stopwords = TRUE,
                                                tolower = T,
                                                minWordLength = 3,
                                                tokenize=ngramTokenizer))
  
  lsa_space <- lsa(tdm, dims = nDim)  # create LSA space
  
  return(lsa_space$dk)
}

###################################################################################
# SUPPORT VECTOR MACHINE

SVM_fun <- function(texts_train, texts_test, classes_train, classes_test = NA,
                    param = data.frame(cost = NA, gamma = NA), verbose = T) {
  # Train a Support Vector Machine classifier on text data
  #
  # Args:
  #   texts_train: list of texts represented as matrix (texts x semantic dimension) (training set)
  #   texts_test: list of texts represented as matrix (texts x semantic dimension) (test set)
  #   classes_train: list of class labels (training set) 
  #   classes_test: list of class labels (test set)
  #   param: parameters sent to svm() function of e1071 library, default is finding these parameters automatically
  #   verbose: If TRUE, print progress messages
  #
  # Returns:
  #   predicted classes in test set and settings

  # Determine optimal SVM parameters in case no parameters given
  if (is.na(param$cost)) {
    if (verbose) print('Determining optimal SVM parameters...')
    svm_tune <- tune(svm, train.x = texts_train,
                     train.y = classes_train,
                     kernel = "radial",
                     scale = T,
                     parallel.core = 2,
                     ranges = list(cost = 10,#
                                 gamma = seq(1, 100, 10)))#
    
    if (verbose) 
      print(svm_tune)
    
    param$cost <- svm_tune$best.parameters$cost
    param$gamma <- svm_tune$best.parameters$gamma
  }
  
  # Train SVM
  if (verbose) 
    print(sprintf('Training SVM classifier using radial kernel with cost %1.2f and gamma %1.2f...',
                  param$cost, param$gamma))
  
  svm_m <- svm(x = texts_train,  # training vectors
              y = classes_train,  # training classification
              kernel = 'radial', probability = T,
              scale = F,
              cost = param$cost,
              gamma = param$gamma)
  
  # performance on test sample
  pred <- predict(svm_m, texts_test, probability = T)
  
  if (verbose) 
    print('Testing sample confusion matrix:')
  
  if (verbose && !is.na(classes_test)) 
    print(table(pred,factor(classes_test)))
  
  return(list(pred = pred,
              testing_sample_size = length(classes_test),
              training_sample_size = length(classes_train),
              nDim = ncol(texts_train)))
}

###################################################################################
# BOOSTING CLASSIFIER

boost_fun <- function(texts_train, texts_test, classes_train, classes_test = NA,
                      nIter = ncol(texts_train), verbose = T) {
  # Train a logitboost classification algorithm using decision stumps
  #
  # Args:
  #   texts_train: list of texts represented as matrix (texts x semantic dimension) (training set)
  #   texts_test: list of texts represented as matrix (texts x semantic dimension) (test set)
  #   classes_train: list of class labels (training set) 
  #   classes_test: list of class labels (test set)
  #   nIter: parameter sent to LogitBoost() function of caTools library, default is number of dimensions in semantic space
  #   verbose: If TRUE, print progress messages
  #
  # Returns:
  #   predicted classes in test set and settings
  
  # Train Boosting algo
  if (verbose) 
    print(sprintf('Training Logitboost classificatioon algorithm...'))
  
  boost_m <- LogitBoost(xlearn = as.matrix(texts_train),#training vectors,
                       ylearn = classes_train,#training classification,
                       nIter = nIter)
  
  # performance on test sample
  pred <- predict(boost_m, texts_test)
  
  if (verbose) 
    print('Testing sample confusion matrix:')
  
  if (verbose && !is.na(classes_test)) 
    print(table(pred,factor(classes_test)))
  
  return(list(pred = pred,
              testing_sample_size = length(classes_test),
              training_sample_size = length(classes_train),
              nIter = nIter))
}