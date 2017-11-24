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

# This script classifies article abstracts as belonging to the topic 'problem based learning' or not.
# It was written by Richard Kunert for the Humboldt University's bologna.lab led by Wolfgang Deicke.
# The bologna.lab is funded by Bundesministerium für Bildung und Forschung.
# For questions and comments, please get in touch with Richard Kunert: rikunert@gmail.com.
# Berlin, November 2017

# This script requires the files RK_fun.R and ALL.txt in the current working directory.
# This script produces some files on the spot which can also be loaded in order to save time:
# texts_vectorized.RData and diagnostics_incl_conf.RData.
# It produces two plots (not saved) and the text file ALL_classified.txt

##############################################################################################
# LOAD LIBRARIES

if(!require(smotefamily)){install.packages('smotefamily')} #Synthetic Minority Oversampling TEchnique
if(!require(FNN)){install.packages('FNN')} #if not installed we get an error
library(smotefamily)

if(!require(ggplot2)){install.packages('ggplot2')}
library(ggplot2)  # plotting

if(!require(gridExtra)){install.packages('gridExtra')}
library(gridExtra)  # combine plots

source('RK_fun.R')  # source custom functions

##############################################################################################
# LOAD DATA

abstracts <- read.delim("ALL.txt",
                       sep = '\t', row.names = NULL, header = T,
                       na.strings = c("", ""), quote = NULL)

abstracts$abstract_raw <- abstracts$abstract  # copy abstracts for later final output

abstracts$abstract <- cleaning_fun(abstracts$abstract)  # remove odd characters and footers, lemmatize texts

##############################################################################################
# TEXT VECTORIZATION

# decide whether vectorization through LSA done on the spot or just load texts.vectorized.RData.

if (T) {  # vectorization done on the spot
  texts_vectorized <- list(LSA_fun(abstracts$abstract, nDim = 10, ngramLength = 1),
                          LSA_fun(abstracts$abstract, nDim = 20, ngramLength = 1),
                          LSA_fun(abstracts$abstract, nDim = 200, ngramLength = 1),
                          LSA_fun(abstracts$abstract, nDim = 2000, ngramLength = 1),
                          LSA_fun(abstracts$abstract, nDim = 10, ngramLength = 2),
                          LSA_fun(abstracts$abstract, nDim = 20, ngramLength = 2),
                          LSA_fun(abstracts$abstract, nDim = 200, ngramLength = 2),
                          LSA_fun(abstracts$abstract, nDim = 2000, ngramLength = 2))
  
  save(texts_vectorized, file = "texts_vectorized.RData")
  
} else {  # vectorization loaded
  
  load("texts_vectorized.RData")
  
}

##############################################################################################
# CLASSIFICATION PERFORMANCE ASSESSMENT

# reduce full sample to those cases in which human raters could determine label
classified_idx <- c(which(abstracts$included == 1),
                   which(abstracts$included == 0))

if (T) {
  
  m <- matrix(NA, n_repetition * n_fold, 12)  # empty matrix
  colnames(m) <- c('rep', 'fold', seq(1, 0.55, -0.05))
  
  coverage <- m  # how many texts classified
  confidence_0 <- m  # classification performance for exclusion
  confidence_1 <- m  # classification performance for inclusion
  
} else {
  
  load('diagnostics_incl_conf.RData')
  
}

# performance assessment parameters
n_fold <- 7  # how many folds in k-fold cross-validation?
n_repetition <- 100  # how many repetitions of a full n_fold cross-validation run?
start_rep <- 1  # should be 1 except if incomplete diagnostics_incl_conf.RData was loaded
counter1 <- (start_rep * n_fold) - n_fold  # starting row in assessment variables

for (repetition in start_rep:n_repetition) {  # for each repetition of complete cross-validation runs
  
  testing_sample_idx <- list()
  training_sample_idx <- list()
  
  for (f in 1:n_fold) {  # for each fold during k-fold cross-validation
    
    counter1 <- counter1 + 1
    
    print(sprintf('Repetition: %d',repetition))
    print(sprintf('Fold: %d', f))
    
    #determine the testing and training subsets  
    if (f < n_fold) {  # if not last fold
      
      testing_sample_idx[[f]] <- sample(classified_idx[!(classified_idx %in% unlist(testing_sample_idx))],
                                       floor(length(classified_idx)/n_fold)) 
      
    } else {  # last fold
      
      testing_sample_idx[[f]] <- classified_idx[!(classified_idx %in% unlist(testing_sample_idx))] #remaining idx 
      
    }
    
    # training subset is all those idx which are not testing subset
    training_sample_idx[[f]] <- classified_idx[!(classified_idx %in% testing_sample_idx[[f]])]
    
    # initialise matrix holding classification output
    diagnostics_summary <- matrix(NA,
                                 length(texts_vectorized) * 2 + 1,
                                 length(testing_sample_idx[[f]]))
    
    diagnostics_summary[1,] <- abstracts$included[testing_sample_idx[[f]]]  # row one is true classification (manual)
    
    counter2 <- 1
    
    for (v in 1:length(texts_vectorized)) {  # for each kind of vectorization (see LSA parameters above)
      
      counter2 <- counter2 +1
      
      # balance the training data set via SMOTE
      texts_SMOTE <- SMOTE(as.data.frame(texts_vectorized[[v]][training_sample_idx[[f]],]),
                          abstracts$included[training_sample_idx[[f]]],
                          K = 5, dup_size = 0)
      
      # restructure balanced data for classifiers
      data_training <- list(texts = as.matrix(texts_SMOTE$data[,1:ncol(texts_vectorized[[v]])]),
                           classes = texts_SMOTE$data[,ncol(texts_vectorized[[v]]) + 1])
      class(data_training$texts) <- 'numeric'
      
      # Support Vector Machine classifier
      diagnostics <- SVM_fun(texts_train = data_training$texts,
                            texts_test = texts_vectorized[[v]][testing_sample_idx[[f]],],
                            classes_train = as.factor(data_training$classes),
                            classes_test = abstracts$included[testing_sample_idx[[f]]],
                            param = data.frame(cost = 10, gamma = 1),
                            verbose = F)
      
      # include SVM classification in matrix holding classification output
      diagnostics_summary[counter2, ] <- as.integer(as.character(diagnostics$pred))
      
      counter2 <- counter2 + 1
      
      # BOOSTING classifier
      diagnostics <- boost_fun(texts_train = data_training$texts,
                              texts_test = texts_vectorized[[v]][testing_sample_idx[[f]],], 
                              classes_train = data_training$classes,
                              classes_test = abstracts$included[testing_sample_idx[[f]]],
                              nIter = 2000,
                              verbose = F)
      
      # include Boosting classification in matrix holding classification output
      diagnostics_summary[counter2, ] <- as.integer(as.character(diagnostics$pred))
      
    }
    
    # combine votes from different vectorizations and classifiers
    voting_mean <- colMeans(diagnostics_summary[2:length(texts_vectorized) + 1,], na.rm = T)
    
    # note position in folding-repetition cycle
    coverage[counter1, 1] <- precision_0[counter1, 1] <- precision_1[counter1, 1] <- confidence_0[counter1, 1] <- confidence_1[counter1, 1] <- repetition
    coverage[counter1, 2] <- precision_0[counter1, 2] <- precision_1[counter1, 2] <- confidence_0[counter1, 2] <- confidence_1[counter1, 2] <- f
    
    counter3 <- 0
    
    for (a in seq(1, 0.55, -0.05)) {#for each model agreement rate
    
      counter3 <- counter3 + 1

      # note classifier performance
      coverage[counter1, counter3 + 2] <- (sum(voting_mean <= abs(a-1), na.rm = T) +
                                            sum(voting_mean >= a, na.rm = T)) / 
        sum(!is.na(voting_mean))
      
      confidence_0[counter1, counter3 + 2] <- sum(diagnostics_summary[1,voting_mean <= round(abs(a-1), digits = 2) & 
                                                                       voting_mean > round(abs(a-1), digits = 2) - 0.14] == 0, na.rm = T) /
        sum(voting_mean <= round(abs(a-1), digits = 2) & 
              voting_mean > round(abs(a-1), digits = 2)-0.14, na.rm = T)
      
      confidence_1[counter1, counter3 + 2] <- sum(diagnostics_summary[1, voting_mean >= a & 
                                                                       voting_mean < a + 0.14] == 1, na.rm = T) /
        sum(voting_mean >= a & 
              voting_mean < a + 0.14, na.rm = T)

    }
  }
  
  # save
  print('save what we have so far...')
  save(coverage, precision_0, precision_1, confidence_0, confidence_1, file = "diagnostics_incl_conf.RData")
  
}

##############################################################################################
# CLASSIFICATION PERFORMANCE VISUALISATION

# custom function for violin plot of results (folds combined)
viol_plot <- function(dat_viol, p_title, p_ylable, p_ylim = c(0.5, 1)) {
  # Plot data as violin plot.
  #
  # Args:
  #   dat_viol: data.frame to be plotted, includes variables x and y
  #   p_title: plot's title
  #   p_ylable: plot's y-axis label
  #   p_ylim: plot's y-axis limits
  #
  # Returns:
  #   violin plot (ggplot2 object)
  
  p <- ggplot(data = dat_viol, aes(x = factor(x), y = y)) +
    geom_violin(aes(fill = x)) +
    ylim(p_ylim) +
    theme(legend.position = "none") +
    labs(x = 'Classifier agreement', y = p_ylable)+
    ggtitle(p_title)
  
  return(p)
}

# plot exclusion confidence
x <- aggregate(confidence_0[, 3:12], list(confidence_0[,1]), mean)  # take mean across folds
dat_viol <- data.frame(y = unlist(x[2:dim(x)[2]]),
                      x = rep(colnames(x[2:length(colnames(x))]), each = nrow(x)))
dat_viol$x <- factor(dat_viol$x, levels = colnames(x[2:length(colnames(x))]))
p1 <- viol_plot(dat_viol = dat_viol,
               p_title = 'Article exclusion via multi-classifier approach',
               p_ylable = 'Exclusion accuracy',
               p_ylim = c(0.5, 1))

# plot inclusion confidence
x <- aggregate(confidence_1[, 6:12], list(confidence_1[,1]), mean)  # inclusions only from 85% agreement onwards
dat_viol <- data.frame(y = unlist(x[2:dim(x)[2]]),
                      x = rep(colnames(x[2:length(colnames(x))]), each = nrow(x)))
dat_viol$x <- factor(dat_viol$x, levels = colnames(x[2:length(colnames(x))]))
p2 <- viol_plot(dat_viol = dat_viol,
               p_title = 'Article inclusion via multi-classifier approach',
               p_ylable = 'Inclusion accuracy')

# plot coverage
x <- aggregate(coverage[, 3:12], list(coverage[,1]), mean)
dat_viol <- data.frame(y = unlist(x[2:dim(x)[2]]),
                      x = rep(colnames(x[2:length(colnames(x))]), each = nrow(x)))
dat_viol$x <- factor(dat_viol$x, levels = colnames(x[2:length(colnames(x))]))
p3 <- viol_plot(dat_viol = dat_viol,
               p_title = 'Article coverage via multi-classifier approach',
               p_ylable = 'Coverage')
p3

grid.arrange(grobs = list(p1,p2, p3), ncol = 3)

##############################################################################################
# FINAL CLASSIFICATION

classification_summary <- matrix(NA,  # place holder value
                             length(texts_vectorized) * 2,  # vectorizations * classifiers
                             dim(texts_vectorized[[1]])[1])  # each abstract
counter2 <- 0
for (v in 1:length(texts_vectorized)) {  # for each kind of vectorization
  counter2 <- counter2 + 1
  
  # balance training data  
  texts_SMOTE <- SMOTE(as.data.frame(texts_vectorized[[v]][classified_idx,]),
                      abstracts$included[classified_idx],
                      K = 5, dup_size = 0)
  
  # restructure balanced data for classifiers
  data_training <- list(texts = as.matrix(texts_SMOTE$data[,1:ncol(texts_vectorized[[v]])]),
                       classes = texts_SMOTE$data[,ncol(texts_vectorized[[v]]) + 1])
  class(data_training$texts) <- 'numeric'
  
  diagnostics <- SVM_fun(texts_train = data_training$texts,
                        texts_test = texts_vectorized[[v]],
                        classes_train = as.factor(data_training$classes),
                        classes_test = NA,
                        param = data.frame(cost = 10, gamma = 1),#irrelevant if dtm dim > 2000
                        verbose = F)
  
  classification_summary[counter2, ] <- as.integer(as.character(diagnostics$pred))
  
  counter2 <- counter2 + 1
  
  diagnostics <- boost_fun(texts_train = data_training$texts,
                          texts_test = texts_vectorized[[v]], 
                          classes_train = data_training$classes,
                          classes_test = NA,
                          nIter = 2000,
                          verbose = F)
  
  classification_summary[counter2, ] <- as.integer(as.character(diagnostics$pred))
  
}
# combine votes from different vectorizations and classifiers
voting_mean <- colMeans(classification_summary, na.rm = T)

# prepare final vote
voting_mean_round <- round(voting_mean)  # simple majority vote
voting_mean_round[voting_mean == 0.5] <- 0  # when the classifiers cannot agree at all, go for base rate (exclusion more likely)

# confidence
confidence <- rep(NA, length(voting_mean))
for (a in seq(1, 0.55, -0.05)) {  # for each model agreement rate
  
  # exclusion confidence
  confidence[voting_mean <= round(abs(a-1), digits = 2) &
               voting_mean > round(abs(a-1), digits = 2)-0.05] <-
    mean(confidence_0[,colnames(confidence_0) == a], na.rm = T)
  
  # inclusion confidence
  if (a > 0.85) {  # above 85% agreement, too few cases for meaningful confidence, so just revert to 85% agreement value
    
    confidence[voting_mean >= a & voting_mean < a + 0.05] <- mean(confidence_1[,colnames(confidence_1) == 0.85], na.rm = T)
    
  } else {
    
    confidence[voting_mean >= a & voting_mean < a + 0.05] <- mean(confidence_1[,colnames(confidence_1) == a], na.rm = T)
    
  }
  
}

confidence[voting_mean < 0.55 & voting_mean > 0.45] <- 0  # when classifiers cannot agree at all, no confidence at all

# finalise final vote
classification_final <- voting_mean_round
classification_final[confidence < 0.98] <- 0.5  # everything with low confidence should be looked at by hand (vote = 0.5)

# actual coverage of final classification
coverage <- rep(NA, length(seq(1, 0, -0.05)))
counter <- 0
for (a in seq(1, 0, -0.05)) {  # for each model agreement rate
  counter <- counter + 1
  
  coverage[counter] <- sum(voting_mean[is.na(abstracts$included)] >= a &
                            voting_mean[is.na(abstracts$included)] < a + 0.04999999)
  
}

# plot coverage (histogram)
p <- ggplot(data = data.frame(x = seq(1, 0, -0.05),
                             y = coverage),
           aes(x = x, y = y)) +
  geom_histogram(stat = 'identity') +
  scale_x_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1),
                   labels = c('1', '0.75', 'no agreement', '0.75', '1')) +
  labs(x = 'Exclusion agreement                                                                                                    Inclusion agreement',
       y = 'Count')+
  ggtitle('Coverage')
p

##############################################################################################
# SAVE FINAL CLASSIFICATION

# put everything together
abstracts_out <- data.frame(
  classification_final = classification_final,
  classification_classifier = voting_mean_round,
  confidence = confidence,
  classification_manual = abstracts$included,
  title_classification = abstracts$classification_title,
  title_raw = abstracts$title,
  abstract_classification = abstracts$abstract,
  abstract_raw = abstracts$abstract_raw)

# save to disk
write.table(abstracts_out, file = "ALL_classified.txt", sep = "\t", dec = ",")  # note German convention: tab delimited with comma as decimal symbol