
# One of the consequences of being editable by anyone is that some people vandalize pages. 
# This can take the form of removing content, adding promotional or inappropriate content, 
# or more subtle shifts that change the meaning of the article. With this many articles and 
# edits per day it is difficult for humans to detect all instances of vandalism and revert (undo) 
# them. As a result, Wikipedia uses bots - computer programs that automatically revert edits that 
# look like vandalism. In this assignment we will attempt to develop a vandalism detector that uses 
# machine learning to distinguish between a valid edit and vandalism.

wiki = read.csv("wiki.csv", stringsAsFactors=FALSE)
wiki$Vandal = as.factor(wiki$Vandal)
table(wiki$Vandal)
library(tm)
corpusAdded = VCorpus(VectorSource(wiki$Added))

corpusAdded = tm_map(corpusAdded, removeWords, stopwords("english"))

corpusAdded = tm_map(corpusAdded, stemDocument)

dtmAdded = DocumentTermMatrix(corpusAdded)
sparseAdded = removeSparseTerms(dtmAdded, 0.997)


wordsAdded = as.data.frame(as.matrix(sparseAdded))

colnames(wordsAdded) = paste("A", colnames(wordsAdded))


corpusRemoved = VCorpus(VectorSource(wiki$Removed))

corpusRemoved = tm_map(corpusRemoved, removeWords, stopwords("english"))

corpusRemoved = tm_map(corpusRemoved, stemDocument)

dtmRemoved = DocumentTermMatrix(corpusRemoved)

sparseRemoved = removeSparseTerms(dtmRemoved, 0.997)

wordsRemoved = as.data.frame(as.matrix(sparseRemoved))

colnames(wordsRemoved) = paste("R", colnames(wordsRemoved))
wikiWords = cbind(wordsAdded, wordsRemoved)
wikiWords$Vandal = wiki$Vandal

library(caTools)

set.seed(123)

spl = sample.split(wikiWords$Vandal, SplitRatio = 0.7)

wikiTrain = subset(wikiWords, spl==TRUE)

wikiTest = subset(wikiWords, spl==FALSE)
table(wikiTest$Vandal)
library(ROCR)
library(rpart)
wikiCART = rpart(Vandal ~ ., data=wikiTrain, method="class")
testPredictCART = predict(wikiCART, newdata=wikiTest, type="class")
table(wikiTest$Vandal, testPredictCART)
library(rpart.plot)
prp(wikiCART)

#=================================================================
  wikiWords2 = wikiWords
wikiWords2$HTTP = ifelse(grepl("http",wiki$Added,fixed=TRUE), 1, 0)
table(wikiWords2$HTTP)

wikiTrain2 = subset(wikiWords2, spl==TRUE)
wikiTest2 = subset(wikiWords2, spl==FALSE)
wikiCART2 = rpart(Vandal ~ ., data=wikiTrain2, method="class")
testPredictCART2 = predict(wikiCART2, newdata=wikiTest2, type="class")
table(wikiTest2$Vandal, testPredictCART2)

wikiWords2$NumWordsAdded = rowSums(as.matrix(dtmAdded))
wikiWords2$NumWordsRemoved = rowSums(as.matrix(dtmRemoved))
mean(wikiWords2$NumWordsAdded)
#======================================================================
  wikiTrain3 = subset(wikiWords2, spl==TRUE)
wikiTest3 = subset(wikiWords2, spl==FALSE)

wikiCART3 = rpart(Vandal ~ ., data=wikiTrain3, method="class")
testPredictCART3 = predict(wikiCART3, newdata=wikiTest3, type="class")
table(wikiTest3$Vandal, testPredictCART3)
#=======================================================================
wikiWords3 = wikiWords2
wikiWords3$Minor = wiki$Minor
wikiWords3$Loggedin = wiki$Loggedin


wikiTrain4 = subset(wikiWords3, spl==TRUE)
wikiTest4 = subset(wikiWords3, spl==FALSE)
wikiCART4 = rpart(Vandal ~ ., data=wikiTrain4, method="class")
predictTestCART4 = predict(wikiCART4, newdata=wikiTest4, type="class")
table(wikiTest4$Vandal, predictTestCART4)
prp(wikiCART4)

# By adding new independent variables, we were able to significantly improve our 
# accuracy without making the model more complicated!
