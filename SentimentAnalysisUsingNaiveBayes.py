import pandas as p
import nltk
import numpy as np
from nltk.stem import porter
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

class NaiveBayesClassifier(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.stemmer = porter.PorterStemmer()
        self.data = []
        self.label = []

    def processFile(self):
        dataframe = p.read_table(self.dataset, sep='\t', header=None, names=['id', 'sentiment', 'review'])
        dataframe['review'] = dataframe.review.str.replace('<[^<]+?>', '')
        dataframe['review'] = dataframe.review.map(lambda x: x.lower())
        dataframe['review'] = dataframe.review.str.replace('[^\w\s]', ' ')
        dataframe['review'] = dataframe['review'].apply(nltk.word_tokenize)
        dataframe['review'] = dataframe['review'].apply(lambda x: [self.stemmer.stem(y) for y in x])
        dataframe['review'] = dataframe['review'].apply(lambda x: ' '.join(x))
        self.data = dataframe['review'][1:]
        self.label = dataframe['sentiment'][1:]

    def initCountsBOWs(self):
        vectorizer = CountVectorizer()
        countsBOWs = vectorizer.fit_transform(self.data)
        return countsBOWs

    def initCountsTFIDF(self, counts):
        transformer = TfidfTransformer().fit(counts)
        countsTFIDF = transformer.transform(counts)
        return countsTFIDF

    def trainModel(self, counts):
        xTrain, xTest, yTrain, yTest = train_test_split(counts, self.label, test_size=0.25, random_state=54)
        model = MultinomialNB().fit(xTrain, yTrain)
        predict = model.predict(xTest)
        return yTest, predict

    def getAccuracy(self, yTrue, yPred):
        print(np.mean(yPred == yTrue))
        stats = confusion_matrix(yTrue, yPred)
        print("Correctly Classified Positive Reviews: %d"%stats[0][0])
        print("Correctly Classified Negative Reviews: %d"%stats[1][1])
        print("Wrongly Classified Positive Reviews: %d"%stats[0][1])
        print("Wrongly Classified Negative Reviews: %d"%stats[1][0])


if __name__ == "__main__":
    object = NaiveBayesClassifier("Question2 Dataset.tsv")
    object.processFile()

    rawCounts = object.initCountsBOWs()
    countTFIDF = object.initCountsTFIDF(rawCounts)

    testRC, predictRC = object.trainModel(rawCounts)
    testTFIDF, predictTFIDF = object.trainModel(countTFIDF)

    print("Accuracy for Naive Bayes Classification using Bag of Words Raw Counts Model: ")
    object.getAccuracy(testRC, predictRC)

    """
    0.84416
    True Positives = 2739
    False Positives = 367
    True Negatives = 2537
    False Negatives = 607
    """

    print("Accuracy for Naive Bayes Classification using Bag of Words TF-IDF Model: ")
    object.getAccuracy(testTFIDF, predictTFIDF)

    """
    0.85744 True
    Positives = 2758
    False Positives = 348
    True Negatives = 2601
    False Negatives = 543
    """
