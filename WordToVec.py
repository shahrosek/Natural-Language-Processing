import nltk
import re
import gensim
import gensim.models
import numpy as np
from gensim import parsing
import string

class wordTovec(object):
    def __init__(self, dataset, stopwords):
        self.dataset = dataset
        self.stopwords = stopwords
        self.preProcessedData = []
        self.cachedStopWords = []

    def loadStopwords(self):
        with open(self.stopwords, "r") as f:
            for word in f:
                word = word.split('\n')
                self.cachedStopWords.append(word[0])
        f.close()

    def processFile(self):
        with open(self.dataset, "r", encoding='utf-8', errors='ignore') as file:
            for line in file:
                line = line.split('\n')
                self.preProcessedData.append(str(self.preProcessData(line)).split())
        file.close()
        return self.preProcessedData

    def findSimilarity(self, model, words):
        for word in words:
            print("Most Similar words for %s are: "%word.capitalize())
            for text in model.wv.most_similar(word):
                print(text)

    def preProcessData(self, text):
        data = str(text).lower()
        data = re.sub('[^\w\s]', ' ', data)
        data = gensim.corpora.textcorpus.strip_multiple_whitespaces(data)
        newWords = [word for word in data.split() if word not in self.cachedStopWords]
        newWords = gensim.corpora.textcorpus.remove_short(newWords, minsize=3)
        data = " ".join(newWords)
        data = gensim.parsing.preprocessing.strip_punctuation2(str(data))
        data = gensim.corpora.textcorpus.strip_multiple_whitespaces(data)
        return data

if __name__ == "__main__":
    vectorObject = wordTovec(dataset = "Question1.txt", stopwords = "stoplist.txt")
    vectorObject.loadStopwords()
    preProcessedData = vectorObject.processFile()
    words = ['clean', 'unclean', 'amazed', 'friendly']
    model = gensim.models.Word2Vec(preProcessedData, size = 300, window = 5, min_count = 3, workers = 5)
    vectorObject.findSimilarity(model, words)

"""
Most Similar words for Clean are:
('spotless', 0.6597861051559448)
('immaculate', 0.5840664505958557)
('nice', 0.4794475734233856)
('spacious', 0.4718148708343506)
('appointed', 0.45556050539016724)
('amenities', 0.4404897689819336)
('beds', 0.4402987062931061)
('smallish', 0.43827879428863525)
('cleaned', 0.4249002933502197)
('amazingly', 0.4237271249294281)
Most Similar words for Unclean are:
('grubby', 0.7753480672836304)
('dusty', 0.7686685919761658)
('grimy', 0.7657902240753174)
('threadbare', 0.7597383260726929)
('dingy', 0.752859890460968)
('damp', 0.7518466711044312)
('moldy', 0.7361340522766113)
('soiled', 0.7335741519927979)
('frayed', 0.7296897172927856)
('stained', 0.7227389812469482)
Most Similar words for Amazed are:
('unbelievable', 0.5424623489379883)
('impressed', 0.5392130613327026)
('thrilled', 0.5363041162490845)
('shocked', 0.5224851965904236)
('delighted', 0.5127214193344116)
('remarkable', 0.5035429000854492)
('surprising', 0.4901426136493683)
('incredible', 0.48136234283447266)
('surprised', 0.47276172041893005)
('suprised', 0.45328670740127563)
Most Similar words for Friendly are:
('freindly', 0.7455536127090454)
('courteous', 0.6897550821304321)
('polite', 0.6792160868644714)
('gracious', 0.6236710548400879)
('curteous', 0.619521975517273)
('cordial', 0.6194512248039246)
('pleasant', 0.6164703965187073)
('friendliest', 0.6057222485542297)
('attentive', 0.583128809928894)
('pleasent', 0.5698697566986084)
"""
