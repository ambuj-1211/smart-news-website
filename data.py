import spacy
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import Dataset

class PreProcessor:

    def __init__(self, lang = 'english'):
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = set(stopwords.words(lang))
        self.vectorizer = TfidfVectorizer()

    def remove_special_chars(self, doc):

        doc = " ".join(re.findall(r'[a-zA-Z0-9]+', doc))
        return doc.lower()

    def lemmatize(self, doc):
        doc = self.nlp(doc)
        tokenized = [token.lemma_ for token in doc]
        return tokenized

    def remove_stop_words(self, tokenized_doc):

        tokenized_doc_no_stopwords = []

        for word in tokenized_doc:
            if word not in self.stop_words:
                tokenized_doc_no_stopwords.append(word)

        return tokenized_doc_no_stopwords

    def forward(self, doc):
        doc = self.remove_special_chars(doc)
        doc = self.lemmatize(doc)
        doc = self.remove_stop_words(doc)

        return " ".join(doc)


class dataset(Dataset):

    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.length = self.x.shape[0]
 
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]  
    
    def __len__(self):
        return self.length
