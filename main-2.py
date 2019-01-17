# Using out-of-core learning
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import pyprind


stop=stopwords.words('english')
def tokenizer(text):
    """ Cleaning unprocessed text data """
    text=re.sub('<[^>]*>','',text)
    emoticons=re.findall('(?::|;|=) (?:-)?(?:\)|\(|D|P)',text.lower())
    text=re.sub('[\W]+',' ',text.lower())+' '.join(emoticons).replace('-','')
    tokenized=[w for w in text.split() if w not in stop]
    return tokenized

def stream_docs(path):
    """ Reads in and returns one doc at a time """
    with open(path,'r',encoding='utf-8') as csv:
        next(csv) # skip header
        for line in csv:
            text, label=line[:-3], int(line[-2])
            yield text, label

def get_minibatch(doc_stream, size):
    """ To get a batch of documents """
    docs, y=[], []
    try:
        for _ in range(size):
            text, label=next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

vect=HashingVectorizer(decode_error='ignore',n_features=2**21, preprocessor=None,tokenizer=tokenizer)
clf=SGDClassifier(loss='log',random_state=1, n_iter=1)
doc_stream=stream_docs(path='movie_data.csv')

# Performing out-of-core learning
pbar=pyprind.ProgBar(45)
classes=np.array([0,1])
for _ in range(45):
    X_train, y_train=get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train=vect.transform(X_train)
    clf.partial_fit(X_train,y_train,classes=classes)
    pbar.update()

# evaluating performance of model
X_test,y_test=get_minibatch(doc_stream,size=5000)
X_test=vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))

# Updating the model with last 5000 docs
clf.partial_fit(X_test, y_test)
