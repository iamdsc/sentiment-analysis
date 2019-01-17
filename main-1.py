# Using grid search
import pandas as pd
import re
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def preprocessor(text):
    """ Removing punctuation marks except emoticons """
    text=re.sub('<[^>]*>','',text) # Removing HTML markup
    emoticons=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    # Removing non-word characters and convert text to lowercase
    text=(re.sub('[\W]+',' ', text.lower())+' '.join(emoticons).replace('-',''))
    return text

def tokenizer(text):
    """" Splitting at white spaces """
    return text.split()

def tokenizer_porter(text):
    """ Applying Porter Stemming Algorithm """
    porter=PorterStemmer()
    return [porter.stem(word) for word in text.split()]

df=pd.read_csv('movie_data.csv', encoding='utf-8')
#print(preprocessor(df.loc[0,'review'][-50:]))

# Applying preprocessing to movie reviews in dataframe
df['review']=df['review'].apply(preprocessor)
#print(df.head())

# Dividing the DataFrame of cleaned text documents into train and test
X_train=df.loc[:25000,'review'].values
y_train=df.loc[:25000,'sentiment'].values
X_test=df.loc[25000:,'review'].values
y_test=df.loc[25000:,'sentiment'].values

tfidf=TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
stop=stopwords.words('english')

# Using GridSearchCV object to find optimal set of parameters using 5-fold stratified cross-validation
param_grid=[{'vect__ngram_range':[(1,1)],'vect__stop_words':[stop, None],'vect__tokenizer':[tokenizer,tokenizer_porter],'clf__penalty':['l1','l2'],'clf__C':[1.0,10.0,100.0]},
                    {'vect__ngram_range':[(1,1)],'vect__stop_words':[stop, None],'vect__tokenizer':[tokenizer,tokenizer_porter],'vect__use_idf':[False],'vect__norm':[None],'clf__penalty':['l1','l2'],'clf__C':[1.0,10.0,100.0]}]

lr_tfidf=Pipeline([('vect',tfidf), ('clf',LogisticRegression(random_state=0))])
gs_lr_tfidf=GridSearchCV(lr_tfidf,param_grid,scoring='accuracy',cv=5,verbose=1,n_jobs=1)
gs_lr_tfidf.fit(X_train,y_train)

print('Best parameter set: %s' % gs_lr_tfidf.best_params_)

# Average 5 fold cross-validation accuracy scores on training set 
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
# Classification accuracy on test dataset
print('Test Accuracy: %.3f' % clf.score(X_test,y_test))
