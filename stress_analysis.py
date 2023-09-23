import pandas as pd
from textblob import TextBlob
import warnings
import pickle
warnings.filterwarnings("ignore")


df1=pd.read_csv("dreaddit-train.csv")
df2=pd.read_csv("dreaddit-test.csv")
# print(df1.shape)
# print(df2.shape)

# print(df2.sample())
df=df1._append(df2) 
# print(df.shape)
# print(df.head())
# print(df.isnull().sum())

# print(TextBlob("the best").polarity)
# print(TextBlob("the best").sentiment)
def detect_sentiment(text):
    return TextBlob(text).sentiment.polarity

df3=df[["text"]]
df3["sentiment"]=df3["text"].apply(detect_sentiment)
# print(df3.head())
# print(df3.sentiment.value_counts())

import nltk
import re
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string

stopwords = set(stopwords.words("english"))
# print(stopwords)
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopwords]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

df3["text"] = df3["text"].apply(clean)
# print(df3['text'].head())

df3["label"]=df["label"].map({0: "No Stress", 1: "Stress"})
df3=df3[["text", "label"]]
# print(df3.head())

df3["sentiment"]=df3["text"].apply(detect_sentiment)
# print(df3.head())

x=df3.text
y=df3.label

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


vect=CountVectorizer(stop_words="english")
x=vect.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
mb=MultinomialNB()

mb.fit(x_train,y_train)
# model=mb.fit(x_train,y_train).predict(x_test)
# print(accuracy_score(model,y_test))

# user="I feel like i need some help"
# df3=vect.transform([user]).toarray()
# output=mb.predict(df3)
# print(output)

pickle.dump(mb,open('model.pkl','wb'))
pickle.dump(vect,open('vectorizer.pkl','wb'))


