from flask import Flask, request, jsonify, render_template
import re
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

app = Flask(__name__)

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stopwords_list = stopwords.words('english')
english_stopset = set(stopwords.words('english')).union({
    "things", "that's", "something", "take", "don't", "may", "want", "you're",
    "set", "might", "says", "including", "lot", "much", "said", "know", "good",
    "step", "often", "going", "thing", "things", "think", "back", "actually",
    "better", "look", "find", "right", "example", "verb", "verbs"
})

docs = [
    '''The modern pencil is a writing instrument that uses a core of solid pigment encased in wood. 
    Pencils are widely used in writing, drawing, and shading, making them a versatile tool. 
    The pencil has evolved from wooden versions to mechanical pencils.''',
    
    '''Olive oil is a staple in Mediterranean cuisine and has been used for thousands of years. 
    Extracted from the fruit of the olive tree, olive oil varies in flavor and quality depending on the type of olive. 
    Olive oil is known for its health benefits, including reducing inflammation and improving heart health.''',

    '''Atlantis, a mythical island first mentioned by Plato, has captured imaginations for centuries. 
    In Plato's dialogues, Atlantis was described as a powerful civilization. 
    Despite extensive exploration, no evidence has been found to prove the existence of Atlantis.'''
]

title = ['Pencils', 'Olive Oil', 'Atlantis']

lemmer = WordNetLemmatizer()
document_clean = []

for d in docs:
    document_test = re.sub(r'[^\x00-\x7F]+', ' ', d)
    document_test = re.sub(r'@\w+', '', document_test)
    document_test = document_test.lower()
    document_test = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', document_test)
    document_test = re.sub(r'[0-9]', '', document_test)
    document_test = re.sub(r'\s{2,}', ' ', document_test)
    document_clean.append(document_test)

new = [' '.join([lemmer.lemmatize(word) for word in text.split()]) for text in document_clean]
titles = [' '.join([lemmer.lemmatize(word) for word in title.split()]) for title in title]

vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.002, max_df=0.99, max_features=10000, lowercase=True, stop_words=list(english_stopset))

X = vectorizer.fit_transform(new)
df = pd.DataFrame(X.T.toarray())

def get_similar_sentences(query, df):
    query_vec = vectorizer.transform([query]).toarray().reshape(df.shape[0],)
    sim_threshold = 0.1
    results = []

    for i, doc in enumerate(new):
        doc_sentences = nltk.sent_tokenize(docs[i])
        for sentence in doc_sentences:
            sentence_clean = ' '.join([lemmer.lemmatize(word) for word in re.sub(r'[%s]' % re.escape(string.punctuation), '', sentence.lower()).split()])
            sentence_vec = vectorizer.transform([sentence_clean]).toarray().reshape(df.shape[0],)

            similarity = np.dot(df.loc[:, i].values, query_vec) / (np.linalg.norm(df.loc[:, i]) * np.linalg.norm(query_vec))
            if similarity > sim_threshold:
                results.append({
                    "title": titles[i],
                    "sentence": sentence,
                    "similarity": similarity
                })

    results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    results = get_similar_sentences(query, df, docs)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
