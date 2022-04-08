
from fileinput import filename
import gensim
import gensim.corpora as corpora
import re
import pandas as pd
import os
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from pprint import pprint

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
            if word not in stop_words] for doc in texts]

def preprocess(filename):
    papers = pd.read_csv(filename)
    papers = papers.drop(columns=['category1'], axis=1)

    # Remove punctuation
    papers['paper_text_processed'] = \
    papers['tit'].map(lambda x: re.sub('[,\.!?]', '', x))

    # Convert the titles to lowercase
    papers['paper_text_processed'] = \
    papers['paper_text_processed'].map(lambda x: x.lower())
    
    return papers
if __name__ == '__main__':
    papers = preprocess('cs_2021_00.csv')
    stop_words = stopwords.words('english')
    # stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    data = papers.paper_text_processed.values.tolist()
    data_words = list(sent_to_words(data))
    # remove stop words
    data_words = remove_stopwords(data_words)

    # Create Dictionary
    id2word = corpora.Dictionary(data_words)

    # Create Corpus
    texts = data_words

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # number of topics
    num_topics = 5

    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=num_topics)

    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]