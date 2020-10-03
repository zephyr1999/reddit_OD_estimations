from gensim import corpora,models
from nltk.corpus import PlaintextCorpusReader
from collections import Counter
from sklearn.linear_model import LogisticRegression
import csv
import nltk
import os
import random

random.seed(1)


# https://www.machinelearningplus.com/nlp/gensim-tutorial/
# https://radimrehurek.com/gensim/tut2.html

dictfile = 'data/dictionary_gensim.dict'
cfile = 'data/corpus_gensim.mm'
comments_file = 'data/opiates_comments'

internetfile = 'data/unified_drug_words.txt' 

# read in dict and corpus
#d = corpora.Dictionary.load(dictfile)
#c = corpora.MmCorpus(cfile)


# create a tfidf model
#tfidf = models.TfidfModel(c)

# build gensim lsi topic model with 2D encoding
# https://radimrehurek.com/gensim/tut2.html
# lsi = models.LsiModel(tfidf[c], id2word=d, num_topics=2)

#w2v = models.Word2Vec(corpus_file='data/gensim_sentences.ls', min_count=3, workers=4)

#w2v.save('data/w2v_cleaned_n=100.model')
w2v = models.Word2Vec.load('data/w2v_cleaned_n=100.model')
# access specific numpy vectors like w2v['heroin']

# import nltk corpus for investigating usage
#TODO write corpus to binary objects or pickle?

print("building nltk corpus")
nltkcorpus = PlaintextCorpusReader(os.getcwd()+'/data/', 'gensim_sentences.ls')
text = nltk.Text(nltkcorpus.words())
c = Counter(nltkcorpus.words())
vocab_3 = [w for w in set(text) if c[w] >2]


# mark positive examples
# read in drug words
with open(internetfile) as fl:
    drug_words_raw = [w.strip() for w in fl.readlines()]

#use only drug words that actually appear
drug_words = []
for w in drug_words_raw:
    try:
        w2v[w]
        drug_words.append(w)
    except:
        pass

n = len(drug_words)

print("training logreg")
# test/train split
non_drug_vocab = [w for w in vocab_3 if w not in drug_words]
random.shuffle(drug_words)
y_drugs = [1 for _ in drug_words]
random.shuffle(non_drug_vocab)
y_nondrugs = [0 for _ in non_drug_vocab]

X = [w2v[w] for w in drug_words] + [w2v[w] for w in non_drug_vocab[:n]]
y = y_drugs + y_nondrugs[:n]

#build skl logreg model
clf = LogisticRegression().fit(X,y)
#TODO what selection likelihood is used here?

# get some possible drug words
# use model to filter words
p = [w for w in non_drug_vocab if clf.predict([w2v[w]])[0] == 1]
# write to file 
writetofile = False
outfile = 'data/logreg_words_unlabeled.csv'
if writetofile:
    with open(outfile,'w') as fl:
        r = csv.writer(fl)
        r.writerow(['word','drug?','misspell or slang?','notes'])
        for w in p: r.writerow([w])

#TODO write tests for prec/recall for models on different models
