import csv
import codecs
import re
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim import corpora
from gensim.utils import simple_preprocess

# first task is to load in manual words and drug/slang terms
unionfile = 'data/union_eriktyler_yes_words.csv'
internetfile = 'data/drug_words_internet.txt'
commentsfile = 'data/opiates_comments.json'

print('reading files')
with open(unionfile) as fl:
    u = [r for r in csv.reader(fl)]

with open(internetfile) as fl:
    raw = fl.readlines()

# process data into a single list of drug words
u_words = [l[0].lower().strip() for l in u if l[1]=='y']

internet_words = []
for line in raw[2:]:
    #skip first 2 lines bcuz theyre header
    for w in line.strip().lower().split(','):
        internet_words.append(w.strip())

# join 2 sets of words
drug_words = list(set(u_words).union(set(internet_words)))
# len(drug_words) == 565

# build dataset, test train split
with open(commentsfile) as fl:
    comments = json.load(fl)

# get count of drug-containing comments
p = re.compile('|'.join(drug_words))
drug_coms = [c for c in comments if p.match(c['body'].lower())]

print("building corpus")
# http://www.nltk.org/api/nltk.tokenize.html
# cat all comment text into one big string
# preprocessing:
#   lower case
#   remove punctuation at the end
com_string = '\n'.join([c['body'] for c in comments])
sents = [' '.join(simple_preprocess(c)) for c in sent_tokenize(com_string)]

#convert to gensim LineSentence format https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.LineSentence
# thats what word2vec needs 

with open('data/gensim_sentences.ls','w') as fl:
    fl.write('\n'.join(sents))

#create gensim dictionary and corpus
#d = corpora.Dictionary([word_tokenize(s) for s in sents])
#c = [d.doc2bow(word_tokenize(s)) for s in sents]

#save dict and corpus to file
#d.save('data/dictionary_gensim.dict')
#corpora.MmCorpus.serialize('data/corpus_gensim.mm',c)

