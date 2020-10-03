import csv
from collections import defaultdict, Counter
from nltk.corpus import PlaintextCorpusReader
import os
import term_freq_analysis
#from nltk.book import FreqDist
import nltk
import numpy as np
import matplotlib.pyplot as plt
import re

run_examples = False
if run_examples: from nltk.book import FreqDist

comments_raw, author_lists, top_authors = term_freq_analysis.load_data()

# write all top authors comments to a corpus for nltk
wr = False
if wr: 
    print("writing corpus")
    t = '\n'.join(['\n'.join([c[0] for c in l]) for l in [author_lists[a] for a in top_authors]])
    with open(outfile,'w') as fl:
        fl.write(t)

# read from a file
print("reading corpus")
corpus_root = os.getcwd() + "/data/"
file_ids = ".*.txt"
corpus = PlaintextCorpusReader(corpus_root, file_ids)

print("building nltk text obj")
text = nltk.Text(corpus.words())
V = set(text)
c = Counter(corpus.words())

#use this option to run examples
if run_examples:
    print("calculating colocations & freqdist")
    print(text.collocations())
    f1 = FreqDist(text)
    f1.plot(50, cumulative=True)

    print("vocab exploration examples")
    print([w for w in V if len(w) > 15][:100])
    print(sorted([w for w in V if len(w) > 7 and f1[w] > 7]))

    print("common contexts and concordances")
    # quit harm addict help pain clean sober high
    print(text.common_contexts(['sober','high']))
    print(text.concordance('sober'))

    print("d/dx comment freq vs words")
    p = term_freq_analysis.multi_author_postfreqterm('I ', [author_lists[a] for a in top_authors])
    p2 = term_freq_analysis.multi_author_postfreqterm('sober', [author_lists[a] for a in top_authors])

    t1,d1 = zip(*p)
    t2,d2 = zip(*p2)

    print(np.mean(d1), np.mean(d2))

    print("regression analysis")
    print(term_freq_analysis.linreg_with_term('I ', [author_lists[a] for a in top_authors]))
    print(term_freq_analysis.linreg_with_term('sober', [author_lists[a] for a in top_authors]))

def avg_delta_plotter(word_limit=None,min_count=3):
    #plot average delta vs word count
    avg_deltas = {}
    l =  [(word,c[word]) for word in V if c[word]>=min_count]
    if word_limit is not None:
        l = l[:word_limit]
    for i,(w,count) in enumerate(l):
        print(i,w)
        #skip any non-alphanumeric words
        if bool(re.match('^[a-zA-Z0-9]+$',w)):
            try:
                f,d = term_freq_analysis.calc_deltas_withoutplot(w,[author_lists[a] for a in top_authors])
                avg_deltas[w] = np.mean(d)
            except:
                print("error")
    plt.plot([c[w] for w in avg_deltas.keys()],[avg_deltas[w] for w in avg_deltas.keys()],'bo')
    plt.show()


