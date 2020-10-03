import csv
from collections import defaultdict
import random

random.seed(99)

datafile = "data/opiates_comments.csv"
header = True #file has a header as the first row
author_index  = 4
post_id_index = 9

print("reading file")
with open(datafile) as fl:
    # skip "deleted" authors
    comments_raw = [r for r in csv.reader(fl) if r[author_index]!="[deleted]"][1:]

print("building author lists")
# group comments by author
author_lists = defaultdict(lambda : list())
for com in comments_raw:
    author_lists[com[author_index]].append(com)

print("calculating list lengths")
#store as a tuple list on author for easy lookup and sorting
author_lens = [(a,len(author_lists[a])) for a in author_lists.keys()]

#sort based on lengths, sort in place, based on descending lengths
author_lens.sort(key=lambda x: x[1], reverse=True)

#get 100 most prolific users
#author_lens[0:99]
# get 10 randoms with ~100
# get 10 randoms with ~30


