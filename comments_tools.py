import csv
from collections import defaultdict
#import random
#from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
#import matplotlib.pyplot as plt
import pickle
#random.seed(99)

#datafile = "data/opiates_comments.csv"
#header = True #file has a header as the first row
author_index  = 4
post_id_index = 9

def data_file(s = "data/opiates_comments.csv"):
    datafile = s
    return datafile

def get_comments_raw(s="data/opiates_comments.csv",header=True):
    print("reading file")
    with open(data_file(s)) as fl:
        # skip "deleted" authors
        if header: comments_raw = [r for r in csv.reader(fl) if r[author_index]!="[deleted]"][1:]
        else: comments_raw = [r for r in csv.reader(fl) if r[author_index]!="[deleted]"]
    return comments_raw

def get_author_lists(comments_raw):
    print("building author lists")
    # group comments by author
    author_lists = defaultdict(lambda : list())
    for com in comments_raw:
        author_lists[com[author_index]].append(com)
    return author_lists



def get_top_authors(author_lists,n=1000):
    # use authors with at least 1000 comments for investigation
    top_authors = [a for a in author_lists.keys() if len(author_lists[a]) >=n]
    return top_authors

def get_all(s="data/opiates_comments.csv",header=True):
    cm = get_comments_raw(s=s, header=header)
    al = get_author_lists(cm)
    ta = get_top_authors(al)
    return cm,al,ta

#usage is as simple now as
#import comments_tools
#cm,al,ta = comments_tools.get_all()

