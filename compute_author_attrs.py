import csv
import networkx as nx
from collections import defaultdict
from itertools import combinations

comment_file = "data/opiates_comments.csv"
author_index  = 4
header=True
post_id_index = 9
author_topic_file = "data/thetas_by_user.csv"
outfile = "author_attributes.csv"

print("reading comments")
with open(comment_file) as fl:
    if header: comments_raw = [r for r in csv.reader(fl) if r[author_index]!="[deleted]"][1:]
    else: comments_raw = [r for r in csv.reader(fl) if r[author_index]!="[deleted]"]

#print("building author lists")
#author_lists = defaultdict(lambda : list())
#for com in comments_raw:
#    author_lists[com[author_index]].append(com)

print("building post list")
# make list of posts for more efficiency
post_lists = defaultdict(lambda : set())
for com in comments_raw:
    post_lists[com[post_id_index]].add(com[author_index])

del comments_raw

print("building network")
net = nx.Graph()
#for _, post_list in post_lists.items():
#    for i in range(len(post_list)-1):
#        for j in range(i+1,len(post_list)):
#            #avoid self-edges
#            if post_list[i][author_index] != post_list[j][author_index]:
#                net.add_edge(post_list[i][author_index], post_list[j][author_index])

# instead, just loop over set of authors
for _, post_list in post_lists.items():
    #author_list = set([c[author_index] for c in post_list])
    for a1,a_2 in combinations(post_list, 2):
        net.add_edge(a1,a_2)

print("reading authors")
with open(author_topic_file) as fl:
    author_topics = [r for r in csv.reader(fl)]

print("building author attributes")
author_attrs = []
for line in author_topics:
    # 0 = author name, 1 = num_comments, 2:26 are topic levels
    num_comments = int(line[1])
    topic_levels = [float(i) for i in line[2:]]
    degree = net.degree(line[0])
    if degree == '{}': degree = 0
    author_attrs.append([line[0],num_comments,degree]+topic_levels)

print("writing out")
with open(outfile, 'w') as fl:
    w = csv.writer(fl)
    for line in author_attrs:
        w.writerow(line)

nx.write_graphml(net, "usergraph.graphml")
# read with newgraph = nx.read_graphml("usergraph.graphml")
