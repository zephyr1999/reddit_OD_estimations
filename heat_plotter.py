# necessary for mac plotting
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random
import csv

#TEST
#unif = np.random.rand(10,12)
#following args turn off ticks and the color bar
#ax = sns.heatmap(unif, cbar=False, xticklabels=False,yticklabels=False)

#TEST2
fake_data = {}
fake_topics = {}
for i in range(100):
    fake_data[i] = random.random()
    fake_topics[i] = fake_topics[i] = list(np.random.rand(1,30)[0])

def load_test_data():
    with open('data/thetas_by_user.csv') as fl:
        raw = [r for r in csv.reader(fl)]
    attrs = {r[0]:int(r[1]) for r in raw}
    topics = {r[0]:[float(k) for k in r[2:]] for r in raw}

    return attrs,topics

counts,count_topics = load_test_data()

def load_degree_and_centrality(topics):
    with open('data/opiates_directed_info.csv') as fl:
        raw = [r for r in csv.reader(fl)]
    # format is user,in degree,out degree,eigenvector centrality
    # user set may not match topics, so we make a new set
    in_degrees = {r[0]:int(r[1]) for r in raw[1:]}
    centralities = {r[0]:float(r[3]) for r in raw[1:]}

    # make sure keys set are the same
    modified_topics = {}
    modified_degrees = {}
    modified_centralities = {}

    for author,topic in topics.items():
        if author in in_degrees.keys(): modified_topics[author] = topic

    for author,attr in in_degrees.items():
        if author in modified_topics.keys(): modified_degrees[author] = attr

    for author,attr in centralities.items():
        if author in modified_topics.keys(): modified_centralities[author] = attr

    return modified_topics,modified_degrees,modified_centralities

mod_topics,degrees,centralities = load_degree_and_centrality(count_topics)

# goal is to write this as a util library.
# inputs will be a csv of author, topic levels
# and an atribute file, a csv of author, attribute

# this util should aggregate authors into buckets by the attribute
# it should sort the topics by prevalance in that bucket and then assign
# colors based on the largest- (or smallest-) -valued bucket
# then produce the heatmap

def group_up(data,eqsize=False, num_buckets=50):
    #datais assumed to be dict where k=authorname and v=numerical_attribute to be plotted

    assignments = {}

    if eqsize:
        # sort data tuples by increasing attribute and then cut list at equal intervals
        srt = sorted(data.items(), key=lambda x:x[1])
        bucket_size = int(len(data)/num_buckets)
        ast=0
        for i,(author,_) in enumerate(srt):
            if i>0 and i%bucket_size==0 and ast<num_buckets-1: ast += 1
            assignments[author] = ast

    else:
        #calculate bucket size by finding range of attribute and dividing bu num_buckets
        bucket_size = (max(data.values()) - min(data.values()) * 1.0) / num_buckets

        # calculate bucket assignments as a dict k=author, v= bucket_assignment (0 to num-buckets-1)
        #assignments = {}
        for author,attr in data.items():
            assignments[author] = math.floor(attr/bucket_size)
            if assignments[author]>=num_buckets:
                #sometimes this happens due to decimal rounding error, so we should just put this 
                # into the last bucket
                assignments[author]=num_buckets-1

    return assignments

def aggregate(assignments,topic_levels,num_buckets=50):
    # assignments come from group_up method, and topic_levels 
    # are a dict  with form {author:[topic1,topic2,...]}
    # returns a 2d list where each row corresponds to an aggregated group
    # and each column corresponds to a topic
    num_topics = len(list(topic_levels.values())[0])
    topic_matrix = np.zeros((num_buckets,num_topics))

    # simply sum each successive author's topic levels to the corresponding bucket row
    for author,topics in topic_levels.items():
        # loop over each topic
        for i,topic_level in enumerate(topics):
            topic_matrix[assignments[author]][i] += topic_level

    return topic_matrix

def generate_color_matrix(topic_matrix,last_row=True):
    # given the aggregated topic matrix from aggregate,
    # now convert this to a color matrix
    # last row causes the colors to be assigned by rank on last bucket
    # I.e. most frequent commenters, most central nodes, etc
    # when set to False, it uses the first row instead

    #initialize color matrix to same size as topic matrix
    color_matrix = np.zeros(topic_matrix.shape)

    if last_row: ind=-1
    else: ind = 0

    # this is made very easy by the np.argsort function, which returns original indicies after sorting
    sorted_ind = np.argsort(topic_matrix)

    #generate color map based on sorting of row indicated by ind
    color_map = {}
    for i,topic in enumerate(sorted_ind[ind]):
        # i corresponds to color because its increasing along row ind
        # topic is the original index in the ind row
        color_map[topic] = i

    # use mapping to assign colors to all rows
    for i,row in enumerate(color_matrix):
        for j,color in enumerate(row):
            color_matrix[i,j] = color_map[sorted_ind[i][j]]

    return color_matrix,color_map




def plot(data,topic_levels,annotate=False,last_row=True,eqsize=False,num_buckets=50,c_map='magma'):
    # for other colormaps, see https://matplotlib.org/examples/color/colormaps_reference.html
    # requirements for data and topic levels are given above.

    asts = group_up(data,eqsize,num_buckets)

    top_mat = aggregate(asts,topic_levels,num_buckets)

    color_mat,color_map = generate_color_matrix(top_mat,last_row)

    ax = sns.heatmap(color_mat, cmap=c_map, annot=annotate,cbar=False, xticklabels=False,yticklabels=False)
    plt.show()

    return asts,top_mat,color_mat,color_map

def plot_with_annotate(data,topic_levels,annotate_list=[],last_row=True,eqsize=False,num_buckets=50,c_map='magma'):
    # for other colormaps, see https://matplotlib.org/examples/color/colormaps_reference.html
    # requirements for data and topic levels are given above.

    asts = group_up(data,eqsize,num_buckets)

    top_mat = aggregate(asts,topic_levels,num_buckets)

    color_mat,color_map = generate_color_matrix(top_mat,last_row)

    #generate inverse map
    inverse_map = {v:k for k,v in color_map.items()}

    #use that to find topic numbers on each
    new_annot = [[inverse_map[i]+1 for i in row] for row in color_mat]

    #need to make this into numpy array for annot to work
    na = np.array(new_annot)
    form = 'd'

    #if annotate_list is set, only plot those topics
    if annotate_list != []:
        for i in range(len(new_annot)):
            for j in range(len(new_annot[0])):
                    if new_annot[i][j] in annotate_list: new_annot[i][j]=str(new_annot[i][j])
                    else: new_annot[i][j] = ' '
        na = np.array(new_annot)
        form='s'


        
    ax = sns.heatmap(color_mat, cmap=c_map, fmt=form,annot=na,cbar=False, xticklabels=False,yticklabels=False)
    plt.show()

    return asts,top_mat,color_mat,color_map


# example useage, data loaded above
plot_with_annotate(counts,count_topics,c_map="Blues_r")
plot_with_annotate(degrees,mod_topics,c_map="Reds_r")
