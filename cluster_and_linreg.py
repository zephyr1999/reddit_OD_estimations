import csv
from sklearn.cluster import KMeans
import numpy as np
from sklearn import linear_model
from collections import defaultdict
import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from random import randint
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel,ttest_ind
from bootstrap_routines import *
from itertools import combinations

attr_file = 'author_attributes.csv'
comment_file = "data/opiates_comments.csv"
header=True
author_index = 4

print("reading attributes")
with open(attr_file) as fl:
    attrs = []
    for row in csv.reader(fl):
        author = row[0]
        num_comments = int(row[1])
        degree = row[2]
        if degree == '{}': degree = 0
        degree = int(degree)
        topics = [float(i) for i in row[3:]]
        attrs.append([author,num_comments,degree]+topics)

# strip author names
X = np.array([l[1:] for l in attrs])
X_scaler = StandardScaler().fit(X)
print("fitting")
kmeans = KMeans(n_clusters=2, random_state=0).fit(X_scaler.transform(X))

#print("centers")
#print([X_scaler.inverse_transform(i) for i in kmeans.cluster_centers_])

print("reading comments")
with open(comment_file) as fl:
    if header: comments_raw = [r for r in csv.reader(fl) if r[author_index]!="[deleted]"][1:]
    else: comments_raw = [r for r in csv.reader(fl) if r[author_index]!="[deleted]"]

# dont need lists for each author
#print("building author lists")
#author_lists = defaultdict(lambda : list())
#for com in comments_raw:
#    author_lists[com[author_index]].append(com)

print("grouping authors")
author_groups = {}
for a in attrs:
    author_groups[a[0]] = kmeans.predict(X_scaler.transform([a[1:]]))[0]

print("cluster sizes")
print("0: ", len([k for k in kmeans.labels_ if k==0]))
print("1: ", len([k for k in kmeans.labels_ if k==1]))

red = defaultdict(lambda : list())
#aggregate by year
for c in comments_raw[1:]:
    if c[author_index] in author_groups.keys():
        d = datetime.datetime.fromtimestamp(int(c[7])).isocalendar()
        y = d[0]
        red[y].append(c)

print("analyzing comments by year")
#define drug category words
# see https://www.kff.org/other/state-indicator/opioid-overdose-deaths-by-type-of-opioid/
natural = ['morphine','codeine','oxycodone', 'hydrocodone', 'hydromorphone', 'oxymorphone']
synthetic = ['tramadol','fentanyl','carfentanil','butyrfentanyl']
methadone = ['methadone']
heroin = ['heroin', ' H ', ' h ']

# 0 or 1 indicates cluster
drug_mentions_0 = {}
drug_mentions_1 = {}
for year, comlist in red.items():

    comlist_0 = list(filter(lambda x: author_groups[x[4]]==0, comlist))
    comlist_1 = list(filter(lambda x: author_groups[x[4]]==1, comlist))

    
    nat_count_0 = len([comment for comment in comlist_0 if any([term in comment[0].lower() for term in natural])])
    syn_count_0 = len([comment for comment in comlist_0 if any([term in comment[0].lower() for term in synthetic])])
    met_count_0 = len([comment for comment in comlist_0 if any([term in comment[0].lower() for term in methadone])])
    h_count_0 = len([comment for comment in comlist_0 if any([term in comment[0].lower() for term in heroin])])

    # use this as the holder for total num of "drug" comments no matter category
    #s = nat_count + syn_count + met_count + h_count

    drug_mentions_0[year] = [nat_count_0,syn_count_0,met_count_0,h_count_0]

    nat_count_1 = len([comment for comment in comlist_1 if any([term in comment[0].lower() for term in natural])])
    syn_count_1 = len([comment for comment in comlist_1 if any([term in comment[0].lower() for term in synthetic])])
    met_count_1 = len([comment for comment in comlist_1 if any([term in comment[0].lower() for term in methadone])])
    h_count_1 = len([comment for comment in comlist_1 if any([term in comment[0].lower() for term in heroin])])

    # use this as the holder for total num of "drug" comments no matter category
    #s = nat_count + syn_count + met_count + h_count

    drug_mentions_1[year] = [nat_count_1,syn_count_1,met_count_1,h_count_1]

# now analyze num users
user_mentions_0 = defaultdict(lambda : list())
user_mentions_1 = defaultdict(lambda : list())
for year in sorted(list(red.keys())):
    comlist_0 = list(filter(lambda x: author_groups[x[4]]==0, red[year]))
    comlist_1 = list(filter(lambda x: author_groups[x[4]]==1, red[year]))

    for drugclass in [natural, synthetic, methadone, heroin]:
        count_0 = len(list(set([comment[4] for comment in comlist_0 if any([term in comment[0].lower() for term in drugclass])])))
        count_1 = len(list(set([comment[4] for comment in comlist_1 if any([term in comment[0].lower() for term in drugclass])])))
        user_mentions_0[year].append(count_0)
        user_mentions_1[year].append(count_1)

# updated to use user counts
#retrieved via big query
reddit_vol ={2010:690914,
        2011:1601459,
        2012:3461843,
        2013:4969048,
        2014:6717292,
        2015:8414146,
        2016:10076395,
        2017:11546887}

h_deaths = {2010:3036,
        2011:4397,
        2012:5925,
        2013:8257,
        2014:10574,
        2015:12989,
        2016:15469}

years = list(range(2010,2017))
#X = [[drug_mentions_0[year][3] + drug_mentions_1[year][3],user_mentions_0[year][3],user_mentions_1[year][3],reddit_vol[year]] for year in years]

#normalize variables so coefficients are comprable
dm_0 = [drug_mentions_0[year][3] for year in years]
dm_1 = [drug_mentions_1[year][3] for year in years]
dm = [drug_mentions_0[year][3] + drug_mentions_1[year][3] for year in years]
um_0 = [user_mentions_0[year][3] for year in years]
um_1 = [user_mentions_1[year][3] for year in years]
um = [user_mentions_1[year][3] + user_mentions_0[year][3] for year in years]
rv = [reddit_vol[year] for year in years]

# subtract mean and divide by std for each of above
def standardize(l):
    mean = sum(l)*1.0/len(l)
    s = np.std(l)
    return [(i-mean)/s for i in l]

dm_0_s = standardize(dm_0)
dm_1_s = standardize(dm_1)
dm_s = standardize(dm)
um_0_s = standardize(um_0)
um_1_s = standardize(um_1)
um_s = standardize(um)
rv_s = standardize(rv)


def scorer(model,X,y):
    # asssumed model has been trained holding out last value of X and y
    print("r^2 train")
    rtr = model.score(X[:-1],y[:-1])
    print(rtr)
    print("rmse train")
    rmt = sqrt(mean_squared_error(y[:-1],model.predict(X[:-1])))
    print(rmt)
    print("r^2 test")
    rte = model.score([X[-1]],[y[-1]])
    print(rte)
    print("rmse test")
    rmte = sqrt(mean_squared_error([y[-1]],model.predict([X[-1]])))
    print(rmte)
    #return in above order
    return (rtr,rmt,rte,rmte)

y = [h_deaths[i] for i in years]

def mean_abs_per_err(y_true,y_pred):
    # https://stats.stackexchange.com/q/62511
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def multi_scorer(X,y):
    results = []
    #calculate 1-holdout
    for i in range(len(X)):
        # i indicates holdout index
        X_tr = X[:i]+X[i+1:]
        y_tr = y[:i] + y[i+1:]
        X_test = [X[i]]
        y_test = [y[i]]
        model = linear_model.Ridge(normalize=False).fit(X_tr,y_tr)
        rtr = model.score(X_tr,y_tr)
        rmt = sqrt(mean_squared_error(y_tr,model.predict(X_tr)))
        rte = model.score(X_test,y_test)
        rmte = sqrt(mean_squared_error(y_test,model.predict(X_test)))
        mape = mean_abs_per_err(y_test,model.predict(X_test))
        results.append((rtr,rmt,rte,rmte,mape))
    #calculate 2-holdout
    for (i,j) in combinations([_ for _ in range(len(X))], 2):
        # i and j are withheld from training
        X_tr = X[:i]+X[i+1:j]+X[j+1:]
        y_tr = y[:i]+y[i+1:j]+y[j+1:]
        X_test = [X[i],X[j]]
        y_test = [y[i],y[j]]
        model = linear_model.Ridge(normalize=False).fit(X_tr,y_tr)
        rtr = model.score(X_tr,y_tr)
        rmt = sqrt(mean_squared_error(y_tr,model.predict(X_tr)))
        rte = model.score(X_test,y_test)
        rmte = sqrt(mean_squared_error(y_test,model.predict(X_test)))
        mape = mean_abs_per_err(y_test,model.predict(X_test))
        results.append((rtr,rmt,rte,rmte,mape))

    #print("avg r^2 train")
    #print(sum([r[0] for r in results])/len(X))
    #print("avg rmse train")
    #print(sum([r[1] for r in results])/len(X))
    #print("avg r^2 test")
    #print(sum([r[2] for r in results])/len(X))
    print("avg rmse test")
    print(sum([r[3] for r in results])/len(X))
    
    # first len(X) are 1-holdout, rest are 2-holdout
    return results

print("model with only reddit volume-----")
X = [[rv_s[i]] for i in range(len(rv_s))]
reddit_results = multi_scorer(X,y)

print("model with unclustered user volume-----")
X = [[um_s[i]] for i in range(len(dm_s))]
unclusterd_comment_results = multi_scorer(X,y)

print("model with clustered user volume-----")
X = [[um_0_s[i],um_1_s[i]] for i in range(len(dm_s))]
clustered_comment_results = multi_scorer(X,y)

#print("model with unclustered user counts-----")
#X = [[um_s[i]] for i in range(len(dm_s))]
##unclustered_user_results = multi_scorer(X,y)
#
#print("model with clustered user counts-----")
#X = [[um_0_s[i],um_1_s[i]] for i in range(len(dm_s))]
#clustered_user_results = multi_scorer(X,y)
#
#print("master model with clustered user counts and clustered comment counts-----")
#X = [[um_0_s[i],um_1_s[i],dm_0_s[i],dm_1_s[i]] for i in range(len(dm_s))]
#master_results = multi_scorer(X,y)
#
print("reddit volume AND clustered user counts model")
X = [[um_0_s[i],um_1_s[i],rv_s[i]] for i in range(len(dm_s))]
reddit_plus_clustered_users_results = multi_scorer(X,y)

# t test on the test rmse values results[3]
print(" t test on clustered and unclustered users")
print(ttest_rel([r[3] for r in unclusterd_comment_results[7:]],[r[3] for r in clustered_comment_results[7:]]))

print("ttest on reddit volume and reddit PLUS clustered user counts")
print(ttest_rel([r[3] for r in reddit_results[7:]],[r[3] for r in reddit_plus_clustered_users_results[7:]]))

##bootstrapping
#print("95% conf interval on reddit volume model")
#print(bootci([r[3] for r in reddit_results]))
#print("95% conf interval on master model (clustered users and comments)")
#print(bootci([r[3] for r in master_results]))
#
#print("confidence interval of master_mean MINUS reddit_mean")
#print(bootci_diff([r[3] for r in master_results],[r[3] for r in reddit_results]))
#
#print("confidence interval of reddit_and_volume model")
#print(bootci([r[3] for r in reddit_plus_clustered_users_results]))
#
#print("confidence interval of (reddit PLUS clusters model mean) MINUS (reddit model mean)")
#print(bootci_diff([r[3] for r in reddit_plus_clustered_users_results],[r[3] for r in reddit_results], alpha=0.35))
#
## above should all be same length, sanity check
##print("lengths of normed attrs should be same")
#for l in [dm_s, um_0_s, um_1_s, rv_s]: print(len(l))


#X = [[dm_s[i], um_0_s[i], um_1_s[i], rv_s[i]] for i in range(len(dm_s))]

# to see coefficients with unclustered users, uncomment below
#um_unified = [user_mentions_0[year][3] + user_mentions_1[year][3] for year in years]
#um_u_s = standardize(um_unified)
#X = [[dm_s[i], um_u_s[i], rv_s[i]] for i in range(len(dm_s))]


#print("training linear model")
#lr = linear_model.Ridge(normalize=False).fit(X,y)
#print("coefficients")
#print(lr.coef_)
#print("r squared")
#print(lr.score(X,y))


