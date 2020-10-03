import csv
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import datetime
import re
import scipy.stats
import numpy as np

datafile = "data/opiates_comments.csv"
header = True #file has a header as the first row
author_index  = 4
post_id_index = 9

def load_data():
    print("reading file")
    with open(datafile) as fl:
        # skip "deleted" authors
        comments_raw = [r for r in csv.reader(fl) if r[author_index]!="[deleted]"][1:]

    print("building author lists")
    # group comments by author
    author_lists = defaultdict(lambda : list())
    for com in comments_raw:
        author_lists[com[author_index]].append(com)

    print("calculating top authors")
    # use authors with at least 1000 comments for investigation
    top_authors = [a for a in author_lists.keys() if len(author_lists[a]) >=1000]

    return comments_raw, author_lists, top_authors

def author_date_plotter(comments,comment_index=0,date_index=7,window_size=10):
    #comments should be a list of RAW comment objects, i,e, with date
    # get a list of comments and their dates
    dates = []
    for c in comments:
        dates.append((c[0],datetime.datetime.fromtimestamp(float(c[7]))))
        #gives a format like datetime.datetime(2016, 2, 17, 17, 51, 3) ~ year, month, day, time
    date_monthints = []
    #date_weekints = []
    #covert dates to ints that can be sorted
    #both month,year combo and year,week
    for c,d in dates:
        monthint = str(d.year)
        #insert a zero to keep things aligned
        if d.month < 10: monthint += '0'
        monthint += str(d.month)
        date_monthints.append(int(monthint))
        #weekint = str(d.year)
        #if d.isocalendar()[1] < 10: weekint += '0'
        #weekint += str(d.isocalendar()[1])
        #date_weekints.append(int(weekint))
    #now make a sequence of all possible year_month and year_week combinations between min and max
    month_seq = []
    #week_seq = []
    minmonth = min(date_monthints)
    #minweek = min(date_weekints)
    maxm = max(date_monthints)
    #maxw = max(date_weekints)
    while minmonth <= maxm:
        #format is yyyymm
        month_seq.append(minmonth)
        m = int(str(minmonth)[-2:]) + 1
        if m > 12:
            newyear = int(str(minmonth)[:4])+1
            minmonth = int(str(newyear)+'01')
        elif m < 10:
            minmonth = int(str(minmonth)[:4] + '0' + str(m))
        else:
            minmonth = int(str(minmonth)[:4] + str(m))

    fig,ax = plt.subplots()
    c = Counter(date_monthints)
    ind = list(range(1,len(month_seq)+1))
    ax.bar(ind,[c[month] for month in month_seq])
    #ax.set_xticklabels((str(m) for m in month_seq))
    #plt.xticks(ind,[c[month] for month in month_seq])

    #calibrate xticks by using built in calculations
    num_labels = len(ax.get_xticks())
    # divide month sequence into that many chuncks and use that as xticks
    plt.xticks(ax.get_xticks(),[month_seq[i] for i in range(0,len(month_seq),int(len(month_seq)/num_labels))])

    plt.show()

    #TODO do same with weeks
    #return c,month_seq,ax
    return 0

def term_freq_plotter(term,comments,comment_index=0,date_index=7,window_size=10):
    #comments should be a list of RAW comment objects, i,e, with date
    # get a list of comments and their dates
    dates = []
    for c in comments:
        dates.append((c[0],datetime.datetime.fromtimestamp(float(c[7]))))
        #gives a format like datetime.datetime(2016, 2, 17, 17, 51, 3) ~ year, month, day, time
    date_monthints = []
    # use this to keep track of occurances of term by month
    term_freqs = defaultdict(int)
    #covert dates to ints that can be sorted
    for c,d in dates:
        monthint = str(d.year)
        #insert a zero to keep things aligned
        if d.month < 10: monthint += '0'
        monthint += str(d.month)
        date_monthints.append(int(monthint))
        # calculate number of times term appears here and add to running total
        # the regex counts the number of times the term appears in the comment
        occurances = re.subn(term.lower(), '', c.lower())[1] 
        term_freqs[int(monthint)] += occurances
    #now make a sequence of all possible year_month and year_week combinations between min and max
    month_seq = []
    minmonth = min(date_monthints)
    maxm = max(date_monthints)
    while minmonth <= maxm:
        #format is yyyymm
        month_seq.append(minmonth)
        m = int(str(minmonth)[-2:]) + 1
        if m > 12:
            newyear = int(str(minmonth)[:4])+1
            minmonth = int(str(newyear)+'01')
        elif m < 10:
            minmonth = int(str(minmonth)[:4] + '0' + str(m))
        else:
            minmonth = int(str(minmonth)[:4] + str(m))

    fig,ax = plt.subplots()
    c = Counter(date_monthints)
    ind = list(range(1,len(month_seq)+1))
    ax.bar(ind,[c[month] for month in month_seq])
    #calibrate xticks by using built in calculations
    num_labels = len(ax.get_xticks())
    # divide month sequence into that many chuncks and use that as xticks
    # check to make sure theres enough plotting room
    if int(len(month_seq)/num_labels) > 0:
        plt.xticks(ax.get_xticks(),[month_seq[i] for i in range(0,len(month_seq),int(len(month_seq)/num_labels))])
    else: plt.xticks(ax.get_xticks(),[month_seq[i] for i in range(0,len(month_seq),1)])

    # add term freq plot on top with twinx https://matplotlib.org/gallery/api/two_scales.html
    ax2 = ax.twinx()
    ax2.plot(ind, [term_freqs[month] for month in month_seq], color='red')
    fig.tight_layout()
    plt.title(term)
    plt.show()
    return 0

def post_freq_vs_term(term,comments,comment_index=0,date_index=7,window_size=10):
    #comments should be a list of RAW comment objects, i,e, with date
    # get a list of comments and their dates
    dates = []
    for c in comments:
        dates.append((c[0],datetime.datetime.fromtimestamp(float(c[7]))))
        #gives a format like datetime.datetime(2016, 2, 17, 17, 51, 3) ~ year, month, day, time
    date_monthints = []
    # use this to keep track of occurances of term by month
    term_freqs = defaultdict(int)
    #covert dates to ints that can be sorted
    for c,d in dates:
        monthint = str(d.year)
        #insert a zero to keep things aligned
        if d.month < 10: monthint += '0'
        monthint += str(d.month)
        date_monthints.append(int(monthint))
        # calculate number of times term appears here and add to running total
        # the regex counts the number of times the term appears in the comment
        occurances = re.subn(term.lower(), '', c.lower())[1]
        term_freqs[int(monthint)] += occurances
    #now make a sequence of all possible year_month and year_week combinations between min and max
    month_seq = []
    minmonth = min(date_monthints)
    maxm = max(date_monthints)
    while minmonth <= maxm:
        #format is yyyymm
        month_seq.append(minmonth)
        m = int(str(minmonth)[-2:]) + 1
        if m > 12:
            newyear = int(str(minmonth)[:4])+1
            minmonth = int(str(newyear)+'01')
        elif m < 10:
            minmonth = int(str(minmonth)[:4] + '0' + str(m))
        else:
            minmonth = int(str(minmonth)[:4] + str(m))

    # calculate deltas between post frequency for consecutive months
    c = Counter(date_monthints)
    deltas = {}
    for i in range(len(month_seq)-1):
        deltas[month_seq[i]] = c[month_seq[i+1]] - c[month_seq[i]]
    # hypothesis is terms occuring in a month will change the delta for the next month
    # only go up to last element because deltas will be 1 short
    plt.plot([term_freqs[month] for month in month_seq[:-1]],[deltas[month] for month in month_seq[:-1]], 'bo')
    plt.show()
    return 0

def remove_outliers(data,m=2):
    x,y = zip(*data)
    mu_y = np.mean(y)
    mu_x = np.mean(x)
    s_y = np.std(y)
    s_x = np.std(x)
    f = [e for e in data if (mu_y - m*s_y < e[1]) and (mu_y + m*s_y > e[1])]
    return [e for e in f if (mu_x - m*s_x < e[0]) and (mu_x + m*s_x > e[0])]

def multi_author_postfreqterm(term, comments_lists, comment_index=0,date_index=7,window_size=10, reject_outliers=True):
    # comments_list is a list of LISTS of raw comments objects, i.e. comments lists for multiple authors
    # goal is to replicate post_freq plotting but throw away zero term frequency points and aggregate
    # across multiple authors
    # we'll store data points here as we calculate across authors. each point should be a tuble (term_freq, delta)
    points = []
    for l in comments_lists:
        dates = []
        for c in l:
            dates.append((c[0],datetime.datetime.fromtimestamp(float(c[7]))))
        date_monthints = []
        term_freqs = defaultdict(int)
        for c,d in dates:
            monthint = str(d.year)
            if d.month < 10: monthint += '0'
            monthint += str(d.month)
            date_monthints.append(int(monthint))
            occurances = re.subn(term.lower(), '', c.lower())[1]
            term_freqs[int(monthint)] += occurances
        month_seq = []     
        minmonth = min(date_monthints)
        maxm = max(date_monthints)
        while minmonth <= maxm:
            #format is yyyymm
            month_seq.append(minmonth)
            m = int(str(minmonth)[-2:]) + 1
            if m > 12:
                newyear = int(str(minmonth)[:4])+1
                minmonth = int(str(newyear)+'01')
            elif m < 10:
                minmonth = int(str(minmonth)[:4] + '0' + str(m))
            else:
                minmonth = int(str(minmonth)[:4] + str(m))

        # calculate deltas between post frequency for consecutive months
        c = Counter(date_monthints)
        deltas = {}
        for i in range(len(month_seq)-1):
            deltas[month_seq[i]] = c[month_seq[i+1]] - c[month_seq[i]]
        # use the counter to calculate a normalized term frequency
        # i.e. nf = f / c[month] or the proportion of comment containing the word
        normalized_freqs = {}
        for month in month_seq:
            if c[month] > 0: normalized_freqs[month] = term_freqs[month] * 1.0 / c[month]
            else: normalized_freqs[month] = 0.0
        for f,d in zip([normalized_freqs[month] for month in month_seq[:-1]],[deltas[month] for month in month_seq[:-1]]):
            if f > 0 and abs(d) > 0: points.append((f,d))
    # points just needs to be unzipped and plotted now
    # throw away outliers
    if reject_outliers: points = remove_outliers(points)
    freqs,deltas = zip(*points)
    plt.plot(freqs,deltas,'bo')
    # solid red line at 0
    plt.plot([0, max(freqs)],[0,0],color='red')
    plt.show()
    return points

def linreg_with_term(term, comments_lists):
    p =  multi_author_postfreqterm(term, comments_lists)
    freqs, deltas = zip(*p)
    return scipy.stats.linregress(freqs, deltas)

#linreg_with_term('quit', [author_lists[a] for a in top authors])

def calc_deltas_withoutplot(term,comments_lists,reject_outliers=True):
    # use this method to feed into others when only the data points are wanted
    points = []
    for l in comments_lists:
        dates = []
        for c in l:
            dates.append((c[0],datetime.datetime.fromtimestamp(float(c[7]))))
        date_monthints = []
        term_freqs = defaultdict(int)
        for c,d in dates:
            monthint = str(d.year)
            if d.month < 10: monthint += '0'
            monthint += str(d.month)
            date_monthints.append(int(monthint))
            occurances = re.subn(term.lower(), '', c.lower())[1]
            term_freqs[int(monthint)] += occurances
        month_seq = []
        minmonth = min(date_monthints)
        maxm = max(date_monthints)
        while minmonth <= maxm:
            #format is yyyymm
            month_seq.append(minmonth)
            m = int(str(minmonth)[-2:]) + 1
            if m > 12:
                newyear = int(str(minmonth)[:4])+1
                minmonth = int(str(newyear)+'01')
            elif m < 10:
                minmonth = int(str(minmonth)[:4] + '0' + str(m))
            else:
                minmonth = int(str(minmonth)[:4] + str(m))

        # calculate deltas between post frequency for consecutive months
        c = Counter(date_monthints)
        deltas = {}
        for i in range(len(month_seq)-1):
            deltas[month_seq[i]] = c[month_seq[i+1]] - c[month_seq[i]]
        # use the counter to calculate a normalized term frequency
        # i.e. nf = f / c[month] or the proportion of comment containing the word
        normalized_freqs = {}
        for month in month_seq:
            if c[month] > 0: normalized_freqs[month] = term_freqs[month] * 1.0 / c[month]
            else: normalized_freqs[month] = 0.0
        for f,d in zip([normalized_freqs[month] for month in month_seq[:-1]],[deltas[month] for month in month_seq[:-1]]):
            if f > 0 and abs(d) > 0: points.append((f,d))
    # points just needs to be unzipped and plotted now
    # throw away outliers
    if reject_outliers: points = remove_outliers(points)
    return list(zip(*points))

