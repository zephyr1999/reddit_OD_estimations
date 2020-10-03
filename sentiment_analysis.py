import csv
from collections import defaultdict, Counter
import random
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import datetime

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

#TODO preprocess comments in standard ways with nltk
# https://towardsdatascience.com/nlp-extracting-the-main-topics-from-your-dataset-using-lda-in-minutes-21486f5aa925
# i.e. tokenize, lemmatize, stem, remove stop words
def process_comments_list(author):
    #takes in an author string and text-processes their comments
    # and sorts them into date order
    date_index  = 7
    cleaned = sorted(author_lists[author], key=lambda x: int(x[7]))
    #TODO refine cleaning stragegy
    return cleaned

# use authors with at least 1000 comments for investigation
top_authors = [a for a in author_lists.keys() if len(author_lists[a]) >=1000]

# sentiment analyzing object
sia = SIA()

#write csv with sentiment scores for tyler to check
#print("writing out 150 randoms for sentiment checking")
#outfile = 'data/comments_with_sentiment.csv'
#get random 150 comments
#coms = [c[0] for c in random.sample(comments_raw,150)]
#coms_with_scores = [(c,sia.polarity_scores(c)) for c in coms]
#with open(outfile,'w') as fl:
#    wrt = csv.writer(fl)
#    #header
#    wrt.writerow(['comment','compound','negative','neutral','positive'])
#    for (comment, scores) in coms_with_scores:
#        wrt.writerow([comment,scores['compound'],scores['neg'],scores['neu'],scores['pos']])

def author_plotter(comments,window_size=10):
    # comments is just a list of strings
    # this just plots the sequence and doesnt consider how fast or slowly they commented
    # i.e 100 comments in a day vs 100 comments over a year are treated the same
    # first generate sentiment scores
    scores = [sia.polarity_scores(c) for c in comments]
    # calculate sliding agerages of 'compound' score from VADER
    avgs = [sum([s['compound'] for s in scores[i:i+window_size]]) * 1.0 / window_size for i in range(len(comments)-window_size+1)]
    
    #plot the sequence with matplotlib
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    l, = plt.plot(list(range(len(avgs))),avgs)
    ax1 = plt.axes([0.25, 0.1, 0.65, 0.03])#, facecolor='lightgoldenrodyellow')
    slider1 = Slider(ax1, "window",valmin=1.0,valmax=100.0,valinit=10.0)#,valstep=1.0)
    def update_slider(val):
        ws = int(val)
        newy = [sum([s['compound'] for s in scores[i:i+ws]]) * 1.0 / ws for i in range(len(comments)-ws+1)]
        l.set_ydata(newy)
        l.set_xdata(list(range(len(newy))))
        fig.canvas.draw_idle()
    slider1.on_changed(update_slider)
    plt.show()

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
    num_lables = len(ax.get_xticks())
    # divide month sequence into that many chuncks and use that as xticks
    plt.xticks(ax.get_xticks(),[month_seq[i] for i in range(0,len(month_seq),int(len(month_seq)/num_labels))])

    plt.show()

    #TODO do same with weeks
    #return c,month_seq,ax
    return 0


def author_investigation_plotter(comments,window_size=10):
    # comments is just a list of strings
    # this just plots the sequence and doesnt consider how fast or slowly they commented
    # i.e 100 comments in a day vs 100 comments over a year are treated the same
    # first generate sentiment scores
    scores = [sia.polarity_scores(c) for c in comments]
    # calculate sliding agerages of 'compound' score from VADER
    avgs = [sum([s['compound'] for s in scores[i:i+window_size]]) * 1.0 / window_size for i in range(len(comments)-window_size+1)]

    #use these lines to align the min and max sliders
    minline = [[10,10],[-0.5,0.5]]
    maxline = [[20,20],[-0.5,0.5]]
    lowerind = 10
    upperind = 20
    
    #plot the sequence with matplotlib
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    l1, l2,l3= plt.plot(list(range(len(avgs))),avgs, minline[0],minline[1],'r',maxline[0],maxline[1],'r')
    ax1 = plt.axes([0.25, 0.1, 0.65, 0.03])#, facecolor='lightgoldenrodyellow')
    ax2 = plt.axes([0.25,0.15,0.65,0.03])
    ax3 = plt.axes([0.25,0.2,0.65,0.03])
    slider1 = Slider(ax1, "smoothing",valmin=1.0,valmax=100.0,valinit=10.0)#,valstep=1.0)
    slider2 = Slider(ax2, "first",valmin=0.0,valmax=1.0*len(scores),valinit=10.0)#,valstep=1.0)
    slider3 = Slider(ax3,'last',valmin=1.0,valmax=1.0*len(scores),valinit=20.0)

    def ul(value):
        lowerind = value
        return value

    def uu(value):
        upperind = value
        return value

    def update_slider(val):
        ws = int(val)
        newy = [sum([s['compound'] for s in scores[i:i+ws]]) * 1.0 / ws for i in range(len(comments)-ws+1)]
        l1.set_ydata(newy)
        l1.set_xdata(list(range(len(newy))))
        fig.canvas.draw_idle()
    def update_slider2(val):
        lowerind = ul(int(val))
        l2.set_xdata([val,val])
        fig.canvas.draw_idle()
    def update_slider3(val):
        upperind = uu(int(val))
        l3.set_xdata([val,val])
        #for c in comments[lowerind:upperind]:print(c)
        #print("------------------------------------------------------------------------------------\n---------------\n---------\n---")
        fig.canvas.draw_idle()
    slider1.on_changed(update_slider)
    slider2.on_changed(update_slider2)
    slider3.on_changed(update_slider3)
    
    ax4 = plt.axes([0.25,-0.2,0.65,0.0])
    prtbtn = Button(ax4, "print")
    def btn_click(event):
        for c in comments[lowerind:upperind]:
            print(c)
        print("------------------------------------------------------------------------------------\n---------------\n---------\n---")
    prtbtn.on_clicked(btn_click)
    plt.show()



#find most positive and negative commenters
print("calculating top authors avg sentiments")
#TODO let below run to completion
#author_avgs = [(a, sum([sia.polarity_scores(c[0])['compound'] for c in author_lists[a]]) / len(author_lists[a])) for a in top_authors]
#author_avs.sort(key= lambda x:x[0])
#below will print most positive and negative authors
# author_avgs[:10]
# author_avgs[-10:]


