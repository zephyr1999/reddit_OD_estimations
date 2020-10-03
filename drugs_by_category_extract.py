#import matplotlib.pyplot as plt
import comments_tools
import re
import json

count = False
json_write = False

infile = 'data/drug_slang_dea.txt'
drug_cats_outfile = 'data/drug_mentions.json'
comments_outfile = 'data/opiates_comments.json'

# read in file
with open(infile) as fl:
    data = fl.readlines()

print('processing drug names')

# find all the lines representing a drug category
# note that some lines have multiple names and are indicated with a '/'
drug_cats = {}

for i,l in enumerate(data):
    if l.startswith('--'):
        #indicates a drug name line, so initialize an empty list
        drug_cats[data[i].lower().strip()[2:]] = []

        j = i+1
        line = data[j]
        while (not line.startswith('--')) and (j<len(data)-1):
            drug_cats[data[i].lower().strip()[2:]].append(data[j].lower().strip())
            j = j + 1
            line = data[j]

# process into clean, split words
drug_word_lists = {}
for drug,lines in drug_cats.items():
    drug_word_lists[drug] = (''.join(lines)).split(';')
    # add the drug itself to the list
    # if theres a slash, then theres 2 words to add
    if '/' in drug:
        drug_word_lists[drug].extend(drug.split('/'))
    else:
        drug_word_lists[drug].append(drug)

# one more pass because some words have spaces and some have parenthetical notes
drugs_clean = {}
for drug, words in drug_word_lists.items():
    drugs_clean[drug] = []
    for word in words:
        if '(' in word:
            drugs_clean[drug].append(word.split('(')[0].strip())
        else:
            drugs_clean[drug].append(word.strip())

# get opiates comments and count how many appear
cr = comments_tools.get_comments_raw()

print('scanning for drug names')

# keep counts of appearances
category_counts = {k:0 for k in drugs_clean.keys()}

#loop over comments
# only run if wanting to count
if count:
    for com in cr:
        # comment body = com[0]
        for category, words in drugs_clean.items():
            p = re.compile('|'.join(words))
#            for w in words:
                # if theres a match, increase the count
#                if re.match(w,com[0]):
#                    category_counts[category] += 1
            if p.match(com[0]): category_counts[category] += 1

#output
    for cat, coun in category_counts.items():
        print(cat,'\t',coun)

# use indecies of comments instead of new datastructure
if json_write:
    # generate dict of {large category: {small category: [comment ids]}}
    drug_cats = {k:{w:[] for w in drugs_clean[k]} for k in drugs_clean.keys()}
    # loop over comments
    for com in cr:
        # loop over each drug category
        for cat, words in drugs_clean.items():
            # see if there's any match at all for this comment and category
            p = re.compile('|'.join(words))
            if p.match(com[0]):
                # if theres a match, search for individual words
                for w in words:
                    p2 = re.compile(w)
                    if p2.match(com[0]) : drug_cats[cat][w].append(com[15])
    
    print('writing json')

    with open(drug_cats_outfile,'w') as fl:
        json.dump(drug_cats,fl)
    
    comments_attrs = ['body','score_hidden','archived','name','author','author_flair_text',
            'downs','created_utc','subreddit_id','link_id','parent_id','score','retrieved_on',
            'controversiality','gilded','id','subreddit','ups','distinguished','author_flair_css_class','removal_reason']
    
    cj = []
    for com in cr:
        cj.append({comments_attrs[i]:com[i] for i in range(len(comments_attrs))})
    
    with open(comments_outfile,'w') as fl:
        json.dump(cj,fl)

