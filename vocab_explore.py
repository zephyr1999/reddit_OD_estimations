import csv
import codecs
import re

#this is required because of bs excel export crap
codecs.register_error("strict", codecs.ignore_errors)


erikfile = 'data/vocab_erik_xl.csv'
tylerfile = 'data/Tyler_Words.csv'

with open(erikfile,encoding='utf-8') as fl:
    erik_data = [r for r in csv.reader(fl)]

with open(tylerfile) as fl:
    tyler_data = [r for r in csv.reader(fl)]

#get the intersection and dysunion of both sets of words
erik_yset = list(set([d[0] for d in erik_data if d[1]=='y']))
tyler_yset = list(set([d[0] for d in tyler_data if d[1]=='y']))

union = [e for e in erik_yset if e in tyler_yset]

erik_only = [e for e in erik_yset if e not in tyler_yset]
tyler_only = [t for t in tyler_yset if t not in erik_yset]

# use this regex for searching for dosing, specifically numberals followed by mg or mgs
p = re.compile(r'\b\d+ ?mgs?')
