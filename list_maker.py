import csv

internetfile = 'data/drug_words_internet.txt'
unionfile = 'data/union_eriktyler_yes_words.csv'
outfile = 'data/unified_drug_words.txt'

print('reading files')
with open(unionfile) as fl:
    u = [r for r in csv.reader(fl)]

with open(internetfile) as fl:
    raw = fl.readlines()

u_words = [l[0].lower().strip() for l in u if l[1]=='y']

internet_words = []
for line in raw[2:]:
    #skip first 2 lines bcuz theyre header
    for w in line.strip().lower().split(','):
        internet_words.append(w.strip())

drug_words = list(set(u_words).union(set(internet_words)))

with open(outfile,'w') as fl:
    for w in drug_words:
        fl.write(w.strip().lower() + '\n')
