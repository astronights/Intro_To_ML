original = "./outliers/practice_outliers_net_worths.pkl"
destination = "./outliers/practice_outliers_net_worths_unix.pkl"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

print("Done. Saved %s bytes." % (len(content)-outsize))

from nltk.corpus import stopwords
sq = stopwords.words("english")
print(len(sq))  
