import bs4 as bs
import urllib.request
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
# taking the text from wikipedia
scraped_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Artificial_intelligence')
article = scraped_data.read()

parsed_article = bs.BeautifulSoup(article,'lxml')

paragraphs = parsed_article.find_all('p')

article_text = ""

for p in paragraphs:
    article_text += p.text

# testing on our existing file
'''f = open('test.txt','r')
article_text = f.read()'''
# replacing the numbers & multiple spaces by the single space
article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
article_text = re.sub(r'\s+', ' ', article_text)

# removing special characters
fomatted_article_text = re.sub('[^a-zA-z]', ' ' , article_text)
fomatted_article_text = re.sub(r'\s+', ' ', fomatted_article_text)

# breking into the sentences
sentence_list = nltk.sent_tokenize(article_text)

# counting the frequencies of the each words
word_frequencies = {}
for word in nltk.word_tokenize(fomatted_article_text):
    if word not in set(stopwords.words('english')):
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1

# getting the wieghted frequency
max_frequency = max(word_frequencies.values())
for word in word_frequencies.keys():
    word_frequencies[word] =word_frequencies[word]/max_frequency

# calculating the scores of each sentence
sentence_scores = {}
for sent in sentence_list:
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_frequencies.keys():
            if len(sent.split(' ')) < 30:   # not taking more than 30 words
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]

# getting the summary
import heapq
sentence_summary = heapq.nlargest(7,sentence_scores, key=sentence_scores.get)
summary = ' '.join(sentence_summary)