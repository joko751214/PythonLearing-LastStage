import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import nltk.tokenize as tk
import nltk.corpus as nc
import nltk.stem.snowball as sb
# 主題建模模塊
import gensim.models.ldamodel as gm
import gensim.corpora as gc

doc = []
with open('topic.txt', 'r') as f:
    for line in f.readlines():
        doc.append(line[:-1])
tokenizer = tk.RegexpTokenizer(r'\w+')
stopwords = nc.stopwords.words('english')
stemmer = sb.SnowballStemmer('english')
lines_tokens = []
for line in doc:
    tokens = tokenizer.tokenize(line.lower())
    line_tokens = []
    for token in tokens:
        if token not in stopwords:
            token = stemmer.stem(token)
            line_tokens.append(token)
    lines_tokens.append(line_tokens)
dic = gc.Dictionary(lines_tokens)
bow = []
for line_tokens in lines_tokens:
    row = dic.doc2bow(line_tokens)
    bow.append(row)
print(bow)
n_topics = 2
# Latent Dirichlet Allocation, LDA
# 隐含狄利克雷分布
# passes:通過文獻
model = gm.LdaModel(bow, num_topics=n_topics, id2word=dic,
                    passes=25)
# 最多放置的單詞數:num_words
topics = model.print_topics(num_topics=n_topics, num_words=4)
print(topics)
