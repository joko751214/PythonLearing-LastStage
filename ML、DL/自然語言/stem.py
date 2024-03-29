import nltk.stem.porter as pt
import nltk.stem.lancaster as lc
import nltk.stem.snowball as sb

words = ['table', 'probably', 'wolves', 'playing', 'is',
         'dog', 'the', 'beaches', 'grounded', 'dreamt', 'envision']
stemmer = pt.PorterStemmer()
for word in words:
	# 用stem方法來提取詞干
    stem = stemmer.stem(word)
    print(stem)
print('-' * 72)
stemmer = lc.LancasterStemmer()
for word in words:
    stem = stemmer.stem(word)
    print(stem)
print('-' * 72)
stemmer = sb.SnowballStemmer('english')
for word in words:
    stem = stemmer.stem(word)
    print(stem)
