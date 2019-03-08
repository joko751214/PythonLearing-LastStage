import nltk.stem as ns

words = ['table', 'probably', 'wolves', 'playing', 'is',
         'dog', 'the', 'beaches', 'grounded', 'dreamt', 'envision']
# 創建詞形還原器對象
lemmatizer = ns.WordNetLemmatizer()
for word in words:
	# 利用方法lemmatize按照名詞來還原
	# n:代表名詞
    lemma = lemmatizer.lemmatize(word, 'n')
    print(lemma)
print('-' * 72)
for word in words:
	# v:代表動詞
    lemma = lemmatizer.lemmatize(word, 'v')
    print(lemma)
