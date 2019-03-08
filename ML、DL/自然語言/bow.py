import nltk.tokenize as tk
# 自然語言特性識別器模塊
import sklearn.feature_extraction.text as ft

doc = 'The brown dog is running. The black dog is in the black room. Running in the room is forbidden.'
print(doc)
print('-' * 72)
sentences = tk.sent_tokenize(doc)
print(sentences)
print('-' * 72)
# 矢量化統計器
cv = ft.CountVectorizer()
# 建立詞頻矩陣
tfmat = cv.fit_transform(sentences).toarray()
# 詞表:獲取句子中的所有特徵值(所有單詞,不會重複)
words = cv.get_feature_names()
# 將詞頻矩陣和詞表做結合即為"詞袋模型"
print(words)
print(tfmat)

