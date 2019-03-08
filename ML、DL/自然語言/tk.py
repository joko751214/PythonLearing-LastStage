import nltk.tokenize as tk
doc = "Are you curious about tokenization? Let's see how it works! We need to analyze a couple of sentences with punctuations to see it in action."
print(doc)
print('-' * 72)
# 以句子為單位的拆分器
tokens = tk.sent_tokenize(doc)
for token in tokens:
    print(token)
print('-' * 72)
# 以詞為單位的拆分器
tokens = tk.word_tokenize(doc)
for token in tokens:
    print(token)
print('-' * 72)
# 以詞跟標點符號為單位的拆分器
tokenizer = tk.WordPunctTokenizer()
tokens = tokenizer.tokenize(doc)
for token in tokens:
    print(token)
