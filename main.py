from gensim.models import Word2Vec
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format("model.txt", binary=False)


print(model.most_similar(positive=['обрезать_VERB', 'волосы_NOUN']))

print(model.most_similar(positive=['лук_NOUN']))

print(model.most_similar(positive=['лук_NOUN', 'стрельба_NOUN']))

print(model.most_similar(positive=['лук_NOUN', 'еда_NOUN']))

print(model.most_similar(positive=['печь_NOUN', 'тепло_NOUN']))
