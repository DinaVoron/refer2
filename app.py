import re

import numpy
from corus import load_lenta
import pandas as pd
from pymorphy2 import MorphAnalyzer
import nltk
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import io
import string
import numpy as np

morph = MorphAnalyzer()


def prepare(text):
    lines = []
    line = ""
    # Возьмем текст
    for char in text:
        if char == '':
            break
        elif char == '.' or char == '!' or char == '?':
            line += char
            line = line.rstrip()
            lines.append(line)
            line = ""
        else:
            if char != '\n':
                line += char
    return lines


def fill_with_sum(lines, model):
    dic = []
    for line in lines:
        sum = np.zeros(300)
        new_line = line.lower()
        new_line = new_line.rstrip()
        new_line = new_line.translate(str.maketrans('', '', string.punctuation))
        words = new_line.split(" ")
        for word in words:
            if word != '':
                if morph.parse(word)[0].normal_form and morph.parse(word)[0].tag.POS:
                    word = morph.parse(word)[0].normal_form + "_" + morph.parse(word)[0].tag.POS
                    if word in model:
                        word_vec = np.array(model[word])
                        sum += word_vec
        dic.append(sum)
    return dic


def get_distance(sent_sum, text_sum):
    dis = {}
    i = 0
    for sent in sent_sum:
        cur_dis = np.linalg.norm(text_sum-sent)
        dis[i] = cur_dis
        i += 1
    return dis


def refer(lines, n):
    model = KeyedVectors.load_word2vec_format("model.txt", binary=False)
    sent_sum = fill_with_sum(lines, model)

    # Посчитаем вектор всего текста
    text_sum = 0
    for vec in sent_sum:
        text_sum += vec

    # Посчитаем косинусное расстояние между текстом и каждым предложением
    dist = get_distance(sent_sum, text_sum)

    # Отсортируем массив по возрастанию расстояния
    dist = sorted(dist.items(), key=lambda x: x[1])

    new_size = int(len(lines)*n)

    sentences_to_take = []

    for i in range(new_size):
        sentences_to_take.append(dist[i][0])

    sentences_to_take.sort()

    new_text = []

    for sentence in sentences_to_take:
        new_text.append(lines[sentence])

    return(' '.join(new_text))



# Считываем текст для сжатия
text_file = open('about_max.txt', 'r', encoding='utf-8')
text = text_file.read()
lines = prepare(text)
# Процент сжатия
n = 0.3
print(refer(lines, n))










