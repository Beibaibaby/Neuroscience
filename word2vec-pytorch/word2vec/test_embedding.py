from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import datapath
from pathlib import Path

model = KeyedVectors.load_word2vec_format(Path.cwd() / 'model_test.txt', binary=False)

def hb(input):
    a = 1
    for i in range(20):
        print(input)
        output = model.most_similar(positive=input)
        print(output)
        if i < 6:
            a = a + 1
        else:
            a = a - 1
        input = output[a][0]


input = input("Please enter your input:\n ")
hb(input)



