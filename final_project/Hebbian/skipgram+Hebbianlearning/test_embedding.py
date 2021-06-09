from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import datapath
from pathlib import Path
import random
model = KeyedVectors.load_word2vec_format(Path.cwd() / 'model_test.txt', binary=False)

def hb(input):
    a = 1
    for i in range(20):
        print(input)
        output = model.most_similar(positive=input)
        print(output)
        if a > 5:
            a = int(a/1.23 )+2
        else:
            a = int(a*1.35 + 1)
        input = output[a][0]

#a = random.randint(0,9)



input = input("Please enter your input:\n ")
hb(input)



