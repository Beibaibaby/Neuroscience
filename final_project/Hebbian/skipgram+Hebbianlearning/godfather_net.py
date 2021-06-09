import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages')

from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import datapath
from pathlib import Path

import numpy as np
import seaborn as sns
import random

model = KeyedVectors.load_word2vec_format(Path.cwd() / 'godfather_test.txt', binary=False)
Words_identities = model.index_to_key
Words_vector = model.vectors
for i in range(len(Words_vector)):
    Words_vector[i,:] = Words_vector[i,:] / max(Words_vector[i,:])
N_Words = len(Words_identities)
N_Neurons = len(Words_vector[0])

WordsWeightMatrix = np.zeros((N_Neurons, N_Neurons))

for i in range(N_Words):
    Word_vector = np.array(Words_vector[i]).reshape((N_Neurons, 1))
    Word_vector_neg = np.zeros((N_Neurons, 1))
    Word_vector_neg[Word_vector[:, 0] < 0] = Word_vector[Word_vector[:, 0] < 0]

    WordsWeightMatrix += Word_vector.dot(Word_vector.T) - Word_vector_neg.dot(Word_vector_neg.T)

AssemblyDecoder = np.zeros((N_Words, N_Neurons))
for i in range(N_Words):
    AssemblyDecoder[i, :] = np.array(Words_vector[i])


def WordsEIConnections(Words_vector, maxE=10, maxI=-7, baseE=1.8, baseI=1.1):
    N_Words = len(Words_vector)
    WordsDistances = np.zeros((N_Words, N_Words))

    for i in range(N_Words):
        Word_i = np.array(Words_vector[i])
        #         Word_i[Word_i > 0] = 1
        #         Word_i[Word_i <= 0] = 0

        for j in range(N_Words):
            Word_j = np.array(Words_vector[j])
            #             Word_j[Word_j > 0] = 1
            #             Word_j[Word_j <= 0] = 0

            difference = (Word_i - Word_j).reshape((len(Word_i), 1))
            distance = difference.T.dot(difference)

            WordsDistances[i, j] = distance

    EConnection = np.zeros((N_Words, N_Words))
    IConnection = np.zeros((N_Words, N_Words))

    for i in range(N_Words):
        EInputIndex = WordsDistances[i, :].argsort()[0:10]
        IInputIndex = WordsDistances[i, :].argsort()[0:200]

        EInputDistances = WordsDistances[i, :][EInputIndex]
        IInputDistances = WordsDistances[i, :][IInputIndex]

        EInputStrength = np.power(baseE, -EInputDistances) * maxE
        IInputStrength = np.power(baseI, -IInputDistances) * maxI

        EConnection[i, EInputIndex] = EInputStrength
        IConnection[i, IInputIndex] = IInputStrength

    return EConnection, IConnection, WordsDistances


def CellAssembly(NeuronConnection, AssemblyDecoder, Words_vector, Input, turn=5, sigma=1):
    N_Neurons = NeuronConnection.shape[0]
    N_Words = len(Words_vector)

    for itr in range(turn):

        Output = NeuronConnection.dot(Input)

        AssociatedWordsWeights = AssemblyDecoder.dot(Output)
        AssociatedWordsWeights = AssociatedWordsWeights / max(AssociatedWordsWeights)
        AssociatedWordsWeights = AssociatedWordsWeights + np.random.normal(0, sigma, N_Words).reshape((N_Words, 1))
        #         AssociatedWordsWeights[AssociatedWordsWeights < 0] = 0

        Input = np.zeros((N_Neurons, 1))
        for i in range(N_Words):
            Word = np.array(Words_vector[i]).reshape((N_Neurons, 1))
            Weight = AssociatedWordsWeights[i]
            Input = Input + Word * Weight

    return AssociatedWordsWeights


def WinnerTakeAll(WordsStrength, EConnection, IConnection, turn=4, sigma=1):
    global Words_identities

    Winners = []
    N_Words = len(WordsStrength)

    for t in range(turn):

        WordsStrength = WordsStrength / max(WordsStrength)
        WordsStrength = WordsStrength + np.random.normal(0, sigma, N_Words).reshape((N_Words, 1))
        #         WordsStrength[WordsStrength < 0] = 0

        EInputs = np.zeros((N_Words, 1))
        IInputs = np.zeros((N_Words, 1))

        for i in range(N_Words):
            EInput = sum(WordsStrength[i, 0] * EConnection[i, :])
            IInput = sum(WordsStrength[i, 0] * IConnection[i, :])

            EInputs[i, 0] = EInput
            IInputs[i, 0] = IInput

        WordsStrength = WordsStrength + EInputs + IInputs
        Winners.append(np.array(Words_identities)[WordsStrength[:, 0] > 0].tolist())

        if t == turn - 1:
            WordsStrengthLast = WordsStrength / max(WordsStrength)
    #             WordsStrengthLast[WordsStrengthLast[:,0]<0] = 0
    #             WordsStrengthLast[WordsStrengthLast[:,0]>0] = 1

    return Winners, WordsStrengthLast


def HeteroAssociationChain(NeuronConnection, AssemblyDecoder, Words_vector, ChainLen=1, WordsNum=1, sigma=1,
                           InputWords=None):
    global Words_identities

    N_Words = len(Words_vector)
    N_Neurons = len(Words_vector[0])

    if not InputWords:
        InputWords_index = random.sample(range(N_Words), WordsNum)
        Input = np.zeros((N_Neurons, 1))
        for i in InputWords_index:
            InputWord = np.array(Words_vector[i]).reshape((N_Neurons, 1))
            Input = Input + InputWord
    else:
        InputWords_index = [Words_identities.index(i) for i in InputWords]
        Input = np.zeros((N_Neurons, 1))
        for i in InputWords_index:
            InputWord = np.array(Words_vector[i]).reshape((N_Neurons, 1))
            Input = Input + InputWord

    HeteroAssociationResult = []
    HeteroAssociationResult.append(np.array(Words_identities)[InputWords_index])

    for Chain in range(ChainLen):

        E, I, D = WordsEIConnections(Words_vector)
        AssociatedWordsStrength = CellAssembly(NeuronConnection, AssemblyDecoder, Words_vector, Input, sigma=sigma,
                                               turn=2)
        Winners, WordsStrengthLast = WinnerTakeAll(AssociatedWordsStrength, E, I, turn=5, sigma=sigma)
        WinnersLast = Winners[-1]
        HeteroAssociationResult.append(WinnersLast)

        Input = np.zeros((N_Neurons, 1))
        for i in range(N_Words):
            InputWord = np.array(Words_vector[i]).reshape((N_Neurons, 1)) * WordsStrengthLast[i]
            Input = Input + InputWord

    return HeteroAssociationResult

# %%


HeteroChain1 = HeteroAssociationChain(WordsWeightMatrix, AssemblyDecoder, Words_vector, sigma=0.1,
                                      InputWords=['Corleone', 'Michael', 'gun'])

for i in HeteroChain1:
    print(i, '\n')

for i in range(10):

    HeteroChain = HeteroAssociationChain(WordsWeightMatrix, AssemblyDecoder, Words_vector, sigma=0.1, InputWords=None)

    for j in HeteroChain:
        print(j, '\n')

    print('############')