import tensorflow as tf
import numpy as np



########################### PREPROCESSING
with open("tiny-shakespeare.txt") as file:
    lines = file.readlines()

#removing all strings which are just empty lines
lines = [line for line in lines if line != '\n']


#want to append lines to sequence unless the line ends in : (assuming only character intros end with semicolon)
#if it does end in : then that should begin a new sequence
#pop(0) and insert(0)


#this should split each character's turn into its own string. this assumes that the only lines to end in ':' are lines that denote a character speaking, so it's not perfect
sequences = []
index = -1
try:
    while (line := lines.pop(0)):
        if line[-2] != ':':
            sequences[index] += line
        else:
            sequences.append(line)
            index += 1
except IndexError:
    pass


########################### CREATING INPUTS
characters = set(''.join(lines))

intToChar = dict(enumerate(characters))
charToInt = {character: index for index, character in intToChar.items()}

input_sequence = []
target_sequence = []
for i in range(len(sequences)):
    input_sequence.append(sequences[i][:-1])
    target_sequence.append(sequences[i][1:])

############################ CONSTRUCT ONE-HOTS
for i in range(len(sequences)):
    input_sequence[i] = [charToInt[character] for character in input_sequence[i]]
    target_sequence[i] = [charToInt[character] for character in target_sequence[i]]

vocab_length = len(charToInt)



#tensor of form [batch, timesteps, feature]

rnn = tf.keras.layers.SimpleRNN(vocab_length)