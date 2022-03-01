from re import search

letters = ['q', 'u', 'i', 'p', 'h', 'j', 'k', 'z', 'x', 'v', 'b', 'm']
words = []

for i in letters:
    for j in letters:
        for k in letters:
            words.append(i + 'i' + j + k + 'd')

for word in  words:
    if "vj" in word:
        del words[words.index(word)]
    elif "bx" in word:
        del words[words.index(word)]
    elif "xx" in word:
        del words[words.index(word)]
    else:
        print(word)

print(len(words))