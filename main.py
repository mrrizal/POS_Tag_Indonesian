from nltk.tokenize import word_tokenize
import pickle

file = open('indonesian_pos_tag.pickle', 'rb')
tagger = pickle.load(file)
file.close()

print(tagger.tag(word_tokenize('saya akan mengerjakan tugas dengan rajin')))
