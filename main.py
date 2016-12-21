from nltk.tokenize import word_tokenize
import pickle

file = open('indonesian_pos_tag.pickle', 'rb')
tagger = pickle.load(file)
file.close()

kalimat = 'Kota Bandung merupakan kota metropolitan terbesar di Provinsi Jawa Barat, sekaligus menjadi ibu kota provinsi tersebut'
print(tagger.tag(word_tokenize(kalimat)))
