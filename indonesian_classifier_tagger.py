from nltk.tag.sequential import ClassifierBasedPOSTagger
import pickle

datas = open('Indonesian_Manually_Tagged_Corpus.tsv', 'r').read()
datas = datas.split('\n\n')

train_sents = []

for data in datas:
	train_sents.append(list(tuple(i.split('\t')) for i in data.split('\n')))

tagger = ClassifierBasedPOSTagger(train=train_sents)
tagger_files = open("indonesian_classifier_pos_tag.pickle", "wb")
pickle.dump(tagger, tagger_files)
tagger_files.close()
