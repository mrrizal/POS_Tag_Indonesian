from nltk.tag import tnt, DefaultTagger
import pickle

datas = open('Indonesian_Manually_Tagged_Corpus.tsv', 'r').read()
datas = datas.split('\n\n')

train_sents = []

for data in datas:
	train_sents.append(list(tuple(i.split('\t')) for i in data.split('\n')))

unk = DefaultTagger('NN')
tnt_tagger = tnt.TnT(unk=unk, Trained=True)
tnt_tagger.train(train_sents)
tagger_file = open("indonesian_tnt_pos_tag.pickle","wb")
pickle.dump(tnt_tagger, tagger_file)
tagger_file.close()
