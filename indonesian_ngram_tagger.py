from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger, DefaultTagger
import pickle

datas = open('Indonesian_Manually_Tagged_Corpus.tsv', 'r').read()
datas = datas.split('\n\n')

train_sents = []

for data in datas:
	train_sents.append(list(tuple(i.split('\t')) for i in data.split('\n')))

def backoff_tagger(train_sents, tagger_classes, backoff=None):
	for cls in tagger_classes:
		backoff = cls(train_sents, backoff=backoff)

	return backoff


backoff = DefaultTagger('NN')

tagger = backoff_tagger(train_sents, [UnigramTagger, BigramTagger, TrigramTagger], backoff=backoff)

tagger_files = open("indonesian_pos_tag.pickle", "wb")
pickle.dump(tagger, tagger_files)
tagger_files.close()