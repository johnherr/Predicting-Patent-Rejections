from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Wikipedia text dumps
# https://dumps.wikimedia.org/enwiki/
wiki = WikiCorpus("enwiki-20190720-pages-articles.xml.bz2")

class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield TaggedDocument(content, [title])

documents = TaggedWikiDocument(wiki)

cores = multiprocessing.cpu_count()

model = Doc2Vec(dm=0, dbow_words=1, vector_size=300, window=10, min_count=50, epochs=5, workers=cores, alpha=.1, min_alpha=.0025)

model.build_vocab(documents)

print('Finsihed building vocab')

# model.train(documents, total_examples=model.corpus_count, epochs=5)
# print('Finished Training')

model.save('wikipedia_model')

print('Model Saved')
