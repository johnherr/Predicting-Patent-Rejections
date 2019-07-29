import pandas as pd

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess

import data_processing as processing

df = pd.read_pickle('data/all_art_units.pkl')

train, test = processing.split_train_test(df)

train = processing.split_filed_granted(train)
test = processing.split_filed_granted(test)

tagged_train_data = [TaggedDocument(words=simple_preprocess(claim), tags=[(str(app)+'_'+str(grnt))]) for claim, app, grnt in zip(train['claim'], train['app_id'], train['allowed'])]

model = Doc2Vec(dm=1, dbow_words=1, dm_concat=0, dm_tag_count=1,
                window=10, min_alpha=0.01, alpha=0.025, workers=8, epochs=20)
                
model.build_vocab(tagged_train_data)

model.train(tagged_train_data,
           total_examples=model.corpus_count,
           epochs=model.epochs)
           
model.save('model_all_art_units_window_10')