import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score

import data_processing as processing

df = pd.read_pickle('data/art_unit_2100.pkl')
train, test = processing.split_train_test(df)
train = processing.infer_claim_vectors(train)
test = processing.infer_claim_vectors(test)

combined_df = pd.concat([train,test]).reset_index(drop=True)
combined_df.to_pickle('data/art_unit_2100_vectorized.pkl')

X_train = list(train['claim_vec'].values)
y_train = train['allowed'].values
X_test = list(test['claim_vec'].values)
y_test = test['allowed'].values

rf = RandomForestClassifier(n_estimators =10)
rf.fit(X_train, y_train)

y_predict = rf.predict(X_test)
results = np.array([y1 == y2 for y1, y2 in zip(y_predict,y_test)])

print('Accuracy = ', accuracy_score(y_test, y_predict), '%')
print('Precision = ', precision_score(y_test, y_predict), '%')
print('Recall = ', recall_score(y_test, y_predict), '%')
