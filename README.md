# scikit-learn wrapper to finetune BERT


 A scikit-learn wrapper to finetune [Google's BERT](https://github.com/google-research/bert) model for text and token sequence tasks based on the [huggingface pytorch](https://github.com/huggingface/pytorch-pretrained-BERT) port.

* Includes configurable MLP as final classifier/regressor for text and text pair tasks
* Includes token sequence classifier for NER, PoS, and chunking tasks
* Includes  [**`SciBERT`**](https://github.com/allenai/scibert) and [**`BioBERT`**](https://github.com/dmis-lab/biobert) pretrained models for scientific  and biomedical domains.


Try in [Google Colab](https://colab.research.google.com/drive/1-wTNA-qYmOBdSYG7sRhIdOrxcgPpcl6L)!


## installation

requires python >= 3.5 and pytorch >= 0.4.1

```bash
git clone -b master https://github.com/charles9n/bert-sklearn
cd bert-sklearn
pip install .
```

## basic operation

**`model.fit(X,y)`**  i.e finetune **`BERT`**

* **`X`**: list, pandas dataframe, or numpy array of text, text pairs, or token lists

* **`y`** : list, pandas dataframe, or numpy array of labels/targets

```python3
from bert_sklearn import BertClassifier
from bert_sklearn import BertRegressor
from bert_sklearn import load_model

# define model
model = BertClassifier()         # text/text pair classification
# model = BertRegressor()        # text/text pair regression
# model = BertTokenClassifier()  # token sequence classification

# finetune model
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# make probabilty predictions
y_pred = model.predict_proba(X_test)

# score model on test data
model.score(X_test, y_test)

# save model to disk
savefile='/data/mymodel.bin'
model.save(savefile)

# load model from disk
new_model = load_model(savefile)

# do stuff with new model
new_model.score(X_test, y_test)

# BERT embeddings
new_model.bert_embedding("a message to embed")
new_model.bert_embedding(list_of_messages_to_embed)
```
See [demo](https://github.com/charles9n/bert-sklearn/blob/master/demo.ipynb) notebook.

## model options

```python3
# try different options...
model.bert_model = 'bert-large-uncased'
model.num_mlp_layers = 3
model.max_seq_length = 196
model.epochs = 4
model.learning_rate = 4e-5
model.gradient_accumulation_steps = 4
model.oversampler = "SMOTE"

# finetune
model.fit(X_train, y_train)

# do stuff...
model.score(X_test, y_test)

```
See [options](https://github.com/charles9n/bert-sklearn/blob/master/Options.md)


## hyperparameter tuning

```python3
from sklearn.model_selection import GridSearchCV

params = {'epochs':[3, 4], 'learning_rate':[2e-5, 3e-5, 5e-5]}

# wrap classifier in GridSearchCV
clf = GridSearchCV(BertClassifier(validation_fraction=0),
                    params,
                    scoring='accuracy',
                    verbose=True)

# fit gridsearch
clf.fit(X_train ,y_train)
```
