from bert_sklearn import BertClassifier, LeanBert
import pandas as pd
import numpy as np
import time, joblib

def data_from_df(df):
    X = df["message"].str.lower().tolist()
    y = df["isOOA"].tolist()

    return X, y

def test_bert():
    X, y = data_from_df(pd.read_csv("~/Documents/OOA/v1.1/Data/train.csv")[:100])

    model = BertClassifier()
    model.restore_finetuned_model("/Users/oduwaedoosagie/Desktop/berts/baby_bert/v2/baby_bert2.bin")

    print(model.bert_embedding(["i like pie"]))
    print(model.bert_embedding(["i like pie"]).shape)
    #print(model.bert_embedding(["i like pie"])[0])


def test_lean_bert():
    X, y = data_from_df(pd.read_csv("~/Documents/OOA/v1.1/Data/train.csv")[:100])
    model = LeanBert()
    model.restore_model("bert_test.bin")

    import time
    reps = 100
    total = 0
    a, b, c = model.prepare_message(X[0])
    for i in range(reps):
        t0 = time.time()
        #model.predict_proba(X[:1])
        model.bert(a, b, c)
        t1 = time.time()
        total += t1-t0
    total = total/reps
    print(f"TIME TAKEN: {total}")

test_bert()
# X, y = data_from_df(pd.read_csv("~/Documents/OOA/v1.1/Data/train.csv")[:100])
#
# model = BertClassifier()
# model.restore_finetuned_model("/Users/oduwaedoosagie/Desktop/berts/baby_bert/v2/baby_bert2.bin")
# print(model.predict(["paypal is good"]))


# from bert_sklearn.config import model2config
# config = model2config(model)
# with open("config_test.txt", "w") as f:
#     f.write(str(config))
#
# model.restore_finetuned_model("bert_test.bin")
# model.fit(X, y, False)

# X = ["cake"]*100
# y = ([1]+[0]*98)+[1]


#model = model.fit(X, y)