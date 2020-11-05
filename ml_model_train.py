# -*- coding: utf-8 -*-
# @Time : 2020/11/4 17:12
# @Author : Jclian91
# @File : ml_model_train.py
# @Place : Yangpu, Shanghai
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import joblib
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from albert_zh.extract_feature import BertVector

# 读取文件
train_df = pd.read_csv("data/sougou_mini/sougou_train.csv").dropna()
test_df = pd.read_csv("data/sougou_mini/sougou_test.csv").dropna()

# 利用ALBERT提取向量特征
all_labels = list(train_df.label.unique())
with open("labels.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(all_labels, ensure_ascii=False, indent=2))
y_train = np.array([all_labels.index(_) for _ in train_df['label'].tolist()])
y_test = np.array([all_labels.index(_) for _ in test_df['label'].tolist()])

bert_model = BertVector(pooling_strategy="REDUCE_MEAN", max_seq_len=512)
f = lambda text: bert_model.encode([text])["encodes"][0]
print('begin encoding')
x_train = np.empty(shape=(train_df.shape[0], 312))
i = 0
for content in tqdm(train_df["content"].tolist()):
    x_train[i, :] = f(content)
    i += 1

x_test = np.empty(shape=(test_df.shape[0], 312))
i = 0
for content in tqdm(test_df["content"].tolist()):
    x_test[i, :] = f(content)
    i += 1

print('end encoding.')

# Logistic Regression
lr = LR(random_state=123)
lr.fit(x_train, y_train)


y_pred = lr.predict(x_test)
print("Logistic Regression Model")
print("混淆矩阵", confusion_matrix(y_true=y_test, y_pred=y_pred))
print("正确率：", accuracy_score(y_test, y_pred))
print(classification_report(y_true=y_test, y_pred=y_pred, digits=4, target_names=all_labels))

# 保存模型
joblib.dump(lr, "lr.model")

# Naive Bayes Model
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
print("\nNaive Bayes Model")
print("混淆矩阵", confusion_matrix(y_true=y_test, y_pred=y_pred))
print("正确率：", accuracy_score(y_test, y_pred))
print(classification_report(y_true=y_test, y_pred=y_pred, digits=4, target_names=all_labels))

# SVM model
svc = SVC(kernel="rbf")
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print("\nSVM Model")
print("混淆矩阵", confusion_matrix(y_true=y_test, y_pred=y_pred))
print("正确率：", accuracy_score(y_test, y_pred))
print(classification_report(y_true=y_test, y_pred=y_pred, digits=4, target_names=all_labels))

joblib.dump(svc, "svc.model")
