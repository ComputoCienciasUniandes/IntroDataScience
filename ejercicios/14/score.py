import pandas as pd
import sklearn.metrics
import numpy as np

truth = pd.read_csv("test/truth_test.csv")
predict = pd.read_csv("test/predict_test.csv")

truth  = truth.sort_values('Name')
predict = predict.sort_values('Name')

f1_0 = sklearn.metrics.f1_score(truth['Target'], predict['Target'], pos_label=0)
f1_1 = sklearn.metrics.f1_score(truth['Target'], predict['Target'], pos_label=1)
print(f1_0, f1_1)
