import glob
import numpy as np

files = glob.glob("train/*.jpg")

files_test = glob.glob("test/*.jpg")
n_test = len(files_test)
predict_test = np.int_(np.random.random(n_test)/0.5)

out = open("test/predict_test.csv", "w")
out.write("Name,Target\n")
for f, p in zip(files_test, predict_test):
    print(f.split("/")[-1], p)
    out.write("{},{}\n".format(f.split("/")[-1],p))

out.close()