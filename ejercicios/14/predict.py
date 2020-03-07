import glob
import numpy as np
import matplotlib.pyplot as plt
import sklearn.discriminant_analysis

def load_train():
    n_files = 100
    input_data = []
    target = np.ones(n_files)
    for i in range(n_files):
        filename = "train/{}.jpg".format(i+1)
        target[i] = i%2 
        input_data.append(plt.imread(filename).flatten())
    input_data = np.array(input_data)
    print(np.shape(input_data), np.shape(target))
    return input_data, target

def train():
    x, y  = load_train()
    clf = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    clf.fit(x,y)
    return clf

def predict():
    clf = train()
    
    files_test = glob.glob("test/*.jpg")
    n_test = len(files_test)
    test_data = []
    for file_test in files_test:
        test_data.append(plt.imread(file_test).flatten())
    test_data = np.array(test_data)

    predict_test = clf.predict(test_data)
    out = open("test/predict_test.csv", "w")
    out.write("Name,Target\n")
    for f, p in zip(files_test, predict_test):
        out.write("{},{}\n".format(f.split("/")[-1],p))
    out.close()

    # visualiza el vector discriminante
    plt.figure()
    discr = clf.coef_[0].reshape(100,100,3)
    discr = discr-discr.min()
    discr = np.int_(255*(discr/(discr.max() - discr.min())))
    print(discr.min(), discr.max())
    plt.imshow(discr)
    plt.savefig("discriminant.png")

train()
predict()
