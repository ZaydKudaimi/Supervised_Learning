import time

import numpy as np
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from yellowbrick.model_selection import LearningCurve, ValidationCurve
from sklearn.datasets import load_digits, fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from matplotlib import pyplot as plt

def main():

    print('DT1')
    DTset1()
    print('DT2')
    DTset2()
    print('NN1')
    NNset1()
    print('NN2')
    NNset2()
    print('KNN1')
    KNNset1()
    print('KNN2')
    KNNset2()
    print('Boost1')
    BoostSet1()
    print('Boost2')
    BoostSet2()
    print('SVM1')
    SVM1()
    print('SVM2')
    SVM2()

def SVM1():
    data = pd.read_csv('./adult/adult.data')
    X = data.iloc[:, :14]
    X = OneHotEncoder(handle_unknown='ignore').fit_transform(X)
    y = data.iloc[:, [-1]]
    # # print(X)
    # # print(y)

    sc = make_scorer(modf1)
    # code modified from https://www.geeksforgeeks.org/bar-plot-in-matplotlib/#
    d = {}
    t = {}
    plt.title('validation chart for SVC')
    plt.xlabel('kernel')
    plt.ylabel('score')
    # code modified from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, np.ravel(y), test_size=0.2, random_state=0)
    # end
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        pipe = make_pipeline(SVC(random_state=0, kernel=kernel))
        res = cross_val_score(pipe, X, np.ravel(y), scoring=sc, n_jobs=20)
        print(res.sum()/5)
        d[kernel] = res.sum()/5
        pipe.fit(X_train, y_train)
        t[kernel] = f1_score(y_test, pipe.predict(X_test), pos_label=" >50K")
    b = np.arange(len(d))
    b1 = [x + .3 for x in b]
    plt.bar(b, d.values(), color='g', label='cross validation score', width=.3)
    plt.bar(b1, t.values(), color='b', label='training score', width=.3)
    plt.xticks([a + .3 for a in range(len(d))], ['linear', 'poly', 'rbf', 'sigmoid'])
    plt.ylim(.5, .9)
    plt.legend()
    plt.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(SVC(random_state=0), cv=StratifiedKFold(n_splits=5), n_jobs=20, scoring=sc)
    vis.fit(X, np.ravel(y))
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis = ValidationCurve(SVC(random_state=0, kernel='linear'), param_name="C",
    param_range=np.arange(0,31,3),  cv=StratifiedKFold(n_splits=5), scoring=sc, n_jobs=20)

    vis.fit(X, np.ravel(y))
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(SVC(random_state=0, kernel='linear', C=30), cv=StratifiedKFold(n_splits=5), scoring=sc, n_jobs=20)
    vis.fit(X, np.ravel(y))
    vis.show()
    # end
    data = pd.read_csv('./adult/adult.data',
                       names=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"])
    X = data.iloc[:, :14]

    y = data.iloc[:, [-1]]

    testdata = pd.read_csv('./adult/adult.test',
                           names=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"])
    X_test = testdata.iloc[:, :14]
    X_test = OneHotEncoder(handle_unknown='ignore').fit(pd.concat((X_test, X))).transform(X_test)
    y_test = testdata.iloc[:, [-1]]

    v = SVC(random_state=0, kernel='linear', C=30)
    X = OneHotEncoder(handle_unknown='ignore').fit(pd.concat((testdata.iloc[:, :14], X))).transform(X)

    strt = time.time()
    v.fit(X, y)
    ned = time.time()
    print('training Time: ' + str(ned - strt) + 's')

    strt = time.time()
    ans = v.predict(X_test)
    ned = time.time()
    print('inference Time: ' + str(ned - strt) + 's')

    print('score:')
    print(f1_score(y_test, ans, pos_label=" >50K"))
    print(' ')

def SVM2():
    # code from https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas"
    )
    # end
    # code modified from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)
    # end
    # print(X)
    # print(y)
    # code modified from https://www.geeksforgeeks.org/bar-plot-in-matplotlib/#
    d = {}
    t = {}
    plt.title('validation chart for SVC')
    plt.xlabel('kernel')
    plt.ylabel('score')
    # code modified from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
    X_trai, X_tes, y_trai, y_tes = train_test_split(
        X_train, y_train, test_size=0.2, random_state=0)
    # end
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        pipe = make_pipeline(SVC(random_state=0, kernel=kernel, cache_size=1800))
        res = cross_val_score(pipe, X_train, y_train, scoring='f1_micro', n_jobs=30)
        print(res.sum()/5)
        d[kernel] = res.sum()/5
        pipe.fit(X_trai, y_trai)

        t[kernel] = f1_score(y_tes, pipe.predict(X_tes), average='micro')
    b = np.arange(len(d))
    b1 = [x + .3 for x in b]
    plt.bar(b, d.values(), color='g', label='cross validation score', width=.3)
    plt.bar(b1, t.values(), color='b', label='training score', width=.3)
    plt.xticks([a + .3 for a in range(len(d))], ['linear', 'poly', 'rbf', 'sigmoid'])
    plt.ylim(.7, 1)
    plt.legend()
    plt.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(SVC(random_state=0), cv=StratifiedKFold(n_splits=5), n_jobs=20, scoring='f1_micro')
    vis.fit(X_train, y_train)
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis = ValidationCurve(SVC(random_state=0), param_name="C",
    param_range=np.arange(0,31,3),  cv=StratifiedKFold(n_splits=5), scoring='f1_micro', n_jobs=20)

    vis.fit(X_train, y_train)
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(SVC(random_state=0, C=12), cv=StratifiedKFold(n_splits=5), scoring='f1_micro', n_jobs=20)
    vis.fit(X_train, y_train)
    vis.show()
    # end

    v = SVC(random_state=0, C=12)

    strt = time.time()
    v.fit(X_train, y_train)
    ned = time.time()
    print('training Time: ' + str(ned - strt) + 's')

    strt = time.time()
    ans = v.predict(X_test)
    ned = time.time()
    print('inference Time: ' + str(ned - strt) + 's')

    print('score:')
    print(f1_score(y_test, ans, average='micro'))
    print(' ')

def BoostSet1():

    data = pd.read_csv('./adult/adult.data')
    X = data.iloc[:, :14]
    X = OneHotEncoder(handle_unknown='ignore').fit_transform(X)
    y = data.iloc[:, [-1]]
    # print(X)
    # print(y)

    sc = make_scorer(modf1)
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis =  ValidationCurve(AdaBoostClassifier(random_state=0), cv=StratifiedKFold(n_splits=5), scoring=sc, n_jobs=20, param_name='n_estimators', param_range=np.arange(0,405,20))
    vis.fit(X, np.ravel(y))
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(AdaBoostClassifier(random_state=0), cv=StratifiedKFold(n_splits=5), n_jobs=10, scoring=sc)
    vis.fit(X, np.ravel(y))
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis = ValidationCurve(AdaBoostClassifier(random_state=0, n_estimators=300), param_name="learning_rate",
    param_range=np.arange(0,11),  cv=StratifiedKFold(n_splits=5), scoring=sc, n_jobs=20)

    vis.fit(X, np.ravel(y))
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(AdaBoostClassifier(random_state=0, n_estimators=300), cv=StratifiedKFold(n_splits=5), scoring=sc, n_jobs=20)
    vis.fit(X, np.ravel(y))
    vis.show()
    # end

    data = pd.read_csv('./adult/adult.data',
                       names=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"])
    X = data.iloc[:, :14]

    y = data.iloc[:, [-1]]

    testdata = pd.read_csv('./adult/adult.test', names=["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"])
    X_test = testdata.iloc[:, :14]
    X_test = OneHotEncoder(handle_unknown='ignore').fit(pd.concat((X_test,X))).transform(X_test)
    y_test = testdata.iloc[:, [-1]]

    v = AdaBoostClassifier(random_state=0, n_estimators=300)
    X = OneHotEncoder(handle_unknown='ignore').fit(pd.concat((testdata.iloc[:, :14],X))).transform(X)

    strt = time.time()
    v.fit(X, y)
    ned = time.time()
    print('training Time: ' + str(ned-strt) + 's')

    strt = time.time()
    ans = v.predict(X_test)
    ned = time.time()
    print('inference Time: ' + str(ned-strt) + 's')

    print('score:')
    print(f1_score(y_test, ans, pos_label=" >50K"))
    print(' ')

def BoostSet2():
    # code from https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas"
    )
    # end
    # code modified from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    # end
    # print(X)
    # print(y)
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis =  ValidationCurve(AdaBoostClassifier(random_state=0), cv=StratifiedKFold(n_splits=5), scoring='f1_micro', n_jobs=20, param_name='n_estimators', param_range=np.arange(0,405,20))
    vis.fit(X_train, y_train)
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(AdaBoostClassifier(random_state=0), cv=StratifiedKFold(n_splits=5), n_jobs=20, scoring='f1_micro')
    vis.fit(X_train, y_train)
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis = ValidationCurve(AdaBoostClassifier(random_state=0, n_estimators=80), param_name="learning_rate",
    param_range=np.arange(0,11),  cv=StratifiedKFold(n_splits=5), scoring='f1_micro', n_jobs=20)

    vis.fit(X_train, y_train)
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(AdaBoostClassifier(random_state=0, n_estimators=80), cv=StratifiedKFold(n_splits=5), scoring='f1_micro', n_jobs=20)
    vis.fit(X_train, y_train)
    vis.show()
    # end

    v = AdaBoostClassifier(random_state=0, n_estimators=80)

    strt = time.time()
    v.fit(X_train, y_train)
    ned = time.time()
    print('training Time: ' + str(ned - strt) + 's')

    strt = time.time()
    ans = v.predict(X_test)
    ned = time.time()
    print('inference Time: ' + str(ned - strt) + 's')

    print('score:')
    print(f1_score(y_test, ans, average='micro'))
    print(' ')
    
def KNNset1():

    data = pd.read_csv('./adult/adult.data')
    X = data.iloc[:, :14]
    X = OneHotEncoder(handle_unknown='ignore').fit_transform(X)
    y = data.iloc[:, [-1]]
    # print(X)
    # print(y)


    sc = make_scorer(modf1)

    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis =  ValidationCurve(KNeighborsClassifier(), cv=StratifiedKFold(n_splits=5), scoring=sc, n_jobs=10, param_name='n_neighbors', param_range=np.arange(0,30))
    vis.fit(X, np.ravel(y))
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(KNeighborsClassifier(), cv=StratifiedKFold(n_splits=5), n_jobs=10, scoring=sc)
    vis.fit(X, np.ravel(y))
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis = ValidationCurve(KNeighborsClassifier(n_neighbors=7, n_jobs=10), param_name="weights",
    param_range=['uniform','distance'],  cv=StratifiedKFold(n_splits=5), scoring=sc)

    vis.fit(X, np.ravel(y))
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(KNeighborsClassifier(n_neighbors=7, n_jobs=10, weights='distance'), cv=StratifiedKFold(n_splits=5), scoring=sc)
    vis.fit(X, np.ravel(y))
    vis.show()
    # end

    data = pd.read_csv('./adult/adult.data',
                       names=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"])
    X = data.iloc[:, :14]

    y = data.iloc[:, [-1]]

    testdata = pd.read_csv('./adult/adult.test', names=["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"])
    X_test = testdata.iloc[:, :14]
    X_test = OneHotEncoder(handle_unknown='ignore').fit(pd.concat((X_test,X))).transform(X_test)
    y_test = testdata.iloc[:, [-1]]

    v = KNeighborsClassifier(n_neighbors=7, n_jobs=10, weights='distance')
    X = OneHotEncoder(handle_unknown='ignore').fit(pd.concat((testdata.iloc[:, :14],X))).transform(X)

    strt = time.time()
    v.fit(X, y)
    ned = time.time()
    print('training Time: ' + str(ned-strt) + 's')

    strt = time.time()
    ans = v.predict(X_test)
    ned = time.time()
    print('inference Time: ' + str(ned-strt) + 's')

    print('score:')
    print(f1_score(y_test, ans, pos_label=" >50K"))
    print(' ')

def KNNset2():
    # code from https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas"
    )
    # end
    # code modified from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    # end
    # print(X)
    # print(y)
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis =  ValidationCurve(KNeighborsClassifier(n_jobs=10), cv=StratifiedKFold(n_splits=5), scoring='f1_micro', param_name='n_neighbors', param_range=np.arange(0,30))
    vis.fit(X_train, y_train)
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(KNeighborsClassifier(n_jobs=10), cv=StratifiedKFold(n_splits=5), scoring='f1_micro')
    vis.fit(X_train, y_train)
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis = ValidationCurve(KNeighborsClassifier(n_neighbors=1, n_jobs=10), param_name="weights",
    param_range=['uniform','distance'],  cv=StratifiedKFold(n_splits=5), scoring='f1_micro')

    vis.fit(X_train, y_train)
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(KNeighborsClassifier(n_neighbors=1, n_jobs=10, weights='distance'), cv=StratifiedKFold(n_splits=5), scoring='f1_micro')
    vis.fit(X_train, y_train)
    vis.show()
    # end

    v = KNeighborsClassifier(n_neighbors=1, n_jobs=10, weights='distance')

    strt = time.time()
    v.fit(X_train, y_train)
    ned = time.time()
    print('training Time: ' + str(ned - strt) + 's')

    strt = time.time()
    ans = v.predict(X_test)
    ned = time.time()
    print('inference Time: ' + str(ned - strt) + 's')

    print('score:')
    print(f1_score(y_test, ans, average='micro'))
    print(' ')

def NNset1():

    data = pd.read_csv('./adult/adult.data')
    X = data.iloc[:, :14]
    X = OneHotEncoder(handle_unknown='ignore').fit_transform(X)
    y = data.iloc[:, [-1]]
    # print(X)
    # print(y)


    sc = make_scorer(modf1)

    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis =  ValidationCurve(MLPClassifier(random_state=0, batch_size=50), cv=StratifiedKFold(n_splits=5), scoring=sc, n_jobs=10, param_name='hidden_layer_sizes', param_range=np.arange(0,30,5))
    vis.fit(X, np.ravel(y))
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis =  ValidationCurve(MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(5,5)), cv=StratifiedKFold(n_splits=5), scoring=sc, n_jobs=10, param_name='learning_rate_init', param_range=np.arange(0, 0.023, 0.001))
    vis.fit(X, np.ravel(y))
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(5,5), learning_rate_init=.008), cv=StratifiedKFold(n_splits=5), n_jobs=10, scoring=sc)
    vis.fit(X, np.ravel(y))
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis = ValidationCurve(MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(5,5), learning_rate_init=.008), param_name="max_iter",
    param_range=np.arange(0, 220, 20),  cv=StratifiedKFold(n_splits=5), scoring=sc)

    vis.fit(X, np.ravel(y))
    vis.show()
    # end

    data = pd.read_csv('./adult/adult.data',
                       names=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"])
    X = data.iloc[:, :14]

    y = data.iloc[:, [-1]]

    testdata = pd.read_csv('./adult/adult.test', names=["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"])
    X_test = testdata.iloc[:, :14]
    X_test = OneHotEncoder(handle_unknown='ignore').fit(pd.concat((X_test,X))).transform(X_test)
    y_test = testdata.iloc[:, [-1]]

    v = MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(5,5), learning_rate_init=.008)
    X = OneHotEncoder(handle_unknown='ignore').fit(pd.concat((testdata.iloc[:, :14],X))).transform(X)

    strt = time.time()
    v.fit(X, y)
    ned = time.time()
    print('training Time: ' + str(ned-strt) + 's')

    strt = time.time()
    ans = v.predict(X_test)
    ned = time.time()
    print('inference Time: ' + str(ned-strt) + 's')

    print('score:')
    print(f1_score(y_test, ans, pos_label=" >50K"))
    print(' ')

def NNset2():
    # code from https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas"
    )
    # end
    # code modified from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    # end
    # print(X)
    # print(y)

    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis = ValidationCurve(MLPClassifier(random_state=0, batch_size=50), cv=StratifiedKFold(n_splits=5), scoring='f1_micro', n_jobs=10, param_name='hidden_layer_sizes', param_range=np.arange(0,100,5))
    vis.fit(X_train, np.ravel(y_train))
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis = ValidationCurve(MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(95,95)), cv=StratifiedKFold(n_splits=5),
                          scoring='f1_micro', n_jobs=10, param_name='learning_rate_init',
                          param_range=np.arange(0, 0.021, 0.001))
    vis.fit(X_train, np.ravel(y_train))
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(95,95)), cv=StratifiedKFold(n_splits=5), n_jobs=10, scoring='f1_micro')
    vis.fit(X_train, np.ravel(y_train))
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis = ValidationCurve(MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(95,95)), param_name="max_iter",
    param_range=np.arange(0, 210, 10),  cv=StratifiedKFold(n_splits=5), scoring='f1_micro', n_jobs=10)

    vis.fit(X_train, np.ravel(y_train))
    vis.show()
    # end

    v = MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(95,95))

    strt = time.time()
    v.fit(X_train, y_train)
    ned = time.time()
    print('training Time: ' + str(ned - strt) + 's')

    strt = time.time()
    ans = v.predict(X_test)
    ned = time.time()
    print('inference Time: ' + str(ned - strt) + 's')

    print('score:')
    print(f1_score(y_test, ans, average='micro'))
    print(' ')


def DTset1():

    data = pd.read_csv('./adult/adult.data')
    X = data.iloc[:, :14]
    X = OneHotEncoder(handle_unknown='ignore').fit_transform(X)
    y = data.iloc[:, [-1]]
    print(X)
    print(y)

    sc = make_scorer(modf1)
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(tree.DecisionTreeClassifier(random_state=0), cv=StratifiedKFold(n_splits=5), scoring=sc, n_jobs=10)

    vis.fit(X, y)
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis = ValidationCurve(tree.DecisionTreeClassifier(random_state=0), param_name="max_depth",
    param_range=np.arange(0, 30),  cv=StratifiedKFold(n_splits=5), scoring=sc, n_jobs=10)

    vis.fit(X, y)
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis = ValidationCurve(tree.DecisionTreeClassifier(random_state=0), param_name="ccp_alpha",
                          param_range=np.arange(0, 30), cv=StratifiedKFold(n_splits=5), scoring=sc, n_jobs=10)

    vis.fit(X, y)
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis = ValidationCurve(tree.DecisionTreeClassifier(random_state=0, max_depth=14), param_name="ccp_alpha",
                          param_range=np.arange(0, 30), cv=StratifiedKFold(n_splits=5), scoring=sc, n_jobs=10)

    vis.fit(X, y)
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(tree.DecisionTreeClassifier(random_state=0, max_depth=14, ccp_alpha=0), cv=StratifiedKFold(n_splits=5), scoring=sc, n_jobs=10)
    vis.fit(X, y)
    vis.show()
    # end

    data = pd.read_csv('./adult/adult.data',
                       names=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"])
    X = data.iloc[:, :14]

    y = data.iloc[:, [-1]]

    testdata = pd.read_csv('./adult/adult.test', names=["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"])
    X_test = testdata.iloc[:, :14]
    X_test = OneHotEncoder(handle_unknown='ignore').fit(pd.concat((X_test,X))).transform(X_test)
    y_test = testdata.iloc[:, [-1]]

    v = tree.DecisionTreeClassifier(random_state=0, max_depth=14, ccp_alpha=0)
    X = OneHotEncoder(handle_unknown='ignore').fit(pd.concat((testdata.iloc[:, :14],X))).transform(X)

    strt = time.time()
    v.fit(X, y)
    ned = time.time()
    print('training Time: ' + str(ned-strt) + 's')

    strt = time.time()
    ans = v.predict(X_test)
    ned = time.time()
    print('inference Time: ' + str(ned-strt) + 's')

    print('score:')
    print(f1_score(y_test, ans, pos_label=" >50K"))
    print(' ')




def DTset2():
    # code from https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas"
    )
    # end
    # code modified from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 0)
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(tree.DecisionTreeClassifier(random_state=0), cv=StratifiedKFold(n_splits=5), scoring='f1_micro')
    vis.fit(X_train, y_train)
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis = ValidationCurve(tree.DecisionTreeClassifier(random_state=0), param_name="max_depth",
    param_range=np.arange(0, 30),  cv=StratifiedKFold(n_splits=5), scoring='f1_micro')

    vis.fit(X_train, y_train)
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis = ValidationCurve(tree.DecisionTreeClassifier(random_state=0), param_name="ccp_alpha",
                          param_range=np.arange(0, 30), cv=StratifiedKFold(n_splits=5), scoring='f1_micro')

    vis.fit(X_train, y_train)
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis = ValidationCurve(tree.DecisionTreeClassifier(random_state=0, max_depth=20), param_name="ccp_alpha",
                          param_range=np.arange(0, 30), cv=StratifiedKFold(n_splits=5), scoring='f1_micro', n_jobs=10)

    vis.fit(X_train, y_train)
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(tree.DecisionTreeClassifier(random_state=0, max_depth=20, ccp_alpha=0), cv=StratifiedKFold(n_splits=5), scoring='f1_micro', n_jobs=10)
    vis.fit(X_train, y_train)
    vis.show()
    # end

    v = tree.DecisionTreeClassifier(random_state=0, max_depth=20, ccp_alpha=0)

    strt = time.time()
    v.fit(X_train, y_train)
    ned = time.time()
    print('training Time: ' + str(ned - strt) + 's')

    strt = time.time()
    ans = v.predict(X_test)
    ned = time.time()
    print('inference Time: ' + str(ned - strt) + 's')

    print('score:')
    print(f1_score(y_test, ans, average='micro'))
    print(' ')





def modf1(y_true,
             y_predny,
             *,
             labels = None,
             pos_label = 1,
             average = "binary",
             sample_weight = None,
             zero_division = "warn"):
    return f1_score(y_true, y_predny, pos_label=' >50K')


if __name__ == '__main__':
    main()

