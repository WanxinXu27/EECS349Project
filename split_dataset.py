import os
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import logging
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)


def fillinMatrix(smallArr, colIndicesArr, nCols):
    """
    In case prediction does not have contain all classes
    """

    nRows = smallArr.shape[0]
    fullArr = np.zeros([nRows, nCols])

    for i in range(colIndicesArr.size):
        fullArr[:, colIndicesArr[i]] = smallArr[:, i]

    return fullArr


def classification(trainData, trainLabels, testData, method):
    """Train model with trainData and trainLabels, then predict testLabels given testData.
    Output one hot representation and probability

    Parameters
    ----------
        trainingData:               dataFrame
        trainLabels:                dataFrame
        testData:                   dataFrame

    Return
    ------
        result:                     dataFrame
        probaDf:                    dataFrame

    """

    nClass = 2
    classLabels = [0,1]

    trainLabelsUnqArr = np.unique(trainLabels)

    if method == 'NaiveBayes':
        classifier = GaussianNB()
        model = classifier.fit(trainData, trainLabels)
        result = model.predict(testData)
        proba = model.predict_proba(testData)
        proba = fillinMatrix(proba, trainLabelsUnqArr, nClass)
        probaDf = pd.DataFrame(data=proba, columns=classLabels)
    elif method == 'knnVoting':

        classifier = KNeighborsClassifier(5)
        model = classifier.fit(trainData, trainLabels)

        result = model.predict(testData)

        proba = model.predict_proba(testData)
        proba = fillinMatrix(proba, trainLabelsUnqArr, nClass)
        probaDf = pd.DataFrame(data=proba, columns=classLabels)

    elif method == 'RandomForests':

        classifier = RandomForestClassifier(max_depth=10, random_state=0)
        model = classifier.fit(trainData, trainLabels)

        result = model.predict(testData)

        proba = model.predict_proba(testData)
        proba = fillinMatrix(proba, trainLabelsUnqArr, nClass)
        probaDf = pd.DataFrame(data=proba, columns=classLabels)
        ############################################
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]
        # Print the feature ranking
        print("Feature ranking:")
        for f in range(trainData.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(trainData.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(trainData.shape[1]), indices)
        plt.xlim([-1, trainData.shape[1]])
        plt.show()

    elif method == 'SVM':

        classifier = svm.SVC(C=3, gamma=0.003, probability=True)
        model = classifier.fit(trainData, trainLabels)

        result = model.predict(testData)

        proba = model.predict_proba(testData)
        proba = fillinMatrix(proba, trainLabelsUnqArr, nClass)
        probaDf = pd.DataFrame(data=proba, columns=classLabels)

    elif method == 'AdaBoost':

        classifier = AdaBoostClassifier()
        model = classifier.fit(trainData, trainLabels)

        result = model.predict(testData)

        proba = model.predict_proba(testData)
        proba = fillinMatrix(proba, trainLabelsUnqArr, nClass)
        probaDf = pd.DataFrame(data=proba, columns=classLabels)
        ############################################
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]
        # Print the feature ranking
        print("Feature ranking:")
        for f in range(trainData.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        # Plot the feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(trainData.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(trainData.shape[1]), indices)
        plt.xlim([-1, trainData.shape[1]])
        plt.show()

    elif method == 'NeuralNetwork':
        classifier = MLPClassifier(alpha=1)
        model = classifier.fit(trainData, trainLabels)

        result = model.predict(testData)

        proba = model.predict_proba(testData)
        proba = fillinMatrix(proba, trainLabelsUnqArr, nClass)
        probaDf = pd.DataFrame(data=proba, columns=classLabels)

    elif method == 'LogisticRegression':
        classifier = LogisticRegression()
        model = classifier.fit(trainData, trainLabels)

        result = model.predict(testData)

        proba = model.predict_proba(testData)
        proba = fillinMatrix(proba, trainLabelsUnqArr, nClass)
        probaDf = pd.DataFrame(data=proba, columns=classLabels)

    elif method == 'LinearSVM':
        classifier = LinearSVC(random_state=0)
        model = classifier.fit(trainData, trainLabels)

        result = model.predict(testData)

        ############################################
        importances = model.coef_
        #  std = np.std([tree.feature_importances_ for tree in model.estimators_],
        plt.plot(importances.shape[1])
        plt.ylabel('some numbers')
        plt.show()
    elif method == 'kNN':

        # logger.info(model.coef_)
    # proba = model.predict_proba(testData)
    #        proba = fillinMatrix(proba, trainLabelsUnqArr, nClass)
    #        probaDf = pd.DataFrame(data=proba, columns=classLabels)
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(trainData, trainLabels)

        result=neigh.predict(testData)
        probaDf=neigh.predict_proba(testData)

    # logger.info(method)

    return result, probaDf

def countvec(data):
    # one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
           # 'adCategoryId', 'productId', 'productType']
    vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
    # for feature in one_hot_feature:
    #     try:
    #         data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    #     except:
    #         data[feature] = LabelEncoder().fit_transform(data[feature])
    #

    cv=CountVectorizer()
    for feature in vector_feature:
        print (feature)
        cv.fit(data[feature])
        a = cv.transform(data[feature])
        data[feature] = a.toarray()
        print(a.toarray())
    print('cv prepared !')
    return data


if __name__ == "__main__":

    # Enter path to .csv file containing raw ECG data
    path = r'E:\NU2018spring\349ML\PROJECT\sampled_data'
    # load raw ECG signal
    # df_train = pd.read_csv(os.path.join(path, 'give_train.csv'))
    # df_test = pd.read_csv(os.path.join(path, 'give_test.csv'))
    # truth = pd.read_csv(os.path.join(path, 'test_truth.csv'))
    df_train = pd.read_csv(os.path.join(path, 'mytrain.csv'),nrows=14000,index_col=None)

    df_test = pd.read_csv(os.path.join(path, 'mytest.csv'),nrows=6000,index_col=None)
    df_test.to_csv('test_.csv')
    truth = df_test['label']
    a = df_train['label']
    df_train = df_train.drop(columns=['label'])
    df_test = df_test.drop(columns=['label'])
    df_train = df_train.drop(columns=['ct','marriageStatus','os'])
    df_test = df_test.drop(columns=['ct','marriageStatus','os'])
    df_train=countvec(df_train)
    df_test=countvec(df_test)




    # df_train=df_train.drop(columns=['os','marriageStatus','ct','appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3'])
    # df_test=df_test.drop(columns=['os','marriageStatus','ct','appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3'])

    result, probaDf=classification(df_train,a,df_test,method='kNN')
    print(truth)
    w = np.ones(truth.shape[0])
    print(accuracy_score(truth, result, normalize=True, sample_weight=w))
    confusion = confusion_matrix(truth, result)
    print(confusion)
    f1 = f1_score(truth, result, average='weighted')
    print(f1)
    auc = roc_auc_score(truth, result, average='weighted')
    print(auc)


