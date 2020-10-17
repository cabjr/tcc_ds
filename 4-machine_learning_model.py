import pandas as pd 
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns
import sklearn
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import time
import imblearn
from sklearn.linear_model import LogisticRegression
import multiprocessing
from sklearn.preprocessing import MinMaxScaler


def train_decisiontree(X_train, y_train, X_test, y_test, X_val, y_val):
    start = time.time()
    clf = tree.DecisionTreeClassifier()
    print("Training Decision Tree")
    clf_train = clf.fit(X_train, y_train)
    end = time.time()
    print("Train Finished ("+str(end - start)+" seconds)")
    y_pred = clf_train.predict(X_val)
    end2 = time.time()
    for index, feat_imp in enumerate(clf_train.feature_importances_):
        print("Feature ("+str(dataset.columns[index])+"): ", feat_imp)
    print("Infer Finished (Time: "+str(end2 - end)+" seconds)")
    print("Accuracy Score: ", sklearn.metrics.accuracy_score(y_val, y_pred))
    print("Precision Score: ", sklearn.metrics.precision_score(y_val, y_pred))
    print("Recall Score: ", sklearn.metrics.recall_score(y_val, y_pred))
    print("F1-Score: ", sklearn.metrics.f1_score(y_val, y_pred))
    print("ROC AUC Score: ", sklearn.metrics.roc_auc_score(y_val, y_pred))
    metrics.plot_roc_curve(clf, X_val, y_val)
    plt.savefig("ROC_Curve_Decision_tree.png")

def train_nn(X_train, y_train, X_test, y_test, X_val, y_val):
    start = time.time()
    print("Training Multilayer Perceptron")
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    clf_train = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    end = time.time()
    print("Train Finished ("+str(end - start)+" seconds)")
    y_pred = clf_train.predict(X_val)
    end2 = time.time()
    print("Infer Finished (Time: "+str(end2 - end)+" seconds)")
    print("Accuracy Score: ", sklearn.metrics.accuracy_score(y_val, y_pred))
    print("Precision Score: ", sklearn.metrics.precision_score(y_val, y_pred))
    print("Recall Score: ", sklearn.metrics.recall_score(y_val, y_pred))
    print("F1-Score: ", sklearn.metrics.f1_score(y_val, y_pred))
    print("ROC AUC Score: ", sklearn.metrics.roc_auc_score(y_val, y_pred))
    metrics.plot_roc_curve(clf_train, X_val, y_val)
    plt.savefig("ROC_Curve_MLP.png")


def train_logisticRegr(X_train, y_train, X_test, y_test, X_val, y_val):
    start = time.time()
    print("Training Logistic Regression")
    clf_train = LogisticRegression(random_state=0).fit(X_train, y_train)
    end = time.time()
    print("Train Finished ("+str(end - start)+" seconds)")
    y_pred = clf_train.predict(X_val)
    end2 = time.time()
    print("Infer Finished (Time: "+str(end2 - end)+" seconds)")
    print("Accuracy Score: ", sklearn.metrics.accuracy_score(y_val, y_pred))
    print("Precision Score: ", sklearn.metrics.precision_score(y_val, y_pred))
    print("Recall Score: ", sklearn.metrics.recall_score(y_val, y_pred))
    print("F1-Score: ", sklearn.metrics.f1_score(y_val, y_pred))
    print("ROC AUC Score: ", sklearn.metrics.roc_auc_score(y_val, y_pred))
    metrics.plot_roc_curve(clf_train, X_val, y_val)
    plt.savefig("ROC_Curve_LR.png")


def train_knn(X_train, y_train, X_test, y_test, X_val, y_val):
    start = time.time()
    clf = KNeighborsClassifier(n_neighbors=3)
    print("Training KNN")
    clf_train = clf.fit(X_train, y_train)
    end = time.time()
    print("Train Finished ("+str(end - start)+" seconds)")
    y_pred = clf_train.predict(X_val)
    end2 = time.time()
    print("Infer Finished (Time: "+str(end2 - end)+" seconds)")
    print("Accuracy Score: ", sklearn.metrics.accuracy_score(y_val, y_pred))
    print("Precision Score: ", sklearn.metrics.precision_score(y_val, y_pred))
    print("Recall Score: ", sklearn.metrics.recall_score(y_val, y_pred))
    print("F1-Score: ", sklearn.metrics.f1_score(y_val, y_pred))
    print("ROC AUC Score: ", sklearn.metrics.roc_auc_score(y_val, y_pred))
    metrics.plot_roc_curve(clf, X_val, y_val)
    plt.savefig("ROC_Curve_KNN.png")

def train_svm(X_train, y_train, X_test, y_test, X_val, y_val):
    start = time.time()
    print("Training SVM")
    clf = Pipeline(steps=[('standardscaler', StandardScaler()), ('svc', SVC(gamma='auto'))])
    clf.fit(X_train, y_train)
    end = time.time()
    print("Train Finished ("+str(end - start)+" seconds)")
    clf_train = clf.fit(X_train, y_train)
    print("Train Finished")
    y_pred = clf_train.predict(X_val)
    end2 = time.time()
    print("Infer Finished (Time: "+str(end2 - end)+" seconds)")
    print("Accuracy Score: ", sklearn.metrics.accuracy_score(y_val, y_pred))
    print("Precision Score: ", sklearn.metrics.precision_score(y_val, y_pred))
    print("Recall Score: ", sklearn.metrics.recall_score(y_val, y_pred))
    print("F1-Score: ", sklearn.metrics.f1_score(y_val, y_pred))
    print("ROC AUC Score: ", sklearn.metrics.roc_auc_score(y_val, y_pred))
    metrics.plot_roc_curve(clf, X_val, y_val)
    plt.savefig("ROC_Curve_SVM.png")

data_basics = pd.read_csv('final_result_youtube_stats.csv')
data_basics.drop('Unnamed: 0',1)


print(data_basics.head())
print(data_basics.columns.to_series().groupby(data_basics.dtypes).groups)
data_basics["Y"] = data_basics.apply(lambda x: 1 if x["revenue"] > 1.5*x["budget"] else 0 ,axis=1)
data_basics["likeCount"] = data_basics["likeCount"].apply(pd.to_numeric, errors='coerce')
data_basics["dislikeCount"] = data_basics["dislikeCount"].apply(pd.to_numeric, errors='coerce')
data_basics["commentCount"] = data_basics["commentCount"].apply(pd.to_numeric, errors='coerce')
data_basics["likeCount"].fillna(0, inplace=True)
data_basics["dislikeCount"].fillna(0, inplace=True)
data_basics["commentCount"].fillna(0, inplace=True)
data_basics = data_basics.drop('revenue',1)
data_basics["likeCount"] = data_basics["likeCount"].astype(np.float64)
data_basics["dislikeCount"] = data_basics["dislikeCount"].astype(float)
data_basics["commentCount"] = data_basics["commentCount"].astype(float)
labels = data_basics["Y"].copy()

#one hot encode genres
genres = []

for value in data_basics['genres'].values:
    if ',' in value:
        for subvalue in value.split(','):
            genres.append(subvalue)
    else:
        genres.append(value)

#remove duplicated items
genres = list(set(genres))
for genre in genres:
    data_basics[genre] = [1 if genre in item['genres'] else 0 for index, item in data_basics.iterrows()]
print("Generos: ", genres)
data_basics = data_basics.drop('genres',1)


dataset = data_basics[[ 'runtimeMinutes', 'startYear', 'staffExperience', 'writersExperience', 'directorsExperience', 'budget' ,'viewCount', 'likeCount', 'dislikeCount', 'commentCount'] + [genre for genre in genres]]
print(dataset.count)
print(dataset.head())
oversample = imblearn.over_sampling.RandomOverSampler(sampling_strategy='minority')

X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.3, random_state=42, shuffle=True)
X_train, y_train = oversample.fit_resample(X_train, y_train)
X_test, y_test = oversample.fit_resample(X_test, y_test)

train_nn(X_train, y_train, X_test, y_test, X_test, y_test)
print("*******************")
train_decisiontree(X_train, y_train, X_test, y_test, X_test, y_test)
print("*******************")
train_knn(X_train, y_train, X_test, y_test, X_test, y_test)
print("*******************")
train_svm(X_train, y_train, X_test, y_test, X_test, y_test)
print("*******************")
train_logisticRegr(X_train, y_train, X_test, y_test, X_test, y_test)
