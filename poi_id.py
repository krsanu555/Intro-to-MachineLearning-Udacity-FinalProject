#!/usr/bin/python
import sys
import pickle
sys.path.append("../tools/")

import sklearn.metrics as metrics
import numpy as np
import pandas as pd
from feature_format import featureFormat, targetFeatureSplit
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.naive_bayes import GaussianNB
from operator import itemgetter

from tester import dump_classifier_and_data, test_classifier


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
def select_classifier(n_features, features, labels):

    clf_list = []
    select_k = SelectKBest(f_classif, k=n_features)
    features = select_k.fit_transform(features, labels)
    scores = select_k.scores_
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)  

    
    #Naive Bayes
    clf_gb = GaussianNB()
    clf_gb.fit(features_train, labels_train)
    pred_gb = clf_gb.predict(features_test)
    clf_list.append([metrics.f1_score(pred_gb, labels_test), metrics.accuracy_score(pred_gb, labels_test), n_features, scores, clf_gb])
    
    #DecisionTree
    dt = DecisionTreeClassifier()
    parameters = {'min_samples_split': [2, 3, 4, 5, 6, 7],'max_features': ['auto', 'sqrt', 'log2', None],'criterion': ['gini', 'entropy']}
    clf_dt = GridSearchCV(dt, parameters, scoring='f1')
    clf_dt.fit(features_train, labels_train)
    clf_dt = clf_dt.best_estimator_
    pred_dt = clf_dt.predict(features_test)    
    clf_list.append([metrics.f1_score(pred_dt, labels_test), metrics.accuracy_score(pred_dt, labels_test), n_features, scores, clf_dt])
    
    #RandomForest
    rf = RandomForestClassifier()
    parameters = {'min_samples_split': [2, 3, 4, 5, 6, 7],'max_features': ['auto', 'sqrt', 'log2', None], 'criterion': ['gini', 'entropy'],
                'n_estimators': [2, 3, 4, 5, 6, 7]}                  
    clf_rf = GridSearchCV(rf, parameters, scoring='f1')
    clf_rf.fit(features_train, labels_train)
    clf_rf = clf_rf.best_estimator_
    pred_rf = clf_rf.predict(features_test)
    clf_list.append([metrics.f1_score(pred_rf, labels_test), metrics.accuracy_score(pred_rf, labels_test), n_features, scores, clf_rf])

    #KNeighbors
    knn = KNeighborsClassifier()
    parameters = {'n_neighbors': [1, 3, 5, 7, 9]}
    clf_knn = GridSearchCV(knn, parameters, scoring='f1')
    clf_knn.fit(features_train, labels_train)
    clf_knn = clf_knn.best_estimator_
    pred_knn = clf_knn.predict(features_test)
    clf_list.append([metrics.f1_score(pred_knn, labels_test), metrics.accuracy_score(pred_knn, labels_test), n_features, scores, clf_knn])

    #SVM
    svm = SVC()
    parameters = {'kernel': ['rbf'],'C': [1, 10, 100, 1000, 10000, 100000],'gamma':['scale', 'auto']}
    clf_svm = GridSearchCV(svm, parameters, scoring='f1')
    clf_svm.fit(features_train, labels_train)
    clf_svm = clf_svm.best_estimator_
    pred_svm = clf_svm.predict(features_test)
    clf_list.append([metrics.f1_score(pred_svm, labels_test), metrics.accuracy_score(pred_svm, labels_test), n_features, scores, clf_svm])

    order_clf_list = sorted(clf_list, key=lambda x: x[0],reverse=True)
    return order_clf_list[0]


### Load the dictionary containing the dataset
file="final_project_dataset.pkl"
destination = "final_project_dataset_unix.pkl"
content = ''
outsize = 0
with open(file, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))
words_file_handler = open(destination, "rb")
data_dict = pickle.load(words_file_handler)
words_file_handler.close()


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = list(data_dict['METTS MARK'])
features_list.remove('poi')
features_list.remove('email_address')
features_list.remove('total_payments')
features_list.remove('total_stock_value')
features_list.remove('other')


# Remove columns with > 50% NaN's
df = pd.DataFrame(data_dict).T
df.replace(to_replace='NaN', value=np.nan, inplace=True)
for key in features_list:
    if df[key].isnull().sum() > df.shape[0] * 0.5:
        features_list.remove(key)
features_list = ['poi'] + features_list


### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')


### Store to my_dataset for easy export below.
my_dataset = data_dict


### Task 3: Create new feature(s)
for name in my_dataset:

    data_point = my_dataset[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    
    if from_poi_to_this_person!="NaN" and to_messages!="NaN":
        fraction_from_poi=float(from_poi_to_this_person)/float(to_messages)
    else:
        fraction_from_poi=0
    data_point["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    if from_this_person_to_poi!="NaN" and from_messages!="NaN":
        fraction_to_poi=float(from_this_person_to_poi)/float(from_messages)
    else:
        fraction_to_poi=0
    data_point["fraction_to_poi"] = fraction_to_poi
    
features_list+=['fraction_from_poi','fraction_to_poi']
# print features_list


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
print(features_list)
clf_list = []
for k in range (1, int(len(features_list) / 2)): # Try sets of 1 - number_of_features / 2
    clf_list.append(select_classifier(k, features, labels))
order_clf_list = sorted(clf_list, key=itemgetter(0, 1)) # order by f1-score and accuracy
clf = order_clf_list[len(order_clf_list) - 1][4]


print('\n\nClf: ', clf)

number_of_features = order_clf_list[len(order_clf_list) - 1][2]
print('\n\nNumber of features: ', number_of_features)

print('\n\nFeatures and scores: ')
score_list = order_clf_list[len(order_clf_list) - 1][3]
features = features_list[1:]
features_scores = []
i = 0
for feature in features:
    features_scores.append([feature, score_list[i]])
    i += 1
features_scores = sorted(features_scores, key=itemgetter(1))
print(features_scores[::-1])

print('\n\nFeatures used: ')
new_features_list = []
for feature in features_scores[::-1][:number_of_features]:
    new_features_list.append(feature[0])
print(new_features_list)
print('\n\n')
new_features_list = ['poi'] + new_features_list

## Task 6: Dump your classifier, dataset, and features_list so anyone can
## check your results. You do not need to change anything below, but make sure
## that the version of poi_id.py that you submit can be run on its own and
## generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, new_features_list)