**Project Introduction**

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives.
In this project, you will play detective, and put your machine learning skills to use by building an algorithm to identify Enron Employees who may have committed fraud based on the public Enron financial and email dataset.

**Question 1**

Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those? [Relevant rubric items: “data exploration”, “outlier investigation”]

In this project, I need to build a person of interest (POI) identifier based on financial and email data made public as a result of the Enron scandal. A person of interest is anyone that is indicted, settled without admitting guilt or testified in exchange for immunity. The Enron dataset constitute the largest email and financial details of real people available in the open. ML will accomplish the goal of this project by building classifiers to detect if there are pattern within the emails and financial details of people who were persons of interest in the fraud case and to identify those patterns. This dataset consist of 146 data points, 18 of which were POIs while 128 were Non-POIs. There are also 21 features or attributes per data point i.e. per person.

Removal of outliers:-
Looking at file enron61702insiderpay.pdf, was verified that these values doesn’t belong to a person, but was correspoding to the sum of all people in the dataset. So, the key “TOTAL” as removed of the data_dict. Looking at the file, we also can see that the last line of the file is labled as “THE TRAVEL AGENCY IN THE PARK”. This key also was removed, since it wasn’t a person.


**Question 2**

What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importance of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values. [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

The features which I have used are as follows:-
```
['poi', 'salary', 'to_messages', 'loan_advances', 'bonus', 'deferred_income', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'restricted_stock', 'fraction_from_poi', 'fraction_to_poi']

```
Out of the above features, following two features I have created which is not in actual dataset:-
1. fraction_from_poi
2. fraction_to_poi'

This two features are calculated in this way:-
```
fraction_from_poi=float(from_poi_to_this_person)/float(to_messages)
fraction_to_poi=float(from_this_person_to_poi)/float(from_messages)
```
My intuition behind using these features are that POIs must have strong email connection. In our dataset,18 peoples are marked as POI. So person which has more numbers of emails(sent/received) out of total messages(sent/received) related to POIs person can be suspect as POI.
For example, Suppose Person 1 sends 60 messages to POIs Person out of total 100 messages sent by him.Person 2 sends 40 messages to POIs Person out of total 50 messages sent by him.
Here,Person 2 is more suspected of being an POI as 80% of email are sent to POIs.

This way, new feature 'fraction_from_poi' are created.In the same way, another feature 'fraction_to_poi' are created.

The features were selected by using SelectKBest imported from sklearn.feature_selection module. Using this, following score for features was obtained:-
```
[['exercised_stock_options', 24.25047235452619], ['bonus', 20.25718499812395], ['salary', 17.71787357924329], ['fraction_to_poi', 15.946248696687636], ['deferred_income', 11.184580251839124], ['restricted_stock', 8.94550301526133], ['shared_receipt_with_poi', 8.276138216260644], ['loan_advances', 7.066710861319749], ['expenses', 5.815328001904854], ['from_poi_to_this_person', 5.041257378669385], ['fraction_from_poi', 2.963990314926164], ['from_this_person_to_poi', 2.295183195738003], ['to_messages', 1.5425809046549228], ['from_messages', 0.18121500856156128]]
```

**Question 3**

What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms? [ relevant rubric item: “pick an algorithm”]

I have used Naive Bayes, Decision Tree,Random Forest,K neighbour and SVC.

**Question 4**

What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well? How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your

GridSearch from scikit-learn was used to tune 4 different types of classifiers (Decision Tree, Random Forest, K-NN and SVM). The following sets of parameters was used:
```
decision_tree_parameters = {'min_samples_split': [2, 3, 4, 5, 6, 7],
                            'max_features': ['auto', 'sqrt', 'log2', None],
                            'criterion': ['gini', 'entropy']}

randon_forest_parameters = {'min_samples_split': [2, 3, 4, 5, 6, 7],
                            'max_features': ['auto', 'sqrt', 'log2', None],
                            'criterion': ['gini', 'entropy'],
                            'n_estimators': [2, 3, 4, 5, 6, 7]}

knn_parameters = {'n_neighbors': [1, 3, 5, 7, 9]}

svm_parameters = {'kernel': ['rbf'], 
                  'C': [1, 10, 100, 1000, 10000, 100000],
				  ,'gamma':['scale', 'auto']}
```
Using test_classifier in tester.py ,I got the following result:-
```
        GaussianNB()
        Accuracy: 0.90409       Precision: 0.46055      Recall: 0.32100 F1: 0.37831     F2: 0.34171
        Total predictions: 11000        True positives:  321    False positives:  376   False negatives:  679   True negatives: 9624	
```
**Question 5**

What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis? [relevant rubric items: “discuss validation”, “validation strategy”]

Validation is a method of testing whether or not your algorithm is doing what you want it to do by splitting the data into a training and testing set so you have an independent dataset to use to get an estimate of your algorithms performance. If you do this wrong, you can overfit your data. I used train_test_split cross validation with 30% of the data in the test set and 70% in the training set, fit the training set to the algorithm, created predictions with the algorithm and then evaluated how the predictions compared to the true labels in the test set.


**Question 6**

Give at least 2 evaluation metrics and your average performance for each of them. Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

The evaluation metrics that I used were recall, precision, and f1 score. The recall score was .32, which means out of all actual POIs, the algorithm correctly identified 32% as being a POI (or 68% were incorrectly labeled as a Non-POI when they were actually a POI). The precision score was .46, which means 46% of all people labeled as POI’s were indeed a POI (54% were labeled as a POI, but were actually a Non-POI). The F1 score is a combination of precision and recall, the higher the F1 score, the better the algorithm is doing at classifying.


