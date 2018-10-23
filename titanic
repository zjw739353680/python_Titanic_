# -*- coding: utf-8 -*-

# python2
'''
1,填充缺失值
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

2，对非数值型进行数值化
    titanic["Sex"].unique()#这一句是查看sex这一列有几种可能性  2

3，交叉验证
    kf = KFold(titanic.shape[0], n_folds=3, random_state=1)  #-------------交叉验证得到的是数据的索引，根据索引得到训练集合测试集


    一共有0~890条数据，这些数据都是训练集，将训练集分成三等分，0~296、297~593、594~890
    选择出两份数据作为训练集，一份做出测试集，一共分成三次，因此for循环经历三次，这叫做交叉验证
    最后

    先选择0~296做测试集，其他为训练集，测试结果加入predictions，
    第二次选择297~593做测试集，其他为训练集，测试结果加入predictions，
    第三次选择594~890做测试集，其他为训练集，测试结果加入 predictions，
    因此最后predictions内的内容是第一条数据到最后一条数据的预测结果，
    最后和原始数据进行比较得到误差

    predictions = []
    for train, test in kf:

4，将不同类型的数据化成一列
    原来predictions是一个列表，每个元素为一列numpy数列形式，因此下面一行是将这些元素放在一个numpy里面
    结果predictions是一个890的序列了

    predictions = np.concatenate(predictions, axis=0)

5，交叉验证跑分
    scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
    print(scores.mean())

6，建立新的特征了，这下进行新的预测了

7，用多个分类器，效果好，进行集成分类



'''

import pandas #ipython notebook
titanic = pandas.read_csv("titanic_train.csv")
titanic.head(5)
titanic.describe()

# titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())  #使用age数据的中位数进行填充缺失值
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
titanic.describe()

print titanic["Sex"].unique()  #这一句是查看sex这一列有几种可能性

# Replace all the occurences of male with the number 0.
#计算机无法识别非数字型的内容，因此要将字符型数据转换成数值型
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0   #把男性映射成0，女的映射1
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
titanic.head(5)

print titanic["Embarked"].unique()
titanic["Embarked"] = titanic["Embarked"].fillna('S')
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
titanic.head(5)

# ----------------------------------------使用现行回归分类器进行预测
# Import the linear regression class
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold   #交叉验证

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm class
alg = LinearRegression()
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)  #-------------交叉验证得到的是数据的索引，根据索引得到训练集合测试集

'''
一共有0~890条数据，这些数据都是训练集，将训练集分成三等分，0~296、297~593、594~890
选择出两份数据作为训练集，一份做出测试集，一共分成三次，因此for循环经历三次，这叫做交叉验证
最后

先选择0~296做测试集，其他为训练集，测试结果加入predictions，
第二次选择297~593做测试集，其他为训练集，测试结果加入predictions，
第三次选择594~890做测试集，其他为训练集，测试结果加入 predictions，
因此最后predictions内的内容是第一条数据到最后一条数据的预测结果，
最后和原始数据进行比较得到误差
'''
predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (titanic[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = titanic["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)

# ----------
import numpy as np

# The predictions are in three separate numpy arrays.  Concatenate them into one.
# We concatenate them on axis 0, as they only have one axis.
'''
原来predictions是一个列表，每个元素为一列numpy数列形式，因此下面一行是将这些元素放在一个numpy里面
结果predictions是一个890的序列了
'''
predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
'''
predictions[predictions == titanic["Survived"]]  预测数据与原始数据一致则为1，否则为0
'''
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print accuracy


# -----------------------------------------用logistics回归进行预测
'''
cross_val_score   这里不用fit什么的了
'''
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
# Initialize our algorithm
alg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())


# --------------------------------------前面用的是训练集进行交叉验证，应该用真正的测试集进行测试
#下面对数据进行预处理
titanic_test = pandas.read_csv("test.csv")
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

# -------------------------接下来用随机森林进行预测
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm with the default paramters
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

# Take the mean of the scores (because we have one for each fold)
print(scores.mean())

# 结果：0.785634118967   感觉和前面两种方法效果一样，接下来我要对随机森林进行调优，就是更改参数
# alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)


# -----------------------------------更改树的数量以及叶子节点和深度进行限制，参数选择哪些比较好？慢慢试，时间长

alg = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
kf = cross_validation.KFold(titanic.shape[0], 3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

# Take the mean of the scores (because we have one for each fold)
print(scores.mean())
# 0.814814814815结果提升了
# 接下来还想提升准确率，参数已经达到瓶颈了，接下来要看数据本身了，也就是数据特征了


# 猜测新的特征对结果造成的影响
# ------------------------自制一个新的特征，比如将家人数量加在一起作为新的特征，名字的长度作为一个特征
# Generating a familysize column
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

# The .apply method generates a new series
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))


import re

# A function to get the title from a name.
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Get all the titles and print how often each one occurs.
titles = titanic["Name"].apply(get_title)
print(pandas.value_counts(titles))

# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v

# Verify that we converted everything.
print(pandas.value_counts(titles))

# Add in the title column.
titanic["Title"] = titles
print titanic


# --------------------------------------建立新的特征了，这下进行新的预测了

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "NameLength"]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

# Pick only the four best features.
predictors = ["Pclass", "Sex", "Fare", "Title"]

alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)




# ------------------------------------------------用多个分类器，效果好



from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

# The algorithms we want to ensemble.
# We're using the more linear predictors for the logistic regression, and everything with the gradient boosting classifier.
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title",]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

# Initialize the cross validation folds
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        # Fit the algorithm on the training data.
        alg.fit(titanic[predictors].iloc[train,:], train_target)
        # Select and predict on the test fold.
        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    # Use a simple ensembling scheme -- just average the predictions to get the final classification.
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

# Put all the predictions together into one array.
predictions = np.concatenate(predictions, axis=0)

# Compute accuracy by comparing to the training data.
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print(accuracy)


pass



