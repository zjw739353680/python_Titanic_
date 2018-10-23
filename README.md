# python_Titanic_

python2环境
python_Titanic_prediction_live_or_death


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
