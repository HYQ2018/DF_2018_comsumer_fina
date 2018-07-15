# -*- coding: utf-8 -*-
# @author: HYQ

import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score

def train_xgb(data):
    enc = LabelEncoder()
    onc = OneHotEncoder(sparse=False)
    for kk in ['V2', 'V4', 'V5']:
        try:
            data[kk] = enc.fit_transform(data[kk])
        except:
            data[kk] = enc.fit_transform(data[kk].map(int))
        s1 = onc.fit_transform(data[kk].values.reshape(-1, 1))
        s1 = pd.DataFrame(s1)
        s1.columns = [kk[0] + '_' + str(i) for i in range(s1.shape[1])]
        s1['USRID'] = data['USRID'].values
        data = pd.merge(data, s1, on=['USRID'])
    data = data.drop(['V2', 'V4', 'V5'], axis=1)
    Train = data[data['FLAG'] != -1]
    X = Train.drop(['USRID', 'FLAG'], axis=1).values
    y = Train['FLAG'].values
    test = data[data['FLAG'] == -1]
    test_userid = test['USRID']
    test_X = test.drop(['USRID', 'FLAG'], axis=1).values
    del data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
    xx_cv = []
    xx_pre = []
    Important = {}
    for k, (train_index, test_index) in enumerate(skf.split(X, y)):
        print('Train: %s | test: %s' % (train_index, test_index))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = xgb.XGBClassifier(
            max_depth=5,
            learning_rate=0.05,
            n_estimators=2000,
            gamma=1,
            reg_lambda=1,
            reg_alpha=1,
            colsample_bytree=0.7,
            colsample_bylevel=0.7,
            seed=4396,
        )
        clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc", verbose=50,
                eval_set=[(X_test, y_test)])
        pred_value = clf.predict_proba(X_test)[:, 1]
        AUC = roc_auc_score(y_test, pred_value)
        xx_cv.append(AUC)
        xx_pre.append(clf.predict_proba(test_X)[:, 1])
        Important[k] = clf.feature_importances_
    print('best xx score', np.mean(xx_cv))
    xx_pre_yu = []
    s = 0
    for k, i in enumerate(xx_pre):
        if k == 0:
            xx_pre_yu = xx_pre[k]
        else:
            xx_pre_yu = xx_pre_yu + xx_pre[k]
    xx_pre_yu = xx_pre_yu / 5
    res = pd.DataFrame()
    res['USRID'] = list(test_userid.values)
    res['prob'] = xx_pre_yu
    return res
def train_lgb(data):
    # 5折训练
    enc = LabelEncoder()
    onc = OneHotEncoder(sparse=False)
    for kk in ['V2', 'V4', 'V5', 'app']:
        try:
            data[kk] = enc.fit_transform(data[kk])
        except:
            data[kk] = enc.fit_transform(data[kk].map(int))
        s1 = onc.fit_transform(data[kk].values.reshape(-1, 1))
        s1 = pd.DataFrame(s1)
        s1.columns = [kk[0] + '_' + str(i) for i in range(s1.shape[1])]
        s1['USRID'] = data['USRID'].values
        data = pd.merge(data, s1, on=['USRID'])
    data = data.drop(['app', 'V2', 'V4', 'V5'], axis=1)
    train = data[data['FLAG'] != -1]
    # Target = train[train['FLAG'] == 0].sample(frac = 10*3176/76824)
    # N_T = train[train['FLAG'] == 1]
    # train = pd.concat([Target,N_T])
    # print(train['FLAG'].value_counts())
    test = data[data['FLAG'] == -1]
    print('train', train.shape)
    print('test', test.shape)
    # 构造数据
    # 提取userid和单独把标签赋值一个变量
    train_userid = train.pop('USRID')
    y = train.pop('FLAG')
    y = y.values
    col = list(train.columns)
    X = train.values
    print(X.shape)
    print(len(col))
    test_userid = test.pop('USRID')
    test_y = test.pop('FLAG')
    test = test.values
    N = 5
    skf = StratifiedKFold(n_splits=N, shuffle=False, random_state=2)
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    xx_cv = []
    xx_pre = []
    Important = {}
    for k, (train_in, test_in) in enumerate(skf.split(X, y)):
        X_train, X_test, y_train, y_test = X[train_in], X[test_in], y[train_in], y[test_in]
        gbm = lgb.LGBMClassifier(
            boosting_type='gbdt', num_leaves=14, reg_alpha=20, reg_lambda=20,
            max_depth=-1, n_estimators=3000, objective='binary',
            subsample=0.7, colsample_bytree=0.7, subsample_freq=1,  # colsample_bylevel=0.7,
            learning_rate=0.05, min_child_weight=50, random_state=4396, n_jobs=50, is_unbalance=True
        )
        gbm.fit(X_train, y_train, verbose=50, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric="auc",
                early_stopping_rounds=100)
        y_pred = gbm.predict_proba(X_test)[:, 1]
        # 并且把最高线下分数对应的与之存入字典
        xx_cv.append(roc_auc_score(y_test, y_pred))
        xx_pre.append(gbm.predict_proba(test)[:, 1])
        Important[k] = gbm.feature_importances_
    print('best xx score', np.mean(xx_cv))
    xx_pre_yu = []
    s = 0
    for k, i in enumerate(xx_pre):
        if k == 0:
            xx_pre_yu = xx_pre[k]
        else:
            xx_pre_yu = xx_pre_yu + xx_pre[k]
    xx_pre_yu = xx_pre_yu / 5
    res = pd.DataFrame()
    res['USRID'] = list(test_userid.values)
    res['prob'] = xx_pre_yu
    return res
from data import load_data
from feature1 import *

if __name__ == '__main__':
    path = "/Users/hyq/Desktop/contest/招行卡中心/data/"
    agg, log, flg = load_data(path)
    C_feature = count_feature(log,flg.USRID)
    T_feature = time_feature(log,flg.USRID)
    M_feature = module_feature(log,flg.USRID)

    Feature = pd.merge(T_feature,M_feature,on = ['USRID'])
    Feature = pd.merge(Feature,C_feature,on=['USRID'])

    data = pd.merge(flg,agg,on = ['USRID'])
    data = pd.merge(data,Feature,on = ['USRID'])
    for ii in agg.columns:
        if ii not in ['USRID', 'V2', 'V4', 'V5']:
            te = data[ii].groupby(data['FLAG']).mean()
            data[ii + '_mean_label0'] = data[ii] - te[0]
            data[ii + '_mean_label1'] = data[ii] - te[1]
    res_lgb = train_xgb(data)
    res_XGB = train_lgb(data)
    res = pd.merge(res_XGB,res_lgb,on = ['USRID'])
    res['prob'] = (res['prob_x']+res['prob_y'])/2
    res = res.drop(['prob_x','prob_y'],axis = 1)