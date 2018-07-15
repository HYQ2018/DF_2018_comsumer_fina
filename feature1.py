# -*- coding: utf-8 -*-
# @author: HYQ
import pandas as pd
import numpy as np

"""
统计特征组
3月的统计特征
用户总点击次数、日均点击数、点击模块的数

时间特征组：
用户点击时间间隔平均差、最后一次点击时间、点击最大时差
以周为时间窗口统计用户点击次数

模块特征组：
用户在每个一级模块上的点击数
用户每个具体模块的点击数（取前topN个)
"""

def count_feature(log,User):
    # Feature
    Feature = pd.DataFrame()
    Feature['USRID'] = User
    #统计每个用户的总点击次数
    click_usr = log[['USRID','EVT_LBL']].groupby(['USRID']).count().reset_index()
    click_usr.columns = ['USRID','click_usr']
    Feature = pd.merge(Feature,click_usr,on = 'USRID',how = 'left')
    Feature['click_usr'].fillna(0,inplace = True)
    #统计每个用户的日均点击次数
    log['day'] = log['OCC_TIM'].apply(lambda x:x[0:10])
    s = log[['USRID','day','EVT_LBL']].groupby(['USRID','day']).count().reset_index()
    avg_click = s[['USRID','EVT_LBL']].groupby(['USRID']).mean().reset_index()
    avg_click.columns = ['USRID','avg_click']
    Feature = pd.merge(Feature,avg_click,on = 'USRID',how = 'left')
    Feature['avg_click'].fillna(0,inplace = True)
    #3月上线app的天数
    temp = log[['USRID','day','EVT_LBL']].groupby(['USRID','day']).count().reset_index()
    online = temp[['USRID','day']].groupby(['USRID']).count().reset_index()
    online.columns = ['USRID','count_month']
    Feature = pd.merge(Feature,online,on = 'USRID',how = 'left')
    Feature['count_month'].fillna(0,inplace = True)
    # 每个人点击的不同模块的个数
    EVT_LBL_len = log.groupby(by= ['USRID'], as_index = False)['EVT_LBL'].agg({'EVT_LBL_len':len})
    EVT_LBL_set_len = log.groupby(by= ['USRID'], as_index = False)['EVT_LBL'].agg({'EVT_LBL_set_len':lambda x:len(set(x))})
    log1 = pd.DataFrame()
    log['OCC_TIM1'] = pd.to_datetime(log['OCC_TIM'])
    log1['hour'] = log.OCC_TIM1.map(lambda x:x.hour)
    log1['day'] = log.OCC_TIM1.map(lambda x:x.day)
    Feature = pd.merge(Feature,EVT_LBL_set_len,on = 'USRID',how = 'left')
    Feature.fillna(0,inplace = True)
    # 点击H5的次数
    l = log[log['TCH_TYP'] == 2]
    l1 = l.groupby(['USRID'])['EVT_LBL'].count().reset_index()
    l1.columns = ['USRID', 'type']
    Feature = pd.merge(Feature, l1, on='USRID', how='left')
    Feature.fillna(0, inplace=True)
    return Feature


def time_feature(log,User):
    Feature = pd.DataFrame()
    Feature['USRID'] = User
    # 最后一次点击app
    s1s1 = log[['USRID', 'OCC_TIM']].groupby(['USRID']).max().reset_index()
    s1s1['OCC_TIM'] = pd.to_datetime(s1s1['OCC_TIM'])
    s1s1['last_click_day'] = pd.to_datetime('2018-04-01 00:00:00') - s1s1['OCC_TIM']
    s1s1['last_click_day'] = s1s1['last_click_day'].apply(lambda x: x.days)
    del s1s1['OCC_TIM']
    Feature = pd.merge(Feature, s1s1, on=['USRID'], how='left')
    Feature['last_click_day'] = Feature['last_click_day'].apply(lambda x: 31 if np.isnan(x) else x)
    # 两次点击之间的天数差的平均值
    log['day'] = log['OCC_TIM'].apply(lambda x: x[0:10])
    Time_gap = log.groupby(['USRID'])['day'].apply(
        lambda x: (int(max(x)[-2:]) - int(min(x)[-2:])) / len(set(x))).reset_index()
    Time_gap[Time_gap == 0] = 31
    Time_gap.columns = ['USRID', 'Time_gap']
    Feature = pd.merge(Feature, Time_gap, on='USRID', how='left')
    Feature.fillna(31, inplace=True)  #
    # 点击的天数的最大时长
    click_max = log.groupby(['USRID'])['day'].apply(lambda x: (int(max(x)[-2:]) - int(min(x)[-2:]))).reset_index()
    Feature = pd.merge(Feature, click_max, on='USRID', how='left')
    Feature.fillna(100, inplace=True)
    D = log['day'].drop_duplicates()
    wek = {}
    for i in D:
        x = pd.to_datetime(i).day
        if x <= 7:
            wek[i] = 1
        elif x > 7 and x <= 14:
            wek[i] = 2
        elif x > 14 and x <= 21:
            wek[i] = 3
        elif x > 21:
            wek[i] = 4
            print(wek[i])
        else:
            pass
    log['week'] = log['day'].map(wek)
    swsw = log[['USRID', 'week', 'day']].groupby(['USRID', 'week']).count().reset_index()
    swsw_ = swsw.set_index(['USRID', 'week']).unstack().reset_index()
    # 最后一周点击次数
    # s_4 = swsw[swsw['week'] == 4]
    # s_4.columns = ['USRID','week','last_week_click']
    swsw_.columns = ["USRID"] + ['week' + str(i) for i in range(1, 5)]
    Feature = pd.merge(Feature, swsw_, on=['USRID'], how='left')
    Feature.fillna(0, inplace=True)
    # 最后一天点击次数
    s_5 = log[log['day'] == '2018-03-31']
    s_55 = s_5[['USRID', 'day']].groupby(['USRID']).count().reset_index()
    s_55.columns = ['USRID', 'last_day_click']
    Feature = pd.merge(Feature, s_55[['USRID', 'last_day_click']], on=['USRID'], how='left')
    Feature['last_day_click'].fillna(0, inplace=True)
    # 最后3天点击次数
    s_5 = log[log['day'].isin(['2018-03-31', '2018-03-30', '2018-03-29'])]
    s_55 = s_5[['USRID', 'day']].groupby(['USRID']).count().reset_index()
    s_55.columns = ['USRID', 'last_3day_click']
    Feature = pd.merge(Feature, s_55[['USRID', 'last_3day_click']], on=['USRID'], how='left')
    Feature['last_3day_click'].fillna(0, inplace=True)
    s12w = log[['USRID', 'day']]
    s12w['day'] = s12w['day'].apply(lambda x: x[-2:])
    s12w = s12w.groupby(['USRID'])['day'].apply(lambda x: x.drop_duplicates()).reset_index()
    s12w = s12w.drop(['level_1'], axis=1)
    s12w['day'] = s12w['day'].map(int)
    s12w['next_time'] = s12w.groupby(['USRID'])['day'].diff(-1).apply(np.abs)
    s12w = s12w.groupby(['USRID'], as_index=False)['next_time'].agg({
        'next_time_mean_dat': np.mean,
        'next_time_std_day': np.std,
        'next_time_min_day': np.min,
        'next_time_max_day': np.max
    })
    Feature = pd.merge(Feature, s12w, on=['USRID'], how='left')
    Feature.fillna(0, inplace=True)
    click_usr = log[['USRID','EVT_LBL']].groupby(['USRID']).count().reset_index()
    click_usr.columns = ['USRID','click_usr']
    Feature = pd.merge(Feature,click_usr,on = 'USRID',how = 'left')
    Feature['click_usr'].fillna(0,inplace = True)
    #最后一天点击数占比
    Feature['last_day_click_rate'] = Feature['last_day_click'] / Feature.click_usr
    Feature.fillna(-1, inplace=True)
    del Feature['click_usr']
    return Feature


def module_feature(log,User):
    Feature = pd.DataFrame()
    Feature['USRID'] = User
    # 统计每个一级模块的点击次数
    EL = log['EVT_LBL'].apply(lambda x: x.split('-'))
    EL1 = pd.DataFrame()
    EL1['USRID'] = log['USRID']
    EL1['EVT_LBL'] = EL
    EL1['F1'] = EL1['EVT_LBL'].apply(lambda x: x[0])
    EL1['F2'] = EL1['EVT_LBL'].apply(lambda x: x[1])
    EL1['F3'] = EL1['EVT_LBL'].apply(lambda x: x[2])
    kks = EL1.groupby('USRID')['F1'].value_counts().unstack().reset_index()
    kks.fillna(0, inplace=True)
    Feature = pd.merge(Feature, kks, on='USRID', how='left')
    kks121 = EL1.groupby('USRID')['F1'].apply(lambda x: len(set(x))).reset_index()
    kks121.columns = ['USRID', 'F1_count']
    kks122 = EL1.groupby('USRID')['F2'].apply(lambda x: len(set(x))).reset_index()
    kks122.columns = ['USRID', 'F2_count']
    kks123 = EL1.groupby('USRID')['F3'].apply(lambda x: len(set(x))).reset_index()
    kks123.columns = ['USRID', 'F3_count']
    Feature = pd.merge(Feature, kks121, on='USRID', how='left')
    Feature = pd.merge(Feature, kks122, on='USRID', how='left')
    Feature = pd.merge(Feature, kks123, on='USRID', how='left')
    Feature.fillna(0, inplace=True)
    cooo = log['EVT_LBL'].value_counts().reset_index()
    coool = cooo['index'][cooo['EVT_LBL'] >= 10000]
    kks = log.groupby('USRID')['EVT_LBL'].value_counts().unstack().reset_index()
    kks.fillna(0, inplace=True)
    Feature = pd.merge(Feature, kks[['USRID'] + list(coool)], on='USRID', how='left')
    Feature.fillna(0, inplace=True)
    kks = kks.drop(coool, axis=1)
    kks['s'] = kks.iloc[:, 1:].sum(axis=1)
    Feature = pd.merge(Feature, kks[['USRID', 's']], on='USRID', how='left')
    Feature.fillna(0, inplace=True)
    # 统计最后5天每个一级模块的点击次数
    log = log[log['day'].isin(['2018-03-28', '2018-03-31', '2018-03-30', '2018-03-29'])]
    EL = log['EVT_LBL'].apply(lambda x: x.split('-'))
    EL1 = pd.DataFrame()
    EL1['USRID'] = log['USRID']
    EL1['EVT_LBL'] = EL
    EL1['F1'] = EL1['EVT_LBL'].apply(lambda x: x[0])
    EL1['F2'] = EL1['EVT_LBL'].apply(lambda x: x[1])
    EL1['F3'] = EL1['EVT_LBL'].apply(lambda x: x[2])
    kks = EL1.groupby('USRID')['F1'].value_counts().unstack().reset_index()
    kks.fillna(0, inplace=True)
    Feature = pd.merge(Feature, kks, on='USRID', how='left')
    Feature.fillna(0, inplace=True)
    # app
    APP = pd.DataFrame()
    APP['USRID'] = list(set(log['USRID']))
    APP['app'] = 1
    Feature = pd.merge(Feature, APP, on='USRID', how='left')
    Feature.fillna(0, inplace=True)
    return Feature

# def diff_feature(agg,flg):
#     data = pd.merge(agg,flg,on = ['USRID'])
#     for ii in agg.columns:
#         if ii not in ['USRID', 'V2', 'V4', 'V5']:
#             te = agg[ii].groupby(data['FLAG']).median()
#             data[ii + '_mean_label0'] = data[ii] - te[0]
#             data[ii + '_mean_label1'] = data[ii] - te[1]
#     return data
