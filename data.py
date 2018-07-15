# -*- coding: utf-8 -*-
# @author: HYQ
import pandas as pd

def load_data(path):
    #agg
    train_agg = pd.read_csv(path+"train/train_agg.csv",delimiter='\t')
    test_agg = pd.read_csv(path+"test/test_agg.csv",delimiter='\t')
    agg = pd.concat([train_agg,test_agg])
    #log
    train_log = pd.read_csv(path+"train/train_log.csv",delimiter='\t')
    test_log = pd.read_csv(path+"test/test_log.csv",delimiter = '\t')
    log = pd.concat([train_log,test_log])
    #flg
    train_flg = pd.read_csv(path+"train/train_flg.csv",delimiter='\t')
    test_flg = pd.read_csv(path+'submit_sample.csv',delimiter = '\t')
    test_flg['FLAG'] = -1
    del test_flg['RST']
    flg = pd.concat([train_flg,test_flg])
    return agg,log,flg
