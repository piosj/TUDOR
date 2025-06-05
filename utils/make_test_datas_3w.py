import pickle
import pandas as pd
import torch
from tqdm import tqdm
import os

def make_test_datas(snapshots_num: int):
    test_file_path = 'behaviors.tsv'
    df = pd.read_csv(test_file_path, sep='\t', encoding='utf-8')

    df['click_time'] = pd.to_datetime(df['click_time'])
    
    criteria_time1 = pd.Timestamp('2017-01-23 00:00:00')
    criteria_time2 = pd.Timestamp('2017-01-26 00:00:00')
    test_df = df[(criteria_time1 <= df['click_time']) & (df['click_time'] < criteria_time2)]
    
    test_df = test_df.dropna(subset=['clicked_news'])

    news2int_file_path = 'news2int.tsv'
    news2int = pd.read_csv(news2int_file_path, sep='\t')

    news2int_mapping = dict(zip(news2int['news_id'], news2int['news_int']))
    
    user2int_df = pd.read_csv('user2int.tsv', sep='\t')
    user2int = user2int_df.set_index('user_id')['user_int'].to_dict()
    all_user_ids = user2int_df['user_int'].tolist()   

    test_df['user_int'] = test_df['history_user'].map(user2int)
    test_df['news_int'] = test_df['clicked_news'].map(news2int_mapping)

    test_news = []
    for u_id in tqdm(range(len(all_user_ids))):
        u_news = torch.tensor(test_df[test_df['user_int'] == u_id]['news_int'].values, dtype=torch.long)
        test_news.append(u_news)

    test_time = []
    test_empty_check = []
    for u_id in tqdm(range(len(all_user_ids))):
        u_len = len(test_df[test_df['user_int'] == u_id])
        u_time = torch.tensor([snapshots_num-1 for _ in range(u_len)], dtype=torch.long)  
        test_time.append(u_time)
        if u_len == 0:
            test_empty_check.append(False)
        else:
            test_empty_check.append(True)
    
    return list(zip(test_news, test_time, test_empty_check))
