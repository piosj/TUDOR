import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm
import os

BEH_PATH   = 'Adressa/datas/behaviors.tsv'
PUB_PATH   = 'Adressa/datas/news_publish_times.tsv'
SAVE_DIR1   = 'Adressa/datas/train'          
SAVE_DIR2   = 'Adressa/datas/test'          
os.makedirs(SAVE_DIR1, exist_ok=True)
os.makedirs(SAVE_DIR2, exist_ok=True)

df = pd.read_csv(BEH_PATH, sep='\t', encoding='utf-8')
df['click_time']   = pd.to_datetime(df['click_time'])

publish_df = pd.read_csv(PUB_PATH, sep='\t', encoding='utf-8')
publish_df['publish_time'] = pd.to_datetime(publish_df['publish_time'])

news2time = dict(zip(publish_df['news_id'], publish_df['publish_time']))

user2clicked = (
    df.groupby('history_user')['clicked_news']
      .apply(set)
      .to_dict()
)

train_mask = (df['click_time'] >= '2017-01-20') & (df['click_time'] < '2017-01-23')
test_mask  = (df['click_time'] >= '2017-01-23') & (df['click_time'] < '2017-01-26')

train_df = df.loc[train_mask].copy()
test_df  = df.loc[test_mask].copy()

PUBLISH_TIMES = publish_df['publish_time'].values
NEWS_IDS      = publish_df['news_id'].values

def get_candidates(click_time):
    lower = click_time - timedelta(hours=36)
    mask  = (PUBLISH_TIMES >= lower) & (PUBLISH_TIMES < click_time)
    return NEWS_IDS[mask]

rng = np.random.default_rng(28)       

def sample_negatives(row, k):
    cand = get_candidates(row['click_time'])
    
    user_clicked = user2clicked.get(row['history_user'], set())
    cand = cand[~np.isin(cand, list(user_clicked))]
    choice_idx = rng.choice(len(cand), size=k, replace=False)
    neg_ids    = cand[choice_idx]
    neg_times  = [news2time[n].strftime('%Y-%m-%d %H:%M:%S') for n in neg_ids]
    return neg_ids, neg_times

def attach_negative_samples(df_clicks, k):
    neg_cols, time_cols = [], []
    for _, row in tqdm(df_clicks.iterrows(), total=len(df_clicks) ):
        ids, times = sample_negatives(row, k)
        neg_cols.append(' '.join(ids))          
        time_cols.append(','.join(times))       
    df_clicks['negative_samples'] = neg_cols
    df_clicks['publish_times']    = time_cols
    return df_clicks

train_df = attach_negative_samples(train_df, 4)
test_df  = attach_negative_samples(test_df, 20)

cols = ['history_user', 'click_time', 'clicked_news', 'negative_samples', 'publish_times']
train_df = train_df[cols].rename(columns={'history_user': 'user'})
test_df  = test_df[cols].rename(columns={'history_user': 'user'})

train_path = os.path.join(SAVE_DIR1, 'tkg_train_negative_samples_lt36_ns4.tsv')
test_path  = os.path.join(SAVE_DIR2, 'tkg_test_negative_samples_lt36_ns20.tsv')

train_df.to_csv(train_path, sep='\t', index=False)
test_df.to_csv(test_path,  sep='\t', index=False)

