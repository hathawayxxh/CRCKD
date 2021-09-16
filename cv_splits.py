"""
For the 3662 fundus images in the APTOS training dataset,
split the dataset for 5-fold cross validation.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


def create_k_fold_dataset(splits_path):

    data_csv = './dataset/train_data.csv'
    df = pd.read_csv(data_csv)
    # print(df['id_code'])
    # print(df['diagnosis'])

    labels = df['diagnosis']

    skf = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)


    for k_idx, (tr_idx, val_idx) in enumerate(skf.split(df, labels)):
        # print(list(set(tr_idx).intersection(set(val_idx))))
        tr_sub_df = df.iloc[tr_idx, :]
        val_sub_df = df.iloc[val_idx, :]
        print(tr_sub_df, val_sub_df)

        tr_image = tr_sub_df['id_code'].tolist()
        te_image = val_sub_df['id_code'].tolist()

        overlap_count = 0
        for i in range(len(te_image)):
            if te_image[i] in tr_image:
                overlap_count += 1
        print("number of overlaps:", overlap_count)

        tr_sub_df.to_csv(splits_path + 'split' + str(k_idx+1) + '_train.csv', index=0)
        val_sub_df.to_csv(splits_path + 'split' + str(k_idx+1) + '_test.csv', index=0)


create_k_fold_dataset(splits_path='./5cv_split_dataset/')
