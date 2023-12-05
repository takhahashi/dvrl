# わざわざ別コードにノイズファイルを作成したのは元コードを勝手に変更することによるエラーなどを防ぐため
import pandas as pd
import numpy as np


# 10, 11, 12点を4, 3, 2
# 2, 3, 4点を12, 11, 10に変更
convert={2:12, 3:11, 4:10, 10:4, 11:3, 12:2}
for mode in ["train", "valid", "test"]:
    y_df = pd.read_csv(f"data/y_{mode}.csv")
    y_df['is_noise']=False
    for i in range(len(y_df)):
        if y_df.iloc[i, 0] in [2, 3, 4, 10, 11, 12] and np.random.rand() <= 0.2:
            y_df.iloc[i, 0] = convert[y_df.iloc[i, 0]]
            y_df.loc[i, 'is_noise'] = True # locだと行や列の名前を直接指定、ilocだと行や列のindexを指定
        
    y_df.to_csv(f"data/y_{mode}_noise.csv", index=False) # 行明を入れないようにするためにindex=False
# print(y_train.iloc[i], type(y_train.iloc[i])) # ILOC[i, :]と同じ
    # print("="*10)
    # print(y_train.iloc[i, 0], type(y_train.iloc[i, 0])) # 列まで指定するとnumpy
    # break