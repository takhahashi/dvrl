# わざわざ別コードにノイズファイルを作成したのは元コードを勝手に変更することによるエラーなどを防ぐため
import pandas as pd
import numpy as np

# 10, 11, 12点を4, 3, 2
# 2, 3, 4点を12, 11, 10に変更
convert={0:14, 1:13, 2:12, 12:2, 13:1, 14:0}
for mode in ["train", "valid", "test"]:
    y_df = pd.read_csv(f"data/riken_y_{mode}.csv")
    y_df['is_noise']=False
    for i in range(len(y_df)):
        if y_df.iloc[i, 0] in [0, 1, 2, 12, 13, 14] and np.random.rand() <= 0.2:
            y_df.iloc[i, 0] = convert[y_df.iloc[i, 0]]
            y_df.loc[i, 'is_noise'] = True # locだと行や列の名前を直接指定、ilocだと行や列のindexを指定
        
    y_df.to_csv(f"data/riken_y_{mode}_noise.csv", index=False) # 行名を入れないようにするためにindex=False
# print(y_train.iloc[i], type(y_train.iloc[i])) # ILOC[i, :]と同じ
    # print("="*10)
    # print(y_train.iloc[i, 0], type(y_train.iloc[i, 0])) # 列まで指定するとnumpy
    # break