# rikenデータセットを読み込んでエンベディングを作成
# sentence bertを使用

from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.model_selection import train_test_split


model = SentenceTransformer('all-MiniLM-L6-v2')

# データ読み込み処理
riken = pd.read_json("~/Work/ex/tomikawa/dvrl/data/Y14_1-2_1_3.json")
# データを訓練・検証・テストの３つに分割したい
riken_pre, x_test, riken_pre2, y_test = train_test_split(riken["mecab"], riken["score"], test_size=0.14)
# riken_preを訓練データと検証データに更に分割(これで6:6:2にする)
x_train, x_valid, y_train, y_valid = train_test_split(riken_pre, riken_pre2, test_size=0.50)

sentences_train = x_train.values
embedding_train = model.encode(sentences_train)
sentences_train_df = pd.DataFrame(embedding_train)
sentences_train_df.to_csv("data/riken_x_train.csv", index = False)
y_train.to_csv("data/riken_y_train.csv", index = False)

sentences_valid = x_valid.values
embedding_valid = model.encode(sentences_valid)
sentences_valid_df = pd.DataFrame(embedding_valid)
sentences_valid_df.to_csv("data/riken_x_valid.csv", index = False)
y_valid.to_csv("data/riken_y_valid.csv", index = False)

sentences_test = x_test.values
embedding_test = model.encode(sentences_test)
sentences_test_df = pd.DataFrame(embedding_test)
sentences_test_df.to_csv("data/riken_x_test.csv", index = False)
y_test.to_csv("data/riken_y_test.csv", index = False)

# print(y_train.max)

# # 各ショートアンサーをまとめてlist型として保持
# list_train = train.to_numpy().tolist()
# list_train = [" ".join(s.split(" ")) for s in list_train]  # 空白消す
# list_valid = valid.to_numpy().tolist()
# list_valid = [" ".join(s.split(" ")) for s in list_valid]  # 空白消す
# list_test = test.to_numpy().tolist()
# list_test = [" ".join(s.split(" ")) for s in list_test]  # 空白消す

