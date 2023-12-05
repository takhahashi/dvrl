# ASAPデータセットを読み込んでえんべディングを作成
# sentence bert

from sentence_transformers import SentenceTransformer
import pandas as pd

model = SentenceTransformer('all-MiniLM-L6-v2')

for mode in ["train", "valid", "test"]:
    df = pd.read_csv(f"data/{mode}.tsv", sep="\t")
    df = df[df["essay_set"] == 1]
    print(df)

    #Our sentences we like to encode
    sentences = df["essay"].values
    label = df["domain1_score"]
    

    #Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)
    sentence_df = pd.DataFrame(embeddings)
    sentence_df.to_csv(f"data/x_{mode}.csv", index=False)
    label.to_csv(f"data/y_{mode}.csv", index=False)

    #Print the embeddings
    for sentence, embedding in zip(sentences, embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("")
        print(type(embedding))
        print(sentence_df)
        break

    