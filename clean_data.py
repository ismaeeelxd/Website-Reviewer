import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv("data/train_balanced.csv")

#print(df.head())
texts = df["text"]
labels = df["review"]

def clean_text(t):
    t=t.lower()
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

df["clean_text"]=df["text"].apply(clean_text)
texts = df["clean_text"]
#print(texts.head())

#bnghyr el reviwe mn text le arkam 
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

#print(le.classes_)  
#print(labels_encoded[:10])


vocab_s = 21000  # choose based on previous explanation

tokenizer = Tokenizer(num_words=vocab_s, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

# longest sequence fe el 95% is 355

lengths = [len(seq) for seq in sequences]
max_len = int(np.percentile(lengths, 95)) 
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

#print(max_len)
