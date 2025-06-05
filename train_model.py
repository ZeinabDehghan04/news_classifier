import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# خواندن داده‌ها
df = pd.read_csv("news_sample.csv")
X = df["title"]
y = df["cat"]

# بردار‌سازی متن
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# آموزش مدل با داده های عددی شده
model = MultinomialNB()
model.fit(X_vectorized, y)

# ذخیره مدل
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# ذخیره وکتورایزر
with open("vectorizer.pkl", "wb") as f: #wb=write binary
    pickle.dump(vectorizer, f)

print("✅ مدل و وکتورایزر ذخیره شدند.")

