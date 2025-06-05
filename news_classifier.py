import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ===================
# آموزش مدل
# ===================
st.title("تشخیص نوع خبر (طبقه‌بندی خودکار)")

# خواندن دیتاست
df = pd.read_csv("news_sample.csv")
st.write("📄 چند نمونه از خبرها:")

st.write(df[['title', 'cat']].head())


# آموزش مدل
X = df['title']
y = df['cat']

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

#آموزش مدل با داده های عددی شده
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ===================
# واسط کاربری
# ===================
st.subheader("🔍 پیش‌بینی دسته خبر")

user_input = st.text_input("📝 یک تیتر خبر وارد کن:")

#برسی وارد شدن متن در ورودی و پیش بینی دسته ی مناسب تیتر توسط مدل
if user_input:
    vectorized_input = vectorizer.transform([user_input])
    prediction = model.predict(vectorized_input)[0]
    st.success(f"📢 نوع خبر: **{prediction}**")

# ===================
# گزارش ارزیابی مدل
# ===================
if st.checkbox("📊 نمایش گزارش ارزیابی مدل"):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())


