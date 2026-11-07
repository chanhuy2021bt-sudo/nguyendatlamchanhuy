
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score

st.set_page_config(page_title="Đồ án AI - Định giá & Phân cụm Smartphone", page_icon="📱", layout="wide")
st.title("📱 ĐỒ ÁN AI: DỰ ĐOÁN GIÁ & PHÂN CỤM ĐIỆN THOẠI")
st.markdown("**Nhóm 3 - ĐH Giao thông Vận tải TP.HCM (UTH)**")

@st.cache_resource
def load_and_train():
    df = pd.read_excel('Nhóm3_Giá điện thoại.xlsx')
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    df_clean['screen_pixels'] = df_clean['resolution'].apply(
        lambda x: float(str(x).lower().split('x')[0]) * float(str(x).lower().split('x')[1]) if 'x' in str(x).lower() else 0.0
    )
    ann_year = pd.to_datetime(df_clean['announcement_date'], errors='coerce').dt.year
    df_clean['announce_year'] = ann_year.fillna(ann_year.median())

    categorical_cols = ['brand', 'os', 'battery_type']
    existing_cat = [c for c in categorical_cols if c in df_clean.columns]
    df_encoded = pd.get_dummies(df_clean, columns=existing_cat, drop_first=True)

    drop_cols = ['phone_name', 'resolution', 'announcement_date', 'price(USD)']
    X = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns], errors='ignore')
    y = df_encoded['price(USD)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    return df, X.columns, lr, kmeans, scaler

df_raw, X_cols, lr, kmeans, scaler = load_and_train()

st.sidebar.header("🎛️ Chọn thông số cấu hình máy")
brand = st.sidebar.selectbox("Thương hiệu", df_raw['brand'].unique())
ram = st.sidebar.slider("RAM (GB)", 1.0, 16.0, 8.0)
storage = st.sidebar.slider("Bộ nhớ (GB)", 8.0, 512.0, 128.0)
battery = st.sidebar.slider("Pin (mAh)", 1000, 7000, 4500)
inches = st.sidebar.slider("Màn hình (Inches)", 4.0, 7.5, 6.5)

input_dict = {
    'ram(GB)': ram, 'storage(GB)': storage, 'battery': battery,
    'inches': inches, 'screen_pixels': 1080*2400, 'announce_year': 2025,
    'brand': brand
}
input_df = pd.DataFrame([input_dict])
input_encoded = pd.get_dummies(input_df).reindex(columns=X_cols, fill_value=0)

col1, col2 = st.columns(2)
with col1:
    st.subheader("📈 Dự đoán giá (Linear Regression)")
    price_pred = lr.predict(input_encoded)[0]
    st.success(f"### Giá dự báo: **${price_pred:,.2f} USD**")

with col2:
    st.subheader("🏷️ Phân cụm thị trường (K-Means)")
    input_scaled = scaler.transform(input_encoded)
    cluster = kmeans.predict(input_scaled)[0]
    cluster_names = {0: "🔥 Cụm 0: Cao cấp", 1: "📱 Cụm 1: Phổ thông", 2: "🔋 Cụm 2: Tầm trung"}
    st.info(f"### Phân khúc: **{cluster_names[cluster]}**")

st.markdown("---")
st.subheader("📊 Biểu đồ phân cụm thị trường")
fig, ax = plt.subplots(figsize=(8, 4))
sns.scatterplot(data=df_raw, x='ram(GB)', y='price(USD)', hue='Cluster', palette='Set1', ax=ax)
ax.scatter([ram], [price_pred], color='black', s=200, marker='*', label='Máy đang chọn')
ax.legend()
st.pyplot(fig)
