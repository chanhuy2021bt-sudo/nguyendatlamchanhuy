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

# ==========================================
# CẤU HÌNH GIAO DIỆN TRANG WEB
# ==========================================
st.set_page_config(
    page_title="Đồ án AI - Định giá & Phân cụm Smartphone",
    page_icon="📱",
    layout="wide"
)

st.title("📱 HỆ THỐNG AI: DỰ ĐOÁN GIÁ & PHÂN CỤM ĐIỆN THOẠI")
st.markdown("**Đồ án Trí tuệ Nhân tạo & Học máy — Nhóm 3**")
st.markdown("---")

# ==========================================
# HÀM TẢI DỮ LIỆU VÀ HUẤN LUYỆN MÔ HÌNH (CÓ CACHE)
# ==========================================
@st.cache_resource
def load_and_train_model():
    file_path = 'Nhóm3_Giá điện thoại.xlsx'
    df = pd.read_excel(file_path)
    df_clean = df.copy()

    # 1. Xử lý giá trị khuyết (Missing Values)
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    # 2. Trích xuất đặc trưng pixel màn hình từ cột resolution
    def calc_pixels(res_str):
        try:
            parts = str(res_str).lower().split('x')
            return float(parts[0]) * float(parts[1])
        except:
            return 0.0

    df_clean['screen_pixels'] = df_clean['resolution'].apply(calc_pixels)
    
    # 3. Trích xuất năm ra mắt từ announcement_date
    ann_year = pd.to_datetime(df_clean['announcement_date'], errors='coerce').dt.year
    df_clean['announce_year'] = ann_year.fillna(ann_year.median())

    # 4. Mã hóa biến định tính (One-Hot Encoding)
    categorical_cols = ['brand', 'os', 'battery_type']
    existing_cat = [c for c in categorical_cols if c in df_clean.columns]
    df_encoded = pd.get_dummies(df_clean, columns=existing_cat, drop_first=True)

    # 5. Tách tập dữ liệu X, y
    drop_cols = ['phone_name', 'resolution', 'announcement_date', 'price(USD)']
    X = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns], errors='ignore')
    y = df_encoded['price(USD)']

    # 6. Chia tập Train / Test (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 7. Huấn luyện Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 8. Chuẩn hóa & Huấn luyện K-Means Clustering (K=3)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_clean['Cluster'] = kmeans_model.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, df_clean['Cluster'])

    return df_clean, X.columns, lr_model, kmeans_model, scaler, (mae, rmse, r2, sil_score)

# Tải mô hình an toàn
try:
    df_raw, X_cols, lr, kmeans, scaler, metrics = load_and_train_model()
    mae, rmse, r2, sil_val = metrics
except Exception as e:
    st.error(f"Lỗi khi tải dữ liệu hoặc huấn luyện mô hình: {e}")
    st.stop()

# ==========================================
# GIAO DIỆN THANH BÊN (SIDEBAR) - NHẬP LIỆU
# ==========================================
st.sidebar.header("🎛️ Tùy chỉnh thông số thiết bị")

brand_options = sorted(df_raw['brand'].unique()) if 'brand' in df_raw.columns else ['Apple', 'Samsung']
brand = st.sidebar.selectbox("Thương hiệu (Brand)", options=brand_options)

ram = st.sidebar.slider("Dung lượng RAM (GB)", 1.0, 16.0, 8.0, step=1.0)
storage = st.sidebar.slider("Bộ nhớ trong (GB)", 16.0, 512.0, 128.0, step=16.0)
battery = st.sidebar.slider("Dung lượng Pin (mAh)", 2000, 7000, 4500, step=100)
inches = st.sidebar.slider("Kích thước màn hình (Inches)", 4.0, 7.2, 6.5, step=0.1)
resolution = st.sidebar.selectbox("Độ phân giải màn hình", options=['1080x2400', '720x1600', '1440x3200', '1080x1920'])
announce_year = st.sidebar.selectbox("Năm ra mắt", options=[2022, 2023, 2024, 2025, 2026], index=3)

# Xử lý dữ liệu đầu vào người dùng
res_parts = resolution.split('x')
pixels = float(res_parts[0]) * float(res_parts[1])

input_dict = {
    'ram(GB)': ram, 
    'storage(GB)': storage, 
    'battery': battery,
    'inches': inches, 
    'screen_pixels': pixels, 
    'announce_year': announce_year,
    'brand': brand
}

input_df = pd.DataFrame([input_dict])
input_encoded = pd.get_dummies(input_df).reindex(columns=X_cols, fill_value=0)

# ==========================================
# KHU VỰC HIỂN THỊ KẾT QUẢ CHÍNH
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Dự đoán giá (Linear Regression)")
    price_pred = lr.predict(input_encoded)[0]
    st.success(f"### Giá dự kiến: **${price_pred:,.2f} USD**")
    
    with st.expander("Xem chi tiết thông số kiểm thử mô hình"):
        st.write(f"- **Sai số tuyệt đối (MAE):** `${mae:.2f}`")
        st.write(f"- **Sai số bình phương (RMSE):** `${rmse:.2f}`")
        st.write(f"- **Hệ số xác định ($R^2$ Score):** `{r2*100:.2f}%`")

with col2:
    st.subheader("🏷️ Phân khúc thị trường (K-Means)")
    input_scaled = scaler.transform(input_encoded)
    cluster_pred = kmeans.predict(input_scaled)[0]
    
    cluster_names = {
        0: "🔥 Cụm 0: Phân khúc Cao cấp (Flagship)",
        1: "📱 Cụm 1: Phân khúc Phổ thông / Giá rẻ",
        2: "🔋 Cụm 2: Phân khúc Tầm trung / Pin trâu"
    }
    st.info(f"### Phân khúc: **{cluster_names.get(cluster_pred, f'Cụm {cluster_pred}')}**")
    st.write(f"- **Điểm Silhouette Score:** `{sil_val:.4f}`")

st.markdown("---")

# ==========================================
# TRỰC QUAN HÓA BIỂU ĐỒ
# ==========================================
st.subheader("📊 Biểu đồ phân cụm thị trường (RAM vs Giá)")
fig, ax = plt.subplots(figsize=(10, 4.5))

sns.scatterplot(
    data=df_raw, 
    x='ram(GB)', 
    y='price(USD)', 
    hue='Cluster', 
    palette='Set1', 
    ax=ax, 
    alpha=0.75, 
    s=60
)

# Đánh dấu vị trí thiết bị của người dùng chọn trên biểu đồ
ax.scatter([ram], [price_pred], color='black', s=250, marker='*', label='Thiết bị của bạn')
ax.set_title("Mối quan hệ RAM và Giá sản phẩm theo các cụm K-Means")
ax.legend(loc='upper left')
st.pyplot(fig)
