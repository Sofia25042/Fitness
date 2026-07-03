import streamlit as st
import pandas as pd
import numpy as np
import requests
import tensorflow as tf
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# --- CẤU HÌNH TRANG CHUYÊN NGHIỆP ---
st.set_page_config(
    page_title="Hệ Thống Trí Tuệ Nhân Tạo Hỗ Trợ Sức Khỏe Toàn Diện",
    layout="wide",
    page_icon="💪",
    initial_sidebar_state="expanded"
)

# --- TÙY CHỈNH GIAO DIỆN BẰNG CSS ĐÃ ĐƯỢC CHUẨN HÓA ---
st.markdown("""
    <style>
    .main-title { font-size: 38px !important; font-weight: 800; color: #FF4B4B; text-align: center; margin-bottom: 20px; }
    .sub-title { font-size: 16px !important; text-align: center; color: #6D7A96; margin-bottom: 40px; }
    .section-header { font-size: 24px !important; font-weight: 700; color: #1E293B; border-left: 5px solid #FF4B4B; padding-left: 10px; margin-bottom: 20px; }
    .card { background-color: #F8FAFC; padding: 20px; border-radius: 12px; border: 1px solid #E2E8F0; margin-bottom: 15px; line-height: 1.6; }
    .story-title { font-size: 18px !important; font-weight: bold; color: #10B981; margin-bottom: 8px; display: block; }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# 🤖 KHỐI TẢI MÔ HÌNH HỌC MÁY (ĐÃ SỬA LỖI TÌM FILE DỮ LIỆU)
# =========================================================
@st.cache_resource
def load_kaggle_model():
    possible_paths = [
        Path(__file__).parent / "model.keras",
        Path("SPCK/model.keras"),
        Path("model.keras")
    ]
    for path in possible_paths:
        if path.exists():
            try:
                return tf.keras.models.load_model(str(path))
            except Exception as e:
                st.error(f"❌ Lỗi định dạng file tại {path}: {e}")
                return None
    st.error("❌ Không tìm thấy file 'model.keras' trong cùng thư mục! Vui lòng kiểm tra lại cấu trúc thư mục SPCK.")
    return None

model = load_kaggle_model()

# =========================================================
# 📊 KHỐI THÔNG SỐ CHUẨN HÓA (STANDARD SCALER)
# =========================================================
MEANS = np.array([0.496, 42.79, 174.46, 74.96, 15.53, 95.51, 40.02], dtype=np.float32)
STDS  = np.array([0.500, 16.98,  14.26, 15.04,  8.32,  9.58,  1.41], dtype=np.float32)

def scale_input_data(raw_features):
    return (raw_features - MEANS) / STDS

# =========================================================
# 🗂️ TÁI TẠO TẬP DỮ LIỆU ĐỂ VẼ ĐỒ THỊ CHUẨN EDA KAGGLE
# =========================================================
@st.cache_data
def get_kaggle_dataset():
    possible_paths = [
        Path(__file__).parent / "calories_dataset.csv",
        Path("SPCK/calories_dataset.csv"),
        Path("calories_dataset.csv")
    ]
    for path in possible_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                df_translated = pd.DataFrame({
                    "Giới tính": df["Gender"].map({"male": "Nam", "female": "Nữ", "Nam": "Nam", "Nữ": "Nữ"}),
                    "Tuổi": df["Age"],
                    "Chiều cao (cm)": df["Height"],
                    "Cân nặng (kg)": df["Weight"],
                    "Thời gian tập (Phút)": df["Duration"],
                    "Nhịp tim (BPM)": df["Heart_Rate"],
                    "Nhiệt độ (°C)": df["Body_Temp"],
                    "Calories": df["Calories"]
                })
                return df_translated.dropna().reset_index(drop=True)
            except Exception as e:
                st.error(f"❌ Lỗi đọc file calories_dataset.csv: {e}")
                
    st.error("❌ Không tìm thấy file 'calories_dataset.csv'! Đang dùng dữ liệu ngẫu nhiên thay thế.")
    np.random.seed(42)
    size = 600
    gender = np.random.choice(["Nam", "Nữ"], size=size, p=[0.496, 0.504])
    age = np.random.normal(42.79, 16.98, size).clip(14, 80)
    height = np.random.normal(174.46, 14.26, size).clip(135, 210)
    weight = np.random.normal(74.96, 15.04, size).clip(38, 125)
    duration = np.random.normal(15.53, 8.32, size).clip(1, 30)
    heart_rate = np.random.normal(95.51, 9.58, size).clip(60, 150)
    body_temp = np.random.normal(40.02, 1.41, size).clip(36, 42)
    
    calories = (duration * 6.5) + (heart_rate * 1.2) + (weight * 0.35) - 110 + np.random.normal(0, 4, size)
    calories = calories.clip(3, 290)
    
    return pd.DataFrame({
        "Giới tính": gender, "Tuổi": age, "Chiều cao (cm)": height, "Cân nặng (kg)": weight,
        "Thời gian tập (Phút)": duration, "Nhịp tim (BPM)": heart_rate, "Nhiệt độ (°C)": body_temp, "Calories": calories
    })

df_clean = get_kaggle_dataset()

# =========================================================
# 🗂️ CẤU TRÚC ĐIỀU HƯỚNG THANH SIDEBAR & QUẢN LÝ API KEY
# =========================================================
with st.sidebar:
    st.markdown("<div style='text-align: center; margin-bottom: 20px;'><h2>⚙️ MENU HỆ THỐNG</h2></div>", unsafe_allow_html=True)
    menu = st.radio(
        label="Lựa chọn không gian làm việc:",
        options=["🏡 Tổng Quan Dự Án (Home)", "📊 Biểu Đồ Thống Kê & EDA (Chart)", "🤖 Tư Vấn Chuyên Sâu (Chat PT)", "🚀 Dự Đoán Thực Tế (Predict)"],
        index=0
    )

    st.write("---")
    st.caption("🤖 Trạng thái mô hình học máy:")
    if model is not None:
        st.success("🟢 model.keras đã sẵn sàng")
    else:
        st.error("🔴 Chưa kết nối được mô hình")

# =========================================================
# 🏡 1. GIAO DIỆN TRANG CHỦ (HOME)
# =========================================================
if "Home" in menu:
    st.markdown("<div class='main-title'>💪 Fitness AI Assistant Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Ứng dụng mạng thần kinh nhân tạo kết hợp mô hình ngôn ngữ lớn để quản lý năng lượng cơ thể</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='section-header'>📈 Chỉ Số Tổng Overview Bộ Dữ Liệu</div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="Tổng số mẫu dữ liệu", value="15,000", delta="+100%")
    with m2:
        st.metric(label="Thời gian tập trung bình", value="15.53 Phút", delta="Duration Avg")
    with m3:
        st.metric(label="Nhịp tim tập trung phổ biến", value="95.51 BPM", delta="Heart_rate Avg")
    with m4:
        st.metric(label="Độ chính xác mô hình (R2)", value="99.2%", delta="Mạng Thần Kinh")
        
    st.write("---")
    
    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown("<div class='section-header'>🎯 Mục Tiêu Sản Phẩm</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>Hệ thống này được phát triển nhằm mục đích cá nhân hóa hoàn toàn lộ trình tiêu hao năng lượng của người tập. Công nghệ Deep Learning giúp phân tích sâu các chỉ số sinh học kết hợp cường độ vận động tức thời nhằm đưa ra kết quả chính xác nhất cho từng cá nhân.</div>", unsafe_allow_html=True)
    with col_r:
        st.markdown("<div class='section-header'>🛠️ Hướng Dẫn Sử Dụng Nhanh</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><ul><li><b>Bước 1:</b> Vào tab <b>Chart</b> để xem phân tích khoa học đằng sau dữ liệu lớn.</li><li><b>Bước 2:</b> Vào tab <b>Predict</b> nhập chỉ số cá nhân để nhận kết quả tính calo từ AI.</li><li><b>Bước 3:</b> Sử dụng không gian <b>Chat PT</b> để lên thực đơn ăn uống phục hồi cơ bắp.</li></ul></div>", unsafe_allow_html=True)

# =========================================================
# 📈 2. GIAO DIỆN BIỂU ĐỒ & EDA CHI TIẾT (CHART)
# =========================================================
elif "Chart" in menu:
    st.markdown("<div class='main-title'>📊 Không Gian Khám Phá Dữ Liệu Lớn (EDA)</div>", unsafe_allow_html=True)
    
    tab_dist, tab_rel, tab_heat, tab_story = st.tabs([
        "📊 1. Phân Phối Đơn Biến", 
        "📈 2. Phân Tích Đa Biến Tương Quan", 
        "🌡️ 3. Ma Trận Hệ Số Nhiệt",
        "📖 4. Câu Chuyện Dữ Liệu Chi Tiết"
    ])
    
    with tab_dist:
        st.markdown("<div class='section-header'>🎯 Khảo sát phân bố của từng thuộc tính riêng lẻ</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.histogram(df_clean, x="Tuổi", nbins=20, color_discrete_sequence=["#3498DB"])
            st.plotly_chart(fig1, use_container_width=True)
            fig2 = px.histogram(df_clean, x="Chiều cao (cm)", nbins=20, color_discrete_sequence=["#2ECC71"])
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            fig3 = px.histogram(df_clean, x="Cân nặng (kg)", nbins=20, color_discrete_sequence=["#E67E22"])
            st.plotly_chart(fig3, use_container_width=True)
            fig4 = px.histogram(df_clean, x="Calories", nbins=20, color_discrete_sequence=["#E74C3C"])
            st.plotly_chart(fig4, use_container_width=True)

    with tab_rel:
        st.markdown("<div class='section-header'>📈 Khảo sát mối tương quan đa biến phức tạp</div>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            fig5 = px.scatter(df_clean, x="Thời gian tập (Phút)", y="Calories", color="Calories", color_continuous_scale="Viridis")
            st.plotly_chart(fig5, use_container_width=True)
            fig6 = px.scatter(df_clean, x="Nhịp tim (BPM)", y="Calories", color="Calories", color_continuous_scale="Plasma")
            st.plotly_chart(fig6, use_container_width=True)
        with c4:
            fig7 = px.box(df_clean, x="Giới tính", y="Calories", color="Giới tính", color_discrete_sequence=["#2980B9", "#9B59B6"])
            st.plotly_chart(fig7, use_container_width=True)
            fig8 = px.box(df_clean, y="Calories", color_discrete_sequence=["#1ABC9C"])
            st.plotly_chart(fig8, use_container_width=True)

    with tab_heat:
        st.markdown("<div class='section-header'>🌡️ Ma trận hệ số tương quan tuyến tính (Pearson Correlation)</div>", unsafe_allow_html=True)
        num_cols = ["Tuổi", "Chiều cao (cm)", "Cân nặng (kg)", "Thời gian tập (Phút)", "Nhịp tim (BPM)", "Nhiệt độ (°C)", "Calories"]
        corr_matrix = df_clean[num_cols].corr().values
        fig9 = go.Figure(data=go.Heatmap(z=corr_matrix, x=num_cols, y=num_cols, colorscale="YlOrRd", zmin=-1, zmax=1))
        st.plotly_chart(fig9, use_container_width=True)

    with tab_story:
        st.markdown("<div class='section-header'>📖 Toàn Bộ 4 Giai Đoạn Câu Chuyện Dữ Liệu Kaggle</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><span class='story-title'>🚀 Giai đoạn 1: Nguồn gốc của các con số</span>Tập dữ liệu gốc bao gồm thông tin cá nhân của 15.000 người tập và nhật ký cảm biến sinh học. Thuật toán tiền xử lý <b>StandardScaler</b> được kích hoạt nhằm đưa mọi biến về cùng phân phối chuẩn có trung bình bằng 0 và độ lệch chuẩn bằng 1.</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><span class='story-title'>🔍 Giai đoạn 2: Khám phá bí ẩn từ biểu đồ đơn biến</span>Nhìn vào nhóm 4 đồ thị phân phối, ta thấy Tuổi, Chiều cao và Cân nặng phân bố đều hình chuông. Tuy nhiên, đồ thị Calories tiêu hao lại lệch phải rõ rệt, chứng minh đa số người tập lựa chọn các bài tập cường độ ngắn và trung bình.</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><span class='story-title'>📈 Giai đoạn 3: Cuộc chiến của các trọng số đặc trưng (Features)</span>Khi tiến hành vẽ ma trận nhiệt Heatmap, thuật toán bóc tách ra <b>Thời gian tập (Duration)</b> và <b>Nhịp tim (Heart Rate)</b> có hệ số tương quan tuyến tính thực tế cực kỳ cao (>0.89) với lượng calo tiêu hao.</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'><span class='story-title'>🧠 Giai đoạn 4: Từ dữ liệu đến mô hình Mạng Thần Kinh (Neural Network)</span>Mô hình mạng thần kinh nhân tạo gồm các tầng ẩn kết hợp hàm kích hoạt phi tuyến <b>ReLU</b> giúp AI tự học và nhận diện được sự phối hợp ngầm giữa các chỉ số sinh học phức tạp này.</div>", unsafe_allow_html=True)

# =========================================================
# 🤖 3. GIAO DIỆN TRÒ CHUYỆN PT ẢO (CHAT PT) - ĐÃ BẢO MẬT KEY
# =========================================================
elif "Chat" in menu:
    st.markdown("<div class='main-title'>🤖 PT Ảo Chuyên Nghiệp - Trợ Lý Sức Khỏe AI</div>", unsafe_allow_html=True)
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
    if user_prompt := st.chat_input("Hỏi PT ảo tại đây..."):
        with chat_container:
            with st.chat_message("user"): st.markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
            except Exception:
                api_key = None
        
        if not api_key:
            ai_response = "❌ **Thiếu API Key:** Vui lòng thiết lập GEMINI_API_KEY trong file secrets.toml để sử dụng chatbot."
        else:
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key
            }
            payload = {
                "system_instruction": {
                    "parts": [{
                        "text": "Bạn là một Huấn luyện viên cá nhân (PT) và chuyên gia dinh dưỡng cao cấp. Hãy trả lời thân thiện, chi tiết, khoa học bằng tiếng Việt."
                    }]
                },
                "contents": [{
                    "parts": [{"text": user_prompt}]
                }]
            }
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=30)
                if response.status_code == 200:
                    res_json = response.json()
                    ai_response = res_json['candidates'][0]['content']['parts'][0]['text']
                elif response.status_code in (400, 401, 403):
                    ai_response = f"❌ **Lỗi API Gemini (Mã lỗi: {response.status_code}):** API Key sai hoặc không hợp lệ."
                else:
                    error_detail = response.text[:300] if response.text else "Không có chi tiết lỗi."
                    ai_response = f"❌ Lỗi hệ thống API (Mã: {response.status_code}): {error_detail}"
            except Exception as e:
                ai_response = f"❌ Lỗi kết nối mạng: {str(e)}"
        
        with chat_container:
            with st.chat_message("assistant"): st.markdown(ai_response)
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

# =========================================================
# 🚀 4. GIAO DIỆN DỰ ĐOÁN MÔ HÌNH KAGGLE (PREDICT)
# =========================================================
elif "Predict" in menu:
    st.markdown("<div class='main-title'>🚀 Tính Toán Năng Lượng Tiêu Hao Bằng Deep Learning</div>", unsafe_allow_html=True)
    col_in, col_out = st.columns([1.3, 1], gap="large")
    
    with col_in:
        st.markdown("<div class='section-header'>📋 Nhập Thông Số Buổi Tập Luyện</div>", unsafe_allow_html=True)
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            gender_input = st.radio("Giới tính sinh học:", ("Nam", "Nữ"))
            age_input = st.slider("Độ tuổi hiện tại (Tuổi):", min_value=10, max_value=90, value=25)
            height_input = st.number_input("Chiều cao cơ thể (cm):", min_value=100, max_value=220, value=170)
            weight_input = st.number_input("Cân nặng thực tế (kg):", min_value=30, max_value=150, value=65)
        with sub_c2:
            duration_input = st.number_input("Thời gian vận động (Phút):", min_value=1, max_value=120, value=20)
            heart_rate_input = st.slider("Nhịp tim đo được (BPM):", min_value=50, max_value=200, value=100)
            body_temp_input = st.slider("Nhiệt độ cơ thể (°C):", min_value=35.0, max_value=42.0, value=37.0, step=0.1)
        gender_encoded = 1 if gender_input == "Nam" else 0
        
    with col_out:
        st.markdown("<div class='section-header'>🔮 Phân Tích & Dự Báo Từ Mô Hình</div>", unsafe_allow_html=True)
        if st.button("🔥 KÍCH HOẠT DỰ ĐOÁN AI", type="primary", use_container_width=True):
            if model is not None:
                raw_data = np.array([gender_encoded, age_input, height_input, weight_input, duration_input, heart_rate_input, body_temp_input], dtype=np.float32)
                scaled_data = scale_input_data(raw_data)
                input_matrix = scaled_data.reshape(1, -1)
                
                prediction = model.predict(input_matrix)
                calo_result = float(prediction[0][0])
                if calo_result < 0: calo_result = 0.0
                
                st.success("### 🎉 Kết quả tính toán hoàn tất!")
                st.metric(label="LƯỢNG CALO TIÊU HAO THỰC TẾ", value=f"{calo_result:.2f} kcal")
            else:
                st.error("Mô hình chưa được tải thành công. Vui lòng kiểm tra lại file model.keras")
