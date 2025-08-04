import streamlit as st
import pandas as pd
import json
from keyword_cannibalization import run_cannibalization_analysis
import google.generativeai as genai

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="تحلیل هم‌نوع‌خواری SEO",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS for a cleaner look ---
st.markdown("""
<style>
    .stButton>button {
        border-radius: 10px;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    .stFileUploader>div>div>button {
        border-radius: 10px;
    }
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    h1, h2, h3 {
        color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Sidebar for Inputs ---
with st.sidebar:
    st.image("https://i.imgur.com/g4f4g4a.png", width=100) # Replace with your logo if you have one
    st.title("تنظیمات تحلیل")
    st.markdown("پارامترهای تحلیل خود را در اینجا وارد کنید.")

    # Gemini API Key Input
    api_key = st.text_input(
        "🔑 کلید Google AI API",
        type="password",
        placeholder="AIzaSy...",
        help="برای استفاده از مدل Gemini جهت تحلیل موضوعی، کلید خود را از [Google AI Studio](https://aistudio.google.com/app/apikey) دریافت کنید."
    )

    # Website Domain Input
    site_url = st.text_input(
        "🌐 آدرس وب‌سایت",
        placeholder="https://aiprovider.org",
        help="آدرس کامل وب‌سایت مورد تحلیل."
    )

    # Country Selection
    country = st.selectbox(
        "🌍 کشور",
        ['ir', 'us', 'de', 'fr', 'es', 'it', 'uk', 'ca', 'au'],
        index=0,
        help="کشور مورد نظر برای بررسی رتبه‌بندی."
    )

    # File Uploader
    uploaded_file = st.file_uploader(
        "📂 فایل اکسل (Keyword & URL)",
        type=['xlsx'],
        help="فایل اکسل باید دو ستون به نام‌های 'Keyword' و 'URL' داشته باشد."
    )

    st.markdown("---")
    # Analysis Button
    run_button = st.button("🚀 شروع تحلیل", type="primary", use_container_width=True, disabled=(not uploaded_file or not api_key or not site_url))

# --- 4. Main Panel with Tabs ---
st.title("🔍 داشبورد تحلیل هم‌نوع‌خواری کلمات کلیدی")

tab1, tab2, tab3 = st.tabs(["📊 نتایج تحلیل", "📄 راهنمای استفاده", "💡 درباره ابزار"])

with tab1:
    st.header("نتایج تحلیل شما")
    if run_button:
        # --- Run Analysis ---
        try:
            # Configure the Gemini client
            genai.configure(api_key=api_key)

            with st.spinner("لطفاً صبر کنید... در حال دریافت داده‌ها و تحلیل با هوش مصنوعی... این فرآیند ممکن است زمان‌بر باشد."):
                df = pd.read_excel(uploaded_file)
                st.info(f"✅ فایل با موفقیت خوانده شد. **{len(df)}** کلمه کلیدی برای تحلیل یافت شد.")
                
                # We pass the api_key via config, so keyword_cannibalization.py can use it
                config = {
                    "api_key": api_key,
                    "site_url": site_url,
                    "country": country,
                    "model_provider": "gemini" # Let the backend know we are using Gemini
                }

                result_df, analysis_summary = run_cannibalization_analysis(df, config)

            st.success("✅ تحلیل با موفقیت به پایان رسید!")

            # --- Display Summary Metrics ---
            st.subheader("خلاصه نتایج")
            col1, col2, col3 = st.columns(3)
            col1.metric("کل کلمات کلیدی", analysis_summary.get('total_keywords', 'N/A'))
            col2.metric("مشکلات یافت‌شده", analysis_summary.get('total_issues_found', 'N/A'), help="تعداد گروه‌هایی که بیش از یک URL برای یک موضوع دارند.")
            col3.metric("موضوعات منحصر به فرد", analysis_summary.get('unique_topics', 'N/A'), help="تعداد دسته‌بندی‌های موضوعی که توسط Gemini شناسایی شد.")
            
            # --- Display Full Results ---
            st.subheader("جدول نتایج کامل")
            st.dataframe(result_df, use_container_width=True)

            # --- Download Button ---
            @st.cache_data
            def convert_df_to_csv(df_to_convert):
                return df_to_convert.to_csv(index=False).encode('utf-8-sig') # Use utf-8-sig for better Excel compatibility

            csv = convert_df_to_csv(result_df)
            st.download_button(
                label="📥 دانلود نتایج به صورت CSV",
                data=csv,
                file_name='seo_cannibalization_results.csv',
                mime='text/csv',
                use_container_width=True
            )

        except Exception as e:
            st.error(f"❌ متاسفانه در حین تحلیل خطایی رخ داد:")
            st.exception(e)
    else:
        st.info("اطلاعات را در پنل سمت راست وارد کرده و روی دکمه 'شروع تحلیل' کلیک کنید.")

with tab2:
    st.header("راهنمای استفاده از ابزار")
    st.markdown("""
    برای استفاده صحیح از این ابزار، لطفاً مراحل زیر را دنبال کنید:

    **۱. آماده‌سازی فایل اکسل:**
    - یک فایل اکسل (`.xlsx`) ایجاد کنید.
    - دو ستون با نام‌های دقیق `Keyword` و `URL` بسازید.
    - در ستون `Keyword`، کلمات کلیدی مورد نظر خود را وارد کنید.
    - در ستون `URL`، آدرس صفحه‌ای که برای آن کلمه کلیدی هدف‌گذاری کرده‌اید را وارد کنید.
    
    *نمونه فایل:*
    | Keyword                 | URL                                     |
    |-------------------------|-----------------------------------------|
    | خرید هاست ارزان         | https://aiprovider.org/hosting          |
    | بهترین هاست وردپرس     | https://aiprovider.org/wordpress-hosting|
    | هاست لینوکس ایران      | https://aiprovider.org/hosting          |

    **۲. دریافت کلید API گوگل:**
    - به [Google AI Studio](https://aistudio.google.com/app/apikey) بروید و یک کلید API جدید بسازید.
    - کلید ساخته شده را در فیلد مربوطه در پنل کناری وارد کنید. این کلید برای استفاده از مدل هوش مصنوعی Gemini جهت دسته‌بندی موضوعی کلمات کلیدی شما ضروری است.

    **۳. وارد کردن اطلاعات:**
    - کلید API، آدرس کامل وب‌سایت و کشور مورد نظر را در پنل کناری وارد کنید.
    - فایل اکسل آماده شده را آپلود کنید.

    **۴. شروع تحلیل:**
    - روی دکمه "شروع تحلیل" کلیک کنید و منتظر بمانید تا نتایج نمایش داده شوند.
    """)

with tab3:
    st.header("درباره این ابزار")
    st.markdown("""
    این ابزار برای کمک به متخصصان سئو جهت شناسایی و حل مشکل **هم‌نوع‌خواری کلمات کلیدی (Keyword Cannibalization)** طراحی شده است.
    
    **هم‌نوع‌خواری چه زمانی رخ می‌دهد؟**
    زمانی که چندین صفحه از وب‌سایت شما برای یک کلمه کلیدی یا یک موضوع مشابه در نتایج جستجوی گوگل با یکدیگر رقابت می‌کنند. این امر باعث سردرگمی موتورهای جستجو شده و می‌تواند به رتبه هر دو صفحه آسیب بزند.
    
    **این ابزار چگونه کار می‌کند؟**
    ۱. **دسته‌بندی موضوعی:** با استفاده از هوش مصنوعی Google Gemini، کلمات کلیدی شما را بر اساس موضوع و قصد کاربر (Intent) دسته‌بندی می‌کند.
    ۲. **تحلیل URLها:** در هر گروه موضوعی، URLهایی که شما برای کلمات کلیدی آن گروه هدف‌گذاری کرده‌اید را بررسی می‌کند.
    ۳. **شناسایی مشکل:** اگر برای یک موضوع واحد، بیش از یک URL منحصر به فرد پیدا شود، آن را به عنوان یک مشکل "هم‌نوع‌خواری" شناسایی و گزارش می‌کند.
    """)
