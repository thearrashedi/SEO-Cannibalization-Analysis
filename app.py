import streamlit as st
import pandas as pd
import io
import json
from keyword_cannibalization import run_cannibalization_analysis
from typing import Optional

# --- Page Configuration ---
st.set_page_config(
    page_title="SEO Cannibalization Analysis",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Main Application UI ---

st.title("📊 SEO Keyword Cannibalization Analysis Tool")
st.markdown("""
    این ابزار به شما کمک می‌کند تا مشکلات "هم‌نوع‌خواری کلمات کلیدی" را در وب‌سایت خود شناسایی کنید. 
    یک فایل اکسل حاوی کلمات کلیدی و URLهای مربوطه را آپلود کنید تا تحلیل شروع شود.
""")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ تنظیمات تحلیل")

    # 1. File Upload
    uploaded_file = st.file_uploader(
        "۱. فایل اکسل خود را آپلود کنید",
        type=['xlsx'],
        help="فایل اکسل باید حداقل دو ستون به نام‌های 'Keyword' و 'URL' داشته باشد."
    )

    # 2. OpenAI API Key
    openai_api_key = st.text_input(
        "۲. کلید OpenAI API",
        type="password",
        placeholder="sk-...",
        help="برای استفاده از مدل‌های زبان برای دسته‌بندی موضوعی کلمات کلیدی ضروری است."
    )

    # 3. Website Domain
    site_url = st.text_input(
        "۳. آدرس وب‌سایت",
        placeholder="https.example.com",
        help="آدرس کامل وب‌سایت خود را وارد کنید (مثلاً https://www.aiprovider.org)."
    )

    # 4. Country Selection
    country = st.selectbox(
        "۴. کشور مورد نظر برای جستجو",
        ['ir', 'us', 'de', 'fr', 'es', 'it', 'uk', 'ca', 'au'],
        index=0,  # Default to 'ir'
        help="کشوری که می‌خواهید رتبه‌بندی کلمات کلیدی در آن بررسی شود."
    )

    # 5. Analysis Button
    st.markdown("---")
    run_button = st.button("🚀 شروع تحلیل", type="primary", use_container_width=True)

# --- Main Panel for Outputs ---

if run_button:
    # Input validation
    if not uploaded_file:
        st.error("لطفاً یک فایل اکسل آپلود کنید.")
    elif not openai_api_key:
        st.error("لطفاً کلید OpenAI API خود را وارد کنید.")
    elif not site_url:
        st.error("لطفاً آدرس وب‌سایت خود را وارد کنید.")
    else:
        try:
            with st.spinner("لطفاً صبر کنید، تحلیل در حال انجام است. این فرآیند ممکن است چند دقیقه طول بکشد..."):
                # Read the uploaded file into a pandas DataFrame
                df = pd.read_excel(uploaded_file)
                st.info(f"فایل شما با موفقیت خوانده شد. تعداد {len(df)} کلمه کلیدی برای تحلیل یافت شد.")

                # Prepare the configuration dictionary
                config = {
                    "openai_api_key": openai_api_key,
                    "site_url": site_url,
                    "country": country
                }

                # Run the main analysis function
                result_df, analysis_summary = run_cannibalization_analysis(df, config)

                # --- Display Results ---
                st.success("✅ تحلیل با موفقیت به پایان رسید!")

                st.subheader("📝 خلاصه تحلیل")
                st.json(analysis_summary)

                st.subheader("📄 نتایج کامل")
                st.dataframe(result_df)

                # Provide a download button for the results
                @st.cache_data
                def convert_df_to_csv(df_to_convert):
                    return df_to_convert.to_csv(index=False).encode('utf-8')

                csv = convert_df_to_csv(result_df)

                st.download_button(
                    label="📥 دانلود نتایج به صورت CSV",
                    data=csv,
                    file_name='seo_cannibalization_results.csv',
                    mime='text/csv',
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"متاسفانه در حین تحلیل خطایی رخ داد:")
            st.exception(e)

else:
    st.info("لطفاً تنظیمات را در منوی سمت چپ وارد کرده و روی دکمه 'شروع تحلیل' کلیک کنید.")
