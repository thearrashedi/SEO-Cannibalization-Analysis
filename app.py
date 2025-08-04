import streamlit as st
import pandas as pd
import google.generativeai as genai
from keyword_cannibalization import run_cannibalization_analysis
import altair as alt

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="تحلیل هوشمند هم‌نوع‌خواری SEO",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Session State Initialization ---
# This is crucial for a professional app. It preserves results across reruns.
if 'results' not in st.session_state:
    st.session_state.results = None
if 'running' not in st.session_state:
    st.session_state.running = False

# --- 3. Custom UI Enhancements ---
st.markdown("""
<style>
    /* General Style Improvements */
    .stApp {
        background-color: #FFFFFF;
    }
    .stButton>button {
        border-radius: 8px;
        border: 2px solid #FF4B4B;
        color: #FF4B4B;
        background-color: transparent;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        border-color: #FF4B4B;
        background-color: #FF4B4B;
        color: white;
    }
    .stButton>button:disabled {
        border: 1px solid #E0E0E0;
        background-color: #F0F2F6;
        color: #A0A0A0;
    }
    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 1px solid #E6E6E6;
    }
    [data-testid="stMetric"] {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. Helper function to make URLs in DataFrame clickable ---
def make_urls_clickable(df):
    """Adds clickable HTML links to the 'URLs' column."""
    if 'URLs' in df.columns:
        df['URLs'] = df['URLs'].apply(
            lambda x: ' | '.join([f'<a href="{url.strip()}" target="_blank">{url.strip()}</a>' for url in x.split('|')]) if isinstance(x, str) else x
        )
    return df

# --- 5. Sidebar UI for Inputs ---
with st.sidebar:
    st.image("https://i.imgur.com/g4f4g4a.png", width=120) # A slightly larger logo
    st.title("پارامترهای تحلیل")
    
    api_key = st.text_input("🔑 کلید Google AI API", type="password", placeholder="AIzaSy...", help="کلید خود را از [Google AI Studio](https://aistudio.google.com/app/apikey) دریافت کنید.")
    site_url = st.text_input("🌐 دامنه وب‌سایت", placeholder="aiprovider.org", help="دامنه اصلی وب‌سایت بدون https.")
    
    uploaded_file = st.file_uploader("📂 فایل اکسل", type=['xlsx'], help="فایل اکسل باید ستون‌هایی برای کلمات کلیدی و URLها داشته باشد.")

    st.markdown("---")
    
    # The button now just toggles the 'running' state
    if st.button("🚀 شروع تحلیل", type="primary", use_container_width=True, disabled=(not uploaded_file or not api_key or not site_url)):
        st.session_state.running = True

# --- 6. Main Panel Logic ---
st.title("🧠 داشبورد هوشمند تحلیل هم‌نوع‌خواری")

# A. If analysis is triggered
if st.session_state.running:
    with st.status("در حال اجرای تحلیل...", expanded=True) as status:
        try:
            status.update(label="مرحله ۱: اعتبارسنجی ورودی‌ها و پیکربندی Gemini...", state="running")
            genai.configure(api_key=api_key)
            df = pd.read_excel(uploaded_file)
            
            config = {
                "api_key": api_key,
                "site_url": site_url,
                "country": "ir", # Country is fixed for now, can be added to sidebar later if needed
                "model_provider": "gemini"
            }
            status.update(label=f"مرحله ۲: خواندن {len(df)} ردیف از فایل و شروع دسته‌بندی با هوش مصنوعی...", state="running")

            # Run the robust backend analysis function
            result_df, analysis_summary = run_cannibalization_analysis(df, config)
            
            # Add a Severity Score for prioritization
            result_df['Severity'] = (result_df['Unique URLs'] - 1) * result_df['Keyword Count']
            
            # Store results in session state
            st.session_state.results = {'df': result_df, 'summary': analysis_summary}
            
            status.update(label="تحلیل با موفقیت کامل شد!", state="complete", expanded=False)

        except Exception as e:
            status.update(label="خطا در حین تحلیل!", state="error", expanded=True)
            st.exception(e)
        
        # Reset the running flag and rerun the script to display results
        st.session_state.running = False
        st.rerun()

# B. If results exist in the session state, display them
if st.session_state.results:
    summary = st.session_state.results['summary']
    results_df = st.session_state.results['df']

    # --- Display Summary Metrics ---
    st.subheader("📊 خلاصه مدیریتی")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("کل کلمات کلیدی تحلیل‌شده", summary.get('total_keywords', 'N/A'))
    col2.metric("🚨 مشکلات هم‌نوع‌خواری", summary.get('total_issues_found', 'N/A'), help="تعداد موضوعاتی که بیش از یک URL برای آن‌ها هدف‌گذاری شده است.")
    col3.metric("🧠 موضوعات منحصر به فرد", summary.get('unique_topics', 'N/A'), help="تعداد دسته‌بندی‌های موضوعی شناسایی‌شده توسط Gemini.")
    col4.metric("⏱️ زمان تحلیل (ثانیه)", summary.get('analysis_duration_seconds', 'N/A'))

    st.markdown("---")
    
    # --- Interactive Results Exploration ---
    st.subheader("👇 نتایج را بررسی و فیلتر کنید")
    filter_option = st.selectbox("نمایش نتایج:", ["🚨 فقط مشکلات", "همه موضوعات"])
    
    display_df = results_df.copy()
    if filter_option == "🚨 فقط مشکلات":
        display_df = display_df[display_df['Cannibalization Issue'].str.contains('بله')]

    if not display_df.empty:
        # --- Data Visualization ---
        chart_df = display_df[display_df['Severity'] > 0].nlargest(10, 'Severity').sort_values('Severity', ascending=True)
        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X('Severity:Q', title='امتیاز شدت مشکل'),
            y=alt.Y('Topic:N', sort='-x', title='موضوع'),
            tooltip=['Topic', 'Keyword Count', 'Unique URLs', 'Severity']
        ).properties(
            title='۱۰ موضوع با بیشترین شدت هم‌نوع‌خواری'
        )
        st.altair_chart(chart, use_container_width=True)

        # --- Actionable Data Table ---
        # Make URLs clickable before displaying
        display_df_html = make_urls_clickable(display_df)
        st.write("جدول کامل نتایج (مرتب‌شده بر اساس شدت مشکل):")
        st.write(display_df_html.to_html(escape=False, index=False), unsafe_allow_html=True)

        # --- Download Button ---
        csv = results_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 دانلود همه نتایج به صورت CSV",
            data=csv,
            file_name=f'cannibalization_results_{site_url}.csv',
            mime='text/csv',
            use_container_width=True
        )
    else:
        st.success("🎉 هیچ مشکلی یافت نشد!")

# C. Initial state of the app
else:
    tab_guide, tab_about = st.tabs(["📄 راهنمای سریع", "💡 درباره ابزار"])
    with tab_guide:
        st.info("برای شروع، پارامترهای تحلیل را در پنل کناری وارد کرده و روی دکمه 'شروع تحلیل' کلیک کنید.")
        st.markdown("""
        **راهنمای فایل اکسل:**
        - فایل شما می‌تواند خروجی مستقیم **Google Search Console** باشد.
        - این ابزار به طور هوشمند ستون‌های حاوی `Query` یا `Keyword` را به عنوان کلمه کلیدی، و ستون‌های حاوی `Page` یا `URL` را به عنوان آدرس شناسایی می‌کند.
        """)
    with tab_about:
        st.markdown("""
        این ابزار با استفاده از هوش مصنوعی **Google Gemini**، کلمات کلیدی شما را بر اساس **قصد کاربر (Search Intent)** به موضوعات مختلف دسته‌بندی می‌کند. سپس با بررسی URLهای هدف‌گذاری شده برای هر موضوع، مشکلات هم‌نوع‌خواری را شناسایی کرده و به شما در اولویت‌بندی و حل آن‌ها کمک می‌کند.
        """)
