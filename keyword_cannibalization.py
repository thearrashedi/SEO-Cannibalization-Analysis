import pandas as pd
import json
import time
import logging
from typing import Tuple, Dict, Any, List
import google.generativeai as genai

# --- 1. Configuration & Constants ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants for column names to avoid typos and for easier maintenance
KEYWORD_COL = 'Keyword'
URL_COL = 'URL'
TOPIC_COL = 'Topic'

def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and validates the input DataFrame.
    - Standardizes column names.
    - Removes empty rows and duplicates.
    """
    if df is None or df.empty:
        raise ValueError("فایل اکسل آپلود شده خالی است.")

    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from column names

    # Flexible column mapping to find keyword and URL columns
    column_mapping = {}
    keyword_aliases = ['keyword', 'کلمه کلیدی', 'query', 'کوئری']
    url_aliases = ['url', 'آدرس', 'permalink', 'page', 'صفحه']

    for col in df.columns:
        col_lower = col.lower()
        if not column_mapping.get(KEYWORD_COL) and any(alias in col_lower for alias in keyword_aliases):
            column_mapping[col] = KEYWORD_COL
        elif not column_mapping.get(URL_COL) and any(alias in col_lower for alias in url_aliases):
            column_mapping[col] = URL_COL
            
    df = df.rename(columns=column_mapping)

    # Final validation check
    if KEYWORD_COL not in df.columns or URL_COL not in df.columns:
        error_message = (
            f"فایل اکسل شما باید دارای ستون‌هایی برای کلمه کلیدی و آدرس (URL) باشد. "
            f"نام‌های ستون شناسایی‌شده: {list(df.columns)}"
        )
        raise ValueError(error_message)

    # Clean data: drop rows with missing keywords or URLs, and remove duplicate keywords
    df.dropna(subset=[KEYWORD_COL, URL_COL], inplace=True)
    df.drop_duplicates(subset=[KEYWORD_COL], inplace=True)
    
    if df.empty:
        raise ValueError("پس از پاکسازی، هیچ ردیف معتبری در فایل اکسل باقی نماند.")

    return df

def get_topic_from_gemini(keywords: List[str], api_key: str, model_name: str = 'gemini-1.5-flash-latest') -> Dict[str, str]:
    """
    Groups keywords into topics using Google's Gemini model with batch processing.
    """
    if not api_key:
        logging.warning("کلید Gemini API ارائه نشده. از دسته‌بندی عمومی استفاده می‌شود.")
        return {kw: "General" for kw in keywords}

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    all_topic_map = {}
    batch_size = 200  # Process 200 keywords at a time to stay within API limits
    num_batches = (len(keywords) + batch_size - 1) // batch_size

    for i in range(0, len(keywords), batch_size):
        batch_keywords = keywords[i:i + batch_size]
        current_batch_num = (i // batch_size) + 1
        logging.info(f"پردازش دسته {current_batch_num}/{num_batches} از کلمات کلیدی...")

        keyword_list_str = "\n".join([f"- {kw}" for kw in batch_keywords])
        prompt = f"""
        Based on user search intent, group the following SEO keywords into concise, relevant topics.
        The topic name should be 1-3 words. The goal is to group keywords that a user would search for to solve the same problem.
        The output MUST be a valid JSON object where keys are the keywords and values are the topic names.
        Example: {{"buy cheap hosting": "Hosting Purchase", "best wordpress hosting": "Hosting Purchase", "what is seo": "SEO Information"}}

        Keywords to categorize:
        {keyword_list_str}

        JSON Output:
        """
        
        try:
            response = model.generate_content(prompt)
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
            
            try:
                batch_topic_map = json.loads(cleaned_response)
                all_topic_map.update(batch_topic_map)
            except json.JSONDecodeError:
                logging.error(f"پاسخ Gemini در دسته {current_batch_num} فرمت JSON معتبر نداشت. پاسخ دریافتی: {cleaned_response}")
                # Fallback for this batch
                for kw in batch_keywords:
                    all_topic_map[kw] = "JSON Format Error"

        except Exception as e:
            logging.error(f"خطا در ارتباط با Gemini API در دسته {current_batch_num}: {e}")
            for kw in batch_keywords:
                all_topic_map[kw] = "API Error"
        
        time.sleep(1) # Be respectful to the API rate limits

    # Ensure all keywords have a topic, even if the API missed some
    for kw in keywords:
        if kw not in all_topic_map:
            all_topic_map[kw] = "Uncategorized"
            
    return all_topic_map

def _analyze_topics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs the core cannibalization analysis on a DataFrame that already has topics.
    """
    analysis_results = []
    grouped_by_topic = df.groupby(TOPIC_COL)

    for topic, group in grouped_by_topic:
        unique_urls = group[URL_COL].dropna().unique().tolist()
        issue_exists = len(unique_urls) > 1
        
        analysis_results.append({
            'Topic': topic,
            'Keyword Count': len(group),
            'Unique URLs': len(unique_urls),
            'URLs': " | ".join(unique_urls),
            'Keywords': ", ".join(group[KEYWORD_COL].tolist()),
            'Cannibalization Issue': '🚨 بله' if issue_exists else '✅ خیر'
        })

    result_df = pd.DataFrame(analysis_results)
    # Sort to show issues first, then by keyword count
    result_df = result_df.sort_values(by=['Cannibalization Issue', 'Keyword Count'], ascending=[False, False])
    return result_df

def run_cannibalization_analysis(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main function to run the cannibalization analysis.
    """
    start_time = time.time()
    
    # --- 1. Preparation & Validation ---
    logging.info("شروع پاکسازی و اعتبارسنجی دیتافریم...")
    df = _prepare_dataframe(df)
    logging.info("پاکسازی و اعتبارسنجی با موفقیت انجام شد.")

    keywords = df[KEYWORD_COL].tolist()

    # --- 2. Topic Modeling using AI ---
    logging.info(f"شروع دسته‌بندی موضوعی برای {len(keywords)} کلمه کلیدی...")
    topic_map = get_topic_from_gemini(keywords, config.get("api_key"))
    df[TOPIC_COL] = df[KEYWORD_COL].map(topic_map)
    logging.info("دسته‌بندی موضوعی با موفقیت به پایان رسید.")

    # --- 3. Cannibalization Analysis ---
    logging.info("شروع تحلیل هم‌نوع‌خواری...")
    result_df = _analyze_topics(df)
    logging.info("تحلیل با موفقیت انجام شد.")
    
    end_time = time.time()
    
    # --- 4. Create Summary ---
    analysis_summary = {
        "total_keywords": len(df),
        "unique_topics": df[TOPIC_COL].nunique(),
        "total_issues_found": int(result_df['Cannibalization Issue'].str.contains('بله').sum()),
        "analysis_duration_seconds": round(end_time - start_time, 2)
    }

    return result_df, analysis_summary
