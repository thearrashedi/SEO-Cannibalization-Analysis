import pandas as pd
import json
import time
import logging
from typing import Tuple, Dict, Any
import google.generativeai as genai
from langchain_openai import OpenAIEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_topic_from_gemini(keywords: list, api_key: str) -> Dict[str, str]:
    """
    Groups keywords into topics using Google's Gemini model.
    """
    if not api_key:
        logging.warning("Gemini API key not provided. Falling back to general categorization.")
        return {kw: "General" for kw in keywords}
    
    try:
        # Configuration is done in app.py
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        # Create a single prompt for batch processing for efficiency
        keyword_list_str = "\n".join([f"- {kw}" for kw in keywords])
        
        prompt = f"""
        Based on user search intent, group the following SEO keywords into concise, relevant topics.
        The topic name should be 1-3 words. The goal is to group keywords that a user would search for to solve the same problem.
        The output MUST be a valid JSON object where keys are the keywords and values are the topic names.
        Example: {{"buy cheap hosting": "Hosting Purchase", "best wordpress hosting": "Hosting Purchase", "what is seo": "SEO Information"}}

        Keywords to categorize:
        {keyword_list_str}

        JSON Output:
        """
        
        response = model.generate_content(prompt)
        # Clean the response to ensure it's valid JSON
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        topic_map = json.loads(cleaned_response)
        
        # Ensure all keywords from input are in the output map
        for kw in keywords:
            if kw not in topic_map:
                topic_map[kw] = "Uncategorized"
                
        return topic_map
        
    except Exception as e:
        logging.error(f"Error with Gemini API: {e}")
        # Fallback to a general topic if API fails
        return {kw: "API Error - General" for kw in keywords}

def get_topic_from_openai(keywords: list, api_key: str) -> Dict[str, str]:
    """
    Placeholder for OpenAI topic modeling. 
    For this version, we will just use a simplified logic or you can integrate a full OpenAI call.
    """
    logging.warning("OpenAI topic modeling is not fully implemented in this version. Using general categorization.")
    # You can add a full OpenAI API call here if needed in the future.
    return {kw: "General (OpenAI)" for kw in keywords}

def run_cannibalization_analysis(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main function to run the cannibalization analysis.
    This version is simplified to focus on topic modeling and URL analysis.
    """
    start_time = time.time()
    
    # Extract config values
    api_key = config.get("api_key")
    model_provider = config.get("model_provider", "gemini") # Default to Gemini

    # --- 1. Smart Input Validation & Column Standardization ---
    df.columns = df.columns.str.strip() # Remove leading/trailing spaces from column names
    
    # Flexible column mapping
    column_mapping = {}
    found_keyword = False
    found_url = False

    for col in df.columns:
        col_lower = col.lower()
        if not found_keyword and ('keyword' in col_lower or 'کلمه کلیدی' in col_lower):
            column_mapping[col] = 'Keyword'
            found_keyword = True
        elif not found_url and ('url' in col_lower or 'آدرس' in col_lower or 'permalink' in col_lower):
            column_mapping[col] = 'URL'
            found_url = True
            
    df = df.rename(columns=column_mapping)

    if 'Keyword' not in df.columns or 'URL' not in df.columns:
        raise ValueError("فایل اکسل شما باید دارای ستون‌هایی برای 'کلمه کلیدی' و 'آدرس (URL)' باشد. نام‌های فعلی ستون‌های شما: " + str(list(df.columns)))
        

    keywords = df['Keyword'].dropna().unique().tolist()
    if not keywords:
        raise ValueError("هیچ کلمه کلیدی معتبری در فایل اکسل شما یافت نشد.")

    # --- 2. Topic Modeling using AI ---
    logging.info(f"Starting topic modeling for {len(keywords)} keywords using {model_provider}...")
    
    topic_map = {}
    if model_provider == "gemini":
        topic_map = get_topic_from_gemini(keywords, api_key)
    else: # Fallback to OpenAI or other models in the future
        topic_map = get_topic_from_openai(keywords, api_key)
    
    df['Topic'] = df['Keyword'].map(topic_map)
    logging.info("Topic modeling complete.")

    # --- 3. Cannibalization Analysis ---
    analysis_results = []
    # Group by the generated topic
    grouped_by_topic = df.groupby('Topic')

    for topic, group in grouped_by_topic:
        # Find all unique URLs targeted for this single topic
        unique_urls = group['URL'].dropna().unique().tolist()
        
        # A cannibalization issue exists if more than one unique URL targets the same topic
        issue_exists = len(unique_urls) > 1
        
        analysis_results.append({
            'Topic': topic,
            'Keyword Count': len(group),
            'Unique URLs': len(unique_urls),
            'URLs': " | ".join(unique_urls),
            'Keywords': ", ".join(group['Keyword'].tolist()),
            'Cannibalization Issue': '🚨 Yes' if issue_exists else '✅ No'
        })

    result_df = pd.DataFrame(analysis_results)
    # Sort to show issues first
    result_df = result_df.sort_values(by='Cannibalization Issue', ascending=False)
    
    end_time = time.time()
    
    # --- 4. Create Summary ---
    analysis_summary = {
        "total_keywords": len(df),
        "unique_topics": df['Topic'].nunique(),
        "total_issues_found": int(result_df['Cannibalization Issue'].str.contains('Yes').sum()),
        "analysis_duration_seconds": round(end_time - start_time, 2)
    }

    return result_df, analysis_summary
