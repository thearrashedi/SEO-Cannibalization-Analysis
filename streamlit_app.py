import streamlit as st
import pandas as pd
import requests
import json
from io import StringIO, BytesIO
import time
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import htmlmin
import os
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config with custom theme
st.set_page_config(
    page_title="Keyword Cannibalization Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 1rem;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
        gap: 1rem;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 4px;
    }
    .stDataFrame {
        padding: 1rem;
        border-radius: 4px;
    }
    .stPlotlyChart {
        padding: 1rem;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Cache the method options and models
@st.cache_data
def get_method_options() -> Dict[str, str]:
    return {
        "tfidf": "TF-IDF",
        "sentence_transformers": "Sentence Transformers",
        "thefuzz": "Levenshtein Distance",
        "openai": "OpenAI Embeddings"
    }

@st.cache_data
def get_sentence_transformer_models() -> Dict[str, str]:
    return {
        "all-MiniLM-L6-v2": "MiniLM (Fast, 384d)",
        "all-mpnet-base-v2": "MPNet (Balanced, 768d)",
        "all-MiniLM-L12-v2": "MiniLM (Balanced, 384d)",
        "paraphrase-multilingual-MiniLM-L12-v2": "Multilingual MiniLM (384d)",
        "paraphrase-multilingual-mpnet-base-v2": "Multilingual MPNet (768d)"
    }

# Title and description with better styling
st.title("🔍 Keyword Cannibalization Analyzer")
st.markdown("""
<div style='text-align: center; padding: 1rem;  border-radius: 4px; margin-bottom: 2rem;'>
    <h3 style='color: #2c3e50;'>Analyze and Optimize Your Content Strategy</h3>
    <p style='color: #34495e;'>Upload your content data to identify potential keyword cannibalization issues and improve your SEO performance.</p>
</div>
""", unsafe_allow_html=True)

def generate_html_report(results_df: pd.DataFrame, filename: str = "cannibalization_report.html") -> str:
    """Generate HTML report with visualizations and interactive table"""
    try:
        # Create HTML table from DataFrame
        html_table = results_df.to_html(index=False, classes='styled-table', table_id="cannibalization-table")
        
        # Create Plotly figure with better styling
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'bar'}, {'type': 'scatter'}]])
        
        # Bar chart for title similarity
        fig.add_trace(
            go.Bar(
                x=results_df['Permalink_1'],
                y=results_df['Title_Similarity'].str.replace('%', '').astype(float),
                name='Title Similarity',
                marker_color='#4CAF50',
                hovertemplate="<b>%{x}</b><br>Similarity: %{y:.1f}%<extra></extra>"
            ),
            row=1, col=1
        )
        
        # Scatter plot for parameter similarity
        fig.add_trace(
            go.Scatter(
                x=results_df['Permalink_1'],
                y=results_df['Param_Similarity'].str.replace('%', '').astype(float),
                mode='markers',
                name='Parameter Similarity',
                marker=dict(
                    size=15,
                    color='#2196F3',
                    line=dict(width=2, color='#1976D2')
                ),
                hovertemplate="<b>%{x}</b><br>Similarity: %{y:.1f}%<extra></extra>"
            ),
            row=1, col=2
        )
        
        # Update layout with better styling
        fig.update_layout(
            title=dict(
                text='Cannibalization Analysis',
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=24, color='#2c3e50')
            ),
            template='plotly_white',
            hovermode='closest',
            height=600,
            xaxis_tickangle=-45,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=100)
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="URL", row=1, col=1)
        fig.update_xaxes(title_text="URL", row=1, col=2)
        fig.update_yaxes(title_text="Similarity (%)", row=1, col=1)
        fig.update_yaxes(title_text="Similarity (%)", row=1, col=2)
        
        # Convert plot to HTML
        plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Create full HTML template with DataTables
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>SEO Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 2rem;
                    background-color: #f8f9fa;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 2rem;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .styled-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 25px 0;
                    font-size: 0.9em;
                    box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
                    border-radius: 8px;
                    overflow: hidden;
                }}
                .styled-table th {{
                    cursor: pointer;
                    background-color: #4CAF50;
                    color: white;
                    padding: 12px 15px;
                    text-align: left;
                    font-weight: 600;
                }}
                .styled-table td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #dddddd;
                }}
                .styled-table tr:hover {{
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    margin-bottom: 2rem;
                    font-size: 2.5rem;
                }}
                .summary {{
                    background-color: #e8f5e9;
                    padding: 1rem;
                    border-radius: 4px;
                    margin-bottom: 2rem;
                }}
            </style>
            <link rel="stylesheet" href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.min.css">
        </head>
        <body>
            <div class="container">
                <h1>Content Cannibalization Report</h1>
                <div class="summary">
                    <h3>Analysis Summary</h3>
                    <p>Total potential cannibalization issues: {len(results_df)}</p>
                    <p>Average title similarity: {results_df['Title_Similarity'].str.replace('%', '').astype(float).mean():.1f}%</p>
                    <p>Average parameter similarity: {results_df['Param_Similarity'].str.replace('%', '').astype(float).mean():.1f}%</p>
                </div>
                {plot_html}
                {html_table}
                <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
                <script src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>
                <script>
                    $(document).ready(function() {{
                        $('#cannibalization-table').DataTable({{
                            paging: true,
                            pageLength: 100,
                            searching: true,
                            order: [[4, 'desc']],
                            language: {{
                                search: "Search:",
                                lengthMenu: "Show _MENU_ entries",
                                info: "Showing _START_ to _END_ of _TOTAL_ entries",
                                paginate: {{
                                    next: "Next",
                                    previous: "Previous"
                                }}
                            }}
                        }});
                    }});
                </script>
            </div>
        </body>
        </html>
        """
        
        # Minify HTML
        minified_html = htmlmin.minify(full_html, remove_comments=True, remove_empty_space=True)
        
        # Save file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(minified_html)
        return filename
    except Exception as e:
        logger.error(f"Error generating HTML report: {str(e)}")
        raise

# Sidebar configuration with better organization
with st.sidebar:
    st.header("Analysis Configuration")
    
    # Get cached options
    method_options = get_method_options()
    sentence_transformer_models = get_sentence_transformer_models()
    
    # Title comparison method
    title_method = st.selectbox(
        "Title Comparison Method",
        options=list(method_options.keys()),
        format_func=lambda x: method_options[x],
        index=0,
        help="Select the method to compare content titles"
    )
    
    # URL comparison method
    url_method = st.selectbox(
        "URL Comparison Method",
        options=list(method_options.keys()),
        format_func=lambda x: method_options[x],
        index=2,
        help="Select the method to compare URLs"
    )
    
    # Show Sentence Transformer model selection if either method uses it
    if title_method == "sentence_transformers" or url_method == "sentence_transformers":
        st.subheader("Sentence Transformer Model")
        sentence_model = st.selectbox(
            "Select Model",
            options=list(sentence_transformer_models.keys()),
            format_func=lambda x: sentence_transformer_models[x],
            index=0,
            help="""Choose a model based on your needs:
            - MiniLM: Faster but less accurate
            - MPNet: More accurate but slower
            - Multilingual: Better for non-English text"""
        )
    else:
        sentence_model = "all-MiniLM-L6-v2"  # Default model
    
    # Thresholds with better styling
    st.subheader("Similarity Thresholds")
    col1, col2 = st.columns(2)
    with col1:
        title_threshold = st.slider(
            "Title Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Minimum similarity score to consider titles as similar"
        )
    with col2:
        url_threshold = st.slider(
            "URL Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Minimum similarity score to consider URLs as similar"
        )
    
    # Check if OpenAI is selected for either method
    use_openai = title_method == "openai" or url_method == "openai"
    
    # OpenAI configuration
    if use_openai:
        st.subheader("OpenAI Configuration")
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key"
        )
        
        # Default OpenAI base URL
        default_openai_url = "https://api.openai.com/v1"
        openai_base_url = st.text_input(
            "OpenAI Base URL",
            value=default_openai_url,
            help="OpenAI API endpoint URL (default: https://api.openai.com/v1)"
        )
        
        openai_model = st.selectbox(
            "OpenAI Model",
            ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"],
            index=0,
            help="Select the OpenAI embedding model to use"
        )
    else:
        openai_api_key = None
        openai_base_url = None
        openai_model = "text-embedding-ada-002"
    
    # Persian preprocessing
    use_persian_preprocessing = st.checkbox(
        "Use Persian Text Preprocessing",
        value=True,
        help="Enable Persian text preprocessing for better results with Persian content"
    )

# Main content area
st.header("Upload Your Data")
uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Upload a CSV or Excel file containing your content data"
)

if uploaded_file is not None:
    try:
        # Read the file based on its type
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:  # Excel file
            df = pd.read_excel(uploaded_file)
        
        # Display sample data with better styling
        st.subheader("Sample Data")
        st.dataframe(
            df.head(),
            use_container_width=True,
            hide_index=True
        )
        
        # Analysis button with better styling
        if st.button("Analyze Keyword Cannibalization", type="primary"):
            with st.spinner("Analyzing content..."):
                try:
                    # Prepare the file for API request
                    file_content = uploaded_file.getvalue()
                    
                    # Prepare configuration
                    config = {
                        "title_method": title_method,
                        "url_method": url_method,
                        "title_threshold": title_threshold,
                        "url_threshold": url_threshold,
                        "openai_api_key": openai_api_key if use_openai else None,
                        "openai_base_url": openai_base_url if use_openai else None,
                        "openai_model": openai_model if use_openai else "text-embedding-ada-002",
                        "use_persian_preprocessing": use_persian_preprocessing,
                        "sentence_model": sentence_model
                    }
                    
                    # Make API request
                    files = {"file": (uploaded_file.name, file_content, uploaded_file.type)}
                    data = {"config": json.dumps(config)}
                    
                    response = requests.post(
                        "http://localhost:8000/analyze",
                        files=files,
                        data=data
                    )
                    
                    if response.status_code == 200:
                        results = response.json()
                        
                        # Display results with better styling
                        st.subheader("Analysis Results")
                        st.success(f"Found {results['total_matches']} potential keyword cannibalization issues")
                        
                        if results['total_matches'] > 0:
                            results_df = pd.DataFrame(results['results'])
                            
                            # Format the similarity columns to show percentages
                            if 'Title_Similarity' in results_df.columns:
                                results_df['Title_Similarity'] = results_df['Title_Similarity'].apply(
                                    lambda x: f"{float(x.strip('%')):.1f}%" if isinstance(x, str) else f"{x:.1f}%"
                                )
                            if 'Param_Similarity' in results_df.columns:
                                results_df['Param_Similarity'] = results_df['Param_Similarity'].apply(
                                    lambda x: f"{float(x.strip('%')):.1f}%" if isinstance(x, str) else f"{x:.1f}%"
                                )
                            
                            # Create tabs for different views
                            tab1, tab2, tab3 = st.tabs(["📊 Table View", "📈 Visualization", "📑 HTML Report"])
                            
                            with tab1:
                                # Display results with formatted columns
                                st.dataframe(
                                    results_df,
                                    use_container_width=True,
                                    hide_index=True
                                )
                            
                            with tab2:
                                # Create Plotly visualizations
                                fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'bar'}, {'type': 'scatter'}]])
                                
                                # Bar chart for title similarity
                                fig.add_trace(
                                    go.Bar(
                                        x=results_df['Permalink_1'],
                                        y=results_df['Title_Similarity'].str.replace('%', '').astype(float),
                                        name='Title Similarity',
                                        marker_color='#4CAF50',
                                        hovertemplate="<b>%{x}</b><br>Similarity: %{y:.1f}%<extra></extra>"
                                    ),
                                    row=1, col=1
                                )
                                
                                # Scatter plot for parameter similarity
                                fig.add_trace(
                                    go.Scatter(
                                        x=results_df['Permalink_1'],
                                        y=results_df['Param_Similarity'].str.replace('%', '').astype(float),
                                        mode='markers',
                                        name='Parameter Similarity',
                                        marker=dict(
                                            size=15,
                                            color='#2196F3',
                                            line=dict(width=2, color='#1976D2')
                                        ),
                                        hovertemplate="<b>%{x}</b><br>Similarity: %{y:.1f}%<extra></extra>"
                                    ),
                                    row=1, col=2
                                )
                                
                                # Update layout with better styling
                                fig.update_layout(
                                    title=dict(
                                        text='Cannibalization Analysis',
                                        x=0.5,
                                        y=0.95,
                                        xanchor='center',
                                        yanchor='top',
                                        font=dict(size=24, color='#2c3e50')
                                    ),
                                    template='plotly_white',
                                    hovermode='closest',
                                    height=600,
                                    xaxis_tickangle=-45,
                                    showlegend=True,
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="center",
                                        x=0.5
                                    ),
                                    margin=dict(t=100)
                                )
                                
                                # Update axes labels
                                fig.update_xaxes(title_text="URL", row=1, col=1)
                                fig.update_xaxes(title_text="URL", row=1, col=2)
                                fig.update_yaxes(title_text="Similarity (%)", row=1, col=1)
                                fig.update_yaxes(title_text="Similarity (%)", row=1, col=2)
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with tab3:
                                # Generate and download HTML report
                                report_path = generate_html_report(results_df)
                                with open(report_path, 'rb') as f:
                                    st.download_button(
                                        label="Download HTML Report",
                                        data=f,
                                        file_name="cannibalization_report.html",
                                        mime="text/html",
                                        type="primary"
                                    )
                            
                            # Download button for CSV results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="cannibalization_results.csv",
                                mime="text/csv",
                                type="secondary"
                            )
                    else:
                        st.error(f"Error: {response.json()['detail']}")
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    logger.error(f"Analysis error: {str(e)}")
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        logger.error(f"File reading error: {str(e)}")

# Instructions with better styling
with st.sidebar:
    st.markdown("---")
    st.header("📚 Instructions")
    st.markdown("""
1. Upload a CSV or Excel file containing your content data
2. Configure analysis parameters in the sidebar
3. Click 'Analyze' to start the analysis
4. Review and download the results

Required columns:
- Title (or similar column name)
- Permalink/URL (or similar column name)

Supported file formats:
- CSV (.csv)
- Excel (.xlsx, .xls)

Similarity Methods:
- TF-IDF: Term frequency-based similarity
- Sentence Transformers: Deep learning-based semantic similarity
- Levenshtein Distance: Character-based string similarity
- OpenAI Embeddings: Advanced AI-based semantic similarity

Sentence Transformer Models:
- MiniLM: Faster but less accurate (384d)
- MPNet: More accurate but slower (768d)
- Multilingual: Better for non-English text
""") 