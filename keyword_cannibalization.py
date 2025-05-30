import numpy as np
import pandas as pd
import re
from thefuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from hazm import Normalizer, Stemmer, stopwords_list
import time
import logging
from typing import Tuple, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('keyword_cannibalization.log')
    ]
)
logger = logging.getLogger(__name__)

class KeywordCannibalizationDetector:
    def __init__(self, title_method='tfidf', url_method='thefuzz', 
                 title_threshold=0.8, url_threshold=0.8, openai_api_key=None,
                 openai_base_url=None, openai_model="text-embedding-ada-002",
                 use_persian_preprocessing=True, sentence_model="all-MiniLM-L6-v2"):
        """
        Initialize the keyword cannibalization detector
        
        Args:
            title_method: Method for H1/title comparison ('tfidf', 'sentence_transformers', 'thefuzz', 'openai')
            url_method: Method for URL comparison ('tfidf', 'sentence_transformers', 'thefuzz', 'openai')
            title_threshold: Similarity threshold for titles (0.0-1.0)
            url_threshold: Similarity threshold for URLs (0.0-1.0)
            openai_api_key: OpenAI API key if using OpenAI embeddings
            openai_base_url: Custom base URL for OpenAI API (e.g., Azure OpenAI endpoint)
            openai_model: OpenAI model to use for embeddings
            use_persian_preprocessing: Whether to apply Persian text preprocessing (True/False)
            sentence_model: Sentence Transformer model to use
        """
        self.title_method = title_method
        self.url_method = url_method
        self.title_threshold = title_threshold
        self.url_threshold = url_threshold
        self.openai_model = openai_model
        self.use_persian_preprocessing = use_persian_preprocessing
        self.sentence_model = sentence_model
        
        # Initialize models based on selected methods
        self.sentence_transformer = None
        if 'sentence_transformers' in [title_method, url_method]:
            try:
                logger.info(f"Loading Sentence Transformer model: {sentence_model}")
                self.sentence_transformer = SentenceTransformer(sentence_model, trust_remote_code=True)
                logger.info(f"✓ Sentence Transformer model '{sentence_model}' loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Sentence Transformer model: {str(e)}")
                raise
            
        # Initialize LangChain OpenAI Embeddings
        self.openai_embeddings = None
        if openai_api_key and 'openai' in [title_method, url_method]:
            try:
                # Configure OpenAI Embeddings with LangChain
                if openai_base_url:
                    self.openai_embeddings = OpenAIEmbeddings(
                        api_key=openai_api_key,
                        model=openai_model,
                        base_url=openai_base_url,
                        timeout=30,
                        check_embedding_ctx_length=False
                    )
                else:
                    self.openai_embeddings = OpenAIEmbeddings(
                        api_key=openai_api_key,
                        model=openai_model,
                        timeout=30,
                        check_embedding_ctx_length=False
                    )
                logger.info(f"✓ OpenAI Embeddings initialized with model: {openai_model}")
                
            except Exception as e:
                logger.error(f"Error initializing OpenAI Embeddings: {e}")
                self.openai_embeddings = None
        
        # Initialize Persian text processing
        self.normalizer = Normalizer()
        self.stemmer = Stemmer()
        self.persian_stopwords = stopwords_list()
        
        # Log initialization status
        logger.info(f"Initialized with methods - Title: {title_method}, URL: {url_method}")
        logger.info(f"Persian preprocessing: {'enabled' if use_persian_preprocessing else 'disabled'}")

    def standardize_column_names(self, df):
        """Standardize column names"""
        df_copy = df.copy()
        column_mapping = {}
        
        for col in df_copy.columns:
            col_lower = col.lower()
            
            # Check for title column
            if any(keyword in col_lower for keyword in ['عنوان', 'title', 'h1']):
                column_mapping[col] = 'Title'
            
            # Check for URL column
            elif any(keyword in col_lower for keyword in ['آدرس', 'url', 'permalink', 'link']):
                column_mapping[col] = 'Permalink'
        
        # Rename columns
        df_copy = df_copy.rename(columns=column_mapping)
        
        # Check required columns
        if 'Title' not in df_copy.columns:
            raise ValueError("No column found that matches title patterns (عنوان, title, h1)")
        if 'Permalink' not in df_copy.columns:
            raise ValueError("No column found that matches URL patterns (آدرس, url, permalink, link)")
            
        return df_copy

    def preprocess_persian_text(self, text):
        """Preprocess Persian text"""
        if pd.isna(text) or not isinstance(text, str) or text.strip() == '':
            return ''
        
        # If Persian preprocessing is disabled, only do basic cleaning
        if not self.use_persian_preprocessing:
            return str(text).strip()
        
        try:
            # Normalization (remove half-space, fix characters, etc.)
            text = self.normalizer.normalize(text)
            # Remove punctuation
            text = re.sub(r'[!؟.،;:(){}[\]"\'<>]', '', text)
            # Convert hyphen to space
            text = text.replace('‌-', ' ')
            # Remove stopwords and stem
            words = [word for word in text.split() if word not in self.persian_stopwords]
            return ' '.join(words)
        except Exception as e:
            logger.warning(f"Warning: Error preprocessing text: {e}")
            return str(text).strip()

    def process_text_for_analysis(self, text, is_title=True):
        """Process text for analysis based on settings"""
        if pd.isna(text) or not isinstance(text, str) or text.strip() == '':
            return ''
        
        # For titles, use Persian preprocessing if enabled
        if is_title:
            return self.preprocess_persian_text(text)
        else:
            # For URLs, always do basic cleaning
            return str(text).strip()

    def extract_last_param(self, permalink):
        """Extract last parameter from URL"""
        if pd.isna(permalink) or not permalink:
            return ''
        return str(permalink).rstrip('/').split('/')[-1]

    def extract_category(self, permalink):
        """Extract category from URL"""
        if pd.isna(permalink) or not permalink:
            return 'main'
        parts = str(permalink).rstrip('/').split('/')
        parts = [p for p in parts if p.strip() != '']
        return parts[-2] if len(parts) >= 3 else 'main'

    def calculate_tfidf_similarity(self, texts):
        """Calculate similarity using TfidfVectorizer"""
        # Filter empty texts
        texts = [text if text and text.strip() else 'empty_text' for text in texts]
        
        try:
            # Configure TfidfVectorizer based on preprocessing
            if self.use_persian_preprocessing:
                # For preprocessed Persian text
                vectorizer = TfidfVectorizer(
                    stop_words=None,  # stopwords already removed
                    min_df=1,
                    ngram_range=(1, 2),
                    token_pattern=r'\b\w+\b'
                )
            else:
                # For raw text
                vectorizer = TfidfVectorizer(
                    stop_words='english',
                    min_df=1,
                    ngram_range=(1, 2)
                )
            
            vectors = vectorizer.fit_transform(texts)
            return cosine_similarity(vectors)
        except Exception as e:
            logger.error(f"Error in TfidfVectorizer: {e}")
            return np.zeros((len(texts), len(texts)))

    def calculate_sentence_transformers_similarity(self, texts):
        """Calculate similarity using Sentence Transformers"""
        if self.sentence_transformer is None:
            raise ValueError("Sentence Transformers model not initialized")
        
        # Filter empty texts
        texts = [text if text and text.strip() else 'empty text' for text in texts]
        
        try:
            logger.info(f"Calculating similarities using Sentence Transformer model: {self.sentence_model}")
            embeddings = self.sentence_transformer.encode(texts)
            return cosine_similarity(embeddings)
        except Exception as e:
            logger.error(f"Error in Sentence Transformers calculation: {e}")
            return np.zeros((len(texts), len(texts)))

    def calculate_thefuzz_similarity(self, texts):
        """Calculate similarity using TheFuzz"""
        n = len(texts)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                try:
                    text_i = str(texts[i]) if texts[i] is not None else ''
                    text_j = str(texts[j]) if texts[j] is not None else ''
                    similarity = fuzz.ratio(text_i, text_j) / 100
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
                except Exception as e:
                    logger.warning(f"Warning: Error calculating fuzz ratio: {e}")
                    similarity_matrix[i, j] = 0
                    similarity_matrix[j, i] = 0
        
        return similarity_matrix
    
    def calculate_openai_similarity(self, texts):
        """Calculate similarity using LangChain OpenAI Embeddings"""
        if self.openai_embeddings is None:
            raise ValueError("OpenAI Embeddings not initialized")

        # Convert all texts to strings and handle empty values
        texts = [str(text) if text is not None else 'empty text' for text in texts]
        texts = [text.strip() if text.strip() else 'empty text' for text in texts]

        try:
            logger.info(f"Getting OpenAI embeddings for {len(texts)} texts...")

            # Use unified embedding method (works for all batch sizes)
            embeddings = []
            batch_size = 50

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

                # Use embed_documents instead of deprecated methods
                batch_embeddings = self.openai_embeddings.embed_documents(batch)
                embeddings.extend(batch_embeddings)

                # Add delay between batches
                if i < len(texts) - batch_size:
                    time.sleep(0.2)  # Reduced delay

            embedding_matrix = np.array(embeddings)
            return cosine_similarity(embedding_matrix)

        except Exception as e:
            logger.error(f"OpenAI Embedding Error: {str(e)}")
            return np.zeros((len(texts), len(texts)))
        
    def get_similarity_matrix(self, texts, method):
        """Select similarity calculation method"""
        if not texts or len(texts) == 0:
            return np.zeros((0, 0))
            
        logger.info(f"Calculating similarities using {method} for {len(texts)} items...")
        
        if method == 'tfidf':
            return self.calculate_tfidf_similarity(texts)
        elif method == 'sentence_transformers':
            return self.calculate_sentence_transformers_similarity(texts)
        elif method == 'thefuzz':
            return self.calculate_thefuzz_similarity(texts)
        elif method == 'openai':
            return self.calculate_openai_similarity(texts)
        else:
            raise ValueError(f"Unknown method: {method}")

    def analyze_cannibalization(self, df):
        """Main keyword cannibalization analysis"""
        if df is None or len(df) == 0:
            logger.info("Empty dataframe provided")
            return pd.DataFrame(), {}
        
        # Standardize column names
        logger.info("Standardizing column names...")
        df = self.standardize_column_names(df)
        
        # Process data
        logger.info(f"Processing {len(df)} rows...")
        df = df.copy()
        
        # Process titles based on settings
        if self.use_persian_preprocessing:
            logger.info("Applying Persian text preprocessing to titles...")
            df['Processed_Title'] = df['Title'].apply(lambda x: self.process_text_for_analysis(x, is_title=True))
        else:
            logger.info("Using raw titles without Persian preprocessing...")
            df['Processed_Title'] = df['Title'].apply(lambda x: str(x).strip() if pd.notna(x) else '')
        
        # Filter empty rows
        df = df[df['Processed_Title'].str.strip() != '']
        df['Last_Param'] = df['Permalink'].apply(self.extract_last_param)
        df['Category'] = df['Permalink'].apply(self.extract_category)

        logger.info(f"After preprocessing: {len(df)} valid rows")

        if len(df) < 2:
            logger.info("Not enough valid rows for comparison")
            return pd.DataFrame(), {}

        # Calculate similarities
        logger.info(f"Computing title similarities using {self.title_method}...")
        if self.title_method == 'openai':
            # For OpenAI always use raw text (based on research)
            title_sim_matrix = self.get_similarity_matrix(df['Title'].tolist(), self.title_method)
        else:
            # For other methods use Processed_Title
            title_sim_matrix = self.get_similarity_matrix(df['Processed_Title'].tolist(), self.title_method)
        
        logger.info(f"Computing URL similarities using {self.url_method}...")
        param_sim_matrix = self.get_similarity_matrix(df['Last_Param'].tolist(), self.url_method)

        # Extract results
        results = []
        n = len(df)
        
        for i in range(n):
            for j in range(i+1, n):
                title_similarity = title_sim_matrix[i, j]
                param_similarity = param_sim_matrix[i, j]
                
                # Check thresholds and same category
                if ((title_similarity > self.title_threshold or param_similarity > self.url_threshold) and 
                    (df.iloc[i]['Category'] == df.iloc[j]['Category'])):
                    
                    results.append({
                        'Title_1': df.iloc[i]['Title'],
                        'Title_2': df.iloc[j]['Title'],
                        'Permalink_1': df.iloc[i]['Permalink'],
                        'Permalink_2': df.iloc[j]['Permalink'],
                        'Category': df.iloc[i]['Category'],
                        'Title_Similarity': f"{title_similarity * 100:.2f}%",
                        'Param_Similarity': f"{param_similarity * 100:.2f}%",
                        'Cannibalization_Type': self._determine_cannibalization_type(
                            title_similarity, param_similarity
                        )
                    })

        # Create results dataframe
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values(
                by=['Title_Similarity', 'Param_Similarity'],
                ascending=False,
                key=lambda x: x.str.replace('%', '').astype(float)
            )

        return results_df, {
            'title_similarity_matrix': title_sim_matrix,
            'param_similarity_matrix': param_sim_matrix,
            'processed_df': df
        }

    def _determine_cannibalization_type(self, title_sim, param_sim):
        """Determine cannibalization type"""
        if title_sim > self.title_threshold and param_sim > self.url_threshold:
            return "Both Title & URL"
        elif title_sim > self.title_threshold:
            return "Title Only"
        elif param_sim > self.url_threshold:
            return "URL Only"
        else:
            return "Below Threshold"

def run_cannibalization_analysis(df, title_method='tfidf', url_method='thefuzz', 
                                title_threshold=0.8, url_threshold=0.8, 
                                openai_api_key=None, openai_base_url=None,
                                openai_model="text-embedding-ada-002",
                                use_persian_preprocessing=True,
                                sentence_model="all-MiniLM-L6-v2"):
    """
    Main function to run keyword cannibalization analysis
    
    Args:
        df: DataFrame with title and URL columns (flexible naming)
        title_method: 'tfidf', 'sentence_transformers', 'thefuzz', 'openai'
        url_method: 'tfidf', 'sentence_transformers', 'thefuzz', 'openai'
        title_threshold: float (0.0-1.0)
        url_threshold: float (0.0-1.0)
        openai_api_key: string (optional)
        openai_base_url: string (optional) - custom OpenAI endpoint
        openai_model: string - OpenAI model name
        use_persian_preprocessing: bool - whether to apply Persian text preprocessing
        sentence_model: string - Sentence Transformer model name
    """
    
    detector = KeywordCannibalizationDetector(
        title_method=title_method,
        url_method=url_method,
        title_threshold=title_threshold,
        url_threshold=url_threshold,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        openai_model=openai_model,
        use_persian_preprocessing=use_persian_preprocessing,
        sentence_model=sentence_model
    )
    
    results_df, analysis_data = detector.analyze_cannibalization(df)
    
    return results_df, analysis_data 