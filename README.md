# Keyword Cannibalization Analyzer

A powerful tool for detecting and analyzing keyword cannibalization issues in your content. This application uses various similarity methods to identify potential content overlap and provides detailed visualizations and reports.

## Features

- **Multiple Similarity Methods**:
  - TF-IDF: Term frequency-based similarity
  - Sentence Transformers: Deep learning-based semantic similarity
  - Levenshtein Distance: Character-based string similarity
  - OpenAI Embeddings: Advanced AI-based semantic similarity

- **Sentence Transformer Models**:
  - MiniLM (Fast, 384d)
  - MPNet (Balanced, 768d)
  - Multilingual MiniLM (384d)
  - Multilingual MPNet (768d)

- **Interactive Visualizations**:
  - Bar charts for title similarity
  - Scatter plots for parameter similarity
  - Interactive HTML reports with DataTables

- **File Support**:
  - CSV files (.csv)
  - Excel files (.xlsx, .xls)

- **Additional Features**:
  - Persian text preprocessing support
  - Configurable similarity thresholds
  - Detailed HTML reports
  - CSV export functionality

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Danobin/SEO-Cannibalization-Analysis.git
cd keyword-cannibalization
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the FastAPI backend:
```bash
uvicorn app:app --reload
```

2. In a separate terminal, start the Streamlit frontend:
```bash
streamlit run streamlit_app.py
```

3. Open your browser and navigate to `http://localhost:8501`

4. Upload your data file (CSV or Excel) containing:
   - Title column
   - Permalink/URL column

5. Configure analysis parameters:
   - Select similarity methods for title and URL comparison
   - Adjust similarity thresholds
   - Choose Sentence Transformer model if applicable
   - Configure OpenAI settings if using OpenAI embeddings
   - Enable/disable Persian preprocessing

6. Click "Analyze" to start the analysis

7. View results in different formats:
   - Table View: Raw data in tabular format
   - Visualization: Interactive charts
   - HTML Report: Downloadable detailed report

## Input File Format

Your input file should be a CSV or Excel file with at least these columns:
- Title: The title of your content
- Permalink/URL: The URL or permalink of your content

Example CSV format:
```csv
Title,Permalink
"Best SEO Practices 2024","/seo/best-practices-2024"
"SEO Guide for Beginners","/seo/guide-beginners"
```

## Output

The analysis provides:
1. Number of potential keyword cannibalization issues
2. Detailed results showing:
   - Title pairs with similarity scores
   - URL/parameter similarity scores
   - Visual representations of similarities
3. Downloadable reports:
   - HTML report with interactive visualizations
   - CSV file with raw data

## API Endpoints

### POST /analyze
Analyzes content for keyword cannibalization.

**Request:**
- File: CSV or Excel file
- Config: JSON configuration object

**Response:**
```json
{
    "total_matches": 10,
    "results": [
        {
            "Title_1": "Example Title 1",
            "Title_2": "Example Title 2",
            "Permalink_1": "/example-1",
            "Permalink_2": "/example-2",
            "Title_Similarity": "85%",
            "Param_Similarity": "90%"
        }
    ]
}
```

## Dependencies

- FastAPI
- Streamlit
- Pandas
- NumPy
- TheFuzz
- scikit-learn
- sentence-transformers
- plotly
- htmlmin
- openpyxl

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Sentence Transformers library for semantic similarity
- TheFuzz library for string matching
- OpenAI for embedding capabilities
- FastAPI and Streamlit for the web interface 
