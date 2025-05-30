from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from keyword_cannibalization import run_cannibalization_analysis
from pydantic import BaseModel
from typing import Optional
import json

app = FastAPI(title="Keyword Cannibalization API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisConfig(BaseModel):
    title_method: str = 'tfidf'
    url_method: str = 'thefuzz'
    title_threshold: float = 0.8
    url_threshold: float = 0.8
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_model: str = "text-embedding-ada-002"
    use_persian_preprocessing: bool = True

async def read_file_content(file: UploadFile) -> pd.DataFrame:
    """Read file content based on file type"""
    contents = await file.read()
    
    try:
        if file.filename.endswith('.csv'):
            # Read CSV file
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            # Read Excel file
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload a CSV or Excel file."
            )
        
        return df
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error reading file: {str(e)}"
        )

@app.post("/analyze")
async def analyze_cannibalization(
    file: UploadFile = File(...),
    config: str = Form(...)
):
    try:
        # Parse the config JSON string
        config_dict = json.loads(config)
        config_obj = AnalysisConfig(**config_dict)
        
        # Read the uploaded file
        df = await read_file_content(file)
        
        # Run analysis
        results_df, analysis_data = run_cannibalization_analysis(
            df,
            title_method=config_obj.title_method,
            url_method=config_obj.url_method,
            title_threshold=config_obj.title_threshold,
            url_threshold=config_obj.url_threshold,
            openai_api_key=config_obj.openai_api_key,
            openai_base_url=config_obj.openai_base_url,
            openai_model=config_obj.openai_model,
            use_persian_preprocessing=config_obj.use_persian_preprocessing
        )
        
        # Convert results to dict for JSON response
        results = results_df.to_dict(orient='records')
        
        return {
            "status": "success",
            "results": results,
            "total_matches": len(results)
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid config format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Keyword Cannibalization API is running"} 