import requests
import json
import pandas as pd
import os
from datetime import datetime

def test_health_check():
    """Test the health check endpoint"""
    print("\n=== Testing Health Check Endpoint ===")
    try:
        response = requests.get("http://localhost:8000/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def create_test_data():
    """Create a sample DataFrame for testing"""
    data = {
        'Title': [
            'How to Make Persian Rice',
            'Persian Rice Recipe',
            'Best Persian Rice Cooking Guide',
            'SEO Tips for Beginners',
            'Beginner SEO Guide'
        ],
        'Permalink': [
            '/recipes/persian-rice',
            '/recipes/persian-rice-recipe',
            '/recipes/best-persian-rice',
            '/seo/beginner-tips',
            '/seo/guide-beginners'
        ]
    }
    return pd.DataFrame(data)

def save_test_data(df, filename):
    """Save test data to CSV and Excel files"""
    # Create test_data directory if it doesn't exist
    if not os.path.exists('test_data'):
        os.makedirs('test_data')
    
    # Save as CSV
    csv_path = f'test_data/{filename}.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV file: {csv_path}")
    
    # Save as Excel
    excel_path = f'test_data/{filename}.xlsx'
    df.to_excel(excel_path, index=False)
    print(f"Saved Excel file: {excel_path}")
    
    return csv_path, excel_path

def test_analysis_endpoint(file_path):
    """Test the analysis endpoint with a file"""
    print(f"\n=== Testing Analysis Endpoint with {os.path.basename(file_path)} ===")
    
    # Prepare test configuration
    config = {
        "title_method": "tfidf",
        "url_method": "thefuzz",
        "title_threshold": 0.8,
        "url_threshold": 0.8,
        "openai_api_key": None,
        "openai_base_url": None,
        "openai_model": "text-embedding-ada-002",
        "use_persian_preprocessing": True
    }
    
    try:
        # Read file content
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # Prepare request
        files = {
            'file': (os.path.basename(file_path), file_content, 
                    'text/csv' if file_path.endswith('.csv') else 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        }
        data = {'config': json.dumps(config)}
        
        # Make request
        response = requests.post(
            "http://localhost:8000/analyze",
            files=files,
            data=data
        )
        
        # Print results
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            results = response.json()
            print(f"Total Matches: {results['total_matches']}")
            if results['total_matches'] > 0:
                print("\nSample Results:")
                for i, result in enumerate(results['results'][:2], 1):
                    print(f"\nMatch {i}:")
                    print(f"Title 1: {result['Title_1']}")
                    print(f"Title 2: {result['Title_2']}")
                    print(f"Similarity: {result['Title_Similarity']}")
        else:
            print(f"Error: {response.json()['detail']}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def run_tests():
    """Run all tests"""
    print("Starting API Tests...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test health check
    health_check_passed = test_health_check()
    print(f"Health Check Test: {'✓ PASSED' if health_check_passed else '✗ FAILED'}")
    
    if not health_check_passed:
        print("Skipping analysis tests due to health check failure")
        return
    
    # Create and save test data
    test_df = create_test_data()
    csv_path, excel_path = save_test_data(test_df, 'test_data')
    
    # Test analysis with CSV
    csv_test_passed = test_analysis_endpoint(csv_path)
    print(f"CSV Analysis Test: {'✓ PASSED' if csv_test_passed else '✗ FAILED'}")
    
    # Test analysis with Excel
    excel_test_passed = test_analysis_endpoint(excel_path)
    print(f"Excel Analysis Test: {'✓ PASSED' if excel_test_passed else '✗ FAILED'}")
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Health Check: {'✓ PASSED' if health_check_passed else '✗ FAILED'}")
    print(f"CSV Analysis: {'✓ PASSED' if csv_test_passed else '✗ FAILED'}")
    print(f"Excel Analysis: {'✓ PASSED' if excel_test_passed else '✗ FAILED'}")
    
    # Clean up test files
    try:
        os.remove(csv_path)
        os.remove(excel_path)
        os.rmdir('test_data')
        print("\nCleaned up test files")
    except Exception as e:
        print(f"\nWarning: Could not clean up test files: {str(e)}")

if __name__ == "__main__":
    run_tests() 