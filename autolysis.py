import os
import sys
import requests
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any
from openai import OpenAI

def validate_openai_credentials():
    try:
        # Check for API key
        aiproxy_token = os.environ.get("AIPROXY_TOKEN")
        
        if not aiproxy_token:
            return False, "No AIPROXY_TOKEN found. Set the environment variable."
        
        # Direct requests-based validation
        proxy_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {aiproxy_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Confirm proxy connection."}
            ],
            "max_tokens": 10
        }
        
        try:
            # Use requests to make a direct API call
            response = requests.post(
                proxy_url, 
                headers=headers, 
                json=payload
            )
            
            # Check response
            if response.status_code == 200:
                return True, "Credentials and proxy connection validated successfully!"
            else:
                return False, f"API Call Failed: {response.status_code} - {response.text}"
        
        except Exception as api_err:
            return False, f"Connection Error: {str(api_err)}"
    
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def load_data(filename: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filename, encoding='utf-8', low_memory=False)
        print(f"Successfully loaded {len(df)} rows from {filename}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def perform_initial_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    analysis = {
        'basic_info': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'column_types': df.dtypes.to_dict()
        },
        'missing_values': df.isnull().sum().to_dict(),
        'descriptive_stats': {
            col: df[col].describe().to_dict() 
            for col in df.select_dtypes(include=['float64', 'int64']).columns
        }
    }
    return analysis

def generate_visualizations(df: pd.DataFrame):
    
    plt.figure(figsize=(16, 8))
    plt.clf()
    
    # Correlation Heatmap
    plt.subplot(1, 2, 1)
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) > 1:
        try:
            correlation_matrix = df[numeric_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title('Correlation Heatmap')
            plt.tight_layout()
        except Exception as e:
            print(f"Error creating correlation heatmap: {e}")
    
    # Distribution of a key numeric column
    plt.subplot(1, 2, 2)
    if len(numeric_columns) > 0:
        try:
            primary_column = numeric_columns[0]
            sns.histplot(df[primary_column], kde=True)
            plt.title(f'Distribution of {primary_column}')
            plt.tight_layout()
        except Exception as e:
            print(f"Error creating distribution plot: {e}")
    
    # Save visualization
    try:
        output_path = os.path.join(os.getcwd(), 'data_analysis.png')
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    
    plt.close()

def ask_llm_for_insights(df: pd.DataFrame, analysis: Dict[str, Any]):
    # First, validate credentials
    is_valid, validation_msg = validate_openai_credentials()
    if not is_valid:
        print(f"Credential Validation Failed: {validation_msg}")
        return None

    try:
        # Use validated credentials
        aiproxy_token = os.environ.get("AIPROXY_TOKEN")
        
        # Prepare payload for direct API call
        proxy_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {aiproxy_token}",
            "Content-Type": "application/json"
        }
        
        # Prepare prompt
        system_prompt = "You are a data storyteller. Help create a narrative from data analysis."
        user_prompt = f"""
        Analyze this dataset. Here's the context:
        - Total Rows: {analysis['basic_info']['total_rows']}
        - Columns: {', '.join(analysis['basic_info']['columns'])}
        - Missing Values: {json.dumps(analysis['missing_values'])}
        
        Create a Markdown README that includes:
        1. Brief data description
        2. Key insights from the analysis
        3. Potential implications or recommendations
        
        Use a storytelling approach. Make it engaging!
        """

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        # Make API call
        response = requests.post(proxy_url, headers=headers, json=payload)
        
        # Check response
        if response.status_code == 200:
            # Extract narrative from response
            narrative = response.json()['choices'][0]['message']['content']
            
            # Write to README
            readme_path = os.path.join(os.getcwd(), 'README.md')
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(narrative)
            
            print("Narrative generated successfully!")
            return narrative
        else:
            print(f"API Call Failed: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def main():
    """
    Main execution function for the data analysis script.
    """
    # Check for correct usage
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)
    
    # Load the dataset
    dataset_path = sys.argv[1]
    df = load_data(dataset_path)
    
    # Perform analysis
    analysis = perform_initial_analysis(df)
    
    # Generate visualizations
    generate_visualizations(df)
    
    # Get LLM narrative
    ask_llm_for_insights(df, analysis)

if __name__ == "__main__":
    main()