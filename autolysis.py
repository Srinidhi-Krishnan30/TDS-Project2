import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from typing import Dict, Any
import openai

# Set up OpenAI API (using AI Proxy)
openai.api_base = "https://api.aiproxy.io/v1"
openai.api_key = os.environ.get("AIPROXY_TOKEN")

def load_data(filename: str) -> pd.DataFrame:
    """Load CSV data with error handling."""
    try:
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def perform_initial_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform generic data analysis."""
    analysis = {
        'basic_info': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'column_types': df.dtypes.to_dict()
        },
        'missing_values': df.isnull().sum().to_dict(),
        'descriptive_stats': df.describe().to_dict()
    }
    return analysis

def generate_visualizations(df: pd.DataFrame):
    """Create multiple visualizations based on data."""
    plt.figure(figsize=(10, 6))
    
    # Correlation Heatmap
    plt.subplot(1, 2, 1)
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numeric_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    
    # Distribution of a key numeric column (if exists)
    plt.subplot(1, 2, 2)
    if len(numeric_columns) > 0:
        primary_column = numeric_columns[0]
        sns.histplot(df[primary_column], kde=True)
        plt.title(f'Distribution of {primary_column}')
    
    plt.tight_layout()
    plt.savefig('data_analysis.png')
    plt.close()

def ask_llm_for_insights(df: pd.DataFrame, analysis: Dict[str, Any]):
    """Use GPT-4o to generate narrative insights."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data storyteller. Help create a narrative from data analysis."},
                {"role": "user", "content": f"""
                Analyze this dataset. Here's the context:
                - Total Rows: {analysis['basic_info']['total_rows']}
                - Columns: {', '.join(analysis['basic_info']['columns'])}
                - Missing Values: {json.dumps(analysis['missing_values'])}
                
                Create a Markdown README that includes:
                1. Brief data description
                2. Key insights from the analysis
                3. Potential implications or recommendations
                
                Use a storytelling approach. Make it engaging!
                """}
            ]
        )
        
        narrative = response.choices[0].message.content
        
        with open('README.md', 'w') as f:
            f.write(narrative)
        
    except Exception as e:
        print(f"Error generating narrative: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py <dataset.csv>")
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