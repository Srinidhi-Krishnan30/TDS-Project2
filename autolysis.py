import os
import sys
import requests
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
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

def generate_categorical_visualizations(df: pd.DataFrame):
    """
    Generate visualizations for categorical columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame to visualize
    """
    # Select categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_columns) == 0:
        print("No categorical columns found for visualization.")
        return
    
    plt.figure(figsize=(15, 5 * ((len(categorical_columns) + 1) // 2)))
    
    for i, col in enumerate(categorical_columns, 1):
        plt.subplot(((len(categorical_columns) + 1) // 2), 2, i)
        
        # Count plot for categorical columns
        value_counts = df[col].value_counts()
        value_counts[:10].plot(kind='bar')  # Top 10 categories
        plt.title(f'Top 10 Categories in {col}', fontsize=10)
        plt.xlabel('Categories', fontsize=8)
        plt.ylabel('Count', fontsize=8)
        plt.xticks(rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    
    try:
        cat_output_path = os.path.join(os.getcwd(), 'categorical_analysis.png')
        plt.savefig(cat_output_path, dpi=300, bbox_inches='tight')
        print(f"Categorical visualization saved to {cat_output_path}")
    except Exception as e:
        print(f"Error saving categorical visualization: {e}")
    
    plt.close()

def generate_visualizations(df: pd.DataFrame):
    """
    Generate multiple diverse visualizations to explore the dataset comprehensively.
    
    Args:
        df (pd.DataFrame): Input DataFrame to visualize
    """
    # Select numeric columns for analysis
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_columns) == 0:
        print("No numeric columns found for visualization.")
        return
    
    # Create a figure with multiple subplots
    plt.figure(figsize=(20, 15))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # 1. Kernel Density Estimation (KDE) Plot for Multiple Numeric Columns
    plt.subplot(2, 3, 1)
    try:
        # Select first 3-4 numeric columns for KDE
        selected_cols = numeric_columns[:min(4, len(numeric_columns))]
        plot_df = df[selected_cols]
        
        for col in selected_cols:
            sns.kdeplot(data=plot_df[col], label=col)
        
        plt.title('Kernel Density Estimation of Numeric Columns', fontsize=10)
        plt.xlabel('Values', fontsize=8)
        plt.legend()
    except Exception as e:
        print(f"Error creating KDE plot: {e}")
    
    # 2. Cumulative Distribution Function (CDF) Plot
    plt.subplot(2, 3, 2)
    try:
        # Select first 3-4 numeric columns for CDF
        selected_cols = numeric_columns[:min(4, len(numeric_columns))]
        plot_df = df[selected_cols]
        
        for col in selected_cols:
            sorted_data = np.sort(plot_df[col])
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            plt.plot(sorted_data, cdf, label=col)
        
        plt.title('Cumulative Distribution Function', fontsize=10)
        plt.xlabel('Values', fontsize=8)
        plt.ylabel('Cumulative Probability', fontsize=8)
        plt.legend()
    except Exception as e:
        print(f"Error creating CDF plot: {e}")
    
    # 3. Scatter Plot Matrix for First few Numeric Columns
    plt.subplot(2, 3, 3)
    try:
        # Select first 3-4 numeric columns for scatter matrix
        selected_cols = numeric_columns[:min(4, len(numeric_columns))]
        plot_df = df[selected_cols]
        
        # Create a simple scatter plot between first two columns
        plt.scatter(plot_df.iloc[:, 0], plot_df.iloc[:, 1], alpha=0.5)
        plt.title(f'Scatter Plot: {selected_cols[0]} vs {selected_cols[1]}', fontsize=10)
        plt.xlabel(selected_cols[0], fontsize=8)
        plt.ylabel(selected_cols[1], fontsize=8)
    except Exception as e:
        print(f"Error creating scatter plot: {e}")
    
    # 4. Box Plot of Numeric Columns
    plt.subplot(2, 3, 4)
    try:
        df[numeric_columns].boxplot(vert=False)
        plt.title('Box Plot of Numeric Columns', fontsize=10)
        plt.xlabel('Values', fontsize=8)
        plt.tight_layout()
    except Exception as e:
        print(f"Error creating box plot: {e}")
    
    # 5. Violin Plot to Show Distribution
    plt.subplot(2, 3, 5)
    try:
        # Select first 3-4 numeric columns for violin plot
        selected_cols = numeric_columns[:min(4, len(numeric_columns))]
        df_melted = df[selected_cols].melt(var_name='Column', value_name='Value')
        sns.violinplot(x='Column', y='Value', data=df_melted)
        plt.title('Violin Plot of Numeric Columns', fontsize=10)
        plt.xticks(rotation=45, ha='right', fontsize=8)
    except Exception as e:
        print(f"Error creating violin plot: {e}")
    
    # 6. Bar Plot of Column Means or Totals
    plt.subplot(2, 3, 6)
    try:
        column_means = df[numeric_columns].mean()
        column_means.plot(kind='bar')
        plt.title('Mean Values of Numeric Columns', fontsize=10)
        plt.xlabel('Columns', fontsize=8)
        plt.ylabel('Mean Value', fontsize=8)
        plt.xticks(rotation=45, ha='right', fontsize=8)
    except Exception as e:
        print(f"Error creating bar plot: {e}")
    
    # Save visualizations
    try:
        output_path = os.path.join(os.getcwd(), 'comprehensive_data_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive visualization saved to {output_path}")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    
    plt.close()

    # Optional: Generate categorical visualizations
    generate_categorical_visualizations(df)

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