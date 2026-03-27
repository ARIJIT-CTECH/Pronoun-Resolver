"""
Utility functions for the pronoun resolution application
"""

import time
import re
from typing import Dict, List, Tuple, Any
import streamlit as st
import pandas as pd


def format_time(seconds: float) -> str:
    """Format time in seconds to a readable string."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.1f} μs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    else:
        return f"{seconds:.2f} s"


def highlight_pronouns_in_text(text: str, resolutions: Dict[str, str], color: str = "yellow") -> str:
    """
    Highlight pronouns in text with their resolutions.
    
    Args:
        text: Original text
        resolutions: Dictionary of pronoun -> antecedent mappings
        color: Highlight color
    
    Returns:
        Text with highlighted pronouns
    """
    highlighted_text = text
    
    # Sort by length to avoid partial replacements
    sorted_pronouns = sorted(resolutions.keys(), key=len, reverse=True)
    
    for pronoun in sorted_pronouns:
        antecedent = resolutions[pronoun]
        # Create highlighted version
        pattern = r'\b' + re.escape(pronoun) + r'\b'
        replacement = f'<mark style="background-color: {color}; color: black;">{pronoun} → [{antecedent}]</mark>'
        highlighted_text = re.sub(pattern, replacement, highlighted_text, flags=re.IGNORECASE)
    
    return highlighted_text


def create_comparison_table(heuristic_result: Dict, dl_result: Dict) -> pd.DataFrame:
    """
    Create a comparison table between heuristic and deep learning results.
    
    Args:
        heuristic_result: Results from heuristic approach
        dl_result: Results from deep learning approach
    
    Returns:
        Pandas DataFrame with comparison
    """
    comparison_data = []
    
    # Get all unique pronouns from both results
    all_pronouns = set(heuristic_result.get('resolutions', {}).keys()) | set(dl_result.get('resolutions', {}).keys())
    
    for pronoun in sorted(all_pronouns):
        heuristic_antecedent = heuristic_result.get('resolutions', {}).get(pronoun, "Not found")
        dl_antecedent = dl_result.get('resolutions', {}).get(pronoun, "Not found")
        
        agreement = "✅" if heuristic_antecedent == dl_antecedent and heuristic_antecedent != "Not found" else "❌"
        
        comparison_data.append({
            'Pronoun': pronoun,
            'Heuristic': heuristic_antecedent,
            'Deep Learning': dl_antecedent,
            'Agreement': agreement
        })
    
    return pd.DataFrame(comparison_data)


def calculate_metrics(heuristic_result: Dict, dl_result: Dict) -> Dict[str, Any]:
    """
    Calculate comparison metrics between approaches.
    
    Args:
        heuristic_result: Results from heuristic approach
        dl_result: Results from deep learning approach
    
    Returns:
        Dictionary with metrics
    """
    heuristic_resolutions = heuristic_result.get('resolutions', {})
    dl_resolutions = dl_result.get('resolutions', {})
    
    # Count of resolved pronouns
    heuristic_count = len(heuristic_resolutions)
    dl_count = len(dl_resolutions)
    
    # Agreement count
    common_pronouns = set(heuristic_resolutions.keys()) & set(dl_resolutions.keys())
    agreement_count = sum(1 for pronoun in common_pronouns 
                         if heuristic_resolutions[pronoun] == dl_resolutions[pronoun])
    
    # Agreement rate
    agreement_rate = (agreement_count / len(common_pronouns) * 100) if common_pronouns else 0
    
    # Processing times
    heuristic_time = heuristic_result.get('processing_time', 0)
    dl_time = dl_result.get('processing_time', 0)
    
    return {
        'heuristic_resolutions': heuristic_count,
        'dl_resolutions': dl_count,
        'agreement_count': agreement_count,
        'agreement_rate': agreement_rate,
        'heuristic_time': heuristic_time,
        'dl_time': dl_time,
        'speedup': heuristic_time / dl_time if dl_time > 0 else float('inf')
    }


def get_sample_texts() -> List[str]:
    """Get sample texts for testing."""
    return [
        "John went to the store. He bought milk. Then he met Sarah and she greeted him.",
        "The company announced its quarterly earnings. It exceeded expectations. The CEO said they were very pleased.",
        "Mary and her sister went to the park. They played on the swings. She pushed her gently.",
        "The students studied for their exams. They hoped to pass. The professor gave them advice.",
        "The dog chased the ball. It was having fun. Its owner watched happily.",
        "When the team won the championship, they celebrated wildly. Their coach was proud of them.",
        "Sarah read the book. She found it interesting. The author wrote about her experiences."
    ]


def validate_input(text: str) -> Tuple[bool, str]:
    """
    Validate user input text.
    
    Args:
        text: Input text to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Please enter some text to analyze."
    
    if len(text.strip()) < 10:
        return False, "Text is too short. Please enter at least 10 characters."
    
    if len(text) > 5000:
        return False, "Text is too long. Please limit to 5000 characters for best performance."
    
    # Check if text contains any pronouns
    pronouns = ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'its', 'their', 'theirs']
    text_lower = text.lower()
    
    if not any(pronoun in text_lower for pronoun in pronouns):
        return False, "No pronouns found in the text. Please enter text with pronouns like 'he', 'she', 'they', etc."
    
    return True, ""


def create_download_data(heuristic_result: Dict, dl_result: Dict, original_text: str) -> Dict[str, str]:
    """
    Create downloadable data for results.
    
    Args:
        heuristic_result: Results from heuristic approach
        dl_result: Results from deep learning approach
        original_text: Original input text
    
    Returns:
        Dictionary with formatted data for download
    """
    # Create CSV data
    comparison_df = create_comparison_table(heuristic_result, dl_result)
    csv_data = comparison_df.to_csv(index=False)
    
    # Create JSON data
    json_data = {
        'original_text': original_text,
        'heuristic_result': heuristic_result,
        'deep_learning_result': dl_result,
        'metrics': calculate_metrics(heuristic_result, dl_result)
    }
    
    import json
    json_str = json.dumps(json_data, indent=2)
    
    # Create text report
    metrics = calculate_metrics(heuristic_result, dl_result)
    report = f"""
PRONOUN RESOLUTION ANALYSIS REPORT
==================================

Original Text:
{original_text}

HEURISTIC APPROACH
------------------
Resolutions: {heuristic_result.get('resolutions', {})}
Processing Time: {format_time(metrics['heuristic_time'])}

DEEP LEARNING APPROACH
----------------------
Resolutions: {dl_result.get('resolutions', {})}
Processing Time: {format_time(metrics['dl_time'])}

COMPARISON METRICS
-----------------
Total Pronouns Found: {metrics['heuristic_resolutions'] + metrics['dl_resolutions']}
Agreement Rate: {metrics['agreement_rate']:.1f}%
Speedup (DL vs Heuristic): {metrics['speedup']:.2f}x

DETAILED COMPARISON
------------------
{comparison_df.to_string(index=False)}
"""
    
    return {
        'csv': csv_data,
        'json': json_str,
        'report': report
    }


def display_metrics(metrics: Dict[str, Any]):
    """
    Display metrics in a nice format.
    
    Args:
        metrics: Dictionary with metrics
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Heuristic Resolutions",
            metrics['heuristic_resolutions'],
            delta=None
        )
    
    with col2:
        st.metric(
            "DL Resolutions",
            metrics['dl_resolutions'],
            delta=None
        )
    
    with col3:
        st.metric(
            "Agreement Rate",
            f"{metrics['agreement_rate']:.1f}%",
            delta=None
        )
    
    with col4:
        if metrics['dl_time'] > 0:
            speedup_text = f"{metrics['speedup']:.1f}x"
        else:
            speedup_text = "N/A"
        
        st.metric(
            "DL Speedup",
            speedup_text,
            delta=None
        )


def setup_custom_css():
    """Setup custom CSS for the Streamlit app."""
    st.markdown("""
    <style>
    .pronoun-highlight {
        background-color: #FFE5B4;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }
    
    .heuristic-result {
        border-left: 4px solid #FF6B6B;
        padding-left: 10px;
        margin: 10px 0;
    }
    
    .dl-result {
        border-left: 4px solid #4ECDC4;
        padding-left: 10px;
        margin: 10px 0;
    }
    
    .comparison-header {
        background-color: #F0F2F6;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 5px 0;
    }
    
    .sample-text {
        background-color: #F8F9FA;
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #007bff;
        margin: 5px 0;
        cursor: pointer;
    }
    
    .sample-text:hover {
        background-color: #E9ECEF;
    }
    </style>
    """, unsafe_allow_html=True)


def error_handler(func):
    """
    Decorator for error handling in functions.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return None
    return wrapper


def cache_models():
    """
    Cache models for better performance.
    This function should be decorated with st.cache_resource in the main app.
    """
    pass


if __name__ == "__main__":
    # Test utility functions
    sample_text = "John went to the store. He bought milk."
    resolutions = {"he": "John"}
    
    highlighted = highlight_pronouns_in_text(sample_text, resolutions)
    print(f"Original: {sample_text}")
    print(f"Highlighted: {highlighted}")
    
    # Test sample texts
    samples = get_sample_texts()
    print(f"\nSample texts: {len(samples)}")
    for i, sample in enumerate(samples[:2], 1):
        print(f"{i}. {sample}")
