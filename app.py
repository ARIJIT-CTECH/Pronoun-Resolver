"""
Streamlit Web Application for Pronoun Resolution
Comparing Heuristic vs Deep Learning Approaches
"""

import streamlit as st
import time
import pandas as pd
from io import BytesIO
import json

# Import our custom modules
from heuristic import HeuristicPronounResolver
from deep_learning import DeepLearningPronounResolver
from utils import (
    format_time, highlight_pronouns_in_text, create_comparison_table,
    calculate_metrics, get_sample_texts, validate_input,
    create_download_data, display_metrics, setup_custom_css
)

# Configure Streamlit page
st.set_page_config(
    page_title="Pronoun Resolution Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache models for better performance
@st.cache_resource(show_spinner=False)
def load_heuristic_model():
    """Load and cache the heuristic pronoun resolver."""
    return HeuristicPronounResolver()

@st.cache_resource(show_spinner=False)
def load_dl_model():
    """Load and cache the deep learning pronoun resolver."""
    return DeepLearningPronounResolver()

def main():
    """Main application function."""
    
    # Setup custom CSS
    setup_custom_css()
    
    # Header
    st.title("🔍 Pronoun Resolution Analyzer")
    st.markdown("""
    Compare **heuristic-based** vs **deep learning** approaches for pronoun coreference resolution.
    Enter text below to see how different methods resolve pronouns to their antecedents.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Configuration")
        
        # Model selection
        st.subheader("Model Settings")
        use_heuristic = st.checkbox("Heuristic Approach", value=True, help="Rule-based pronoun resolution")
        use_dl = st.checkbox("Deep Learning Approach", value=True, help="Neural network-based resolution")
        
        # Display options
        st.subheader("Display Options")
        show_clusters = st.checkbox("Show Clusters", value=True)
        show_comparison = st.checkbox("Show Comparison", value=True)
        show_metrics = st.checkbox("Show Metrics", value=True)
        highlight_text = st.checkbox("Highlight Text", value=True)
        
        # Sample texts
        st.subheader("📝 Sample Texts")
        if st.button("Load Sample Text"):
            sample_texts = get_sample_texts()
            selected_sample = st.selectbox("Choose a sample:", sample_texts, index=0)
            st.session_state.input_text = selected_sample
    
    # Main content area
    st.header("📄 Input Text")
    
    # Text input area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Initialize session state
        if 'input_text' not in st.session_state:
            st.session_state.input_text = ""
        
        input_text = st.text_area(
            "Enter or paste your text here:",
            value=st.session_state.input_text,
            height=150,
            placeholder="Example: John went to the store. He bought milk. Then he met Sarah and she greeted him.",
            help="Enter text containing pronouns (he, she, they, it, etc.)"
        )
    
    with col2:
        st.write("**Quick Actions:**")
        
        # Clear button
        if st.button("🗑️ Clear", help="Clear the input text"):
            st.session_state.input_text = ""
            st.rerun()
        
        # Sample text buttons
        st.write("**Try Examples:**")
        sample_texts = get_sample_texts()[:3]
        for i, sample in enumerate(sample_texts, 1):
            if st.button(f"Example {i}", key=f"sample_{i}"):
                st.session_state.input_text = sample
                st.rerun()
    
    # Analysis button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🚀 Resolve Pronouns",
            type="primary",
            use_container_width=True,
            help="Analyze the text and resolve pronouns using selected methods"
        )
    
    # Results section
    if analyze_button:
        # Validate input
        is_valid, error_message = validate_input(input_text)
        
        if not is_valid:
            st.error(error_message)
            return
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize results storage
        if 'results' not in st.session_state:
            st.session_state.results = {}
        
        # Load models
        status_text.text("Loading models...")
        progress_bar.progress(10)
        
        heuristic_resolver = None
        dl_resolver = None
        
        try:
            if use_heuristic:
                heuristic_resolver = load_heuristic_model()
                status_text.text("Heuristic model loaded...")
            
            if use_dl:
                dl_resolver = load_dl_model()
                status_text.text("Deep learning model loaded...")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return
        
        progress_bar.progress(30)
        
        # Process with heuristic approach
        heuristic_result = None
        if use_heuristic and heuristic_resolver:
            status_text.text("Analyzing with heuristic approach...")
            progress_bar.progress(40)
            
            start_time = time.time()
            heuristic_result = heuristic_resolver.resolve_pronouns(input_text)
            heuristic_result['processing_time'] = time.time() - start_time
            
            progress_bar.progress(60)
        
        # Process with deep learning approach
        dl_result = None
        if use_dl and dl_resolver:
            status_text.text("Analyzing with deep learning approach...")
            progress_bar.progress(70)
            
            dl_result = dl_resolver.resolve_pronouns(input_text)
            
            progress_bar.progress(90)
        
        # Store results
        st.session_state.results = {
            'heuristic': heuristic_result,
            'deep_learning': dl_result,
            'input_text': input_text
        }
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
    
    # Display results
    if 'results' in st.session_state and st.session_state.results:
        results = st.session_state.results
        input_text = results['input_text']
        
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        # Create columns for side-by-side comparison
        if show_comparison and results['heuristic'] and results['deep_learning']:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🧠 Heuristic Approach")
                heuristic_result = results['heuristic']
                
                # Display processing time
                st.info(f"⏱️ Processing Time: {format_time(heuristic_result.get('processing_time', 0))}")
                
                # Display resolutions
                if heuristic_result.get('resolutions'):
                    st.write("**Pronoun Resolutions:**")
                    for pronoun, antecedent in heuristic_result['resolutions'].items():
                        st.write(f"• **{pronoun}** → {antecedent}")
                else:
                    st.write("No pronouns resolved.")
                
                # Display highlighted text
                if highlight_text and heuristic_result.get('highlighted_text'):
                    st.write("**Resolved Text:**")
                    st.markdown(heuristic_result['highlighted_text'], unsafe_allow_html=True)
                
                # Display clusters if available
                if show_clusters and heuristic_result.get('clusters'):
                    with st.expander("🔍 View Clusters"):
                        for cluster_id, entities in heuristic_result['clusters'].items():
                            st.write(f"• {cluster_id}: {', '.join(entities)}")
            
            with col2:
                st.subheader("🤖 Deep Learning Approach")
                dl_result = results['deep_learning']
                
                # Display processing time
                st.info(f"⏱️ Processing Time: {format_time(dl_result.get('processing_time', 0))}")
                
                # Display method used
                method = dl_result.get('method', 'unknown')
                st.write(f"**Method:** {method.replace('_', ' ').title()}")
                
                # Display resolutions
                if dl_result.get('resolutions'):
                    st.write("**Pronoun Resolutions:**")
                    for pronoun, antecedent in dl_result['resolutions'].items():
                        st.write(f"• **{pronoun}** → {antecedent}")
                else:
                    st.write("No pronouns resolved.")
                
                # Display highlighted text
                if highlight_text and dl_result.get('highlighted_text'):
                    st.write("**Resolved Text:**")
                    st.markdown(dl_result['highlighted_text'], unsafe_allow_html=True)
                
                # Display clusters
                if show_clusters and dl_result.get('clusters'):
                    with st.expander("🔍 View Clusters"):
                        for cluster_id, entities in dl_result['clusters'].items():
                            st.write(f"• {cluster_id}: {', '.join(entities)}")
        
        # Show metrics
        if show_metrics and results['heuristic'] and results['deep_learning']:
            st.markdown("---")
            st.subheader("📈 Comparison Metrics")
            
            metrics = calculate_metrics(results['heuristic'], results['deep_learning'])
            display_metrics(metrics)
            
            # Detailed comparison table
            if results['heuristic'].get('resolutions') or results['deep_learning'].get('resolutions'):
                st.subheader("📋 Detailed Comparison")
                comparison_df = create_comparison_table(results['heuristic'], results['deep_learning'])
                st.dataframe(comparison_df, use_container_width=True)
        
        # Download options
        st.markdown("---")
        st.subheader("💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Download Report", key="download_report"):
                download_data = create_download_data(
                    results.get('heuristic', {}),
                    results.get('deep_learning', {}),
                    input_text
                )
                st.download_button(
                    label="Download Text Report",
                    data=download_data['report'],
                    file_name="pronoun_resolution_report.txt",
                    mime="text/plain"
                )
        
        with col2:
            if st.button("📊 Download CSV", key="download_csv"):
                download_data = create_download_data(
                    results.get('heuristic', {}),
                    results.get('deep_learning', {}),
                    input_text
                )
                st.download_button(
                    label="Download CSV",
                    data=download_data['csv'],
                    file_name="pronoun_resolution_comparison.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("🔧 Download JSON", key="download_json"):
                download_data = create_download_data(
                    results.get('heuristic', {}),
                    results.get('deep_learning', {}),
                    input_text
                )
                st.download_button(
                    label="Download JSON",
                    data=download_data['json'],
                    file_name="pronoun_resolution_data.json",
                    mime="application/json"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 📚 About This App
    
    This application demonstrates two different approaches to pronoun resolution (coreference resolution):
    
    - **🧠 Heuristic Approach**: Uses rule-based logic including nearest noun matching, gender agreement, 
      number agreement, and subject preference.
    
    - **🤖 Deep Learning Approach**: Uses neural network models (transformers) for more sophisticated 
      coreference resolution.
    
    **Technologies Used**: Python, Streamlit, spaCy, NLTK, Transformers, PyTorch
    """)
    
    # Info section
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1. **Enter Text**: Type or paste text containing pronouns in the input area
        2. **Configure**: Choose which approaches to compare in the sidebar
        3. **Analyze**: Click "Resolve Pronouns" to process the text
        4. **Review**: Compare results side-by-side with highlighted text and metrics
        5. **Export**: Download results in various formats for further analysis
        
        **Tips:**
        - Use longer texts (50+ words) for better results
        - Try the sample texts to see examples
        - Compare processing times and accuracy between approaches
        - Check the agreement rate to see how often methods agree
        """)


if __name__ == "__main__":
    main()
