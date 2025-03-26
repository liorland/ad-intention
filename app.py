"""
Streamlit UI for Ad Intention Classification System

This application provides a user interface for the Ad Intention Classification System,
allowing users to classify URLs into Brand Awareness or Call to Action categories
using various classification methods.
"""

import os
import sys
import pandas as pd
import streamlit as st
import logging
from typing import Dict, Any, List, Union, Optional
import tempfile
import json
from pathlib import Path
import base64

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import ensemble classifier
try:
    from ad_intention.enhanced.ensemble_classifier import EnsembleClassifier
    from ad_intention.config import (
        BRAND_AWARENESS, CALL_TO_ACTION, 
        OPENAI_API_KEY, GOOGLE_API_KEY
    )
    from ad_intention.enhanced.cli import classify_urls_from_list, classify_from_file
except ImportError:
    st.error("Failed to import required modules. Make sure the ad_intention package is installed.")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="Ad Intention Classification",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to get the encoded image
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Function to set API keys in environment
def set_api_keys(openai_key, google_key):
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if google_key:
        os.environ["GOOGLE_API_KEY"] = google_key

# Function to create the ensemble classifier
def create_classifier(options: Dict[str, Any]) -> EnsembleClassifier:
    """Create an ensemble classifier based on the provided options."""
    # Basic classifier parameters
    classifier_params = {
        "use_sentiment": options["use_sentiment"],
        "use_bert": options["use_bert"],
        "use_gemini": options["use_gemini"],
        "use_openai_judge": options["use_openai_judge"],
        "use_vader": options["use_vader"],
        "use_textblob": options["use_textblob"],
        "sentiment_weight": options["sentiment_weight"],
        "bert_weight": options["bert_weight"],
        "gemini_weight": options["gemini_weight"],
        "rules_weight": options["rules_weight"],
        "majority_threshold": options["majority_threshold"],
        "openai_api_key": OPENAI_API_KEY,
        "google_api_key": GOOGLE_API_KEY,
        "debug": options["debug"]
    }
    
    # Note: We track individual sentiment analyzer weights in the UI,
    # but the EnsembleClassifier only accepts a single sentiment_weight parameter.
    # Individual analyzer weights (vader_weight, textblob_weight, lexicon_weight) are not
    # passed to the classifier as it doesn't support these parameters.
    
    # Note: The judge configuration parameters (verbosity, depth, etc.) are not supported
    # directly by the EnsembleClassifier. We keep track of them in the UI, but they are
    # not passed to the classifier. The classifier only supports the use_openai_judge flag.
    
    return EnsembleClassifier(**classifier_params)

# Function to classify URLs
def classify_urls(
    urls: List[str], 
    options: Dict[str, Any]
) -> pd.DataFrame:
    """Classify a list of URLs and return the results as a DataFrame."""
    
    # Set API keys
    set_api_keys(OPENAI_API_KEY, GOOGLE_API_KEY)
    
    # Create the classifier
    ensemble = create_classifier(options)
    
    # Classify the URLs
    results = classify_urls_from_list(
        urls,
        ensemble,
        options["with_details"],
        options["separate_sentiment"],
        options["debug"],
        options["n_jobs"],
        options["backend"]
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df

# Function to classify URLs from a file
def classify_from_file_wrapper(
    file_path: str,
    options: Dict[str, Any]
) -> pd.DataFrame:
    """Classify URLs from a file and return the results as a DataFrame."""
    
    # Set API keys
    set_api_keys(OPENAI_API_KEY, GOOGLE_API_KEY)
    
    # Create the classifier
    ensemble = create_classifier(options)
    
    # Classify the URLs from the file
    output_file = None  # We'll return the DataFrame directly instead of saving to a file
    results_df = classify_from_file(
        file_path, 
        output_file, 
        options["url_column"],
        ensemble,
        options["with_details"],
        options["separate_sentiment"],
        options["debug"]
    )
    
    return results_df

# Main app
def main():
    # Initialize session state to store results across reruns
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'available_cols' not in st.session_state:
        st.session_state.available_cols = []
    if 'input_method' not in st.session_state:
        st.session_state.input_method = None
    if 'options' not in st.session_state:
        st.session_state.options = {}
        
    # Display logo in center of main area at the very top
    try:
        logo_path = Path("images/biscience.png")
        if logo_path.exists():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(str(logo_path), width=300)  # Increased size
        else:
            logger.warning(f"Logo file not found at {logo_path}.")
            st.info("Note: Logo file (images/biscience.png) was not found.")
    except Exception as e:
        logger.error(f"Error displaying logo: {e}")
        st.warning(f"Error displaying logo: {str(e)}")
    
    # Main content - title after logo
    st.title("Ad Intention Classification System")
    
    st.markdown("""
    Classify URLs to determine if they're designed for **Brand Awareness** 
    or **Call to Action**
    """)
    
    # Application description in sidebar
    st.sidebar.markdown("""
    ### About
    This tool classifies ad landing pages into:
    - **Brand Awareness**
    - **Call to Action (CTA)**
    """)
    
    # Classification method selection
    st.sidebar.markdown("### Classification Methods")
    
    options = {}
    
    # Rule-based classifier (first option)
    options["use_rule_based"] = st.sidebar.checkbox("Use Rule-based Classification", value=True, 
                                                  help="Traditional pattern-matching approach using URL structure")
    
    # Sentiment analysis (second option)
    options["use_sentiment"] = st.sidebar.checkbox("Use Sentiment Analysis", value=True)
    
    # Show sentiment analyzer options when sentiment analysis is enabled
    if options["use_sentiment"]:
        # Create columns for sentiment options to save space
        sent_col1, sent_col2 = st.sidebar.columns(2)
        
        with sent_col1:
            options["use_vader"] = st.checkbox("VADER", value=True, help="VADER sentiment analyzer")
            options["use_textblob"] = st.checkbox("TextBlob", value=True, help="TextBlob sentiment analyzer")
        
        with sent_col2:
            options["use_lexicon"] = st.checkbox("Lexicon", value=True, help="Lexicon-based analyzer")
        
        # Sentiment weight applies to the combined sentiment analysis
        options["sentiment_weight"] = st.sidebar.slider(
            "Sentiment Weight", min_value=0.1, max_value=5.0, value=1.0, step=0.1
        )
        
        # Individual sentiment analyzer weights
        with st.sidebar.expander("Sentiment Analyzer Weights"):
            options["vader_weight"] = st.slider("VADER Weight", 0.1, 5.0, 1.0, 0.1)
            options["textblob_weight"] = st.slider("TextBlob Weight", 0.1, 5.0, 0.8, 0.1)
            options["lexicon_weight"] = st.slider("Lexicon Weight", 0.1, 5.0, 2.0, 0.1)
            
        # Option to show separate sentiment results
        options["separate_sentiment"] = True
        
    else:
        # Set default values when sentiment analysis is disabled
        options["use_vader"] = False
        options["use_textblob"] = False
        options["use_lexicon"] = False
        options["sentiment_weight"] = 0.0
        options["vader_weight"] = 0.0
        options["textblob_weight"] = 0.0
        options["lexicon_weight"] = 0.0
        options["separate_sentiment"] = True
    
    # BERT classifier option
    bert_selected = st.sidebar.checkbox("Use BERT Classification", value=False, 
                                      help="Uses BERT-based NLP model (higher accuracy but slower)")
    if bert_selected:
        st.sidebar.error("⚠️ BERT Classification requires permission. Please ask Lior for access.")
        options["use_bert"] = False
    else:
        options["use_bert"] = False
    options["bert_weight"] = 0.0
    
    # Input method selection before Gemini and OpenAI options
    input_method = st.radio("Choose input method:", ["Enter URLs", "Upload CSV File"])
    
    # Store current input method in session state
    st.session_state.input_method = input_method
    
    # Only show these options if using direct URL input, otherwise disable them for CSV
    if input_method != "Upload CSV File":
        # Gemini classifier option
        gemini_selected = st.sidebar.checkbox("Use Gemini Classification", value=False, 
                                            help="Uses Google's Gemini model for classification")
        if gemini_selected:
            st.sidebar.error("⚠️ Gemini Classification requires permission. Please ask Lior for access.")
            options["use_gemini"] = False
        else:
            options["use_gemini"] = False
        options["gemini_weight"] = 0.0
            
        # OpenAI judge option
        openai_selected = st.sidebar.checkbox("Use OpenAI Judge", value=False, 
                                            help="Adds GPT-4o classifier and evaluation")
        if openai_selected:
            st.sidebar.error("⚠️ OpenAI Judge requires permission. Please ask Lior for access.")
            options["use_openai_judge"] = False
        else:
            options["use_openai_judge"] = False
    else:
        # Disable Gemini and OpenAI options for CSV upload but keep them visible in UI
        options["use_gemini"] = False
        options["use_openai_judge"] = False
        options["gemini_weight"] = 0.0
        
        # Show disabled options to inform the user
        st.sidebar.markdown("### Disabled for CSV Upload")
        st.sidebar.warning("These advanced classifiers require permission from Lior and are not available for CSV uploads.")
        st.sidebar.checkbox("Use Gemini Classification", value=False, disabled=True, 
                         help="Requires permission from Lior and not available for CSV upload")
        st.sidebar.checkbox("Use OpenAI Judge", value=False, disabled=True,
                         help="Requires permission from Lior and not available for CSV upload")
    
    # Rule-based weight is always available
    options["rules_weight"] = st.sidebar.slider(
        "Rule-based Weight", min_value=0.1, max_value=5.0, value=2.0, step=0.1
    )
    
    # Thresholds and other settings
    advanced = st.sidebar.expander("Advanced Settings")
    with advanced:
        options["majority_threshold"] = st.slider("Majority Threshold", 0.1, 1.0, 0.6, 0.05)
        options["with_details"] = st.checkbox("Include Detailed Results", value=True)
        options["debug"] = st.checkbox("Debug Mode", value=False)
        options["n_jobs"] = st.slider("Parallel Jobs", -1, 16, -1)
        options["backend"] = st.selectbox("Parallelism Backend", ["loky", "threading", "multiprocessing"], index=0)
        options["url_column"] = st.text_input("URL Column Name (for CSV input)", "url")
    
    # Set default judge configuration (not visible to users)
    options["judge_verbosity"] = "medium"
    options["judge_depth"] = 2
    options["judge_features"] = ["all"]
    options["judge_format"] = "natural"
    options["judge_threshold"] = 0.3
    
    # Modify options based on input method
    if input_method == "Upload CSV File":
        # Disable Gemini and OpenAI options for CSV upload but keep them visible
        if options["use_gemini"]:
            options["use_gemini"] = False
            st.warning("Gemini classifier requires permission from Lior and is not available for CSV uploads.")
            
        if options["use_openai_judge"]:
            options["use_openai_judge"] = False
            st.warning("OpenAI judge requires permission from Lior and is not available for CSV uploads.")
            
        # We're now handling these in the sidebar section above
        # st.sidebar.markdown("### Disabled for CSV Upload")
        # st.sidebar.checkbox("Use Gemini Classification", value=False, disabled=True, 
        #                   help="Disabled for CSV upload to prevent crashes")
        # st.sidebar.checkbox("Use OpenAI Judge", value=False, disabled=True,
        #                   help="Disabled for CSV upload to prevent crashes")
    
    # Store options in session state
    st.session_state.options = options.copy()
    
    # Initialize results variable
    results = st.session_state.results
    
    # Process new classification requests
    if input_method == "Enter URLs":
        url_input = st.text_area("Enter URLs (one per line)")
        
        if st.button("Classify URLs"):
            if url_input:
                urls = [url.strip() for url in url_input.split("\n") if url.strip()]
                if urls:
                    with st.spinner("Classifying URLs..."):
                        try:
                            results = classify_urls(urls, options)
                            # Store in session state
                            st.session_state.results = results
                            st.success(f"Successfully classified {len(urls)} URLs!")
                        except Exception as e:
                            st.error(f"Error classifying URLs: {e}")
                            logger.error(f"Error: {e}", exc_info=True)
                else:
                    st.warning("Please enter at least one URL")
            else:
                st.warning("Please enter at least one URL")
    
    else:  # Upload CSV File
        uploaded_file = st.file_uploader("Upload CSV file with URLs", type="csv")
        
        if uploaded_file is not None:
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            if st.button("Classify URLs from CSV"):
                with st.spinner("Classifying URLs from CSV..."):
                    try:
                        results = classify_from_file_wrapper(tmp_path, options)
                        # Store in session state
                        st.session_state.results = results
                        st.success(f"Successfully classified URLs from CSV!")
                        # Clean up the temporary file
                        os.unlink(tmp_path)
                    except Exception as e:
                        st.error(f"Error classifying URLs from CSV: {e}")
                        logger.error(f"Error: {e}", exc_info=True)
                        # Clean up the temporary file
                        os.unlink(tmp_path)
    
    # Display results
    if results is not None and not results.empty:
        st.subheader("Classification Results")
        
        # Check if we need to recalculate the columns (only when new results are available)
        if st.session_state.results is results and len(st.session_state.available_cols) > 0:
            # Use stored columns from session state
            available_cols = st.session_state.available_cols
        else:
            # Define columns for CSV results view based on user requirements
            if st.session_state.input_method == "Upload CSV File":
                # Start with required columns
                basic_cols = ["url", "classification"]
                
                # Add columns based on which classifiers are enabled AND available in results
                if st.session_state.options.get("use_rule_based", False) and "rule_based_class" in results.columns:
                    basic_cols.append("rule_based_class")
                    
                if st.session_state.options.get("use_sentiment", False):
                    if "sentiment_class" in results.columns:
                        basic_cols.append("sentiment_class")
                    if st.session_state.options.get("use_vader", False) and "vader_class" in results.columns:
                        basic_cols.append("vader_class")
                    if st.session_state.options.get("use_textblob", False) and "textblob_class" in results.columns:
                        basic_cols.append("textblob_class")
                    if st.session_state.options.get("use_lexicon", False) and "lexicon_class" in results.columns:
                        basic_cols.append("lexicon_class")
                        
                if st.session_state.options.get("use_bert", False) and "bert_class" in results.columns:
                    basic_cols.append("bert_class")
                    
                if st.session_state.options.get("use_gemini", False) and "gemini_class" in results.columns:
                    basic_cols.append("gemini_class")
                    
                if st.session_state.options.get("use_openai_judge", False):
                    if "judge_class" in results.columns:
                        basic_cols.append("judge_class")
                    if "judge_explanation" in results.columns:
                        basic_cols.append("judge_explanation")
            else:
                # For direct URL input, use same column format as CSV for consistency
                basic_cols = ["url", "classification"]
                
                # Add classifier columns based on which classifiers are enabled AND available in results
                if st.session_state.options.get("use_rule_based", False) and "rule_based_class" in results.columns:
                    basic_cols.append("rule_based_class")
                    
                if st.session_state.options.get("use_sentiment", False):
                    if "sentiment_class" in results.columns:
                        basic_cols.append("sentiment_class")
                    if st.session_state.options.get("use_vader", False) and "vader_class" in results.columns:
                        basic_cols.append("vader_class")
                    if st.session_state.options.get("use_textblob", False) and "textblob_class" in results.columns:
                        basic_cols.append("textblob_class")
                    if st.session_state.options.get("use_lexicon", False) and "lexicon_class" in results.columns:
                        basic_cols.append("lexicon_class")
                        
                if st.session_state.options.get("use_bert", False) and "bert_class" in results.columns:
                    basic_cols.append("bert_class")
                    
                if st.session_state.options.get("use_gemini", False) and "gemini_class" in results.columns:
                    basic_cols.append("gemini_class")
                    
                if st.session_state.options.get("use_openai_judge", False):
                    if "judge_class" in results.columns:
                        basic_cols.append("judge_class")
                    if "judge_explanation" in results.columns:
                        basic_cols.append("judge_explanation")
                
                # Add confidence to basic columns for all input methods
                if "confidence" in results.columns:
                    basic_cols.append("confidence")
                
                # Add token usage if available (usually only for direct URL input)
                if "openai_total_tokens" in results.columns:
                    basic_cols.extend(["openai_total_tokens", "openai_cost"])
                if "gemini_total_tokens" in results.columns:
                    basic_cols.extend(["gemini_total_tokens", "gemini_cost"])
            
            # Show the dataframe with filtered columns that exist in the results
            available_cols = [col for col in basic_cols if col in results.columns]
            
            # Store in session state for future use
            st.session_state.available_cols = available_cols
        
        # Display the dataframe with the selected columns
        if available_cols:
            # Limit to 50 rows max for initial display to prevent UI lag
            display_df = results[available_cols]
            row_count = len(display_df)
            
            if row_count > 50:
                st.warning(f"Showing first 50 of {row_count} results to improve performance. Download the full results using the buttons below.")
                st.dataframe(display_df.head(50))
            else:
                st.dataframe(display_df)
        else:
            st.error("No columns were selected for display. This may happen if none of the enabled classifiers produced results or the column names in the results don't match what we're looking for.")
            # Fallback to showing all available columns but limit rows
            st.write("Showing all available columns as fallback:")
            st.dataframe(results.head(50))
        
        # Show classifier contributions if available (in an expander to save space)
        if "details" in results.columns and options["with_details"]:
            contrib_expander = st.expander("Classifier Contributions", expanded=True)
            with contrib_expander:
                # Select a URL to visualize with a limit for better performance
                if len(results) > 1:
                    if len(results) > 100:
                        # Limit options to improve performance
                        url_options = results["url"].iloc[:100].tolist()
                        st.warning(f"Showing first 100 of {len(results)} URLs in the dropdown for better performance")
                    else:
                        url_options = results["url"].tolist()
                        
                    selected_url = st.selectbox("Select URL to visualize", url_options)
                    selected_row = results[results["url"] == selected_url].iloc[0]
                else:
                    selected_row = results.iloc[0]
                
                # Get details with error handling
                try:
                    details = selected_row["details"]
                    if isinstance(details, str):
                        try:
                            details = json.loads(details)
                        except:
                            details = {"error": "Could not parse details JSON"}
                    
                    # Extract classifier contributions (more efficiently)
                    contributions = {}
                    classifier_results = {}
                    
                    # Check various possible structures based on code review
                    if isinstance(details, dict):
                        # Process classifier results more efficiently
                        if "classifiers" in details and isinstance(details["classifiers"], dict):
                            for classifier, data in details["classifiers"].items():
                                if isinstance(data, dict):
                                    if "classification" in data:
                                        classifier_results[classifier] = data["classification"]
                                    if "confidence" in data:
                                        contributions[classifier] = data["confidence"]
                        
                        # Process alternative structure (simpler approach)
                        for key in ["rule_based", "sentiment", "bert", "gemini"]:
                            if key in details and isinstance(details[key], dict):
                                if "classification" in details[key]:
                                    classifier_results[key] = details[key]["classification"]
                                if "confidence" in details[key]:
                                    contributions[key] = details[key]["confidence"]
                                elif "weight" in details[key]:
                                    contributions[key] = details[key]["weight"]
                        
                        # Handle judge separately
                        if "judge" in details and isinstance(details["judge"], dict):
                            if "classification" in details["judge"]:
                                classifier_results["judge"] = details["judge"]["classification"]
                    
                    # If we found contributions, visualize them
                    if contributions:
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown(f"**Classifier Decisions for URL:** {selected_row['url']}")
                            
                            # Filter classifier results to remove judge if it's not enabled
                            filtered_results = {k: v for k, v in classifier_results.items() 
                                              if k != "judge" or options["use_openai_judge"]}
                            
                            # Create a table of classifier decisions (more efficient approach)
                            data = []
                            for k, v in filtered_results.items():
                                data.append({"Classifier": k, "Decision": v})
                                
                            # Add judge decision as a special row if available AND if enabled
                            if "judge" in classifier_results and options["use_openai_judge"]:
                                data.append({
                                    "Classifier": "JUDGE DECISION",
                                    "Decision": classifier_results["judge"]
                                })
                            
                            # Add ensemble final decision
                            data.append({
                                "Classifier": "ENSEMBLE FINAL DECISION",
                                "Decision": selected_row["classification"]
                            })
                            
                            # Convert to DataFrame and display
                            classifier_df = pd.DataFrame(data)
                            st.table(classifier_df)
                        
                        with col2:
                            st.markdown("**Classifier Weights/Confidence**")
                            
                            # Create bar chart with limited entries if too many
                            if len(contributions) > 10:
                                # Sort and limit to top 10 for performance
                                top_contribs = sorted(contributions.items(), 
                                                     key=lambda x: x[1], reverse=True)[:10]
                                contribution_df = pd.DataFrame({
                                    "Classifier": [k for k, v in top_contribs],
                                    "Weight/Confidence": [v for k, v in top_contribs]
                                })
                                st.info("Showing top 10 contributions by weight")
                            else:
                                contribution_df = pd.DataFrame({
                                    "Classifier": contributions.keys(),
                                    "Weight/Confidence": contributions.values()
                                })
                            
                            # Sort by weight
                            contribution_df = contribution_df.sort_values("Weight/Confidence", 
                                                                         ascending=False)
                            
                            # Use built-in chart with reduced height for better performance
                            st.bar_chart(contribution_df.set_index("Classifier"), height=300)
                    else:
                        st.warning("No classifier contribution data found for this URL")
                        
                except Exception as e:
                    st.error(f"Error displaying classifier contributions: {str(e)}")
                    if options["debug"]:
                        st.exception(e)
        
        # Show detailed results if available and requested
        if options["with_details"] and "details" in results.columns:
            # Limit the number of detailed results to prevent UI lag
            max_details = 10
            total_results = len(results)
            
            detailed_results = st.expander("Detailed Classification Results")
            with detailed_results:
                if total_results > max_details:
                    st.warning(f"Showing details for the first {max_details} of {total_results} results to improve performance.")
                    result_selection = results.iloc[:max_details]
                else:
                    result_selection = results
                
                # Create tabs for each URL to better organize details
                if len(result_selection) > 1:
                    tabs = st.tabs([f"URL {i+1}" for i in range(len(result_selection))])
                    for i, (tab, (_, row)) in enumerate(zip(tabs, result_selection.iterrows())):
                        with tab:
                            st.markdown(f"**URL**: {row['url']}")
                            st.markdown(f"**Classification**: {row['classification']} (Confidence: {row['confidence']:.2f})")
                            
                            if "details" in row:
                                details = row["details"]
                                if isinstance(details, str):
                                    try:
                                        details = json.loads(details)
                                    except:
                                        pass
                                
                                # Use a checkbox instead of nested expander
                                show_details = st.checkbox(f"Show Details for URL {i+1}", key=f"details_checkbox_{i}")
                                if show_details:
                                    st.write("---")
                                    if isinstance(details, dict):
                                        st.json(details)
                                    else:
                                        st.write(details)
                                    st.write("---")
                else:
                    # Single result case
                    row = result_selection.iloc[0]
                    st.markdown(f"**URL**: {row['url']}")
                    st.markdown(f"**Classification**: {row['classification']} (Confidence: {row['confidence']:.2f})")
                    
                    if "details" in row:
                        details = row["details"]
                        if isinstance(details, str):
                            try:
                                details = json.loads(details)
                            except:
                                pass
                        
                        # Use a checkbox instead of nested expander
                        show_details = st.checkbox("Show Classification Details", key="single_details_checkbox")
                        if show_details:
                            st.write("---")
                            if isinstance(details, dict):
                                st.json(details)
                            else:
                                st.write(details)
                            st.write("---")
        
        # Download options
        st.subheader("Download Results")
        
        # Get results and columns from session state for consistency
        download_results = st.session_state.results
        download_cols = st.session_state.available_cols
        
        # Make sure we have valid data to download
        if download_results is not None and len(download_cols) > 0:
            # CSV download
            csv = download_results[download_cols].to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="ad_intention_results.csv",
                mime="text/csv",
                key="download_csv"  # Unique key helps with state management
            )
            
            # JSON download
            json_str = download_results[download_cols].to_json(orient="records")
            st.download_button(
                label="Download as JSON",
                data=json_str,
                file_name="ad_intention_results.json",
                mime="application/json",
                key="download_json"  # Unique key helps with state management
            )
        else:
            st.warning("No data available for download")

    # Add a disclaimer about API keys
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Advanced Classifiers Information**  
    BERT, Gemini, and OpenAI classifiers require special permission from Lior.
    Please reach out to request access to these features.
    """)

if __name__ == "__main__":
    main() 