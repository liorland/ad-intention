#!/usr/bin/env python
"""
Command-line interface for the Ad Intent Ensemble Classifier.

This module provides a command-line interface for classifying URLs
using the ensemble classifier with various AI-powered approaches.
"""

import argparse
import csv
import json
import os
import sys
import logging
import time
from typing import List, Dict, Any, Optional
import pandas as pd

# Import joblib for parallel processing
import joblib
from joblib import Parallel, delayed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define pricing constants for token-based API models (prices per million tokens)
# These prices are current as of March 2025
API_PRICING = {
    "openai": {
        "gpt4o": {
            "input_tokens": 2.50,  # $2.50 per million input tokens
            "output_tokens": 10.00  # $10.00 per million output tokens
        }
    },
    "gemini": {
        "2.0_flash_lite": {
            "input_tokens": 0.075,  # $0.075 per million input tokens
            "output_tokens": 0.30   # $0.30 per million output tokens
        }
    }
}

# Try to import the ensemble classifier
try:
    from ad_intention.enhanced.ensemble_classifier import EnsembleClassifier
except ImportError:
    logger.error("Failed to import EnsembleClassifier. Make sure the package is installed.")
    sys.exit(1)


def classify_urls_from_list(
    urls: List[str],
    ensemble: EnsembleClassifier,
    with_details: bool = False,
    separate_sentiment: bool = False,
    debug: bool = False,
    n_jobs: int = -1,  # Number of parallel jobs, -1 means use all available cores
    backend: str = "loky"  # Use 'loky' for process-based parallelism
) -> List[Dict[str, Any]]:
    """
    Classify a list of URLs using the ensemble classifier with parallel processing.
    
    Args:
        urls: List of URLs to classify
        ensemble: Ensemble classifier instance
        with_details: Whether to include detailed results
        separate_sentiment: Whether to show separate results for each sentiment analyzer
        debug: Whether to include debug information
        n_jobs: Number of parallel jobs, -1 means use all available cores
        backend: Parallelism backend ('loky', 'threading', etc.)
        
    Returns:
        List of classification results
    """
    # Extract sentiment component weights from the ensemble classifier if available
    sentiment_component_weights = {}
    if hasattr(ensemble, "sentiment_classifier"):
        sentiment_classifier = ensemble.sentiment_classifier
        if hasattr(sentiment_classifier, "component_weights"):
            sentiment_component_weights = sentiment_classifier.component_weights
        # Try alternative attribute names if the standard one isn't found
        elif hasattr(sentiment_classifier, "weights"):
            sentiment_component_weights = sentiment_classifier.weights
        
        # If still empty, look for specific weight attributes
        if not sentiment_component_weights and hasattr(sentiment_classifier, "lexicon_weight"):
            sentiment_component_weights = {
                "lexicon": getattr(sentiment_classifier, "lexicon_weight", 2.0),
                "vader": getattr(sentiment_classifier, "vader_weight", 1.0),
                "textblob": getattr(sentiment_classifier, "textblob_weight", 0.8)
            }
    
    if debug and sentiment_component_weights:
        logger.debug(f"Found sentiment component weights: {sentiment_component_weights}")
    
    # Extract top-level classifier weights from the ensemble classifier if available
    classifier_weights = {}
    if hasattr(ensemble, "classifier_weights"):
        classifier_weights = ensemble.classifier_weights
    
    # If empty, try to extract from individual weights
    if not classifier_weights and hasattr(ensemble, "sentiment_weight"):
        classifier_weights = {
            "sentiment": getattr(ensemble, "sentiment_weight", 1.0),
            "bert": getattr(ensemble, "bert_weight", 1.0),
            "gemini": getattr(ensemble, "gemini_weight", 1.5),
            "rule_based": getattr(ensemble, "rules_weight", 2.0)
        }
    
    if debug and classifier_weights:
        logger.debug(f"Found classifier weights: {classifier_weights}")
    
    # Helper function to process a single URL
    def process_single_url(i: int, url: str) -> Dict[str, Any]:
        # Log progress
        if (i + 1) % 10 == 0 or i == 0 or i == len(urls) - 1:
            logger.info(f"Classifying URL {i+1}/{len(urls)}")
            
        try:
            # Classify the URL
            result = ensemble.classify_with_confidence(url)
            
            # Debug logging to inspect the structure
            if debug:
                logger.debug(f"Result keys: {result.keys()}")
                if "details" in result:
                    logger.debug(f"Details keys: {result['details'].keys()}")
            
            # Create a result dictionary
            result_dict = {
                "url": url,
                "classification": result["classification"],
                "confidence": result["confidence"]
            }
            
            # Extract token usage information if available
            token_usage = {}
            
            # Extract OpenAI token usage
            if hasattr(ensemble, "openai_judge") and ensemble.openai_judge:
                openai_token_counts = getattr(ensemble.openai_judge, "token_counts", {})
                if openai_token_counts:
                    # Calculate costs based on token usage
                    input_tokens = openai_token_counts.get("input_tokens", 0)
                    output_tokens = openai_token_counts.get("output_tokens", 0)
                    
                    # Calculate costs in dollars (convert from per million to per token)
                    input_cost = (input_tokens / 1_000_000) * API_PRICING["openai"]["gpt4o"]["input_tokens"]
                    output_cost = (output_tokens / 1_000_000) * API_PRICING["openai"]["gpt4o"]["output_tokens"]
                    total_cost = input_cost + output_cost
                    
                    token_usage["openai"] = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                        "input_cost": input_cost,
                        "output_cost": output_cost,
                        "total_cost": total_cost
                    }
                    
                    # Add token usage to result dictionary
                    result_dict["openai_input_tokens"] = input_tokens
                    result_dict["openai_output_tokens"] = output_tokens
                    result_dict["openai_total_tokens"] = input_tokens + output_tokens
                    result_dict["openai_cost"] = total_cost
            
            # Extract Gemini token usage
            if hasattr(ensemble, "gemini_classifier") and ensemble.gemini_classifier:
                gemini_token_counts = getattr(ensemble.gemini_classifier, "token_counts", {})
                if gemini_token_counts:
                    # Calculate costs based on token usage
                    input_tokens = gemini_token_counts.get("input_tokens", 0)
                    output_tokens = gemini_token_counts.get("output_tokens", 0)
                    
                    # Calculate costs in dollars (convert from per million to per token)
                    input_cost = (input_tokens / 1_000_000) * API_PRICING["gemini"]["2.0_flash_lite"]["input_tokens"]
                    output_cost = (output_tokens / 1_000_000) * API_PRICING["gemini"]["2.0_flash_lite"]["output_tokens"]
                    total_cost = input_cost + output_cost
                    
                    token_usage["gemini"] = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                        "input_cost": input_cost,
                        "output_cost": output_cost,
                        "total_cost": total_cost
                    }
                    
                    # Add token usage to result dictionary
                    result_dict["gemini_input_tokens"] = input_tokens
                    result_dict["gemini_output_tokens"] = output_tokens
                    result_dict["gemini_total_tokens"] = input_tokens + output_tokens
                    result_dict["gemini_cost"] = total_cost
            
            # Add combined token usage information
            if token_usage:
                result_dict["token_usage"] = token_usage
            
            # Always add details if requested, simplified if separate_sentiment is enabled
            if with_details:
                details = result.get("details", {})
                
                # If separate sentiment is requested, create a simplified version of details
                if separate_sentiment:
                    # Extract classifications from each component without technical details
                    simplified_details = {
                        # First, include the top-level classifications
                        "final_classification": result["classification"],
                        "final_confidence": result["confidence"],
                    }
                    
                    # Debug - log details to find where judge info might be
                    if debug:
                        logger.debug(f"Searching for judge information in result structure...")
                        
                        # Print all keys at each level to help locate judge info
                        def print_nested_keys(d, prefix=""):
                            if isinstance(d, dict):
                                for k, v in d.items():
                                    logger.debug(f"{prefix}{k}: {type(v)}")
                                    if isinstance(v, (dict, list)) and k != "details":
                                        if isinstance(v, dict):
                                            print_nested_keys(v, prefix + "  ")
                                        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                                            print_nested_keys(v[0], prefix + "  [0].")
                        
                        print_nested_keys(result)
                    
                    # Try multiple possible locations for judge data
                    judge_info = None
                    judge_class = "Unknown"
                    judge_confidence = 0.0
                    judge_explanation = ""
                    
                    # Check if the judge is actually enabled in the ensemble
                    openai_judge_enabled = getattr(ensemble, "use_openai_judge", False)
                    
                    if openai_judge_enabled:
                        if debug:
                            logger.debug("OpenAI judge is enabled, ensuring judge classification is set")
                        
                        # First try in details.judge
                        if "judge" in details:
                            judge_info = details["judge"]
                            if judge_info:
                                if debug:
                                    logger.debug(f"Found judge info in details.judge: {judge_info}")
                        
                        # Try in details.openai_judge
                        elif "openai_judge" in details:
                            judge_info = details["openai_judge"]
                            if judge_info:
                                if debug:
                                    logger.debug(f"Found judge info in details.openai_judge: {judge_info}")
                        
                        # Try in details.final_judgment
                        elif "final_judgment" in details:
                            judge_info = details["final_judgment"]
                            if judge_info:
                                if debug:
                                    logger.debug(f"Found judge info in details.final_judgment: {judge_info}")
                        
                        # Try at top level
                        elif "openai_judge" in result:
                            judge_info = result["openai_judge"]
                            if judge_info:
                                if debug:
                                    logger.debug(f"Found judge info in result.openai_judge: {judge_info}")
                        
                        # If judge_info was found, extract the data
                        if judge_info:
                            # Try different possible field names for classification
                            for class_field in ["classification", "class", "decision", "judgment", "label"]:
                                if class_field in judge_info:
                                    judge_class = judge_info[class_field]
                                    break
                            
                            # Try different possible field names for confidence
                            for conf_field in ["confidence", "conf", "score", "probability"]:
                                if conf_field in judge_info:
                                    judge_confidence = judge_info[conf_field]
                                    break
                            
                            # Try different possible field names for explanation
                            for exp_field in ["explanation", "reason", "reasoning", "justification", "notes"]:
                                if exp_field in judge_info:
                                    judge_explanation = judge_info[exp_field]
                                    break
                        
                        # If we still have Unknown judge_class, use final classification as fallback
                        if judge_class == "Unknown":
                            if debug:
                                logger.debug("No judge classification found, using final classification as fallback")
                            judge_class = result["classification"]
                            judge_confidence = result["confidence"]
                    
                    # Add judge information to simplified details and result
                    simplified_details["judge"] = {
                        "classification": judge_class,
                        "confidence": judge_confidence,
                        "explanation": judge_explanation
                    }
                    
                    # Add standalone columns for judge information
                    result_dict["judge_class"] = judge_class
                    result_dict["judge_confidence"] = judge_confidence
                    result_dict["judge_explanation"] = judge_explanation
                    
                    if debug:
                        logger.debug(f"Added judge class '{judge_class}' to result")
                    
                    # Extract normalized weights from the details if available
                    normalized_weights = {}
                    if "weighted_scores" in details:
                        weighted_scores = details["weighted_scores"]
                        total_weight = sum(weighted_scores.values()) if weighted_scores else 1.0
                        
                        if total_weight > 0:
                            normalized_weights = {k: v/total_weight for k, v in weighted_scores.items()}
                            simplified_details["normalized_weights"] = normalized_weights
                            
                            # Add each weight as a separate column with a _weight suffix
                            for classifier, weight in normalized_weights.items():
                                result_dict[f"{classifier}_weight"] = weight
                    
                    # Try to find judge weight in normalized weights
                    if "judge" in normalized_weights:
                        result_dict["judge_weight"] = normalized_weights["judge"]
                    elif "openai_judge" in normalized_weights:
                        result_dict["judge_weight"] = normalized_weights["openai_judge"]
                    
                    # Extract rule-based classification and add it as a separate column
                    if "rule_based" in details:
                        rule_based = details["rule_based"]
                        rule_based_class = rule_based.get("classification", "Unknown")
                        simplified_details["rule_based"] = {
                            "classification": rule_based_class,
                            "confidence": rule_based.get("confidence", 0.0)
                        }
                        
                        # Add standalone column for rule-based classification
                        result_dict["rule_based_class"] = rule_based_class
                    
                    # Extract BERT classification if available
                    if "bert" in details:
                        bert = details["bert"]
                        if bert:  # If bert results exist
                            bert_class = bert.get("classification", "Unknown")
                            simplified_details["bert"] = {
                                "classification": bert_class,
                                "confidence": bert.get("confidence", 0.0)
                            }
                            
                            # Add standalone column for BERT classification
                            result_dict["bert_class"] = bert_class
                    
                    # Extract Gemini classification if available
                    if "gemini" in details:
                        gemini = details["gemini"]
                        if gemini:  # If gemini results exist
                            gemini_class = gemini.get("classification", "Unknown")
                            simplified_details["gemini"] = {
                                "classification": gemini_class,
                                "confidence": gemini.get("confidence", 0.0)
                            }
                            
                            # Add standalone column for Gemini classification
                            result_dict["gemini_class"] = gemini_class
                    
                    # Extract sentiment classification and add it before component classifications
                    if "sentiment" in details:
                        sentiment = details["sentiment"]
                        sentiment_class = sentiment.get("classification", "Unknown")
                        sentiment_confidence = sentiment.get("confidence", 0.0)
                        
                        # Add the overall sentiment classification as a separate column
                        result_dict["sentiment_class"] = sentiment_class
                        
                        # Add component weights to the details - first try to extract from result
                        component_weights = {}
                        
                        # Check if weights are in the sentiment details
                        if "details" in sentiment:
                            sent_details = sentiment["details"]
                            # Try different possible locations
                            if "component_weights" in sent_details:
                                component_weights = sent_details["component_weights"]
                            elif "weights" in sent_details:
                                component_weights = sent_details["weights"]
                        
                        # If not found in the result, use the pre-extracted weights
                        if not component_weights and sentiment_component_weights:
                            component_weights = sentiment_component_weights.copy()
                        
                        # Add the weights to the result_dict for direct access
                        if component_weights:
                            # Normalize the component weights for better interpretability
                            total = sum(component_weights.values()) if component_weights.values() else 1.0
                            if total > 0:
                                normalized_component_weights = {k: v/total for k, v in component_weights.items()}
                                result_dict["sentiment_component_weights"] = normalized_component_weights
                                
                                # Add each component weight as a separate column
                                for component, weight in normalized_component_weights.items():
                                    result_dict[f"{component}_sentiment_weight"] = weight
                        
                        # Store in simplified details
                        simplified_details["sentiment"] = {
                            "classification": sentiment_class,
                            "confidence": sentiment_confidence,
                            "component_weights": component_weights
                        }
                        
                        # Extract sentiment component classifications
                        if "details" in sentiment and "sentiment" in sentiment["details"] and "component_scores" in sentiment["details"]["sentiment"]:
                            component_scores = sentiment["details"]["sentiment"]["component_scores"]
                            
                            # Process sentiment components (lexicon, VADER, TextBlob)
                            component_classifications = {}
                            components_info = {}
                            
                            if "lexicon" in component_scores:
                                lexicon = component_scores["lexicon"]
                                lexicon_class = "Call to Action" if lexicon.get("action_score", 0) > lexicon.get("info_score", 0) else "Brand Awareness"
                                component_classifications["lexicon"] = lexicon_class
                                
                                # Store the detailed component info including action/info scores
                                components_info["lexicon"] = {
                                    "classification": lexicon_class,
                                    "action_score": lexicon.get("action_score", 0),
                                    "info_score": lexicon.get("info_score", 0),
                                    "weight": component_weights.get("lexicon", 2.0) if component_weights else 2.0
                                }
                                
                                # Add standalone column for easy access
                                result_dict["lexicon_class"] = lexicon_class
                            
                            if "vader" in component_scores:
                                vader = component_scores["vader"]
                                vader_class = "Call to Action" if vader.get("action_score", 0) > vader.get("info_score", 0) else "Brand Awareness"
                                component_classifications["vader"] = vader_class
                                
                                # Store the detailed component info including action/info scores
                                components_info["vader"] = {
                                    "classification": vader_class,
                                    "action_score": vader.get("action_score", 0),
                                    "info_score": vader.get("info_score", 0),
                                    "weight": component_weights.get("vader", 1.0) if component_weights else 1.0,
                                    "compound": vader.get("compound", 0),
                                    "pos": vader.get("pos", 0),
                                    "neg": vader.get("neg", 0),
                                    "neu": vader.get("neu", 0)
                                }
                                
                                # Add standalone column for easy access
                                result_dict["vader_class"] = vader_class
                            
                            if "textblob" in component_scores:
                                textblob = component_scores["textblob"]
                                textblob_class = "Call to Action" if textblob.get("action_score", 0) > textblob.get("info_score", 0) else "Brand Awareness"
                                component_classifications["textblob"] = textblob_class
                                
                                # Store the detailed component info including action/info scores
                                components_info["textblob"] = {
                                    "classification": textblob_class,
                                    "action_score": textblob.get("action_score", 0),
                                    "info_score": textblob.get("info_score", 0),
                                    "weight": component_weights.get("textblob", 0.8) if component_weights else 0.8,
                                    "polarity": textblob.get("polarity", 0),
                                    "subjectivity": textblob.get("subjectivity", 0)
                                }
                                
                                # Add standalone column for easy access
                                result_dict["textblob_class"] = textblob_class
                            
                            # Store component classifications and detailed info
                            simplified_details["sentiment_components"] = component_classifications
                            simplified_details["sentiment_component_details"] = components_info
                    
                    # Include weights information
                    if "weighted_scores" in details:
                        simplified_details["classifier_weights"] = details["weighted_scores"]
                    
                    # Include final scores
                    if "final_score" in details:
                        simplified_details["final_score"] = details["final_score"]
                    if "cta_score" in details:
                        simplified_details["cta_score"] = details["cta_score"]
                    if "brand_score" in details:
                        simplified_details["brand_score"] = details["brand_score"]
                    
                    # Store classifications summary for easy access in CSV
                    component_classifications = {
                        "rule_based": simplified_details.get("rule_based", {}).get("classification", "Unknown"),
                        "combined": result["classification"]
                    }
                    
                    # Always add judge classification to summary, using the final classification as fallback
                    if judge_class != "Unknown":
                        component_classifications["judge"] = judge_class
                    else:
                        # Use the final classification as a fallback for judge
                        component_classifications["judge"] = result["classification"]
                    
                    # Add sentiment overall classification to summary
                    if "sentiment" in simplified_details:
                        component_classifications["sentiment"] = simplified_details["sentiment"]["classification"]
                    
                    # Add BERT classification to summary if available
                    if "bert" in simplified_details:
                        component_classifications["bert"] = simplified_details["bert"]["classification"]
                    
                    # Add Gemini classification to summary if available
                    if "gemini" in simplified_details:
                        component_classifications["gemini"] = simplified_details["gemini"]["classification"]
                    
                    # Add component classifications if available
                    if "sentiment_components" in simplified_details:
                        component_classifications.update(simplified_details["sentiment_components"])
                    
                    # Store the classifications summary
                    result_dict["classifications_summary"] = component_classifications
                    
                    # Replace full details with simplified version
                    result_dict["details"] = simplified_details
                else:
                    # Just use the full details if separate sentiment is not requested
                    result_dict["details"] = details
            
            # Ensure judge_class is always present if OpenAI judge is enabled
            if hasattr(ensemble, "use_openai_judge") and ensemble.use_openai_judge and "judge_class" not in result_dict:
                # Use final classification as fallback for judge class
                result_dict["judge_class"] = result_dict["classification"]
                # Add default confidence
                if "judge_confidence" not in result_dict:
                    result_dict["judge_confidence"] = result_dict["confidence"]
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Error classifying URL {url}: {e}")
            # Add a placeholder result for failed URLs
            return {
                "url": url,
                "classification": "Error",
                "confidence": 0.0,
                "error": str(e)
            }
    
    # Display information about parallel processing
    cpu_count = joblib.cpu_count()
    effective_n_jobs = cpu_count if n_jobs == -1 else min(n_jobs, cpu_count)
    logger.info(f"Processing {len(urls)} URLs in parallel using {effective_n_jobs} workers ({backend} backend)")
    
    start_time = time.time()
    
    # Use joblib to parallelize URL processing
    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(process_single_url)(i, url) for i, url in enumerate(urls)
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Parallel processing completed in {elapsed_time:.2f} seconds")
    
    return results


def classify_from_file(
    input_file: str,
    output_file: Optional[str],
    url_column: str,
    ensemble: EnsembleClassifier,
    with_details: bool = False,
    separate_sentiment: bool = False,
    debug: bool = False
) -> pd.DataFrame:
    """
    Classify URLs from a file (CSV) using the ensemble classifier.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (optional)
        url_column: Name of column containing URLs
        ensemble: Ensemble classifier instance
        with_details: Whether to include detailed results
        separate_sentiment: Whether to show separate results for each sentiment analyzer
        debug: Whether to include debug information
        
    Returns:
        DataFrame with classification results
    """
    try:
        # Read the input file
        df = pd.read_csv(input_file)
        
        # Check if URL column exists
        if url_column not in df.columns:
            logger.error(f"Column '{url_column}' not found in {input_file}")
            sys.exit(1)
            
        # Get the list of URLs
        urls = df[url_column].tolist()
        
        # Classify the URLs
        logger.info(f"Classifying {len(urls)} URLs from {input_file}")
        results = classify_urls_from_list(urls, ensemble, with_details, separate_sentiment, debug)
        
        if debug:
            # Log some information about the results
            logger.debug(f"Results shape: {len(results)}")
            sample_keys = list(results[0].keys()) if results else []
            logger.debug(f"Sample result keys: {sample_keys}")
            if 'details' in sample_keys:
                logger.debug(f"Details type: {type(results[0]['details'])}")
        
        # Create a DataFrame from the results - ensure all fields are included
        result_df = pd.DataFrame(results)
        
        # Always ensure judge_class is present in the DataFrame if OpenAI judge is enabled
        if hasattr(ensemble, "use_openai_judge") and ensemble.use_openai_judge:
            # Set judge_class column to classification column as fallback
            if "judge_class" not in result_df.columns:
                result_df["judge_class"] = result_df["classification"]
            # If column exists but has NaN or empty values, fill them with classification values
            else:
                result_df["judge_class"] = result_df["judge_class"].fillna(result_df["classification"])
                # Replace empty strings with classification
                result_df.loc[result_df["judge_class"] == "", "judge_class"] = result_df.loc[result_df["judge_class"] == "", "classification"]
            
            # Similarly handle judge_confidence if needed
            if "judge_confidence" not in result_df.columns:
                result_df["judge_confidence"] = result_df["confidence"]
            else:
                result_df["judge_confidence"] = result_df["judge_confidence"].fillna(result_df["confidence"])
        
        if debug:
            # Log DataFrame information
            logger.debug(f"DataFrame columns: {result_df.columns.tolist()}")
            logger.debug(f"DataFrame shape: {result_df.shape}")
        
        # Make sure details are handled properly if requested
        if with_details:
            # Function to safely convert to JSON
            def safe_json_dumps(item):
                try:
                    if item is None:
                        return None
                    return json.dumps(item)
                except Exception as e:
                    logger.warning(f"Failed to convert to JSON: {e}")
                    return str(item)
            
            # Convert dictionary columns to JSON strings for CSV storage
            columns_to_convert = [
                "details", "vader_details", "textblob_details", "classifications_summary",
                "sentiment_component_weights"
            ]
            
            for col in columns_to_convert:
                if col in result_df.columns:
                    if debug:
                        logger.debug(f"Converting {col} to JSON")
                    result_df[col] = result_df[col].apply(safe_json_dumps)
                    
                    # Check if conversion worked
                    if debug and not result_df[col].isnull().all():
                        logger.debug(f"Sample {col}: {result_df[col].iloc[0][:100]}...")
        
        # Ensure classifier columns are included if separate sentiment is requested
        if separate_sentiment and not result_df.empty:
            # If these columns don't exist yet, but classifications_summary does, extract them now
            if ("classifications_summary" in result_df.columns and 
                "rule_based_class" not in result_df.columns):
                
                try:
                    # Extract classifier decisions from classifications_summary
                    if "classifications_summary" in result_df.columns:
                        try:
                            # Function to fix judge classification in the summary
                            def fix_judge_in_summary(summary_json):
                                try:
                                    if not summary_json:
                                        return summary_json
                                    
                                    # Handle both string and dict inputs
                                    if isinstance(summary_json, str):
                                        data = json.loads(summary_json)
                                    else:
                                        data = summary_json
                                    
                                    # If judge is Unknown or missing, set it to the combined classification
                                    if "judge" not in data or data["judge"] == "Unknown":
                                        data["judge"] = data.get("combined", "Unknown")
                                    
                                    if isinstance(summary_json, str):
                                        return json.dumps(data)
                                    else:
                                        return data
                                except:
                                    return summary_json
                            
                            # Fix judge classification in summaries
                            if hasattr(ensemble, "use_openai_judge") and ensemble.use_openai_judge:
                                result_df["classifications_summary"] = result_df["classifications_summary"].apply(
                                    fix_judge_in_summary
                                )
                    
                            # Function to extract classifier decisions
                            def extract_classification(summary_json, classifier_key, default="Unknown"):
                                try:
                                    if not summary_json:
                                        return default
                                    
                                    # Handle both string and dict inputs
                                    if isinstance(summary_json, str):
                                        data = json.loads(summary_json)
                                    else:
                                        data = summary_json
                                        
                                    return data.get(classifier_key, default)
                                except:
                                    return default
                            
                            # Extract and create separate columns for each classifier type
                            classifiers = ["rule_based", "sentiment", "bert", "gemini", "judge", 
                                           "lexicon", "vader", "textblob"]
                            
                            for classifier in classifiers:
                                col_name = f"{classifier}_class"
                                if col_name not in result_df.columns:
                                    result_df[col_name] = result_df["classifications_summary"].apply(
                                        lambda x: extract_classification(x, classifier)
                                    )
                        except Exception as e:
                            logger.warning(f"Error extracting classifications: {e}")
                except Exception as e:
                    logger.warning(f"Error extracting classifications: {e}")
        
        # If the output file is specified, save the results
        if output_file:
            # Ensure we have the details column for CSV storage
            if with_details and "details" not in result_df.columns:
                logger.warning("Details column missing from results - adding empty column")
                result_df["details"] = "{}"
            
            # Reorder columns for a more logical presentation in the CSV
            if separate_sentiment and not result_df.empty:
                # Create lists of columns by type
                base_cols = ["url", "classification", "confidence"]
                
                # Top-level classifier columns
                classifier_cols = []
                for col in ["rule_based_class", "sentiment_class", "bert_class", "gemini_class", "judge_class"]:
                    if col in result_df.columns:
                        classifier_cols.append(col)
                
                # Judge explanation column if available
                if "judge_explanation" in result_df.columns:
                    classifier_cols.append("judge_explanation")
                
                # Top-level weight columns
                weight_cols = []
                for col in result_df.columns:
                    if col.endswith("_weight") and not col.endswith("_sentiment_weight"):
                        weight_cols.append(col)
                
                # Sentiment component columns
                sentiment_component_cols = []
                for col in ["lexicon_class", "vader_class", "textblob_class"]:
                    if col in result_df.columns:
                        sentiment_component_cols.append(col)
                
                # Sentiment component weight columns
                sentiment_weight_cols = []
                for col in result_df.columns:
                    if col.endswith("_sentiment_weight"):
                        sentiment_weight_cols.append(col)
                
                # Get all other columns that aren't in any of the above categories
                all_defined_cols = base_cols + classifier_cols + weight_cols + sentiment_component_cols + sentiment_weight_cols
                other_cols = [col for col in result_df.columns if col not in all_defined_cols]
                
                # Combine all column lists in the desired order
                ordered_columns = base_cols + classifier_cols + weight_cols + sentiment_component_cols + sentiment_weight_cols + other_cols
                
                # Reorder the DataFrame
                result_df = result_df[ordered_columns]
                
                if debug:
                    logger.debug(f"Reordered columns: {result_df.columns.tolist()}")
            
            # Save to CSV
            result_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
            
            # Double-check the saved file
            if debug:
                try:
                    check_df = pd.read_csv(output_file)
                    logger.debug(f"Saved CSV columns: {check_df.columns.tolist()}")
                    if "details" in check_df.columns:
                        logger.debug("Details column successfully saved to CSV")
                    else:
                        logger.warning("Details column missing from saved CSV!")
                except Exception as e:
                    logger.error(f"Error checking saved CSV: {e}")
            
        return result_df
        
    except Exception as e:
        logger.error(f"Error processing file {input_file}: {e}")
        logger.exception("Detailed error:")
        sys.exit(1)


def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Ad Intent Ensemble Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--urls", nargs="+", help="URLs to classify")
    input_group.add_argument("--file", help="Input CSV file with URLs")
    
    # Output options
    parser.add_argument("--output", "-o", help="Output file for results (CSV)")
    parser.add_argument("--url-column", default="url", help="Column name containing URLs (for CSV input)")
    
    # Classifier options
    classifier_group = parser.add_argument_group("Classifier Options")
    classifier_group.add_argument("--use-sentiment", action="store_true", help="Use sentiment analysis")
    classifier_group.add_argument("--use-bert", action="store_true", help="Use BERT zero-shot classification")
    classifier_group.add_argument("--use-gemini", action="store_true", help="Use Gemini LLM-based classification")
    classifier_group.add_argument("--use-openai-judge", action="store_true", help="Use OpenAI GPT-4o as a judge")
    classifier_group.add_argument("--no-openai-judge", action="store_true", 
                                 help="Disable OpenAI GPT-4o judge validation (not recommended)")
    
    # Sentiment analysis options
    sentiment_group = parser.add_argument_group("Sentiment Analysis Options")
    sentiment_group.add_argument("--use-vader", action="store_true", help="Use VADER for sentiment analysis")
    sentiment_group.add_argument("--use-textblob", action="store_true", help="Use TextBlob for sentiment analysis")
    sentiment_group.add_argument("--no-vader", action="store_true", help="Disable VADER for sentiment analysis")
    sentiment_group.add_argument("--no-textblob", action="store_true", help="Disable TextBlob for sentiment analysis")
    sentiment_group.add_argument("--separate-sentiment", action="store_true", 
                               help="Show separate results for VADER and TextBlob (instead of combined)")
    
    # Weight options
    weight_group = parser.add_argument_group("Weight Options")
    weight_group.add_argument("--sentiment-weight", type=float, default=2.2, help="Weight for sentiment classifier")
    weight_group.add_argument("--bert-weight", type=float, default=1.0, help="Weight for BERT classifier")
    weight_group.add_argument("--gemini-weight", type=float, default=1.5, help="Weight for Gemini classifier")
    weight_group.add_argument("--rules-weight", type=float, default=2.0, help="Weight for rule-based classifier")
    weight_group.add_argument("--vader-weight", type=float, default=1.0, help="Weight for VADER within sentiment analysis")
    weight_group.add_argument("--textblob-weight", type=float, default=0.8, help="Weight for TextBlob within sentiment analysis")
    weight_group.add_argument("--lexicon-weight", type=float, default=2.0, help="Weight for lexicon-based analysis within sentiment analysis")
    
    # Threshold options
    parser.add_argument("--majority-threshold", type=float, default=0.5, help="Threshold for majority voting")
    
    # Parallel processing options
    parallel_group = parser.add_argument_group("Parallel Processing Options")
    parallel_group.add_argument("--n-jobs", type=int, default=-1, 
                               help="Number of parallel jobs. -1 uses all cores, 1 disables parallelism")
    parallel_group.add_argument("--backend", type=str, choices=["loky", "threading", "multiprocessing"], default="loky",
                               help="Joblib parallel backend. Use 'loky' for CPU-bound tasks, 'threading' for IO-bound tasks")
    
    # API keys
    api_group = parser.add_argument_group("API Keys")
    api_group.add_argument("--openai-api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    api_group.add_argument("--google-api-key", help="Google API key for Gemini (or set GOOGLE_API_KEY env var)")
    
    # Other options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--with-details", action="store_true", help="Include detailed results")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Process VADER and TextBlob flags
    use_vader = True
    use_textblob = True
    
    if args.no_vader:
        use_vader = False
    elif args.use_vader:
        use_vader = True
        
    if args.no_textblob:
        use_textblob = False
    elif args.use_textblob:
        use_textblob = True
    
    # Initialize sentiment setup if requested
    if args.use_sentiment and (use_vader or use_textblob):
        try:
            from ad_intention.enhanced.nltk_setup import setup_nltk
            setup_nltk(quiet=not args.debug)
        except ImportError:
            if args.debug:
                logger.warning("NLTK setup module not found. VADER may not work properly.")
        
    # Initialize the ensemble classifier
    logger.info("Initializing ensemble classifier...")
    
    # Check the openai judge flag setting
    openai_judge_enabled = not args.no_openai_judge
    if args.use_openai_judge:
        openai_judge_enabled = True
    
    logger.info(f"OpenAI judge enabled: {openai_judge_enabled}")
    
    ensemble = EnsembleClassifier(
        use_sentiment=args.use_sentiment,
        use_bert=args.use_bert,
        use_gemini=args.use_gemini,
        use_openai_judge=openai_judge_enabled,
        use_vader=use_vader,
        use_textblob=use_textblob,
        sentiment_weight=args.sentiment_weight,
        bert_weight=args.bert_weight,
        gemini_weight=args.gemini_weight,
        rules_weight=args.rules_weight,
        majority_threshold=args.majority_threshold,
        openai_api_key=args.openai_api_key,
        google_api_key=args.google_api_key,
        debug=args.debug
    )
    
    # Check for OpenAI API key
    openai_api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not openai_api_key and openai_judge_enabled:
        logger.warning(
            "⚠️ OpenAI API key not found. Judge validation will not be used, which may reduce accuracy. "
            "Set with --openai-api-key or OPENAI_API_KEY environment variable.")
        logger.warning("To proceed without judge validation, use --no-openai-judge flag.")
    elif openai_api_key and openai_judge_enabled:
        logger.info("✓ OpenAI API key found. Judge validation will be used.")
        
        # Check after initialization if the judge was properly set up
        if hasattr(ensemble, "openai_judge") and ensemble.openai_judge:
            logger.info("✓ OpenAI judge successfully initialized.")
        else:
            logger.warning("⚠️ OpenAI judge was not properly initialized despite API key being available.")
    
    # Process input
    if args.urls:
        # Classify URLs directly
        logger.info(f"Classifying {len(args.urls)} URLs...")
        results = classify_urls_from_list(
            args.urls, 
            ensemble, 
            args.with_details, 
            args.separate_sentiment,
            args.debug,
            args.n_jobs,
            args.backend
        )
        
        # Print results to console
        for result in results:
            confidence_str = f"{result['confidence']:.2f}"
            output = f"{result['url']} | {result['classification']} | {confidence_str}"
            
            # Ensure judge_class is always present if OpenAI judge is enabled
            if not args.no_openai_judge and "judge_class" not in result:
                # Use final classification as fallback for judge class
                result["judge_class"] = result["classification"]
            
            # Add separate sentiment results if requested
            if args.separate_sentiment:
                if "classifications_summary" in result:
                    summary = result["classifications_summary"]
                    # Add judge to summary if not present
                    if "judge" not in summary and "judge_class" in result:
                        summary["judge"] = result["judge_class"]
                        
                    output += " | Summary: "
                    for classifier, decision in summary.items():
                        # Skip combined as it's already shown as the main classification
                        if classifier != "combined":
                            output += f"{classifier}:{decision[:2]} "  # Use first two letters to save space
                elif "vader_classification" in result:
                    vader_conf = f"{result['vader_confidence']:.2f}"
                    output += f" | VADER: {result['vader_classification']} ({vader_conf})"
                    
                    if "textblob_classification" in result:
                        textblob_conf = f"{result['textblob_confidence']:.2f}"
                        output += f" | TextBlob: {result['textblob_classification']} ({textblob_conf})"
            
            # Always show judge classification if available (even without separate_sentiment)
            if "judge_class" in result:
                judge_class = result["judge_class"]
                if judge_class != "Unknown":
                    judge_conf = result.get("judge_confidence", 0.7)  # Use a default confidence if not available
                    output += f" | Judge: {judge_class} ({judge_conf:.2f})"
            
            print(output)
            
        # Save to file if specified
        if args.output:
            logger.info(f"Saving results to {args.output}...")
            df = pd.DataFrame(results)
            
            # Always ensure judge_class and judge_confidence are present when OpenAI judge is used
            if not args.no_openai_judge:
                # Create judge_class column if it doesn't exist
                if "judge_class" not in df.columns:
                    df["judge_class"] = df["classification"]
                # Fill NaN or empty values with classification
                else:
                    df["judge_class"] = df["judge_class"].fillna(df["classification"])
                    # Replace empty strings with classification
                    df.loc[df["judge_class"] == "", "judge_class"] = df.loc[df["judge_class"] == "", "classification"]
                
                # Do the same for judge_confidence
                if "judge_confidence" not in df.columns:
                    df["judge_confidence"] = df["confidence"]
                else:
                    df["judge_confidence"] = df["judge_confidence"].fillna(df["confidence"])
            
            # Additionally, let's debug print the columns to verify
            logger.info(f"Columns in output DataFrame: {df.columns.tolist()}")
            
            df.to_csv(args.output, index=False)
            
    elif args.file:
        # Classify URLs from file
        logger.info(f"Classifying URLs from {args.file}...")
        result_df = classify_from_file(
            args.file,
            args.output,
            args.url_column,
            ensemble,
            args.with_details,
            args.separate_sentiment,
            args.debug
        )
        
        # Print summary to console
        logger.info("Classification summary:")
        class_counts = result_df["classification"].value_counts()
        for classification, count in class_counts.items():
            percentage = count / len(result_df) * 100
            logger.info(f"{classification}: {count} ({percentage:.1f}%)")
    
    # Display OpenAI API call statistics if applicable
    if not args.no_openai_judge and ensemble.openai_judge:
        try:
            # Check if the OpenAI judge has been initialized correctly
            logger.info("Checking OpenAI judge configuration...")
            
            # Get OpenAI import status
            has_openai_failed = getattr(ensemble.openai_judge, "openai_import_failed", False)
            if has_openai_failed:
                logger.warning("OpenAI module import failed. Please install it with: pip install openai")
                logger.info("API statistics may be inaccurate due to OpenAI module import failure.")
            
            # Debug - check if the attributes we expect exist
            if hasattr(ensemble.openai_judge, "api_calls_attempted"):
                logger.info(f"OpenAI judge api_calls_attempted attribute exists: {ensemble.openai_judge.api_calls_attempted}")
            else:
                logger.warning("OpenAI judge api_calls_attempted attribute does not exist")
            
            if hasattr(ensemble.openai_judge, "token_counts"):
                logger.info(f"OpenAI judge token_counts attribute exists: {ensemble.openai_judge.token_counts}")
            else:
                logger.warning("OpenAI judge token_counts attribute does not exist")
            
            # Check api_calls_failed and see if it reflects the failures
            if hasattr(ensemble.openai_judge, "api_calls_failed"):
                failed_calls = ensemble.openai_judge.api_calls_failed
                logger.info(f"OpenAI judge api_calls_failed: {failed_calls}")
                if failed_calls > 0:
                    logger.info("API calls are failing, check error logs above for details.")
            
            # Get API stats and token stats
            api_stats = ensemble.openai_judge.get_api_stats()
            token_stats = ensemble.openai_judge.get_token_stats()
            
            logger.info("=" * 50)
            logger.info("OPENAI API CALL STATISTICS")
            logger.info("=" * 50)
            logger.info(f"Total API calls attempted: {api_stats['api_calls_attempted']}")
            logger.info(f"Successful API calls: {api_stats['api_calls_successful']}")
            logger.info(f"Failed API calls: {api_stats['api_calls_failed']}")
            logger.info(f"Success rate: {api_stats['success_rate']}%")
            
            if 'avg_call_time' in api_stats and api_stats['api_calls_successful'] > 0:
                logger.info(f"Average API call time: {api_stats['avg_call_time']} seconds")
            
            # Display token usage and cost statistics
            logger.info("-" * 50)
            logger.info("TOKEN USAGE AND COST")
            logger.info("-" * 50)
            logger.info(f"Input tokens: {token_stats['input_tokens']:,}")
            logger.info(f"Output tokens: {token_stats['output_tokens']:,}")
            logger.info(f"Total tokens: {token_stats['total_tokens']:,}")
            logger.info(f"Input cost: ${token_stats['input_cost']:.6f}")
            logger.info(f"Output cost: ${token_stats['output_cost']:.6f}")
            logger.info(f"Total cost: ${token_stats['total_cost']:.6f}")
            
        except Exception as e:
            logger.error(f"Error retrieving API statistics: {e}")
            
    # Display Gemini API call statistics if applicable
    if args.use_gemini and hasattr(ensemble, "gemini_classifier") and ensemble.gemini_classifier:
        try:
            # Check if the Gemini classifier has been initialized correctly
            logger.info("Checking Gemini classifier configuration...")
            
            # Debug - check if the attributes we expect exist
            if hasattr(ensemble.gemini_classifier, "api_calls_attempted"):
                logger.info(f"Gemini classifier api_calls_attempted attribute exists: {ensemble.gemini_classifier.api_calls_attempted}")
            else:
                logger.warning("Gemini classifier api_calls_attempted attribute does not exist")
            
            if hasattr(ensemble.gemini_classifier, "token_counts"):
                logger.info(f"Gemini classifier token_counts attribute exists: {ensemble.gemini_classifier.token_counts}")
            else:
                logger.warning("Gemini classifier token_counts attribute does not exist")
            
            # Get API stats and token stats
            gemini_stats = ensemble.gemini_classifier.get_api_stats()
            token_stats = ensemble.gemini_classifier.get_token_stats()
            
            logger.info("=" * 50)
            logger.info("GEMINI API CALL STATISTICS")
            logger.info("=" * 50)
            logger.info(f"API calls attempted: {gemini_stats['api_calls_attempted']}")
            logger.info(f"API calls successful: {gemini_stats['api_calls_successful']}")
            logger.info(f"API calls failed: {gemini_stats['api_calls_failed']}")
            logger.info(f"API success rate: {gemini_stats['api_success_rate']}")
            
            logger.info(f"JSON parse attempts: {gemini_stats['json_parse_attempts']}")
            logger.info(f"JSON parse successful: {gemini_stats['json_parse_successful']}")
            logger.info(f"JSON parse failed: {gemini_stats['json_parse_failed']}")
            logger.info(f"JSON success rate: {gemini_stats['json_success_rate']}")
            
            # Display token usage and cost statistics
            logger.info("-" * 50)
            logger.info("TOKEN USAGE AND COST")
            logger.info("-" * 50)
            logger.info(f"Input tokens (estimated): {token_stats['input_tokens']:,}")
            logger.info(f"Output tokens (estimated): {token_stats['output_tokens']:,}")
            logger.info(f"Total tokens (estimated): {token_stats['total_tokens']:,}")
            logger.info(f"Input cost: ${token_stats['input_cost']:.6f}")
            logger.info(f"Output cost: ${token_stats['output_cost']:.6f}")
            logger.info(f"Total cost: ${token_stats['total_cost']:.6f}")
            
        except Exception as e:
            logger.error(f"Error retrieving Gemini API statistics: {e}")

    # Calculate and display token usage and cost statistics from CSV/individual results
    if args.urls:
        results_collection = results
    elif args.file and 'result_df' in locals():
        results_collection = result_df.to_dict('records')
    else:
        results_collection = []

    if results_collection:
        # Initialize counters
        openai_total_input_tokens = 0
        openai_total_output_tokens = 0
        openai_total_tokens = 0
        openai_total_cost = 0.0
        
        gemini_total_input_tokens = 0
        gemini_total_output_tokens = 0
        gemini_total_tokens = 0
        gemini_total_cost = 0.0
        
        # Collect token usage from all results
        for result in results_collection:
            # OpenAI token usage
            openai_total_input_tokens += result.get("openai_input_tokens", 0)
            openai_total_output_tokens += result.get("openai_output_tokens", 0)
            openai_total_tokens += result.get("openai_total_tokens", 0)
            openai_total_cost += result.get("openai_cost", 0.0)
            
            # Gemini token usage
            gemini_total_input_tokens += result.get("gemini_input_tokens", 0)
            gemini_total_output_tokens += result.get("gemini_output_tokens", 0)
            gemini_total_tokens += result.get("gemini_total_tokens", 0)
            gemini_total_cost += result.get("gemini_cost", 0.0)
        
        # Display token usage and cost summary
        if openai_total_tokens > 0 or gemini_total_tokens > 0:
            logger.info("=" * 50)
            logger.info("TOKEN USAGE AND COST SUMMARY")
            logger.info("=" * 50)
            
            if openai_total_tokens > 0:
                logger.info("OPENAI TOKEN USAGE:")
                logger.info(f"Total input tokens: {openai_total_input_tokens:,}")
                logger.info(f"Total output tokens: {openai_total_output_tokens:,}")
                logger.info(f"Total tokens: {openai_total_tokens:,}")
                logger.info(f"Estimated cost: ${openai_total_cost:.6f}")
            
            if gemini_total_tokens > 0:
                logger.info("GEMINI TOKEN USAGE:")
                logger.info(f"Total input tokens: {gemini_total_input_tokens:,}")
                logger.info(f"Total output tokens: {gemini_total_output_tokens:,}")
                logger.info(f"Total tokens: {gemini_total_tokens:,}")
                logger.info(f"Estimated cost: ${gemini_total_cost:.6f}")
            
            # Calculate combined cost
            total_cost = openai_total_cost + gemini_total_cost
            logger.info("-" * 50)
            logger.info(f"TOTAL ESTIMATED API COST: ${total_cost:.6f}")

            # Also add the costs from the API trackers if available
            api_tracker_total = 0.0

            # Add OpenAI costs if available
            if not args.no_openai_judge and hasattr(ensemble, "openai_judge") and ensemble.openai_judge:
                try:
                    openai_token_stats = ensemble.openai_judge.get_token_stats()
                    api_tracker_total += openai_token_stats.get("total_cost", 0.0)
                except Exception:
                    pass

            # Add Gemini costs if available
            if args.use_gemini and hasattr(ensemble, "gemini_classifier") and ensemble.gemini_classifier:
                try:
                    gemini_token_stats = ensemble.gemini_classifier.get_token_stats()
                    api_tracker_total += gemini_token_stats.get("total_cost", 0.0)
                except Exception:
                    pass

            # Display grand total including API trackers
            grand_total = total_cost + api_tracker_total
            logger.info("=" * 50)
            logger.info(f"GRAND TOTAL API COST: ${grand_total:.6f}")
            logger.info("=" * 50)
        
    logger.info("Classification complete!")
    
    
if __name__ == "__main__":
    main() 