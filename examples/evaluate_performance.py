#!/usr/bin/env python
"""
Evaluate the performance of the Ad Intent Ensemble Classifier.

This script evaluates the performance of the ensemble classifier
against manual validations in a CSV file.
"""

import os
import sys
import logging
import pandas as pd
import argparse
import json
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Make sure the ad_intention package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import the ensemble classifier
try:
    from ad_intention.enhanced.ensemble_classifier import EnsembleClassifier
    from ad_intention.constants import BRAND_AWARENESS, CALL_TO_ACTION
except ImportError:
    logger.error("Failed to import EnsembleClassifier. Make sure the package is installed.")
    sys.exit(1)


def load_validation_data(csv_file: str, url_col: str, intent_col: str) -> pd.DataFrame:
    """
    Load validation data from a CSV file.
    
    Args:
        csv_file: Path to the CSV file
        url_col: Name of the column containing URLs
        intent_col: Name of the column containing manual validations
        
    Returns:
        DataFrame with validation data
    """
    try:
        df = pd.read_csv(csv_file)
        
        # Check if required columns exist
        if url_col not in df.columns:
            logger.error(f"URL column '{url_col}' not found in the CSV file")
            sys.exit(1)
        
        if intent_col not in df.columns:
            logger.error(f"Intent column '{intent_col}' not found in the CSV file")
            sys.exit(1)
        
        # Only keep URLs with manual validation
        df = df.dropna(subset=[intent_col])
        
        logger.info(f"Loaded {len(df)} validated URLs from {csv_file}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        sys.exit(1)


def classify_urls(
    urls: List[str], 
    ensemble: EnsembleClassifier,
    with_details: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Classify URLs using the ensemble classifier.
    
    Args:
        urls: List of URLs to classify
        ensemble: Ensemble classifier instance
        with_details: Whether to include detailed results
        
    Returns:
        Dictionary of URL to classification result
    """
    results = {}
    
    for i, url in enumerate(urls):
        try:
            # Log progress
            if (i + 1) % 10 == 0 or i == 0 or i == len(urls) - 1:
                logger.info(f"Classifying URL {i+1}/{len(urls)}")
            
            # Classify the URL
            if with_details:
                result = ensemble.classify_with_confidence(url)
            else:
                result = {
                    "classification": ensemble.classify_url(url),
                    "confidence": 0.0  # Placeholder
                }
            
            results[url] = result
            
        except Exception as e:
            logger.error(f"Error classifying URL {url}: {e}")
            # Add a placeholder result
            results[url] = {
                "classification": "Error",
                "confidence": 0.0
            }
    
    return results


def calculate_metrics(true_intents: List[str], pred_intents: List[str]) -> Dict[str, Any]:
    """
    Calculate classification metrics.
    
    Args:
        true_intents: List of true intent classifications
        pred_intents: List of predicted intent classifications
        
    Returns:
        Dictionary of metrics
    """
    # Generate classification report
    report = classification_report(
        true_intents, pred_intents, 
        target_names=[BRAND_AWARENESS, CALL_TO_ACTION],
        output_dict=True
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(true_intents, pred_intents)
    
    # Calculate specific metrics
    metrics = {
        "accuracy": report["accuracy"],
        "brand_awareness_precision": report[BRAND_AWARENESS]["precision"],
        "brand_awareness_recall": report[BRAND_AWARENESS]["recall"],
        "brand_awareness_f1": report[BRAND_AWARENESS]["f1-score"],
        "call_to_action_precision": report[CALL_TO_ACTION]["precision"],
        "call_to_action_recall": report[CALL_TO_ACTION]["recall"],
        "call_to_action_f1": report[CALL_TO_ACTION]["f1-score"],
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
    
    return metrics


def plot_confusion_matrix(cm: np.ndarray, save_path: str = None):
    """
    Plot a confusion matrix.
    
    Args:
        cm: Confusion matrix as a numpy array
        save_path: Path to save the plot to
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=[BRAND_AWARENESS, CALL_TO_ACTION],
        yticklabels=[BRAND_AWARENESS, CALL_TO_ACTION]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()


def analyze_errors(
    validation_df: pd.DataFrame,
    results: Dict[str, Dict[str, Any]],
    url_col: str,
    intent_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze classification errors.
    
    Args:
        validation_df: DataFrame with validation data
        results: Dictionary of classification results
        url_col: Name of the column containing URLs
        intent_col: Name of the column containing manual validations
        
    Returns:
        Tuple of (error_df, classifier_agreement_df)
    """
    # Create a DataFrame with results
    data = []
    for index, row in validation_df.iterrows():
        url = row[url_col]
        true_intent = row[intent_col]
        
        if url in results:
            result = results[url]
            pred_intent = result["classification"]
            confidence = result.get("confidence", 0.0)
            
            # Get classifier details if available
            details = {}
            if "details" in result:
                details = result["details"]
            
            data.append({
                "url": url,
                "true_intent": true_intent,
                "pred_intent": pred_intent,
                "confidence": confidence,
                "is_correct": true_intent == pred_intent,
                "details": details
            })
    
    result_df = pd.DataFrame(data)
    
    # Find errors
    error_df = result_df[~result_df["is_correct"]].copy()
    
    # Analyze classifier agreement if details are available
    classifier_agreement = []
    for _, row in result_df.iterrows():
        if not isinstance(row["details"], dict):
            continue
            
        details = row["details"]
        if "weighted_scores" not in details:
            continue
            
        agreement_data = {
            "url": row["url"],
            "true_intent": row["true_intent"],
            "pred_intent": row["pred_intent"],
            "is_correct": row["is_correct"],
            "confidence": row["confidence"]
        }
        
        # Get individual classifier predictions
        for clf_name, clf_result in details.items():
            if clf_name in ["rule_based", "sentiment", "bert", "gemini"]:
                if isinstance(clf_result, dict) and "classification" in clf_result:
                    agreement_data[f"{clf_name}_pred"] = clf_result["classification"]
                    agreement_data[f"{clf_name}_conf"] = clf_result.get("confidence", 0.0)
        
        classifier_agreement.append(agreement_data)
    
    agreement_df = pd.DataFrame(classifier_agreement)
    
    return error_df, agreement_df


def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate the Ad Intent Ensemble Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--csv", required=True, help="Path to the CSV file with manual validations")
    parser.add_argument("--url-col", default="url", help="Name of the column containing URLs")
    parser.add_argument("--intent-col", default="manual_validation", help="Name of the column containing manual validations")
    
    # Output arguments
    parser.add_argument("--output", "-o", help="Path to save the evaluation results")
    parser.add_argument("--plot", help="Path to save the confusion matrix plot")
    
    # Classifier options
    parser.add_argument("--use-sentiment", action="store_true", help="Use sentiment analysis")
    parser.add_argument("--use-bert", action="store_true", help="Use BERT zero-shot classification")
    parser.add_argument("--use-gemini", action="store_true", help="Use Gemini LLM-based classification")
    parser.add_argument("--use-openai-judge", action="store_true", help="Use OpenAI GPT-4o as a judge")
    
    # Weight options
    parser.add_argument("--sentiment-weight", type=float, default=1.0, help="Weight for sentiment classifier")
    parser.add_argument("--bert-weight", type=float, default=1.0, help="Weight for BERT classifier")
    parser.add_argument("--gemini-weight", type=float, default=1.5, help="Weight for Gemini classifier")
    parser.add_argument("--rules-weight", type=float, default=2.0, help="Weight for rule-based classifier")
    
    # Threshold options
    parser.add_argument("--majority-threshold", type=float, default=0.5, help="Threshold for majority voting")
    
    # Other options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--with-details", action="store_true", help="Include detailed results")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load validation data
    logger.info(f"Loading validation data from {args.csv}")
    validation_df = load_validation_data(args.csv, args.url_col, args.intent_col)
    
    # Get the list of URLs
    urls = validation_df[args.url_col].tolist()
    
    # Check for API keys
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    google_api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    # Warn if API keys are not available
    use_openai = args.use_openai_judge and bool(openai_api_key)
    use_gemini = args.use_gemini and bool(google_api_key)
    
    if args.use_openai_judge and not openai_api_key:
        logger.warning("OpenAI API key not found. OpenAI judge will not be used.")
    if args.use_gemini and not google_api_key:
        logger.warning("Google API key not found. Gemini classifier will not be used.")
    
    # Initialize ensemble classifier
    logger.info("Initializing ensemble classifier...")
    ensemble = EnsembleClassifier(
        use_sentiment=args.use_sentiment,
        use_bert=args.use_bert,
        use_gemini=use_gemini,
        use_openai_judge=use_openai,
        sentiment_weight=args.sentiment_weight,
        bert_weight=args.bert_weight,
        gemini_weight=args.gemini_weight,
        rules_weight=args.rules_weight,
        majority_threshold=args.majority_threshold,
        openai_api_key=openai_api_key,
        google_api_key=google_api_key,
        debug=args.debug
    )
    
    # Classify URLs
    logger.info(f"Classifying {len(urls)} URLs...")
    results = classify_urls(urls, ensemble, args.with_details)
    
    # Prepare data for metrics calculation
    true_intents = []
    pred_intents = []
    
    for index, row in validation_df.iterrows():
        url = row[args.url_col]
        true_intent = row[args.intent_col]
        
        if url in results:
            result = results[url]
            pred_intent = result["classification"]
            
            true_intents.append(true_intent)
            pred_intents.append(pred_intent)
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics = calculate_metrics(true_intents, pred_intents)
    
    # Print metrics summary
    print("\n===== Evaluation Results =====")
    print(f"Total URLs: {len(true_intents)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nClass-specific metrics:")
    print(f"  {BRAND_AWARENESS}:")
    print(f"    Precision: {metrics['brand_awareness_precision']:.4f}")
    print(f"    Recall: {metrics['brand_awareness_recall']:.4f}")
    print(f"    F1-score: {metrics['brand_awareness_f1']:.4f}")
    print(f"  {CALL_TO_ACTION}:")
    print(f"    Precision: {metrics['call_to_action_precision']:.4f}")
    print(f"    Recall: {metrics['call_to_action_recall']:.4f}")
    print(f"    F1-score: {metrics['call_to_action_f1']:.4f}")
    
    # Print confusion matrix
    cm = np.array(metrics["confusion_matrix"])
    print("\nConfusion Matrix:")
    print(f"                   Predicted {BRAND_AWARENESS}  Predicted {CALL_TO_ACTION}")
    print(f"True {BRAND_AWARENESS}:     {cm[0][0]}                    {cm[0][1]}")
    print(f"True {CALL_TO_ACTION}:      {cm[1][0]}                    {cm[1][1]}")
    
    # Plot confusion matrix
    if args.plot:
        plot_confusion_matrix(cm, args.plot)
    
    # Analyze errors
    logger.info("Analyzing classification errors...")
    error_df, agreement_df = analyze_errors(
        validation_df, results, args.url_col, args.intent_col
    )
    
    # Print error analysis
    if len(error_df) > 0:
        print("\n===== Classification Errors =====")
        print(f"Total errors: {len(error_df)}")
        
        # Print errors
        for _, row in error_df.iterrows():
            print(f"\nURL: {row['url']}")
            print(f"  True intent: {row['true_intent']}")
            print(f"  Predicted intent: {row['pred_intent']}")
            print(f"  Confidence: {row['confidence']:.2f}")
    else:
        print("\nNo classification errors found!")
    
    # Print classifier agreement analysis
    if len(agreement_df) > 0 and not agreement_df.empty:
        print("\n===== Classifier Agreement Analysis =====")
        
        # Calculate agreement rates
        classifier_cols = [col for col in agreement_df.columns if col.endswith("_pred")]
        
        print("\nAgreement with ground truth:")
        for col in classifier_cols:
            classifier_name = col.replace("_pred", "")
            agreement_rate = (agreement_df[col] == agreement_df["true_intent"]).mean()
            print(f"  {classifier_name}: {agreement_rate:.4f}")
        
        # Calculate inter-classifier agreement
        print("\nInter-classifier agreement matrix:")
        for col1 in classifier_cols:
            for col2 in classifier_cols:
                if col1 >= col2:
                    continue
                
                clf1_name = col1.replace("_pred", "")
                clf2_name = col2.replace("_pred", "")
                
                agreement_rate = (agreement_df[col1] == agreement_df[col2]).mean()
                print(f"  {clf1_name} vs {clf2_name}: {agreement_rate:.4f}")
    
    # Save results if an output file is specified
    if args.output:
        logger.info(f"Saving evaluation results to {args.output}")
        
        output_data = {
            "metrics": metrics,
            "errors": error_df.to_dict(orient="records") if not error_df.empty else [],
            "classifier_agreement": agreement_df.to_dict(orient="records") if not agreement_df.empty else []
        }
        
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
    
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main() 