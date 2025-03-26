#!/usr/bin/env python
"""
Debug script for Ad Intention Classifier with Gemini.

This script provides an easy way to debug the Gemini component
of the ad intention classifier using PyCharm's debugger.
"""

import os
import sys
import logging
from ad_intention.enhanced.ensemble_classifier import EnsembleClassifier
from ad_intention.enhanced.cli import classify_urls_from_list

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_gemini_classifier():
    """Run the classifier with Gemini enabled for debugging."""
    # Configuration settings
    config = {
        # URL to classify - add more URLs to the list if needed
        "urls": ["https://www.example.com"],
        
        # Classifier settings
        "use_sentiment": True,
        "use_bert": False,  # Set to True if you want to compare with BERT
        "use_gemini": True,  # Enable Gemini for debugging
        "use_openai_judge": False,  # Disable OpenAI to focus on Gemini
        
        # Sentiment settings
        "use_vader": True,
        "use_textblob": True,
        "separate_sentiment": True,
        
        # Weights - adjust as needed
        "sentiment_weight": 1.0,
        "bert_weight": 1.0,
        "gemini_weight": 1.5,
        "rules_weight": 2.0,
        "vader_weight": 1.0,
        "textblob_weight": 0.8,
        "lexicon_weight": 2.0,
        
        # Other settings
        "majority_threshold": 0.5,
        "debug": True,
        "with_details": True,
    }
    
    # Configure Gemini-specific logging
    gemini_logger = logging.getLogger('ad_intention.enhanced.gemini_classifier')
    gemini_logger.setLevel(logging.DEBUG)
    
    # Add a special handler to log Gemini responses in detail
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - GEMINI - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    gemini_logger.addHandler(handler)
    
    # Check for Google API key - required for Gemini
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        logger.warning("⚠️ Google API key not found. Set GOOGLE_API_KEY environment variable.")
        logger.warning("Gemini classifier will not work without an API key.")
        
        # Uncomment and set your key for testing if needed
        # google_api_key = "YOUR_GOOGLE_API_KEY"
    
    # Initialize the ensemble classifier
    logger.info("Initializing ensemble classifier...")
    try:
        # Place a breakpoint here to examine classifier initialization
        ensemble = EnsembleClassifier(
            use_sentiment=config["use_sentiment"],
            use_bert=config["use_bert"],
            use_gemini=config["use_gemini"],
            use_openai_judge=config["use_openai_judge"],
            use_vader=config["use_vader"],
            use_textblob=config["use_textblob"],
            sentiment_weight=config["sentiment_weight"],
            bert_weight=config["bert_weight"],
            gemini_weight=config["gemini_weight"],
            rules_weight=config["rules_weight"],
            majority_threshold=config["majority_threshold"],
            google_api_key=google_api_key,  # Pass Google API key for Gemini
            debug=config["debug"]
        )
        
        # Place a breakpoint here to examine the Gemini classifier if available
        if hasattr(ensemble, "gemini_classifier"):
            logger.info("Gemini classifier is available")
            # You can inspect ensemble.gemini_classifier here in the debugger
        else:
            logger.warning("Gemini classifier not found in the ensemble")
        
        # Classify the URLs
        logger.info(f"Classifying {len(config['urls'])} URLs...")
        
        # Place a breakpoint here to step through the classification process
        results = classify_urls_from_list(
            config["urls"],
            ensemble,
            config["with_details"],
            config["separate_sentiment"],
            config["debug"]
        )
        
        # Print results
        for result in results:
            # Place a breakpoint here to examine the classification results
            confidence_str = f"{result['confidence']:.2f}"
            print(f"\nURL: {result['url']}")
            print(f"Classification: {result['classification']} (Confidence: {confidence_str})")
            
            # Print the Gemini classifier's result if available
            if "gemini_class" in result:
                print(f"Gemini Classification: {result['gemini_class']}")
            
            # Print weight information if available
            if "gemini_weight" in result:
                print(f"Gemini Weight: {result['gemini_weight']:.2f}")
            
            # Print the full details (for debugging)
            if config["with_details"] and "details" in result:
                print("\nDetailed Results:")
                
                # Check if Gemini results are in the details
                details = result["details"]
                if isinstance(details, dict) and "gemini" in details:
                    gemini_details = details["gemini"]
                    print(f"Gemini Details: {gemini_details}")
                
                # Check for weight information
                if "normalized_weights" in details:
                    print(f"Normalized Weights: {details['normalized_weights']}")
            
            # Print the classified summary if available
            if "classifications_summary" in result:
                print("\nClassifications Summary:")
                for classifier, decision in result["classifications_summary"].items():
                    print(f"  {classifier}: {decision}")
        
        return results
        
    except Exception as e:
        logger.exception(f"Error in debug script: {e}")
        return None

if __name__ == "__main__":
    # This is the entry point for PyCharm debugging
    # Set breakpoints in the debug_gemini_classifier function to examine the Gemini component
    
    print("Starting Ad Intention Classifier debug session...")
    results = debug_gemini_classifier()
    
    if results:
        print("\nDebug session completed successfully")
    else:
        print("\nDebug session failed - check the logs for errors")