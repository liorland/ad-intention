#!/usr/bin/env python
"""
Example script demonstrating the Ad Intent Ensemble Classifier.

This script shows how to use the ensemble classifier with various
AI-powered components to classify URLs by ad intent.
"""

import os
import sys
import logging
from typing import Dict, Any

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
    from ad_intention.enhanced.nltk_setup import setup_nltk, is_vader_available
except ImportError:
    logger.error("Failed to import EnsembleClassifier. Make sure the package is installed.")
    sys.exit(1)

# Sample URLs for testing
SAMPLE_URLS = [
    # Brand Awareness URLs (informational)
    "http://www.example.com/",
    "https://brand.example.com/about",
    "https://www.example.org/company-history",
    
    # Call to Action URLs (actionable)
    "https://shop.example.com/buy-now",
    "https://www.example.com/signup",
    "https://example.org/register?source=campaign",
    
    # URLs that were previously misclassified
    "http://www.sandiego.org/?utm_campaign=MC_Spring2014&utm_medium=campaign",
    "http://www.lonestartexasgrill.com/?gclid=cmsv14by4locffa7mgodf0saua",
    "https://signup.ebay.com/",
    "https://www.airbnb.com/become-a-host"
]

def print_classification_result(url: str, result: Dict[str, Any], show_details: bool = True):
    """
    Print the classification result in a readable format.
    
    Args:
        url: URL that was classified
        result: Classification result dictionary
        show_details: Whether to show detailed results
    """
    print(f"\n{'=' * 80}")
    print(f"URL: {url}")
    print(f"Classification: {result['classification']}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    if show_details and "details" in result:
        print("\nClassifier Details:")
        details = result["details"]
        
        # Print rule-based classifier result
        if "rule_based" in details:
            rule_result = details["rule_based"]
            print(f"  Rule-based: {rule_result['classification']} "
                  f"(confidence: {rule_result['confidence']:.2f})")
        
        # Print sentiment classifier result
        if "sentiment" in details:
            sentiment_result = details["sentiment"]
            print(f"  Sentiment: {sentiment_result['classification']} "
                  f"(confidence: {sentiment_result['confidence']:.2f})")
            
            # Print sentiment analysis details if available
            if "details" in sentiment_result and isinstance(sentiment_result["details"], dict):
                sent_details = sentiment_result["details"]
                print("\n  Sentiment Analysis Details:")
                
                # Print lexicon scores
                if "lexicon_scores" in sent_details:
                    lexicon = sent_details["lexicon_scores"]
                    print(f"    Lexicon: action={lexicon['action']:.3f}, info={lexicon['info']:.3f}")
                
                # Print VADER scores
                if "vader_scores" in sent_details:
                    vader = sent_details["vader_scores"]
                    print(f"    VADER: action={vader['action']:.3f}, info={vader['info']:.3f}")
                    print(f"      Raw: pos={vader['raw']['pos']:.3f}, neg={vader['raw']['neg']:.3f}, "
                          f"neu={vader['raw']['neu']:.3f}, compound={vader['raw']['compound']:.3f}")
                
                # Print TextBlob scores
                if "textblob_scores" in sent_details:
                    textblob = sent_details["textblob_scores"]
                    print(f"    TextBlob: action={textblob['action']:.3f}, info={textblob['info']:.3f}")
                    print(f"      Polarity: {textblob['polarity']:.3f}, Subjectivity: {textblob['subjectivity']:.3f}")
        
        # Print BERT classifier result
        if "bert" in details:
            bert_result = details["bert"]
            print(f"  BERT: {bert_result['classification']} "
                  f"(confidence: {bert_result['confidence']:.2f})")
        
        # Print Gemini classifier result
        if "gemini" in details:
            gemini_result = details["gemini"]
            print(f"  Gemini: {gemini_result['classification']} "
                  f"(confidence: {gemini_result['confidence']:.2f})")
        
        # Print OpenAI judge result
        if "openai_judge" in details:
            judge_result = details["openai_judge"]
            print(f"\nOpenAI Judge:")
            print(f"  Is Correct: {judge_result.get('is_correct', 'N/A')}")
            print(f"  Correct Classification: {judge_result.get('correct_classification', 'N/A')}")
            print(f"  Explanation: {judge_result.get('explanation', 'N/A')}")
            
        # Print weighted scores
        if "weighted_scores" in details:
            print("\nWeighted Scores:")
            for clf, score in details["weighted_scores"].items():
                print(f"  {clf}: {score:.2f}")
            
            # Print majority score
            print(f"  Majority Threshold: {details.get('majority_threshold', 0.5):.2f}")
            print(f"  Final Score: {details.get('final_score', 0.0):.2f}")
    
    print(f"{'=' * 80}")


def compare_sentiment_configurations(urls):
    """
    Compare different sentiment analysis configurations.
    
    Args:
        urls: List of URLs to test
    """
    # Create different ensemble configurations
    configurations = {
        "Base (No Sentiment)": EnsembleClassifier(
            use_sentiment=False,
            use_bert=False,
            use_gemini=False,
            use_openai_judge=False,
            debug=True
        ),
        "Lexicon Only": EnsembleClassifier(
            use_sentiment=True,
            use_vader=False,
            use_textblob=False,
            use_bert=False,
            use_gemini=False,
            use_openai_judge=False,
            debug=True
        ),
        "With VADER": EnsembleClassifier(
            use_sentiment=True,
            use_vader=True,
            use_textblob=False,
            use_bert=False,
            use_gemini=False,
            use_openai_judge=False,
            debug=True
        ),
        "With TextBlob": EnsembleClassifier(
            use_sentiment=True,
            use_vader=False,
            use_textblob=True,
            use_bert=False,
            use_gemini=False,
            use_openai_judge=False,
            debug=True
        ),
        "Full Sentiment": EnsembleClassifier(
            use_sentiment=True,
            use_vader=True,
            use_textblob=True,
            use_bert=False,
            use_gemini=False,
            use_openai_judge=False,
            debug=True
        )
    }
    
    print("\n=== Sentiment Configuration Comparison ===")
    
    for url in urls:
        print(f"\nURL: {url}")
        
        for name, classifier in configurations.items():
            result = classifier.classify_with_confidence(url)
            confidence = result["confidence"]
            classification = result["classification"]
            
            print(f"  {name}: {classification} (confidence: {confidence:.2f})")


def main():
    """Run the ensemble classifier example."""
    # Check for API keys
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    google_api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    # Warn if API keys are not available
    use_openai = bool(openai_api_key)
    use_gemini = bool(google_api_key)
    
    if not use_openai:
        logger.warning("OpenAI API key not found. OpenAI judge will not be used.")
    if not use_gemini:
        logger.warning("Google API key not found. Gemini classifier will not be used.")
    
    # Setup NLTK resources for VADER
    vader_available = is_vader_available()
    if not vader_available:
        setup_nltk(['vader_lexicon'])
        vader_available = is_vader_available()
    
    if vader_available:
        logger.info("VADER sentiment analyzer is available")
    else:
        logger.warning("VADER sentiment analyzer is not available. Some features will be limited.")
    
    # Initialize ensemble classifier with all components
    logger.info("Initializing ensemble classifier...")
    ensemble = EnsembleClassifier(
        use_sentiment=True,
        use_bert=True,
        use_gemini=use_gemini,
        use_openai_judge=use_openai,
        use_vader=vader_available,
        use_textblob=True,
        sentiment_weight=1.0,
        bert_weight=1.0,
        gemini_weight=2.0 if use_gemini else 0.0,
        rules_weight=2.0,
        majority_threshold=0.6,
        openai_api_key=openai_api_key,
        google_api_key=google_api_key,
        debug=True
    )
    
    # Classify each URL
    logger.info(f"Classifying {len(SAMPLE_URLS)} sample URLs...")
    
    # First pass: quick classification without details
    quick_results = {}
    print("\n=== Quick Classification Results ===")
    for url in SAMPLE_URLS:
        # Classification without full details for speed
        result = ensemble.classify_url(url)
        quick_results[url] = result
        print(f"{url} -> {result}")
    
    # Second pass: detailed analysis with all components
    print("\n=== Detailed Classification Results ===")
    for url in SAMPLE_URLS:
        try:
            # Get detailed classification
            result = ensemble.classify_with_confidence(url)
            
            # Print the result
            print_classification_result(url, result)
            
        except Exception as e:
            logger.error(f"Error classifying URL {url}: {e}")
    
    # Compare different sentiment analysis configurations
    test_urls = [
        "https://example.com/signup",
        "https://brand.example.com/about",
        "http://www.sandiego.org/?utm_campaign=MC_Spring2014",
        "https://www.airbnb.com/become-a-host"
    ]
    
    compare_sentiment_configurations(test_urls)
    
    # Print summary
    brand_count = sum(1 for r in quick_results.values() if r == BRAND_AWARENESS)
    cta_count = sum(1 for r in quick_results.values() if r == CALL_TO_ACTION)
    
    print("\n=== Classification Summary ===")
    print(f"Total URLs: {len(SAMPLE_URLS)}")
    print(f"Brand Awareness: {brand_count}")
    print(f"Call to Action: {cta_count}")
    
    logger.info("Example completed successfully!")


if __name__ == "__main__":
    main() 