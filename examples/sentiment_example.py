#!/usr/bin/env python
"""
Example script demonstrating the enhanced sentiment analysis for Ad Intent classification.

This script shows how to use the enhanced sentiment classifier that includes
VADER and TextBlob for more sophisticated URL classification.
"""

import os
import sys
import logging
from tabulate import tabulate
from typing import Dict, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to Python path to import ad_intention package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import sentiment classifier
try:
    from ad_intention.enhanced.sentiment_classifier import SentimentClassifier
    from ad_intention.enhanced.nltk_setup import setup_nltk, NLTK_AVAILABLE
except ImportError:
    logger.error("Failed to import the necessary modules. Make sure the package is installed.")
    sys.exit(1)

# Sample URLs for testing
sample_urls = [
    # Likely Call to Action URLs
    "https://example.com/signup",
    "https://shop.example.com/buy-now?promo=summer",
    "https://example.org/register?source=campaign",
    "https://www.example.com/checkout",
    "http://www.sandiego.org/?utm_campaign=MC_Spring2014&utm_medium=campaign",
    "https://www.airbnb.com/become-a-host",
    
    # Likely Brand Awareness URLs
    "https://example.com/about",
    "https://brand.example.com/company-history",
    "https://www.example.org/blog/article",
    "https://www.example.com/press-release",
    "https://example.org/team",
    "https://www.example.com/careers"
]

def analyze_url_with_sentiment(url: str, classifier: SentimentClassifier, verbose: bool = False) -> Dict:
    """
    Analyze a URL using the sentiment classifier and print the results.
    
    Args:
        url: The URL to analyze
        classifier: The sentiment classifier to use
        verbose: Whether to print verbose results
        
    Returns:
        The sentiment analysis result
    """
    # Get detailed analysis
    result = classifier.analyze_sentiment(url)
    
    # Print verbose details if requested
    if verbose and "details" in result:
        details = result["details"]
        
        print(f"\n=== Detailed Analysis for: {url} ===")
        print(f"Classification: {result['classification']}")
        print(f"Confidence: {result['confidence']:.2f}")
        
        # Print lexicon scores
        if "lexicon_scores" in details:
            lexicon = details["lexicon_scores"]
            print("\nLexicon Analysis:")
            print(f"  Action Score: {lexicon['action']:.3f}")
            print(f"  Info Score: {lexicon['info']:.3f}")
        
        # Print VADER scores
        if "vader_scores" in details:
            vader = details["vader_scores"]
            print("\nVADER Analysis:")
            print(f"  Action Score: {vader['action']:.3f}")
            print(f"  Info Score: {vader['info']:.3f}")
            print(f"  Raw Scores: pos={vader['raw']['pos']:.3f}, neg={vader['raw']['neg']:.3f}, "
                  f"neu={vader['raw']['neu']:.3f}, compound={vader['raw']['compound']:.3f}")
        
        # Print TextBlob scores
        if "textblob_scores" in details:
            textblob = details["textblob_scores"]
            print("\nTextBlob Analysis:")
            print(f"  Action Score: {textblob['action']:.3f}")
            print(f"  Info Score: {textblob['info']:.3f}")
            print(f"  Polarity: {textblob['polarity']:.3f}")
            print(f"  Subjectivity: {textblob['subjectivity']:.3f}")
    
    return result

def compare_classifier_configurations(urls: List[str]):
    """
    Compare different configurations of the sentiment classifier.
    
    Args:
        urls: List of URLs to analyze
    """
    # Create different classifier configurations
    classifiers = {
        "Lexicon Only": SentimentClassifier(
            use_vader=False, 
            use_textblob=False,
            debug=True
        ),
        "With VADER": SentimentClassifier(
            use_vader=True, 
            use_textblob=False,
            vader_weight=1.5,
            debug=True
        ),
        "With TextBlob": SentimentClassifier(
            use_vader=False, 
            use_textblob=True,
            textblob_weight=1.5,
            debug=True
        ),
        "Combined (Full)": SentimentClassifier(
            use_vader=True, 
            use_textblob=True,
            vader_weight=1.0,
            textblob_weight=0.8,
            lexicon_weight=2.0,
            debug=True
        )
    }
    
    # Collect results for each configuration
    results = []
    
    for url in urls:
        row = {"URL": url}
        
        for name, classifier in classifiers.items():
            result = classifier.classify_with_confidence(url)
            label = result["classification"]
            confidence = result["confidence"]
            row[name] = f"{label} ({confidence:.2f})"
        
        results.append(row)
    
    # Print results as a table
    headers = ["URL"] + list(classifiers.keys())
    table_data = []
    
    for result in results:
        row = [result["URL"]]
        for name in classifiers.keys():
            row.append(result[name])
        table_data.append(row)
    
    print("\n=== Classifier Configuration Comparison ===")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def main():
    """Main function to run the example."""
    # Set up NLTK resources
    if NLTK_AVAILABLE:
        setup_nltk(resources=['vader_lexicon'], quiet=False)
    
    # Create the enhanced sentiment classifier
    classifier = SentimentClassifier(
        use_vader=True,
        use_textblob=True,
        vader_weight=1.0,
        textblob_weight=0.8,
        lexicon_weight=2.0,
        debug=True
    )
    
    # Process each URL
    results = []
    
    for url in sample_urls:
        # Analyze the URL
        result = analyze_url_with_sentiment(url, classifier, verbose=False)
        
        # Store the result for the summary table
        results.append({
            "URL": url,
            "Classification": result["classification"],
            "Confidence": result["confidence"]
        })
    
    # Print a summary table
    print("\n=== Sentiment Analysis Results ===")
    headers = ["URL", "Classification", "Confidence"]
    table_data = [[r["URL"], r["Classification"], f"{r['Confidence']:.2f}"] for r in results]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Perform a detailed analysis on one example of each type
    print("\n=== Detailed Analysis Examples ===")
    
    # Example Call to Action URL
    cta_example = "https://shop.example.com/buy-now?promo=summer"
    print("\n>>> Call to Action Example:")
    analyze_url_with_sentiment(cta_example, classifier, verbose=True)
    
    # Example Brand Awareness URL
    ba_example = "https://www.example.org/blog/article"
    print("\n>>> Brand Awareness Example:")
    analyze_url_with_sentiment(ba_example, classifier, verbose=True)
    
    # Compare different classifier configurations
    compare_classifier_configurations([
        "https://example.com/signup",
        "https://example.com/about",
        "http://www.sandiego.org/?utm_campaign=MC_Spring2014",
        "https://www.airbnb.com/become-a-host"
    ])
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main() 