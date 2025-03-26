#!/usr/bin/env python
"""
Example script showing how to use both VADER and TextBlob sentiment analyzers together.
"""

import argparse
import logging
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the classifier
from ad_intention.enhanced.sentiment_classifier import SentimentClassifier

def analyze_urls(urls, debug=False):
    """
    Analyze URLs using both VADER and TextBlob sentiment analyzers.
    
    Args:
        urls: List of URLs to analyze
        debug: Whether to enable debug logging
    """
    # Initialize the classifier with both analyzers enabled
    classifier = SentimentClassifier(
        use_vader=True,         # Enable VADER
        use_textblob=True,      # Enable TextBlob
        vader_weight=1.0,       # Equal weight for VADER
        textblob_weight=1.0,    # Equal weight for TextBlob
        lexicon_weight=1.0,     # Equal weight for lexicon
        debug=debug
    )
    
    results = []
    
    for url in urls:
        # Get the detailed sentiment analysis results
        sentiment_details = classifier.analyze_sentiment(url)
        
        # Get the classification with confidence
        classification = classifier.classify_with_confidence(url)
        
        # Extract useful information
        vader_scores = None
        textblob_scores = None
        
        # Get detailed component scores if available
        if hasattr(classifier, 'last_vader_scores') and classifier.last_vader_scores:
            vader_scores = classifier.last_vader_scores
            
        if hasattr(classifier, 'last_textblob_scores') and classifier.last_textblob_scores:
            textblob_scores = classifier.last_textblob_scores
        
        # Create result record
        result = {
            "url": url,
            "classification": classification["classification"],
            "confidence": classification["confidence"],
        }
        
        # Add VADER scores if available
        if vader_scores:
            result["vader_neg"] = vader_scores.get("neg", 0)
            result["vader_neu"] = vader_scores.get("neu", 0)
            result["vader_pos"] = vader_scores.get("pos", 0)
            result["vader_compound"] = vader_scores.get("compound", 0)
            
        # Add TextBlob scores if available
        if textblob_scores:
            result["textblob_polarity"] = textblob_scores[0]
            result["textblob_subjectivity"] = textblob_scores[1]
        
        results.append(result)
        
        # Print detailed result
        print(f"\nURL: {url}")
        print(f"Classification: {classification['classification']} (confidence: {classification['confidence']:.2f})")
        
        if vader_scores:
            print(f"VADER scores: neg={vader_scores.get('neg', 0):.3f}, pos={vader_scores.get('pos', 0):.3f}, compound={vader_scores.get('compound', 0):.3f}")
            
        if textblob_scores:
            print(f"TextBlob scores: polarity={textblob_scores[0]:.3f}, subjectivity={textblob_scores[1]:.3f}")
    
    # Create a DataFrame for easier viewing
    df = pd.DataFrame(results)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total URLs: {len(df)}")
    print(f"Brand Awareness: {len(df[df['classification'] == 'Brand Awareness'])}")
    print(f"Call to Action: {len(df[df['classification'] == 'Call to Action'])}")
    
    # Return the DataFrame
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Analyze URLs using both VADER and TextBlob sentiment analyzers"
    )
    
    parser.add_argument("--urls", nargs="+", help="URLs to analyze")
    parser.add_argument("--file", help="Input file with URLs (one per line)")
    parser.add_argument("--output", help="Output CSV file for results")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get URLs from file or arguments
    urls = []
    if args.file:
        try:
            with open(args.file, "r") as f:
                urls = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error reading file {args.file}: {e}")
            return
    elif args.urls:
        urls = args.urls
    else:
        # Use some example URLs if none provided
        urls = [
            "https://www.nike.com/",
            "https://www.microsoft.com/about",
            "https://www.apple.com/shop/buy-mac",
            "https://www.airbnb.com/become-a-host",
            "https://www.target.com/?utm_source=google"
        ]
    
    # Analyze the URLs
    results_df = analyze_urls(urls, args.debug)
    
    # Save results if output file specified
    if args.output:
        results_df.to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")

if __name__ == "__main__":
    main() 