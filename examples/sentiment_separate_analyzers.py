#!/usr/bin/env python
"""
Example script showing separate classifications using VADER and TextBlob sentiment analyzers.
This provides distinct results from each analyzer rather than a combined approach.
"""

import argparse
import logging
import pandas as pd
import re
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the classifier components
from ad_intention.enhanced.sentiment_classifier import (
    SentimentClassifier, 
    VADER_AVAILABLE, 
    TEXTBLOB_AVAILABLE,
    ACTION_WORDS,
    INFO_WORDS
)

class SeparateSentimentClassifier:
    """Extended classifier that provides separate results for each analyzer"""
    
    def __init__(self, debug=False):
        # Create the base classifier (we'll use its tokenization and base methods)
        self.base_classifier = SentimentClassifier(
            use_vader=True,
            use_textblob=True,
            debug=debug
        )
        self.debug = debug
        
        # Check availability
        self.vader_available = VADER_AVAILABLE
        self.textblob_available = TEXTBLOB_AVAILABLE
        
        logger.info(f"VADER available: {self.vader_available}")
        logger.info(f"TextBlob available: {self.textblob_available}")
    
    def classify_url(self, url: str) -> Dict[str, Any]:
        """
        Classify a URL using separate VADER and TextBlob analysis.
        
        Args:
            url: The URL to classify
            
        Returns:
            Dictionary with separate classifications
        """
        # Get the tokens from the URL
        tokens = self.base_classifier.tokenize_url(url)
        
        # Create readable text for sentiment analysis
        readable_text = self.base_classifier.create_readable_text(url, tokens)
        
        # Count action and info words (lexicon approach)
        action_count = sum(1 for token in tokens if token in ACTION_WORDS)
        info_count = sum(1 for token in tokens if token in INFO_WORDS)
        
        # VADER-specific classification
        vader_result = self._classify_with_vader(readable_text, action_count, info_count)
        
        # TextBlob-specific classification
        textblob_result = self._classify_with_textblob(readable_text, action_count, info_count)
        
        # Lexicon-only classification
        lexicon_result = self._classify_with_lexicon(url, tokens, action_count, info_count)
        
        # Combined result (similar to original classifier)
        combined_result = self.base_classifier.classify_with_confidence(url)
        
        # Aggregate all results
        result = {
            "url": url,
            "tokens": tokens,
            "vader": vader_result,
            "textblob": textblob_result,
            "lexicon": lexicon_result,
            "combined": combined_result
        }
        
        return result
    
    def _classify_with_vader(self, text: str, action_count: int, info_count: int) -> Dict[str, Any]:
        """VADER-specific classification"""
        if not self.vader_available:
            return {"available": False}
        
        # Get VADER scores
        vader_scores = self.base_classifier.analyze_vader_sentiment(text)
        
        # Interpret VADER scores for CTA vs Brand Awareness
        # Positive sentiment often correlates with action-oriented content
        vader_action_score = vader_scores["pos"] * 0.7 + vader_scores["compound"] * 0.3
        vader_info_score = vader_scores["neu"] * 0.7 + (1 - abs(vader_scores["compound"])) * 0.3
        
        # Factor in lexicon counts (with lower weight)
        if action_count + info_count > 0:
            lexicon_action_score = action_count / (action_count + info_count)
            lexicon_info_score = info_count / (action_count + info_count)
            
            # Blend with 70% VADER, 30% lexicon
            vader_action_score = vader_action_score * 0.7 + lexicon_action_score * 0.3
            vader_info_score = vader_info_score * 0.7 + lexicon_info_score * 0.3
        
        # Determine classification based on scores
        if vader_action_score > vader_info_score:
            classification = "Call to Action"
            confidence = min(1.0, 0.5 + (vader_action_score - vader_info_score))
        else:
            classification = "Brand Awareness"
            confidence = min(1.0, 0.5 + (vader_info_score - vader_action_score))
        
        return {
            "available": True,
            "classification": classification,
            "confidence": round(confidence, 2),
            "scores": vader_scores,
            "action_score": round(vader_action_score, 3),
            "info_score": round(vader_info_score, 3)
        }
    
    def _classify_with_textblob(self, text: str, action_count: int, info_count: int) -> Dict[str, Any]:
        """TextBlob-specific classification"""
        if not self.textblob_available:
            return {"available": False}
        
        # Get TextBlob scores
        polarity, subjectivity = self.base_classifier.analyze_textblob_sentiment(text)
        
        # Interpret TextBlob scores
        # Higher polarity (more positive) often indicates action-oriented content
        # Higher subjectivity can indicate persuasive language used in CTAs
        textblob_action_score = max(0, polarity) * 0.6 + subjectivity * 0.4
        textblob_info_score = (1 - abs(polarity)) * 0.7 + (1 - subjectivity) * 0.3
        
        # Factor in lexicon counts (with lower weight)
        if action_count + info_count > 0:
            lexicon_action_score = action_count / (action_count + info_count)
            lexicon_info_score = info_count / (action_count + info_count)
            
            # Blend with 70% TextBlob, 30% lexicon
            textblob_action_score = textblob_action_score * 0.7 + lexicon_action_score * 0.3
            textblob_info_score = textblob_info_score * 0.7 + lexicon_info_score * 0.3
        
        # Determine classification based on scores
        if textblob_action_score > textblob_info_score:
            classification = "Call to Action"
            confidence = min(1.0, 0.5 + (textblob_action_score - textblob_info_score))
        else:
            classification = "Brand Awareness"
            confidence = min(1.0, 0.5 + (textblob_info_score - textblob_action_score))
        
        return {
            "available": True,
            "classification": classification,
            "confidence": round(confidence, 2),
            "polarity": round(polarity, 3),
            "subjectivity": round(subjectivity, 3),
            "action_score": round(textblob_action_score, 3),
            "info_score": round(textblob_info_score, 3)
        }
    
    def _classify_with_lexicon(self, url: str, tokens: List[str], action_count: int, info_count: int) -> Dict[str, Any]:
        """Lexicon-only classification"""
        # Special patterns that strongly indicate CTA
        has_cta_pattern = False
        cta_patterns = [
            r'(sign|log)in',
            r'sign-?up',
            r'check-?out',
            r'shop.?cart',
            r'add.?to.?cart',
            r'become.?a.?',
            r'get.?started',
            r'try.?now',
            r'buy.?now',
            r'free.?trial'
        ]
        
        for pattern in cta_patterns:
            if re.search(pattern, url.lower()):
                has_cta_pattern = True
                action_count += 2  # Give extra weight to these patterns
        
        # Check for query parameters that indicate CTA
        cta_params = ['gclid', 'utm_', 'cid', 'promo', 'coupon', 'discount', 'ref', 'affiliate']
        for param in cta_params:
            if param in url.lower():
                action_count += 1
        
        # Calculate lexicon-based scores
        total = max(1, action_count + info_count)  # Avoid division by zero
        lexicon_action_score = action_count / total
        lexicon_info_score = info_count / total
        
        # Strong pattern override
        if has_cta_pattern:
            lexicon_action_score = max(lexicon_action_score, 0.7)
        
        # Determine classification based on scores
        if lexicon_action_score > lexicon_info_score:
            classification = "Call to Action"
            confidence = min(1.0, 0.5 + (lexicon_action_score - lexicon_info_score))
        else:
            classification = "Brand Awareness"
            confidence = min(1.0, 0.5 + (lexicon_info_score - lexicon_action_score))
        
        return {
            "classification": classification,
            "confidence": round(confidence, 2),
            "action_count": action_count,
            "info_count": info_count,
            "action_score": round(lexicon_action_score, 3),
            "info_score": round(lexicon_info_score, 3),
            "has_cta_pattern": has_cta_pattern
        }

def analyze_urls(urls, debug=False):
    """
    Analyze URLs using separate analyzers.
    
    Args:
        urls: List of URLs to analyze
        debug: Whether to enable debug logging
    """
    classifier = SeparateSentimentClassifier(debug=debug)
    
    results = []
    
    for url in urls:
        # Get separate classifications for the URL
        result = classifier.classify_url(url)
        
        # Print detailed results
        print(f"\n{'='*80}")
        print(f"URL: {url}")
        print(f"{'='*80}")
        
        # Print lexicon results
        lexicon = result["lexicon"]
        print(f"\n[LEXICON ANALYZER]")
        print(f"Classification: {lexicon['classification']} (confidence: {lexicon['confidence']:.2f})")
        print(f"Action words: {lexicon['action_count']}, Info words: {lexicon['info_count']}")
        print(f"Action score: {lexicon['action_score']:.3f}, Info score: {lexicon['info_score']:.3f}")
        
        # Print VADER results if available
        vader = result["vader"]
        if vader["available"]:
            print(f"\n[VADER ANALYZER]")
            print(f"Classification: {vader['classification']} (confidence: {vader['confidence']:.2f})")
            scores = vader["scores"]
            print(f"Negative: {scores['neg']:.3f}, Neutral: {scores['neu']:.3f}, Positive: {scores['pos']:.3f}, Compound: {scores['compound']:.3f}")
            print(f"Action score: {vader['action_score']:.3f}, Info score: {vader['info_score']:.3f}")
        else:
            print("\n[VADER ANALYZER] Not available")
        
        # Print TextBlob results if available
        textblob = result["textblob"]
        if textblob["available"]:
            print(f"\n[TEXTBLOB ANALYZER]")
            print(f"Classification: {textblob['classification']} (confidence: {textblob['confidence']:.2f})")
            print(f"Polarity: {textblob['polarity']:.3f}, Subjectivity: {textblob['subjectivity']:.3f}")
            print(f"Action score: {textblob['action_score']:.3f}, Info score: {textblob['info_score']:.3f}")
        else:
            print("\n[TEXTBLOB ANALYZER] Not available")
        
        # Print combined results
        combined = result["combined"]
        print(f"\n[COMBINED RESULT]")
        print(f"Classification: {combined['classification']} (confidence: {combined['confidence']:.2f})")
        
        # Add results to list for DataFrame
        row = {
            "url": url,
            "lexicon_classification": lexicon["classification"],
            "lexicon_confidence": lexicon["confidence"],
            "vader_classification": vader.get("classification", "N/A") if vader.get("available", False) else "N/A",
            "vader_confidence": vader.get("confidence", 0.0) if vader.get("available", False) else 0.0,
            "textblob_classification": textblob.get("classification", "N/A") if textblob.get("available", False) else "N/A",
            "textblob_confidence": textblob.get("confidence", 0.0) if textblob.get("available", False) else 0.0,
            "combined_classification": combined["classification"],
            "combined_confidence": combined["confidence"]
        }
        
        results.append(row)
    
    # Create a DataFrame with the results
    df = pd.DataFrame(results)
    
    # Print summary
    print(f"\n{'='*80}")
    print("CLASSIFICATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total URLs analyzed: {len(df)}")
    
    # Count agreement between different classifiers
    agreement_count = sum(
        (df["lexicon_classification"] == df["vader_classification"]) & 
        (df["vader_classification"] == df["textblob_classification"])
    )
    
    if len(df) > 0:
        agreement_pct = agreement_count / len(df) * 100
        print(f"All classifiers agree: {agreement_count}/{len(df)} ({agreement_pct:.1f}%)")
    
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Analyze URLs using separate sentiment analyzers"
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