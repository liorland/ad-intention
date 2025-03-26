"""
Example script demonstrating OpenAI API call tracking.

This script processes multiple URLs through the ensemble classifier
and displays statistics about OpenAI API calls.
"""

import os
import logging
import time
from typing import List

from ad_intention.enhanced.ensemble_classifier import EnsembleClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample URLs to classify
SAMPLE_URLS = [
    "http://www.example.com",
    "http://www.example.com/shop/buy-now",
    "http://www.example.com/about",
    "http://www.example.com/landing/summer-sale",
    "http://signup.example.com",
    "http://www.example.com/blog/our-story",
    "http://www.example.com/contact",
    "http://www.example.com/products?utm_source=google",
    "http://www.example.com/cart",
    "http://www.example.com/?gclid=abc123"
]

def process_urls(urls: List[str], use_openai: bool = True) -> None:
    """
    Process a list of URLs and track OpenAI API calls.
    
    Args:
        urls: List of URLs to classify
        use_openai: Whether to use OpenAI judge
    """
    # Initialize the classifier
    classifier = EnsembleClassifier(
        use_sentiment=True,
        use_bert=True,
        use_gemini=False,  # Set to True if you have a Google API key
        use_openai_judge=use_openai,
        debug=True
    )
    
    results = []
    start_time = time.time()
    
    # Process each URL
    for i, url in enumerate(urls, 1):
        logger.info(f"Processing URL {i}/{len(urls)}: {url}")
        result = classifier.classify_with_confidence(url)
        results.append({
            "url": url,
            "classification": result["classification"],
            "confidence": result["confidence"]
        })
    
    # Calculate processing time
    processing_time = time.time() - start_time
    avg_time_per_url = processing_time / len(urls)
    
    # Display results
    logger.info("=" * 50)
    logger.info("CLASSIFICATION RESULTS")
    logger.info("=" * 50)
    
    for result in results:
        logger.info(f"URL: {result['url']}")
        logger.info(f"Classification: {result['classification']}")
        logger.info(f"Confidence: {result['confidence']}")
        logger.info("-" * 50)
    
    # Display OpenAI API statistics
    if use_openai and classifier.openai_judge:
        api_stats = classifier.openai_judge.get_api_stats()
        logger.info("=" * 50)
        logger.info("OPENAI API STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total API calls attempted: {api_stats['api_calls_attempted']}")
        logger.info(f"Successful API calls: {api_stats['api_calls_successful']}")
        logger.info(f"Failed API calls: {api_stats['api_calls_failed']}")
        logger.info(f"Success rate: {api_stats['success_rate']}%")
        
        if 'avg_call_time' in api_stats:
            logger.info(f"Average API call time: {api_stats['avg_call_time']} seconds")
    
    # Display overall performance
    logger.info("=" * 50)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Processed {len(urls)} URLs in {processing_time:.2f} seconds")
    logger.info(f"Average time per URL: {avg_time_per_url:.2f} seconds")
    
    # Count classifications
    brand_awareness_count = sum(1 for r in results if r["classification"] == "Brand Awareness")
    call_to_action_count = sum(1 for r in results if r["classification"] == "Call to Action")
    
    logger.info(f"Brand Awareness URLs: {brand_awareness_count}")
    logger.info(f"Call to Action URLs: {call_to_action_count}")

if __name__ == "__main__":
    # Check if OpenAI API key is available
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        logger.warning("No OpenAI API key found. Set the OPENAI_API_KEY environment variable for OpenAI judge.")
        logger.warning("Proceeding with fallback judgment mechanism.")
    
    process_urls(SAMPLE_URLS, use_openai=bool(openai_api_key)) 