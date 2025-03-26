#!/usr/bin/env python
"""
Example script for tracking OpenAI API calls with the OpenAIJudge component.

This script demonstrates how to track the success rate, timing, and cost
of OpenAI API calls when using the OpenAIJudge for validation.
"""

import argparse
import csv
import json
import os
import sys
import time
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import the OpenAIJudge
try:
    from ad_intention.enhanced.openai_judge import OpenAIJudge
    from ad_intention.enhanced.ensemble_classifier import EnsembleClassifier
except ImportError:
    logger.error("Failed to import required modules. Make sure the package is installed.")
    sys.exit(1)

# Sample URLs for testing
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

def process_with_judge(
    urls: List[str],
    initial_classification: str = "Brand Awareness",
    api_key: str = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Process URLs with OpenAI judge and track API calls.
    
    Args:
        urls: List of URLs to judge
        initial_classification: Initial classification to validate
        api_key: OpenAI API key
        debug: Whether to enable debug logging
    
    Returns:
        Dictionary with results and API statistics
    """
    # Initialize the judge
    judge = OpenAIJudge(api_key=api_key, debug=debug)
    
    results = []
    start_time = time.time()
    
    # Process each URL
    for i, url in enumerate(urls, 1):
        logger.info(f"Processing URL {i}/{len(urls)}: {url}")
        
        try:
            # Get the judge's evaluation
            judgment = judge.judge_classification(url, initial_classification)
            
            # Store the result
            results.append({
                "url": url,
                "initial_classification": initial_classification,
                "is_correct": judgment["is_correct"],
                "correct_classification": judgment["correct_classification"],
                "confidence": judgment["confidence"],
                "explanation": judgment["explanation"]
            })
            
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            results.append({
                "url": url,
                "error": str(e)
            })
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Get API statistics
    api_stats = judge.get_api_stats()
    
    return {
        "results": results,
        "api_stats": api_stats,
        "processing_time": processing_time
    }

def process_with_ensemble(
    urls: List[str],
    api_key: str = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Process URLs with the ensemble classifier and track OpenAI API calls.
    
    Args:
        urls: List of URLs to classify
        api_key: OpenAI API key
        debug: Whether to enable debug logging
    
    Returns:
        Dictionary with results and API statistics
    """
    # Initialize the ensemble classifier
    ensemble = EnsembleClassifier(
        use_sentiment=True,
        use_bert=True,
        use_gemini=False,
        use_openai_judge=True,
        openai_api_key=api_key,
        debug=debug
    )
    
    results = []
    start_time = time.time()
    
    # Process each URL
    for i, url in enumerate(urls, 1):
        logger.info(f"Processing URL {i}/{len(urls)}: {url}")
        
        try:
            # Classify with the ensemble
            classification = ensemble.classify_with_confidence(url)
            
            # Store the result
            results.append({
                "url": url,
                "classification": classification["classification"],
                "confidence": classification["confidence"],
                "judge_details": classification["details"].get("openai_judge", {})
            })
            
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            results.append({
                "url": url,
                "error": str(e)
            })
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Get API statistics
    api_stats = ensemble.openai_judge.get_api_stats() if ensemble.openai_judge else {}
    
    return {
        "results": results,
        "api_stats": api_stats,
        "processing_time": processing_time
    }

def display_statistics(stats: Dict[str, Any]) -> None:
    """
    Display statistics about the API calls.
    
    Args:
        stats: Statistics dictionary
    """
    api_stats = stats["api_stats"]
    results = stats["results"]
    processing_time = stats["processing_time"]
    
    # Display OpenAI API statistics
    logger.info("=" * 50)
    logger.info("OPENAI API CALL STATISTICS")
    logger.info("=" * 50)
    logger.info(f"Total API calls attempted: {api_stats.get('api_calls_attempted', 0)}")
    logger.info(f"Successful API calls: {api_stats.get('api_calls_successful', 0)}")
    logger.info(f"Failed API calls: {api_stats.get('api_calls_failed', 0)}")
    logger.info(f"Success rate: {api_stats.get('success_rate', 0)}%")
    
    if 'avg_call_time' in api_stats:
        logger.info(f"Average API call time: {api_stats['avg_call_time']} seconds")
    
    # Calculate cost estimate (approximate)
    calls_successful = api_stats.get('api_calls_successful', 0)
    if calls_successful > 0:
        # Approximate cost per call for GPT-4o
        cost_per_call = 0.01  # $0.01 per call is a rough estimate
        estimated_cost = calls_successful * cost_per_call
        logger.info(f"Estimated API cost: ${estimated_cost:.2f} (approximate)")
    
    # Display overall performance
    logger.info("=" * 50)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Processed {len(results)} URLs in {processing_time:.2f} seconds")
    logger.info(f"Average time per URL: {processing_time / len(results):.2f} seconds")
    
    # Only for judge-only mode
    if "initial_classification" in results[0]:
        correct_count = sum(1 for r in results if r.get("is_correct", False))
        logger.info(f"Correct classifications: {correct_count}/{len(results)} ({correct_count/len(results)*100:.1f}%)")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Track OpenAI API calls with OpenAIJudge',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--urls", nargs="+", help="URLs to process")
    input_group.add_argument("--file", help="File containing URLs (one per line)")
    
    # Processing mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--judge-only", action="store_true", 
                           help="Use only OpenAIJudge (not the full ensemble)")
    mode_group.add_argument("--ensemble", action="store_true", default=True,
                           help="Use the full ensemble classifier with OpenAIJudge")
    
    # Other options
    parser.add_argument("--initial-classification", choices=["Brand Awareness", "Call to Action"],
                       default="Brand Awareness", help="Initial classification for judge-only mode")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set up debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("No OpenAI API key provided. Set via --api-key or OPENAI_API_KEY env variable.")
        sys.exit(1)
    
    # Get URLs
    urls = []
    if args.urls:
        urls = args.urls
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error reading file {args.file}: {e}")
            sys.exit(1)
    else:
        # Use sample URLs
        logger.info("No URLs provided. Using sample URLs.")
        urls = SAMPLE_URLS
    
    # Process the URLs
    if args.judge_only:
        logger.info(f"Processing {len(urls)} URLs with OpenAIJudge only...")
        stats = process_with_judge(
            urls, 
            initial_classification=args.initial_classification,
            api_key=api_key,
            debug=args.debug
        )
    else:
        logger.info(f"Processing {len(urls)} URLs with ensemble classifier...")
        stats = process_with_ensemble(
            urls,
            api_key=api_key,
            debug=args.debug
        )
    
    # Display statistics
    display_statistics(stats)
    
    # Save results if requested
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        except Exception as e:
            logger.error(f"Error saving results to {args.output}: {e}")

if __name__ == "__main__":
    main() 