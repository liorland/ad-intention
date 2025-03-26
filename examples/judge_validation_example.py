#!/usr/bin/env python
"""
Example script demonstrating the mandatory OpenAI judge validation for Ad Intent classification.

This script shows how to use the OpenAI GPT-4o judge to validate and correct classification results.
"""

import os
import sys
import logging
from typing import Dict, List, Any
from tabulate import tabulate

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import needed modules
try:
    from ad_intention.classification.classifier import AdIntentClassifier
    from ad_intention.enhanced.openai_judge import OpenAIJudge
    from ad_intention.classification.rules import BRAND_AWARENESS, CALL_TO_ACTION
except ImportError:
    logger.error("Failed to import required modules. Make sure the package is installed.")
    sys.exit(1)

# Sample URLs for testing
SAMPLE_URLS = [
    # URLs that were previously misclassified and need judge verification
    "http://www.sandiego.org/?utm_campaign=MC_Spring2014&utm_medium=campaign",
    "http://www.lonestartexasgrill.com/?gclid=cmsv14by4locffa7mgodf0saua",
    "https://signup.ebay.com/",
    "https://www.airbnb.com/become-a-host",
    
    # Brand Awareness URLs
    "https://www.example.org/about",
    "https://brand.example.com/company-history",
    
    # Call to Action URLs
    "https://shop.example.com/buy-now",
    "https://www.example.com/signup?promo=summer"
]

def get_openai_api_key() -> str:
    """Get the OpenAI API key from environment or prompt the user."""
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("\n⚠️ OpenAI API key not found in environment variables.")
        print("The judge validation step is mandatory for accurate classification.")
        api_key = input("Please enter your OpenAI API key: ").strip()
        
        if api_key:
            # Set the key for this session
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            print("No API key provided. Exiting.")
            sys.exit(1)
            
    return api_key

def validate_classifications(urls: List[str], api_key: str) -> List[Dict[str, Any]]:
    """
    Classify URLs and validate results with the OpenAI judge.
    
    Args:
        urls: List of URLs to classify and validate
        api_key: OpenAI API key
        
    Returns:
        List of validation results
    """
    # Create the classifier and judge
    classifier = AdIntentClassifier()
    judge = OpenAIJudge(api_key=api_key)
    
    results = []
    
    for url in urls:
        logger.info(f"Processing URL: {url}")
        
        # Step 1: Get initial classification
        initial_classification = classifier.classify_url(url)
        
        # Step 2: Validate with the judge
        try:
            judgment = judge.judge_classification(url, initial_classification)
            
            # Step 3: Record the results
            result = {
                "url": url,
                "initial_classification": initial_classification,
                "is_correct": judgment["is_correct"],
                "correct_classification": judgment["correct_classification"],
                "confidence": judgment["confidence"],
                "explanation": judgment.get("explanation", "No explanation provided")
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error validating URL {url}: {e}")
            results.append({
                "url": url,
                "initial_classification": initial_classification,
                "is_correct": False,
                "correct_classification": "Error",
                "confidence": 0.0,
                "explanation": f"Error: {str(e)}"
            })
    
    return results

def main():
    """Main function to run the example."""
    print("\n=== OpenAI Judge Validation Example ===\n")
    print("This example demonstrates the mandatory judge validation step in the classification pipeline.")
    print("The judge validation ensures maximum accuracy by leveraging OpenAI GPT-4o to validate and correct classifications.\n")
    
    # Get the OpenAI API key
    api_key = get_openai_api_key()
    
    # Process the sample URLs
    results = validate_classifications(SAMPLE_URLS, api_key)
    
    # Calculate statistics
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    corrected = sum(1 for r in results if not r["is_correct"] and r["correct_classification"] not in ["Error"])
    errors = sum(1 for r in results if r["correct_classification"] == "Error")
    
    # Print summary statistics
    print("\n=== Validation Summary ===")
    print(f"Total URLs processed: {total}")
    print(f"Correctly classified: {correct} ({correct/total*100:.1f}%)")
    print(f"Corrected by judge: {corrected} ({corrected/total*100:.1f}%)")
    print(f"Errors: {errors}")
    
    # Print detailed results as a table
    headers = ["URL", "Initial Classification", "Is Correct", "Correct Classification", "Confidence"]
    table_data = []
    
    for r in results:
        table_data.append([
            r["url"], 
            r["initial_classification"], 
            "✓" if r["is_correct"] else "✗", 
            r["correct_classification"],
            f"{r['confidence']:.2f}"
        ])
    
    print("\n=== Detailed Results ===")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Print explanations for corrected classifications
    if corrected > 0:
        print("\n=== Judge Explanations for Corrections ===")
        for r in results:
            if not r["is_correct"] and r["correct_classification"] not in ["Error"]:
                print(f"\nURL: {r['url']}")
                print(f"Initial classification: {r['initial_classification']}")
                print(f"Corrected to: {r['correct_classification']}")
                print(f"Explanation: {r['explanation']}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main() 