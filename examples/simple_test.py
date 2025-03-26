"""
Simple test script to demonstrate the Ad Intent Classifier.
"""
import sys
import os
import logging

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO)

from ad_intention.classification.classifier import AdIntentClassifier

def main():
    # Create a classifier instance - use debug=False for cleaner output
    classifier = AdIntentClassifier(debug=False)
    
    # Test URLs
    test_urls = [
        # Brand Awareness examples
        "http://www.assoturcupra.it",
        "http://www.schiebel.net/",
        "http://www.deltaco.com",
        "http://www.nike.com/about",
        "http://mercedes-benz.com/innovation",
        
        # Call to Action examples
        "http://nike.com/cart",
        "http://apple.com/shop/buy-mac",
        "http://cocacola.com/signup?promo=summer",
        "http://starbucks.com/rewards/join",
        "http://bike-kaitori.com/lp/satei_bluelong",
        
        # Edge cases
        "http://www.sandiego.org/?utm_campaign=MC_Spring2014&utm_medium=campaign&utm_source=ABC.com&utm_term=19177028&utm_content=POE Companion Banner",
        "http://www.lonestartexasgrill.com/?gclid=cmsv14by4locffa7mgodf0saua"
    ]
    
    # Classify each URL
    print("URL Classification Results:")
    print("=" * 80)
    print(f"{'URL':<60} | {'Classification':<20}")
    print("-" * 80)
    
    for url in test_urls:
        result = classifier.classify_url(url)
        # Truncate URL if it's too long
        display_url = url[:57] + "..." if len(url) > 60 else url
        print(f"{display_url:<60} | {result:<20}")
    
    print("\n\nConfidence Analysis:")
    print("=" * 80)
    print(f"{'URL':<60} | {'Classification':<20} | {'Confidence':<10}")
    print("-" * 80)
    
    for url in test_urls:
        result = classifier.classify_with_confidence(url)
        # Truncate URL if it's too long
        display_url = url[:57] + "..." if len(url) > 60 else url
        print(f"{display_url:<60} | {result['classification']:<20} | {result['confidence']:<10.2f}")

if __name__ == "__main__":
    main() 