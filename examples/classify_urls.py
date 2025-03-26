#!/usr/bin/env python
"""
Basic example of classifying URLs with the Ad Intent Classifier.
"""
import argparse
import csv
import sys
from typing import List

from ad_intention.classification.classifier import AdIntentClassifier


def classify_from_command_line(urls: List[str], with_confidence: bool = False) -> None:
    """
    Classify URLs provided as command line arguments.
    
    Args:
        urls: List of URLs to classify
        with_confidence: Whether to include confidence scores
    """
    classifier = AdIntentClassifier()
    
    if with_confidence:
        print("URL | Classification | Confidence")
        print("-" * 60)
        
        for url in urls:
            result = classifier.classify_with_confidence(url)
            print(f"{url} | {result['classification']} | {result['confidence']}")
    else:
        print("URL | Classification")
        print("-" * 40)
        
        for url in urls:
            classification = classifier.classify_url(url)
            print(f"{url} | {classification}")


def classify_from_file(input_file: str, output_file: str = None, with_confidence: bool = False) -> None:
    """
    Classify URLs from a file (one URL per line).
    
    Args:
        input_file: Path to input file containing URLs
        output_file: Path to output CSV file (optional)
        with_confidence: Whether to include confidence scores
    """
    classifier = AdIntentClassifier()
    results = []
    
    # Read URLs from file
    with open(input_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    # Classify URLs
    for url in urls:
        if with_confidence:
            result = classifier.classify_with_confidence(url)
            results.append({
                'url': url,
                'classification': result['classification'],
                'confidence': result['confidence']
            })
        else:
            classification = classifier.classify_url(url)
            results.append({
                'url': url,
                'classification': classification
            })
    
    # Write results to output file or stdout
    if output_file:
        with open(output_file, 'w', newline='') as f:
            if with_confidence:
                fieldnames = ['url', 'classification', 'confidence']
            else:
                fieldnames = ['url', 'classification']
                
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Results written to {output_file}")
    else:
        if with_confidence:
            print("URL | Classification | Confidence")
            print("-" * 60)
            for result in results:
                print(f"{result['url']} | {result['classification']} | {result['confidence']}")
        else:
            print("URL | Classification")
            print("-" * 40)
            for result in results:
                print(f"{result['url']} | {result['classification']}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Classify URLs as Brand Awareness or Call to Action')
    
    # Create a group for input sources
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-u', '--urls', nargs='+', help='URLs to classify')
    input_group.add_argument('-f', '--file', help='File containing URLs (one per line)')
    
    # Other arguments
    parser.add_argument('-o', '--output', help='Output file path (CSV)')
    parser.add_argument('-c', '--confidence', action='store_true', help='Include confidence scores')
    
    args = parser.parse_args()
    
    if args.urls:
        classify_from_command_line(args.urls, args.confidence)
    elif args.file:
        classify_from_file(args.file, args.output, args.confidence)


if __name__ == '__main__':
    main() 