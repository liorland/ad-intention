#!/usr/bin/env python
"""
Example of batch processing URLs using the Ad Intent Classifier.
"""
import argparse
import os
import pandas as pd
from tqdm import tqdm

from ad_intention.utils.batch_process import process_dataframe, process_csv_file


def process_excel_file(
    input_file: str,
    url_column: str,
    output_file: str = None,
    add_confidence: bool = False,
    batch_size: int = 1000
) -> pd.DataFrame:
    """
    Process URLs from an Excel file.
    
    Args:
        input_file: Path to the Excel file
        url_column: Name of the column containing URLs
        output_file: Path to save the results (optional)
        add_confidence: Whether to include confidence scores
        batch_size: Number of URLs to process in each batch
        
    Returns:
        DataFrame with classification results
    """
    print(f"Reading Excel file: {input_file}")
    df = pd.read_excel(input_file)
    
    if url_column not in df.columns:
        raise ValueError(f"Column '{url_column}' not found in the Excel file")
        
    print(f"Processing {len(df)} URLs in batches of {batch_size}")
    result_df = process_dataframe(
        df=df,
        url_column=url_column,
        add_confidence=add_confidence,
        batch_size=batch_size,
        show_progress=True
    )
    
    if output_file:
        # Determine output format based on extension
        _, ext = os.path.splitext(output_file)
        
        if ext.lower() == '.csv':
            result_df.to_csv(output_file, index=False)
            print(f"Results saved to CSV: {output_file}")
        elif ext.lower() in ('.xlsx', '.xls'):
            result_df.to_excel(output_file, index=False)
            print(f"Results saved to Excel: {output_file}")
        else:
            print(f"Unsupported output format: {ext}. Saving as CSV instead.")
            csv_path = os.path.splitext(output_file)[0] + '.csv'
            result_df.to_csv(csv_path, index=False)
            print(f"Results saved to: {csv_path}")
    
    return result_df


def analyze_results(df: pd.DataFrame, classification_column: str = 'ad_intent') -> None:
    """
    Analyze and print summary statistics of classification results.
    
    Args:
        df: DataFrame with classification results
        classification_column: Name of the classification column
    """
    if classification_column not in df.columns:
        raise ValueError(f"Column '{classification_column}' not found in the DataFrame")
        
    total = len(df)
    category_counts = df[classification_column].value_counts()
    
    print("\nClassification Summary:")
    print("-" * 40)
    
    for category, count in category_counts.items():
        percentage = (count / total) * 100
        print(f"{category}: {count} ({percentage:.2f}%)")
        
    print("-" * 40)
    print(f"Total URLs: {total}")
    
    # If confidence scores are available, show average confidence per category
    if 'ad_intent_confidence' in df.columns:
        print("\nAverage Confidence Scores:")
        avg_confidence = df.groupby(classification_column)['ad_intent_confidence'].mean()
        
        for category, avg in avg_confidence.items():
            print(f"{category}: {avg:.2f}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Batch process URLs from files')
    
    # Input file and options
    parser.add_argument('input_file', help='Input file path (CSV or Excel)')
    parser.add_argument('url_column', help='Name of the column containing URLs')
    
    # Output options
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-c', '--confidence', action='store_true', 
                        help='Include confidence scores')
    
    # Processing options
    parser.add_argument('-b', '--batch-size', type=int, default=1000,
                       help='Batch size for processing')
    parser.add_argument('-a', '--analyze', action='store_true',
                       help='Print analysis of results')
    
    args = parser.parse_args()
    
    # Determine file type and process accordingly
    _, ext = os.path.splitext(args.input_file)
    
    try:
        if ext.lower() in ('.xlsx', '.xls'):
            df = process_excel_file(
                input_file=args.input_file,
                url_column=args.url_column,
                output_file=args.output,
                add_confidence=args.confidence,
                batch_size=args.batch_size
            )
        elif ext.lower() == '.csv':
            df = process_csv_file(
                filepath=args.input_file,
                url_column=args.url_column,
                output_filepath=args.output,
                add_confidence=args.confidence,
                batch_size=args.batch_size,
                show_progress=True
            )
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
        if args.analyze:
            analyze_results(df)
            
    except Exception as e:
        print(f"Error processing file: {e}")
        return 1
        
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main()) 