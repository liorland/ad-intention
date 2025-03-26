"""
Batch processing utilities for classifying large sets of URLs.
"""

import os
import logging
from typing import List, Dict, Optional, Union, Tuple
import pandas as pd
from tqdm import tqdm

# Remove this import to avoid circular imports
# from ad_intention.classification.classifier import AdIntentClassifier


logger = logging.getLogger(__name__)


def process_dataframe(
    df: pd.DataFrame,
    url_column: str,
    add_confidence: bool = False,
    batch_size: int = 1000,
    show_progress: bool = True,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Process URLs in a DataFrame and add classification results.
    
    Args:
        df: Pandas DataFrame containing URLs
        url_column: Name of the column containing URLs to classify
        add_confidence: If True, include confidence scores
        batch_size: Number of URLs to process in each batch
        show_progress: Whether to show a progress bar
        inplace: If True, modify df in-place, otherwise return a copy
        
    Returns:
        DataFrame with added classification columns
    """
    # Import classifier here to avoid circular imports
    from ad_intention.classification.classifier import AdIntentClassifier
    
    if not inplace:
        df = df.copy()
        
    if url_column not in df.columns:
        raise ValueError(f"Column '{url_column}' not found in DataFrame")
        
    # Initialize the classifier
    classifier = AdIntentClassifier()
    
    # Add columns for classifications
    if 'ad_intent' not in df.columns:
        df['ad_intent'] = None
        
    if add_confidence and 'ad_intent_confidence' not in df.columns:
        df['ad_intent_confidence'] = None
        
    # Process in batches
    total_rows = len(df)
    batch_count = (total_rows + batch_size - 1) // batch_size  # ceiling division
    
    # Set up progress bar if requested
    iterator = range(batch_count)
    if show_progress:
        iterator = tqdm(iterator, desc="Classifying URLs", unit="batch")
        
    for i in iterator:
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total_rows)
        
        # Extract URLs for this batch
        batch_df = df.iloc[start_idx:end_idx]
        urls = batch_df[url_column].tolist()
        
        # Classify URLs
        if add_confidence:
            results = []
            for url in urls:
                if pd.isna(url) or url == '':
                    results.append({'classification': None, 'confidence': None})
                else:
                    results.append(classifier.classify_with_confidence(url))
                    
            # Update DataFrame with results
            df.loc[batch_df.index, 'ad_intent'] = [r['classification'] for r in results]
            df.loc[batch_df.index, 'ad_intent_confidence'] = [r['confidence'] for r in results]
        else:
            classifications = []
            for url in urls:
                if pd.isna(url) or url == '':
                    classifications.append(None)
                else:
                    classifications.append(classifier.classify_url(url))
                    
            # Update DataFrame with classifications
            df.loc[batch_df.index, 'ad_intent'] = classifications
            
    return df


def process_csv_file(
    filepath: str,
    url_column: str,
    output_filepath: Optional[str] = None,
    add_confidence: bool = False,
    batch_size: int = 1000,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Process URLs in a CSV file and save the results.
    
    Args:
        filepath: Path to the input CSV file
        url_column: Name of the column containing URLs
        output_filepath: Path to save the output CSV (if None, won't save)
        add_confidence: If True, include confidence scores
        batch_size: Number of URLs to process in each batch
        show_progress: Whether to show a progress bar
        
    Returns:
        DataFrame with classification results
    """
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    # Read the CSV file
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        raise
        
    # Process the DataFrame
    result_df = process_dataframe(
        df=df,
        url_column=url_column,
        add_confidence=add_confidence,
        batch_size=batch_size,
        show_progress=show_progress
    )
    
    # Save results if output path is provided
    if output_filepath:
        result_df.to_csv(output_filepath, index=False)
        logger.info(f"Results saved to {output_filepath}")
        
    return result_df


def process_url_list(
    urls: List[str],
    add_confidence: bool = False,
    show_progress: bool = True
) -> Union[List[str], List[Dict[str, Union[str, float]]]]:
    """
    Process a list of URLs and return classifications.
    
    Args:
        urls: List of URLs to classify
        add_confidence: If True, include confidence scores
        show_progress: Whether to show a progress bar
        
    Returns:
        List of classifications or dictionaries with classification and confidence
    """
    # Import classifier here to avoid circular imports
    from ad_intention.classification.classifier import AdIntentClassifier
    
    # Initialize the classifier
    classifier = AdIntentClassifier()
    
    # Set up progress bar if requested
    if show_progress:
        urls = tqdm(urls, desc="Classifying URLs", unit="url")
        
    if add_confidence:
        return [classifier.classify_with_confidence(url) for url in urls]
    else:
        return [classifier.classify_url(url) for url in urls] 