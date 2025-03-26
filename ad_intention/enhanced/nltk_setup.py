"""
Utility for setting up NLTK resources needed for sentiment analysis.

This module provides functions to download and set up NLTK resources
required by VADER and other NLTK-based sentiment analysis tools.
"""

import os
import logging
import importlib.util

logger = logging.getLogger(__name__)

# Check if NLTK is available
NLTK_AVAILABLE = importlib.util.find_spec("nltk") is not None

def setup_nltk(resources=None, quiet=False):
    """
    Download required NLTK resources if not already available.
    
    Args:
        resources: List of resources to download (default: vader_lexicon)
        quiet: Whether to suppress output
        
    Returns:
        True if successful, False otherwise
    """
    if not NLTK_AVAILABLE:
        logger.warning("NLTK is not installed. Cannot set up resources.")
        return False
        
    try:
        import nltk
        
        # Set default resources if not provided
        if resources is None:
            resources = ['vader_lexicon']
            
        # Create NLTK data directory if it doesn't exist
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Download each resource if not already available
        for resource in resources:
            try:
                nltk.data.find(f'sentiment/{resource}')
                if not quiet:
                    logger.info(f"NLTK resource '{resource}' is already available")
            except LookupError:
                if not quiet:
                    logger.info(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=quiet)
                
        return True
        
    except Exception as e:
        logger.error(f"Error setting up NLTK resources: {e}")
        return False

def is_vader_available():
    """
    Check if VADER is available and properly set up.
    
    Returns:
        True if VADER is available, False otherwise
    """
    if not NLTK_AVAILABLE:
        return False
        
    try:
        # First, try to import VADER
        import nltk.sentiment.vader
        
        # Then try to find the lexicon
        try:
            nltk.data.find('sentiment/vader_lexicon')
            logger.info("VADER lexicon found - VADER is available")
        except LookupError:
            # If lexicon not found, try to download it
            logger.info("VADER lexicon not found - downloading now")
            nltk.download('vader_lexicon', quiet=True)
        
        # Try to initialize VADER as a final test
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        
        # If we get here, VADER is available
        logger.info("VADER is fully available and initialized")
        return True
        
    except Exception as e:
        logger.warning(f"VADER is not available: {str(e)}")
        return False

# Cache the result to avoid repeated checks
VADER_AVAILABLE_CACHE = is_vader_available()

# Make the cached result available
def is_vader_available():
    return VADER_AVAILABLE_CACHE

# Auto-setup VADER when the module is imported
if NLTK_AVAILABLE:
    setup_nltk(quiet=True) 