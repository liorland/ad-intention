"""
Centralized configuration module for Ad Intention Classification.

This module manages all configuration settings for the project,
loading from environment variables when available.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    # Look for .env file in the project root
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loaded environment variables from {env_path}")
    else:
        logger.warning(f".env file not found at {env_path}. Using default values or environment variables.")
except ImportError:
    logger.warning("python-dotenv not installed. Using default values or environment variables.")

# Helper functions to parse environment variables
def parse_bool(value: Optional[str]) -> bool:
    """Parse string to boolean."""
    if value is None:
        return False
    return value.lower() in ('true', 'yes', '1', 't', 'y')

def parse_float(value: Optional[str], default: float) -> float:
    """Parse string to float with default value."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Classification Settings
USE_SENTIMENT = parse_bool(os.getenv("USE_SENTIMENT", "true"))
USE_BERT = parse_bool(os.getenv("USE_BERT", "true"))
USE_GEMINI = parse_bool(os.getenv("USE_GEMINI", "true"))
USE_OPENAI_JUDGE = parse_bool(os.getenv("USE_OPENAI_JUDGE", "true"))
USE_VADER = parse_bool(os.getenv("USE_VADER", "true"))
USE_TEXTBLOB = parse_bool(os.getenv("USE_TEXTBLOB", "true"))

# Classifier Weights
SENTIMENT_WEIGHT = parse_float(os.getenv("SENTIMENT_WEIGHT"), 1.0)
BERT_WEIGHT = parse_float(os.getenv("BERT_WEIGHT"), 1.0)
GEMINI_WEIGHT = parse_float(os.getenv("GEMINI_WEIGHT"), 1.5)
RULES_WEIGHT = parse_float(os.getenv("RULES_WEIGHT"), 2.0)
VADER_WEIGHT = parse_float(os.getenv("VADER_WEIGHT"), 1.0)
TEXTBLOB_WEIGHT = parse_float(os.getenv("TEXTBLOB_WEIGHT"), 0.8)
LEXICON_WEIGHT = parse_float(os.getenv("LEXICON_WEIGHT"), 2.0)

# Classification Threshold
MAJORITY_THRESHOLD = parse_float(os.getenv("MAJORITY_THRESHOLD"), 0.6)

# Debug Mode
DEBUG = parse_bool(os.getenv("DEBUG", "false"))

# Classification Constants
BRAND_AWARENESS = "Brand Awareness"
CALL_TO_ACTION = "Call to Action"

# Action and Information Word Sets
# These were previously hardcoded in sentiment_classifier.py
ACTION_WORDS = {
    'signup', 'sign-up', 'login', 'register', 'join', 'subscribe', 'buy',
    'purchase', 'order', 'shop', 'cart', 'checkout', 'get', 'download',
    'try', 'demo', 'book', 'reserve', 'apply', 'submit', 'start', 'begin',
    'add', 'activate', 'claim', 'redeem', 'save', 'discount', 'deal', 'sale',
    'special', 'offer', 'promo', 'promotion', 'coupon', 'code', 'limited',
    'exclusive', 'free', 'trial', 'install', 'enroll', 'payment', 'pay',
    'host', 'create', 'manage', 'sell', 'become', 'hire'
}

INFO_WORDS = {
    'about', 'company', 'history', 'mission', 'vision', 'values', 'team',
    'careers', 'jobs', 'press', 'news', 'blog', 'article', 'story', 'learn',
    'discover', 'explore', 'find', 'read', 'watch', 'view', 'meet', 'know',
    'understand', 'info', 'information', 'support', 'help', 'faq', 'contact',
    'locations', 'stores', 'privacy', 'terms', 'policy', 'legal', 'copyright',
    'research', 'development', 'innovation', 'technology', 'sustainability',
    'responsibility', 'community', 'events', 'partners', 'investors', 'annual',
    'report', 'foundation', 'global', 'international', 'local'
}

# Path patterns that strongly indicate CTA
CTA_PATTERNS = [
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

# Query parameters that indicate CTA
CTA_PARAMS = ['gclid', 'utm_', 'cid', 'promo', 'coupon', 'discount', 'ref', 'affiliate']

def get_config_dict() -> Dict[str, Any]:
    """Return a dictionary of all configuration settings."""
    return {
        "api_keys": {
            "openai": OPENAI_API_KEY,
            "google": GOOGLE_API_KEY
        },
        "features": {
            "use_sentiment": USE_SENTIMENT,
            "use_bert": USE_BERT,
            "use_gemini": USE_GEMINI,
            "use_openai_judge": USE_OPENAI_JUDGE,
            "use_vader": USE_VADER,
            "use_textblob": USE_TEXTBLOB
        },
        "weights": {
            "sentiment": SENTIMENT_WEIGHT,
            "bert": BERT_WEIGHT,
            "gemini": GEMINI_WEIGHT,
            "rules": RULES_WEIGHT,
            "vader": VADER_WEIGHT,
            "textblob": TEXTBLOB_WEIGHT,
            "lexicon": LEXICON_WEIGHT
        },
        "thresholds": {
            "majority": MAJORITY_THRESHOLD
        },
        "debug": DEBUG
    }

def print_config() -> None:
    """Print the current configuration (hiding API keys)."""
    config = get_config_dict()
    
    # Mask API keys for security
    if config["api_keys"]["openai"]:
        config["api_keys"]["openai"] = "****" + config["api_keys"]["openai"][-4:] if len(config["api_keys"]["openai"]) > 4 else "****"
    if config["api_keys"]["google"]:
        config["api_keys"]["google"] = "****" + config["api_keys"]["google"][-4:] if len(config["api_keys"]["google"]) > 4 else "****"
    
    print("\n=== Ad Intention Classification Configuration ===")
    print(json.dumps(config, indent=2))
    print("================================================\n")

# Print config at module load when in debug mode
if DEBUG:
    print_config() 