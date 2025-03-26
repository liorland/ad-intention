"""
BERT-based zero-shot classifier for Ad Intent classification.

This module implements a zero-shot classification approach using
a pre-trained BERT model to classify URLs without explicit training.
"""

import re
import logging
from typing import Dict, Union, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Define constants
BRAND_AWARENESS = "Brand Awareness"
CALL_TO_ACTION = "Call to Action"


class BertZeroShotClassifier:
    """
    Classifier that uses a BERT model for zero-shot classification of URLs.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize the BERT classifier.
        
        Args:
            debug: Whether to enable debug logging
        """
        self.debug = debug
        self.model = None
        self.tokenizer = None
        
    def _load_model(self):
        """Load the BERT model and tokenizer on first use."""
        if self.model is None:
            try:
                from transformers import pipeline
                # Use a zero-shot classification pipeline
                self.model = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"  # Smaller and faster than full BERT
                )
                if self.debug:
                    logger.debug("BERT model loaded successfully")
            except ImportError:
                logger.error("Failed to import transformers. Please install with: pip install transformers torch")
                raise
            except Exception as e:
                logger.error(f"Error loading BERT model: {e}")
                raise
    
    def _preprocess_url(self, url: str) -> str:
        """
        Process a URL into a format more suitable for text classification.
        
        Args:
            url: The URL to preprocess
            
        Returns:
            Processed text representation of the URL
        """
        # Remove protocol
        url = re.sub(r'^https?://', '', url)
        
        # Replace special characters with spaces
        url = re.sub(r'[/\-_\.?&=]', ' ', url)
        
        # Remove common TLDs and www
        url = re.sub(r'\b(www|com|org|net|io|co|gov)\b', '', url)
        
        # Clean up multiple spaces
        url = re.sub(r'\s+', ' ', url).strip()
        
        # Create a more natural language representation
        if '?' in url:
            base_part, query_part = url.split('?', 1)
            return f"Visit {base_part} with parameters {query_part}"
        else:
            return f"Visit {url}"
    
    def classify_with_confidence(self, url: str) -> Dict[str, Union[str, float]]:
        """
        Classify a URL with confidence using BERT zero-shot classification.
        
        Args:
            url: The URL to classify
            
        Returns:
            Dictionary with classification and confidence
        """
        self._load_model()
        
        # Preprocess the URL into a format suitable for BERT
        processed_text = self._preprocess_url(url)
        
        # Define classification labels
        candidate_labels = [
            "providing information about a brand",  # Brand Awareness
            "encouraging user action like purchase or signup"  # Call to Action
        ]
        
        try:
            # Run zero-shot classification
            result = self.model(processed_text, candidate_labels, multi_label=False)
            
            if self.debug:
                logger.debug(f"BERT classification for: {url}")
                logger.debug(f"Processed text: {processed_text}")
                logger.debug(f"Classification result: {result}")
            
            # Map the result to our classification schema
            scores = dict(zip(result["labels"], result["scores"]))
            
            info_score = scores["providing information about a brand"]
            action_score = scores["encouraging user action like purchase or signup"]
            
            if action_score > info_score:
                classification = CALL_TO_ACTION
                confidence = action_score
            else:
                classification = BRAND_AWARENESS
                confidence = info_score
                
            return {
                "classification": classification,
                "confidence": round(float(confidence), 2)
            }
            
        except Exception as e:
            logger.error(f"Error in BERT classification: {e}")
            # Fallback to a simple rule-based approach if BERT fails
            if any(kw in url.lower() for kw in ["cart", "checkout", "signup", "sign-up", "buy", "order"]):
                return {"classification": CALL_TO_ACTION, "confidence": 0.7}
            else:
                return {"classification": BRAND_AWARENESS, "confidence": 0.6}
    
    def classify_url(self, url: str) -> str:
        """
        Classify a URL using BERT zero-shot classification.
        
        Args:
            url: The URL to classify
            
        Returns:
            Classification: either "Brand Awareness" or "Call to Action"
        """
        result = self.classify_with_confidence(url)
        return result["classification"]
    
    
# Dummy implementation for environments without transformers
class DummyBertClassifier:
    """A fallback classifier when transformers isn't available"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        logger.warning("Using DummyBertClassifier as transformers package is not available")
        
    def classify_with_confidence(self, url: str) -> Dict[str, Union[str, float]]:
        """Simple rule-based fallback"""
        url = url.lower()
        
        # Simple rules for action-oriented URLs
        action_keywords = ["cart", "checkout", "buy", "order", "purchase", "shop", 
                          "signup", "sign-up", "register", "join", "subscribe", 
                          "download", "get", "try", "demo"]
        
        if any(kw in url for kw in action_keywords):
            return {"classification": CALL_TO_ACTION, "confidence": 0.75}
            
        # Check for tracking params which often indicate ads
        tracking_params = ["utm_", "gclid", "fbclid", "cid", "ref="]
        if any(param in url for param in tracking_params) and not url.endswith('/'):
            return {"classification": CALL_TO_ACTION, "confidence": 0.65}
            
        return {"classification": BRAND_AWARENESS, "confidence": 0.6}
        
    def classify_url(self, url: str) -> str:
        """Return classification only"""
        result = self.classify_with_confidence(url)
        return result["classification"]
        

# Factory function to get the appropriate classifier
def get_bert_classifier(debug: bool = False) -> Union[BertZeroShotClassifier, DummyBertClassifier]:
    """
    Get a BERT classifier or fall back to a dummy implementation if dependencies aren't available.
    
    Args:
        debug: Whether to enable debug logging
        
    Returns:
        A classifier object that supports classify_url and classify_with_confidence methods
    """
    try:
        import transformers
        return BertZeroShotClassifier(debug=debug)
    except ImportError:
        return DummyBertClassifier(debug=debug) 