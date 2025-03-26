"""
Ad Intent Classifier implementation.

This module provides the main classifier for determining whether a URL
indicates a Brand Awareness or Call to Action intent.
"""

import logging
from typing import List, Dict, Union, Optional
from urllib.parse import urlparse

from ad_intention.utils.url_helpers import (
    clean_url,
    normalize_url,
    extract_url_components,
    is_root_domain as is_root_domain_helper
)
from ad_intention.classification.rules import (
    BRAND_AWARENESS,
    CALL_TO_ACTION,
    CTA_PATH_INDICATORS,
    BRAND_AWARENESS_PATH_INDICATORS,
    contains_path_indicator,
    contains_action_keyword,
    has_promotional_params,
    is_landing_page_url,
    is_tracking_only_root_domain
)


# Set up logging
logger = logging.getLogger(__name__)


class AdIntentClassifier:
    """
    Classifier for determining ad intent based on URL structure.
    
    This classifier implements rule-based logic to categorize URLs as either
    "Brand Awareness" or "Call to Action" based on patterns in the URL structure.
    """
    
    def __init__(self, 
                 debug: bool = False, 
                 prioritize_root_domains: bool = True):
        """
        Initialize the classifier.
        
        Args:
            debug: If True, log detailed classification decisions
            prioritize_root_domains: If True, give higher priority to root domains
                with tracking parameters as Brand Awareness
        """
        self.debug = debug
        self.prioritize_root_domains = prioritize_root_domains
        
        # Configure logging
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        
    def classify_url(self, url: str) -> str:
        """
        Classify a single URL as either Brand Awareness or Call to Action.
        
        Args:
            url: The URL to classify
            
        Returns:
            Classification string: either "Brand Awareness" or "Call to Action"
        """
        if not url:
            return BRAND_AWARENESS  # Default for empty URLs
            
        # Preprocess the URL
        url = clean_url(url)
        url = normalize_url(url)
        
        # Extract components for analysis
        domain, path, query_params = extract_url_components(url)
        
        if self.debug:
            logger.debug(f"Classifying URL: {url}")
            logger.debug(f"Domain: {domain}, Path: {path}, Params: {query_params}")
        
        # Special case: Root domain with only tracking parameters
        # This was identified as a potential misclassification in the validation findings
        if self.prioritize_root_domains and is_tracking_only_root_domain(domain, path, query_params):
            if self.debug:
                logger.debug(f"URL is a root domain with only tracking parameters, classifying as Brand Awareness")
            return BRAND_AWARENESS
        
        # Direct path checks for clear CTA indicators - highest priority
        path_lower = path.lower()
        if any(indicator in path_lower for indicator in ['/cart', '/checkout', '/buy', '/shop/bag', '/order', '/payment', '/signup', '/register']):
            if self.debug:
                logger.debug(f"URL contains direct CTA path indicator, classifying as Call to Action")
            return CALL_TO_ACTION
            
        # Check for CTA path indicators
        if contains_path_indicator(path, CTA_PATH_INDICATORS):
            if self.debug:
                logger.debug(f"URL contains CTA path indicator, classifying as Call to Action")
            return CALL_TO_ACTION
            
        # Check for landing page indicators
        if is_landing_page_url(path):
            if self.debug:
                logger.debug(f"URL has landing page indicator, classifying as Call to Action")
            return CALL_TO_ACTION
            
        # Check for action keywords
        if contains_action_keyword(url):
            if self.debug:
                logger.debug(f"URL contains action keyword, classifying as Call to Action")
            return CALL_TO_ACTION
            
        # Check for promotional parameters
        if has_promotional_params(query_params):
            if self.debug:
                logger.debug(f"URL has promotional parameters, classifying as Call to Action")
            return CALL_TO_ACTION
            
        # Check for brand awareness indicators
        is_root = path in ('', '/')
        has_brand_path = contains_path_indicator(path, BRAND_AWARENESS_PATH_INDICATORS)
        
        if is_root or has_brand_path:
            if self.debug:
                logger.debug(f"URL is root domain or has brand awareness path, classifying as Brand Awareness")
            return BRAND_AWARENESS
            
        # If no strong indicators are found, default classification
        # For ambiguous cases, default to Brand Awareness
        if self.debug:
            logger.debug(f"No strong indicators found, defaulting to Brand Awareness")
            
        return BRAND_AWARENESS
        
    def classify_urls(self, urls: List[str]) -> List[str]:
        """
        Classify multiple URLs.
        
        Args:
            urls: List of URLs to classify
            
        Returns:
            List of classifications (one per URL)
        """
        return [self.classify_url(url) for url in urls]
        
    def classify_with_confidence(self, url: str) -> Dict[str, Union[str, float]]:
        """
        Classify a URL with a confidence score.
        
        This is an enhanced version that provides a confidence level
        for the classification.
        
        Args:
            url: The URL to classify
            
        Returns:
            Dictionary with 'classification' and 'confidence' keys
        """
        # Preprocess the URL
        url = clean_url(url)
        url = normalize_url(url)
        
        # Extract components for analysis
        domain, path, query_params = extract_url_components(url)
        
        if self.debug:
            logger.debug(f"Calculating confidence for URL: {url}")
            logger.debug(f"Domain: {domain}, Path: {path}, Params: {query_params}")
        
        # Initialize confidence scores with base values to avoid 0.0 confidence
        cta_confidence = 0.1
        brand_confidence = 0.15  # Slight bias towards Brand Awareness as the default
        
        # Calculate confidence based on different factors
        
        # Root domain is a strong signal for Brand Awareness
        if path in ('', '/'):
            brand_confidence += 0.6
            if self.debug:
                logger.debug(f"Root domain detected: +0.6 to Brand Awareness (total: {brand_confidence})")
            
        # Brand Awareness path indicators
        if contains_path_indicator(path, BRAND_AWARENESS_PATH_INDICATORS):
            brand_confidence += 0.5
            if self.debug:
                logger.debug(f"Brand Awareness path indicator detected: +0.5 to Brand Awareness (total: {brand_confidence})")
            
        # CTA path indicators are strong signals - give them higher weight
        if contains_path_indicator(path, CTA_PATH_INDICATORS):
            cta_confidence += 0.8
            if self.debug:
                logger.debug(f"CTA path indicator detected: +0.8 to CTA (total: {cta_confidence})")
                
        # Check for specific patterns that strongly indicate CTA
        if "cart" in path.lower() or "checkout" in path.lower() or "buy" in path.lower() or "signup" in path.lower():
            cta_confidence += 0.7
            if self.debug:
                logger.debug(f"Direct CTA keyword in path: +0.7 to CTA (total: {cta_confidence})")
            
        # Landing page indicators
        if is_landing_page_url(path):
            cta_confidence += 0.6
            if self.debug:
                logger.debug(f"Landing page indicator detected: +0.6 to CTA (total: {cta_confidence})")
            
        # Action keywords
        if contains_action_keyword(url):
            cta_confidence += 0.5
            if self.debug:
                logger.debug(f"Action keyword detected: +0.5 to CTA (total: {cta_confidence})")
            
        # Promotional parameters
        if has_promotional_params(query_params):
            cta_confidence += 0.5
            if self.debug:
                logger.debug(f"Promotional parameters detected: +0.5 to CTA (total: {cta_confidence})")
            
        # Special case: Root domain with only tracking parameters
        if is_tracking_only_root_domain(domain, path, query_params):
            brand_confidence += 0.6
            cta_confidence -= 0.3  # Reduce CTA confidence for this case
            if self.debug:
                logger.debug(f"Root domain with tracking params detected: +0.6 to Brand, -0.3 to CTA")
            
        # Determine classification and confidence
        classification = CALL_TO_ACTION if cta_confidence > brand_confidence else BRAND_AWARENESS
        confidence = max(0.3, min(1.0, cta_confidence if classification == CALL_TO_ACTION else brand_confidence))
        
        if self.debug:
            logger.debug(f"Final classification: {classification} with confidence {confidence}")
            logger.debug(f"CTA confidence: {cta_confidence}, Brand confidence: {brand_confidence}")
            
        return {
            'classification': classification,
            'confidence': round(confidence, 2)
        } 