"""
Classification rules for Ad Intent classification.

This module defines the rule sets and criteria used to classify URLs
as either "Brand Awareness" or "Call to Action".
"""

from typing import Set, Dict, List, Tuple
import re


# Classification result constants
BRAND_AWARENESS = "Brand Awareness"
CALL_TO_ACTION = "Call to Action"


# Path indicators for Call to Action URLs
CTA_PATH_INDICATORS = {
    '/cart', 
    '/checkout', 
    '/apply', 
    '/register', 
    '/payment', 
    '/subscribe',
    '/buy',
    '/order',
    '/shop',
    '/purchase',
    '/join',
    '/signup',
    '/reservation',
    '/book',
    '/estimate',
    '/quote',
    '/download'
}


# Path indicators for Brand Awareness URLs
BRAND_AWARENESS_PATH_INDICATORS = {
    '/about',
    '/our-story',
    '/company',
    '/blog',
    '/news',
    '/press',
    '/careers',
    '/investors',
    '/history',
    '/values',
    '/mission',
    '/team',
    '/contact',  # Contact page is generally informational
    '/locations',
    '/innovation',
    '/research',
    '/sustainability',
    '/corporate',
    '/information'
}


# Query parameters indicating promotional or tracking for CTA
CTA_QUERY_PARAMETERS = {
    'promo',
    'offer',
    'discount',
    'coupon',
    'gclid',  # Google Click ID
    'fbclid',  # Facebook Click ID
    'utm_source',
    'utm_medium',
    'utm_campaign',
    'utm_content',
    'utm_term',
    'cid',  # Campaign ID
    'ref',  # Referrer
    'source',
    'affid',  # Affiliate ID
    'aff_id',
    'promocode',
    'promo_code'
}


# Action-based keywords that indicate CTA
ACTION_KEYWORDS = {
    'start',
    'buy',
    'join',
    'get',
    'claim',
    'order',
    'apply',
    'register',
    'subscribe',
    'download',
    'shop',
    'purchase',
    'book',
    'reserve',
    'estimate',
    'satei',  # Japanese term for "assessment" or "appraisal"
    'quote',
    'trial',
    'demo',
    'signup',
    'sign-up',
    'login',
    'enroll'
}


# Landing page indicators (parts of URL that suggest it's a landing page)
LANDING_PAGE_INDICATORS = [
    '/lp/',
    'lp-',
    '/landing/',
    'landing-',
    '/campaign/',
    'campaign-',
    '/promo/',
    'promo-',
    '/special/',
    'special-',
    '/offer/'
]


def contains_path_indicator(path: str, indicators: Set[str]) -> bool:
    """
    Check if the URL path contains any of the specified indicators.
    
    Args:
        path: The URL path to check
        indicators: Set of path indicators to check for
        
    Returns:
        True if the path contains any of the indicators, False otherwise
    """
    path = path.lower()
    
    # Ensure path starts with a leading slash for consistent matching
    if not path.startswith('/'):
        path = '/' + path
        
    # Multiple matching strategies for more robust detection:
    
    # 1. Check if indicator exists as a complete path segment
    path_with_slashes = f"{path}/" if not path.endswith('/') else path
    
    # 2. Check if indicator exists anywhere in the path (more lenient)
    for indicator in indicators:
        # Remove any leading slashes from the indicator for consistent matching
        indicator = indicator.lstrip('/')
        
        # Match as a complete path segment
        if f"/{indicator}/" in path_with_slashes or path == f"/{indicator}" or path.endswith(f"/{indicator}"):
            return True
            
        # For cart/checkout/etc. paths, match more liberally to catch variations
        if indicator in ('cart', 'checkout', 'buy', 'order', 'shop', 'subscribe', 'signup', 'register', 'payment'):
            if indicator in path.split('/'):
                return True
            # Also match common variations like /shopping-cart, /my-cart, etc.
            if f"{indicator}" in path:
                return True
    
    return False


def contains_action_keyword(url: str) -> bool:
    """
    Check if the URL contains any action-based keywords.
    
    Args:
        url: The URL to check
        
    Returns:
        True if the URL contains any action keywords, False otherwise
    """
    url_lower = url.lower()
    
    # Check if any action keyword exists as a whole word in the URL
    for keyword in ACTION_KEYWORDS:
        # \b is a word boundary in regex
        pattern = fr'\b{re.escape(keyword)}\b'
        if re.search(pattern, url_lower):
            return True
            
    return False


def has_promotional_params(query_params: Dict[str, Set[str]]) -> bool:
    """
    Check if the URL has promotional or campaign tracking parameters.
    
    Args:
        query_params: Dictionary of query parameters from the URL
        
    Returns:
        True if promotional parameters are present, False otherwise
    """
    # Check if any of the parameter names match CTA query parameters
    for param in CTA_QUERY_PARAMETERS:
        if param in query_params:
            return True
    
    return False


def is_landing_page_url(path: str) -> bool:
    """
    Check if the URL path indicates a landing page.
    
    Args:
        path: The URL path to check
        
    Returns:
        True if it's a landing page, False otherwise
    """
    path_lower = path.lower()
    
    for indicator in LANDING_PAGE_INDICATORS:
        if indicator in path_lower:
            return True
            
    return False


def is_tracking_only_root_domain(domain: str, path: str, query_params: Dict[str, Set[str]]) -> bool:
    """
    Check if the URL is a root domain with only tracking parameters.
    This is a special case identified in validation findings.
    
    Args:
        domain: The URL domain
        path: The URL path
        query_params: The URL query parameters
        
    Returns:
        True if it's a root domain with only tracking parameters, False otherwise
    """
    # Check if it's a root domain (path is empty or just /)
    if path not in ('', '/'):
        return False
        
    # Check if it only has tracking parameters
    tracking_only = False
    
    if query_params:
        # Only check if there are any query params to begin with
        has_utm = any(param.startswith('utm_') for param in query_params)
        has_gclid = 'gclid' in query_params
        
        # Return true if the only query parameters are tracking-related
        all_params = set(query_params.keys())
        tracking_params = {'gclid', 'fbclid', 'utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 'utm_term'}
        
        # Check if all parameters are tracking parameters
        tracking_only = all(param in tracking_params or param.startswith('utm_') for param in all_params)
        
    return tracking_only 