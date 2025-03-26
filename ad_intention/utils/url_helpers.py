"""
URL helper utilities for preprocessing and normalizing URLs.
"""

import re
from typing import Dict, Set, Optional, Tuple
from urllib.parse import urlparse, parse_qs, unquote


def clean_url(url: str) -> str:
    """
    Clean a URL by removing leading/trailing whitespace and normalizing.
    
    Args:
        url: The URL string to clean
        
    Returns:
        Cleaned URL string
    """
    # Remove leading/trailing whitespace
    url = url.strip()
    
    # Ensure URL has a scheme (add http:// if missing)
    if url and not url.startswith(('http://', 'https://')):
        url = f"http://{url}"
        
    return url


def normalize_url(url: str) -> str:
    """
    Normalize a URL by decoding URL-encoded characters and lowercasing.
    
    Args:
        url: The URL string to normalize
        
    Returns:
        Normalized URL string
    """
    # Decode URL-encoded characters
    url = unquote(url)
    
    # Convert to lowercase
    url = url.lower()
    
    return url


def extract_url_components(url: str) -> Tuple[str, str, Dict[str, Set[str]]]:
    """
    Extract the domain, path, and query parameters from a URL.
    
    Args:
        url: The URL to extract components from
        
    Returns:
        Tuple of (domain, path, query_params)
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path
        
        # Parse query parameters into a dictionary
        # Use sets for values to handle multiple parameters with the same name
        query_params = {}
        if parsed.query:
            params = parse_qs(parsed.query)
            for key, values in params.items():
                query_params[key.lower()] = set(v.lower() for v in values if v)
                
        return domain, path, query_params
        
    except Exception as e:
        # Handle malformed URLs gracefully
        return "", "", {}


def is_root_domain(url: str) -> bool:
    """
    Check if the URL is a root domain (homepage).
    
    Args:
        url: The URL to check
        
    Returns:
        True if the URL is a root domain, False otherwise
    """
    try:
        parsed = urlparse(url)
        return parsed.path in ('', '/') and not parsed.query and not parsed.fragment
    except Exception:
        return False


def has_file_extension(path: str) -> bool:
    """
    Check if a URL path has a file extension (e.g., .html, .php).
    
    Args:
        path: The URL path to check
        
    Returns:
        True if the path has a file extension, False otherwise
    """
    # Common web file extensions
    extensions = {'.html', '.htm', '.php', '.asp', '.aspx', '.jsp', '.json', '.xml'}
    
    # Check if the path ends with any of the extensions
    return any(path.endswith(ext) for ext in extensions)


def extract_path_segments(path: str) -> list:
    """
    Extract meaningful segments from a URL path.
    
    Args:
        path: The URL path to process
        
    Returns:
        List of path segments
    """
    # Remove leading and trailing slashes
    path = path.strip('/')
    
    # Split by slashes and filter out empty segments
    return [segment for segment in path.split('/') if segment] 