"""
Tests for utility functions.
"""

import pytest
from ad_intention.utils.url_helpers import (
    clean_url,
    normalize_url,
    extract_url_components,
    is_root_domain,
    has_file_extension,
    extract_path_segments
)


class TestUrlHelpers:
    """Test cases for URL helper functions."""
    
    def test_clean_url(self):
        """Test URL cleaning function."""
        # Test whitespace removal
        assert clean_url("  http://example.com  ") == "http://example.com"
        
        # Test schema addition
        assert clean_url("example.com") == "http://example.com"
        assert clean_url("www.brand.com") == "http://www.brand.com"
        
        # Test existing schemas
        assert clean_url("https://secure.com") == "https://secure.com"
        assert clean_url("http://insecure.com") == "http://insecure.com"
        
        # Test empty input
        assert clean_url("") == ""
        assert clean_url("   ") == ""
        
    def test_normalize_url(self):
        """Test URL normalization function."""
        # Test lowercasing
        assert normalize_url("HTTP://EXAMPLE.COM") == "http://example.com"
        assert normalize_url("Example.COM/Test") == "example.com/test"
        
        # Test URL decoding
        assert normalize_url("http://example.com/%20space") == "http://example.com/ space"
        assert normalize_url("http://site.com/%21%40%23") == "http://site.com/!@#"
        
    def test_extract_url_components(self):
        """Test URL component extraction function."""
        # Basic extraction
        domain, path, params = extract_url_components("https://example.com/path?query=value")
        assert domain == "example.com"
        assert path == "/path"
        assert "query" in params
        assert "value" in params["query"]
        
        # Multiple parameters with same name
        domain, path, params = extract_url_components("https://example.com/path?q=1&q=2")
        assert "q" in params
        assert params["q"] == {"1", "2"}
        
        # Empty URL
        domain, path, params = extract_url_components("")
        assert domain == ""
        assert path == ""
        assert params == {}
        
        # Invalid URL
        domain, path, params = extract_url_components("not a url")
        assert domain == ""
        assert path == ""
        assert params == {}
        
    def test_is_root_domain(self):
        """Test root domain detection function."""
        # Root domains
        assert is_root_domain("http://example.com")
        assert is_root_domain("https://example.com/")
        
        # Non-root domains
        assert not is_root_domain("http://example.com/path")
        assert not is_root_domain("https://example.com/about")
        assert not is_root_domain("http://example.com?query=value")
        assert not is_root_domain("https://example.com/#fragment")
        
    def test_has_file_extension(self):
        """Test file extension detection function."""
        # Paths with file extensions
        assert has_file_extension("/page.html")
        assert has_file_extension("/script.php")
        assert has_file_extension("/data.json")
        assert has_file_extension("/legacy.aspx")
        
        # Paths without file extensions
        assert not has_file_extension("/about")
        assert not has_file_extension("/products/")
        assert not has_file_extension("/category/items")
        
    def test_extract_path_segments(self):
        """Test path segment extraction function."""
        # Simple path
        assert extract_path_segments("/about") == ["about"]
        
        # Multi-segment path
        assert extract_path_segments("/products/category/item") == ["products", "category", "item"]
        
        # Path with trailing slash
        assert extract_path_segments("/blog/") == ["blog"]
        
        # Path with empty segments
        assert extract_path_segments("//category//item/") == ["category", "item"]
        
        # Root path
        assert extract_path_segments("/") == []
        assert extract_path_segments("") == [] 