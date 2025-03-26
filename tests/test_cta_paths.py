"""
Tests specifically for CTA paths like cart and checkout.
"""

import pytest
from ad_intention.classification.classifier import AdIntentClassifier
from ad_intention.classification.rules import BRAND_AWARENESS, CALL_TO_ACTION


class TestCtaPaths:
    """Test cases for CTA path detection."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.classifier = AdIntentClassifier(debug=False)
        
    def test_cart_paths(self):
        """Test classification of URLs with 'cart' in the path."""
        cart_urls = [
            "https://www.amazon.com/gp/cart/view.html",
            "https://www.walmart.com/cart",
            "https://shop.example.com/cart",
            "https://store.com/shopping-cart",
            "https://example.com/my-cart",
            "https://retailer.com/cart/items",
            "https://shop.com/cart?promo=SAVE10"
        ]
        
        for url in cart_urls:
            result = self.classifier.classify_url(url)
            assert result == CALL_TO_ACTION, f"Failed for {url}"
            
            # Also test confidence
            confidence_result = self.classifier.classify_with_confidence(url)
            assert confidence_result['classification'] == CALL_TO_ACTION
            assert confidence_result['confidence'] > 0.5
            
    def test_checkout_paths(self):
        """Test classification of URLs with 'checkout' in the path."""
        checkout_urls = [
            "https://www.amazon.com/checkout",
            "https://shop.example.com/checkout/payment",
            "https://store.com/secure-checkout",
            "https://example.com/checkout/shipping",
            "https://retailer.com/checkout?step=payment",
            "https://www.target.com/checkout"
        ]
        
        for url in checkout_urls:
            result = self.classifier.classify_url(url)
            assert result == CALL_TO_ACTION, f"Failed for {url}"
            
            # Also test confidence
            confidence_result = self.classifier.classify_with_confidence(url)
            assert confidence_result['classification'] == CALL_TO_ACTION
            assert confidence_result['confidence'] > 0.5
            
    def test_other_cta_paths(self):
        """Test classification of URLs with other CTA indicators in the path."""
        cta_urls = [
            "https://www.example.com/signup",
            "https://shop.example.com/buy-now",
            "https://store.com/order/confirm",
            "https://example.com/register",
            "https://retailer.com/payment",
            "https://www.saas.com/subscribe",
            "https://www.apple.com/shop/bag"
        ]
        
        for url in cta_urls:
            result = self.classifier.classify_url(url)
            assert result == CALL_TO_ACTION, f"Failed for {url}"
            
            # Also test confidence
            confidence_result = self.classifier.classify_with_confidence(url)
            assert confidence_result['classification'] == CALL_TO_ACTION
            assert confidence_result['confidence'] > 0.3
            
    def test_edge_cases(self):
        """Test edge cases that might be ambiguous."""
        test_cases = [
            # URL, expected classification
            ("https://www.example.com/category/cart-accessories", BRAND_AWARENESS),  # 'cart' is part of product name
            ("https://blog.example.com/how-to-optimize-checkout", BRAND_AWARENESS),  # blog about checkout
            ("https://www.example.com/checkout-our-new-store", BRAND_AWARENESS),  # 'checkout' used as verb
            ("https://shop.example.com/cart", CALL_TO_ACTION),  # actual cart
            ("https://shop.example.com/mycart", CALL_TO_ACTION),  # variation of cart
        ]
        
        for url, expected in test_cases:
            result = self.classifier.classify_url(url)
            assert result == expected, f"Failed for {url}" 