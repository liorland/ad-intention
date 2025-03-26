"""
Tests for the AdIntentClassifier.
"""

import pytest
from ad_intention.classification.classifier import AdIntentClassifier
from ad_intention.classification.rules import BRAND_AWARENESS, CALL_TO_ACTION


class TestAdIntentClassifier:
    """Test cases for the AdIntentClassifier."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.classifier = AdIntentClassifier(debug=False)
        
    def test_empty_url(self):
        """Test classification of empty URLs."""
        assert self.classifier.classify_url("") == BRAND_AWARENESS
        assert self.classifier.classify_url(None) == BRAND_AWARENESS
        
    def test_root_domains(self):
        """Test classification of root domains."""
        # Root domains should be Brand Awareness
        assert self.classifier.classify_url("http://example.com") == BRAND_AWARENESS
        assert self.classifier.classify_url("https://brand.com/") == BRAND_AWARENESS
        assert self.classifier.classify_url("www.company.org") == BRAND_AWARENESS
        
    def test_brand_awareness_paths(self):
        """Test classification of URLs with Brand Awareness paths."""
        assert self.classifier.classify_url("https://example.com/about") == BRAND_AWARENESS
        assert self.classifier.classify_url("https://brand.com/our-story") == BRAND_AWARENESS
        assert self.classifier.classify_url("https://company.org/blog") == BRAND_AWARENESS
        assert self.classifier.classify_url("https://example.com/company/team") == BRAND_AWARENESS
        assert self.classifier.classify_url("https://brand.com/investors") == BRAND_AWARENESS
        
    def test_cta_paths(self):
        """Test classification of URLs with CTA paths."""
        assert self.classifier.classify_url("https://example.com/cart") == CALL_TO_ACTION
        assert self.classifier.classify_url("https://shop.com/checkout") == CALL_TO_ACTION
        assert self.classifier.classify_url("https://company.org/apply") == CALL_TO_ACTION
        assert self.classifier.classify_url("https://example.com/subscribe/premium") == CALL_TO_ACTION
        assert self.classifier.classify_url("https://brand.com/join-now") == CALL_TO_ACTION
        
    def test_promotional_parameters(self):
        """Test classification of URLs with promotional parameters."""
        assert self.classifier.classify_url("https://example.com?promo=summer") == CALL_TO_ACTION
        assert self.classifier.classify_url("https://brand.com/?discount=20off") == CALL_TO_ACTION
        assert self.classifier.classify_url("https://shop.com/products?coupon=SAVE10") == CALL_TO_ACTION
        
    def test_tracking_parameters(self):
        """Test classification of URLs with tracking parameters."""
        # Root domains with only tracking parameters - check that our special case handling works
        classifier_with_priority = AdIntentClassifier(prioritize_root_domains=True)
        classifier_without_priority = AdIntentClassifier(prioritize_root_domains=False)
        
        # These should be classified as Brand Awareness with prioritize_root_domains=True
        assert classifier_with_priority.classify_url("https://example.com?utm_source=google") == BRAND_AWARENESS
        assert classifier_with_priority.classify_url("https://brand.com/?utm_campaign=spring&utm_medium=social") == BRAND_AWARENESS
        assert classifier_with_priority.classify_url("https://company.org?gclid=abc123") == BRAND_AWARENESS
        
        # These would be classified as Call to Action without the special handling
        assert classifier_without_priority.classify_url("https://example.com?utm_source=google") == CALL_TO_ACTION
        assert classifier_without_priority.classify_url("https://brand.com/?utm_campaign=spring&utm_medium=social") == CALL_TO_ACTION
        assert classifier_without_priority.classify_url("https://company.org?gclid=abc123") == CALL_TO_ACTION
        
        # Non-root domains with tracking parameters should always be CTA
        assert self.classifier.classify_url("https://example.com/products?utm_source=google") == CALL_TO_ACTION
        assert self.classifier.classify_url("https://brand.com/summer-sale?gclid=abc123") == CALL_TO_ACTION
        
    def test_action_keywords(self):
        """Test classification of URLs with action keywords."""
        assert self.classifier.classify_url("https://example.com/buy-now") == CALL_TO_ACTION
        assert self.classifier.classify_url("https://shop.com/get-started") == CALL_TO_ACTION
        assert self.classifier.classify_url("https://brand.com/claim-reward") == CALL_TO_ACTION
        assert self.classifier.classify_url("https://company.org/order-product") == CALL_TO_ACTION
        
    def test_landing_page_indicators(self):
        """Test classification of URLs with landing page indicators."""
        assert self.classifier.classify_url("https://example.com/lp/summer-sale") == CALL_TO_ACTION
        assert self.classifier.classify_url("https://brand.com/landing/new-product") == CALL_TO_ACTION
        assert self.classifier.classify_url("https://shop.com/campaign/holiday") == CALL_TO_ACTION
        assert self.classifier.classify_url("https://company.org/lp-black-friday") == CALL_TO_ACTION
        
    def test_edge_cases(self):
        """Test classification of edge cases from validation findings."""
        # Examples from validation findings
        assert self.classifier.classify_url("http://www.assoturcupra.it") == BRAND_AWARENESS
        assert self.classifier.classify_url("http://www.schiebel.net/") == BRAND_AWARENESS
        assert self.classifier.classify_url("http://www.deltaco.com") == BRAND_AWARENESS
        assert self.classifier.classify_url("http://www.massage-technique.net/") == BRAND_AWARENESS
        assert self.classifier.classify_url("http://southwestmt.com") == BRAND_AWARENESS
        
        assert self.classifier.classify_url("http://www.over70dating.com/search.py?aff_id=google&aff_pg=3&aff_cp=d-c-con&aff_adg=over 70 dating - content - similar to over 70 - content _153479-2&aff_kw=over 70&gclid=ckcfjf3h47wcfa5dmgodxyya_q") == CALL_TO_ACTION
        assert self.classifier.classify_url("http://www.confused.com/campaign/car-insurance/car-insurance?mediacode=324&kw=car+comparison+sites+content&gclid=cnkfo5hhrrycfe2ipaodjmyadg") == CALL_TO_ACTION
        assert self.classifier.classify_url("http://www.nottawasagaresort.com/packages-promotions.html?gclid=cowrmztsnrwcfy1cmgodvjwakw") == CALL_TO_ACTION
        
        # Edge cases with root domains + tracking parameters
        assert self.classifier.classify_url("http://www.sandiego.org/?utm_campaign=MC_Spring2014&utm_medium=campaign&utm_source=ABC.com&utm_term=19177028&utm_content=POE Companion Banner") == BRAND_AWARENESS
        assert self.classifier.classify_url("http://www.lonestartexasgrill.com/?gclid=cmsv14by4locffa7mgodf0saua") == BRAND_AWARENESS
        
    def test_classify_with_confidence(self):
        """Test classification with confidence scoring."""
        # Test strong CTA signals
        result = self.classifier.classify_with_confidence("https://example.com/cart?coupon=SAVE10")
        assert result['classification'] == CALL_TO_ACTION
        assert result['confidence'] >= 0.8  # High confidence for strong CTA signal
        
        # Test strong Brand Awareness signals
        result = self.classifier.classify_with_confidence("https://example.com/about-us")
        assert result['classification'] == BRAND_AWARENESS
        assert result['confidence'] >= 0.6  # Reasonable confidence for Brand Awareness
        
        # Test borderline case
        result = self.classifier.classify_with_confidence("https://example.com/products")
        assert isinstance(result['confidence'], float)
        assert 0 <= result['confidence'] <= 1.0 