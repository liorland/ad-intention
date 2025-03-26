"""
Gemini-based classifier for Ad Intent classification using few-shot learning.

This module implements a classifier using Google's Gemini Flash LLM
to classify URLs based on few-shot examples and classification rules.
"""

import os
import json
import logging
import time
from typing import Dict, Union, List, Any, Optional
import re

logger = logging.getLogger(__name__)

# Constants
BRAND_AWARENESS = "Brand Awareness"
CALL_TO_ACTION = "Call to Action"

# Default API key environment variable name
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"

# Classification rules for the prompt
CLASSIFICATION_RULES = """
CLASSIFICATION RULES:
1. Brand Awareness URLs:
   - Root domains (like "example.com" or "example.com/")
   - Pages about the company (about, team, careers, etc.)
   - Information pages (blog, news, press, research, etc.)
   - Root domains with only tracking parameters (e.g., "example.com/?utm_source=xyz")

2. Call to Action (CTA) URLs:
   - URLs containing action paths: cart, checkout, signup, register, buy, order, etc.
   - URLs with promotional parameters: promo, discount, coupon, etc.
   - URLs with tracking parameters (except when they're on root domains)
   - Landing pages (containing /lp/, /landing/, /campaign/, etc.)
   - URLs containing action keywords: signup, join, get, buy, etc.
   - If the domain or subdomain contains CTA words (like "signup.example.com")
   - URLs containing action-oriented phrases (like "become-a-host", "get-started")
"""

# Few-shot examples from the manually validated dataset
FEW_SHOT_EXAMPLES = [
    {"url": "http://www.assoturcupra.it", "classification": "Brand Awareness"},
    {"url": "http://www.schiebel.net/", "classification": "Brand Awareness"},
    {"url": "http://www.deltaco.com", "classification": "Brand Awareness"},
    {"url": "http://www.nike.com/about", "classification": "Brand Awareness"},
    {"url": "http://mercedes-benz.com/innovation", "classification": "Brand Awareness"},
    {"url": "http://nike.com/cart", "classification": "Call to Action"},
    {"url": "http://apple.com/shop/buy-mac", "classification": "Call to Action"},
    {"url": "http://cocacola.com/signup?promo=summer", "classification": "Call to Action"},
    {"url": "http://bike-kaitori.com/lp/satei_bluelong", "classification": "Call to Action"},
    {"url": "http://www.sandiego.org/?utm_campaign=MC_Spring2014&utm_medium=campaign&utm_source=ABC.com", "classification": "Call to Action"},
    {"url": "http://www.lonestartexasgrill.com/?gclid=cmsv14by4locffa7mgodf0saua", "classification": "Call to Action"},
    {"url": "https://signup.ebay.com/", "classification": "Call to Action"},
    {"url": "https://www.airbnb.com/become-a-host", "classification": "Call to Action"}
]


class GeminiClassifier:
    """
    Classifier that uses Google's Gemini Flash LLM for
    classifying URLs based on few-shot examples.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 examples: Optional[List[Dict[str, str]]] = None,
                 debug: bool = False):
        """
        Initialize the Gemini classifier.
        
        Args:
            api_key: Google API key for Gemini (optional, can be set via environment variable)
            examples: Few-shot examples (optional, default examples will be used if not provided)
            debug: Whether to enable debug logging
        """
        self.api_key = api_key or os.environ.get(GOOGLE_API_KEY_ENV)
        if not self.api_key:
            logger.warning("No Google API key provided for Gemini. Set via constructor or GOOGLE_API_KEY env variable.")
            
        self.examples = examples or FEW_SHOT_EXAMPLES
        self.debug = debug
        self.client = None
        
        # Statistics for tracking API calls
        self.api_calls_attempted = 0
        self.api_calls_successful = 0
        self.api_calls_failed = 0
        self.json_parse_attempts = 0
        self.json_parse_successful = 0
        self.json_parse_failed = 0
        
        # Token usage tracking
        self.token_counts = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
        
    def _initialize_client(self):
        """Initialize the Gemini client if not already initialized."""
        if self.client is None and self.api_key:
            try:
                import google.generativeai as genai
                
                # Initialize the client
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel('gemini-2.0-flash-lite')
                
                if self.debug:
                    logger.debug("Gemini client initialized successfully")
                logger.info("✅ Gemini client initialized successfully")
                    
            except ImportError:
                logger.error("Failed to import Google Generative AI. Install with: pip install google-generativeai")
                raise
            except Exception as e:
                logger.error(f"Error initializing Gemini client: {e}")
                raise
                
    def _format_few_shot_examples(self) -> str:
        """Format few-shot examples for the prompt."""
        formatted_examples = ""
        for i, example in enumerate(self.examples, 1):
            formatted_examples += f"Example {i}:\n"
            formatted_examples += f"URL: {example['url']}\n"
            formatted_examples += f"Classification: {example['classification']}\n\n"
        return formatted_examples
    
    def _create_prompt(self, url: str) -> str:
        """
        Create a prompt for the Gemini model.
        
        Args:
            url: The URL to classify
            
        Returns:
            Formatted prompt string
        """
        few_shot_examples = self._format_few_shot_examples()
        
        prompt = f"""
You are a URL classification expert specializing in ad intent analysis. Your task is to classify URLs into two categories:
1. "Brand Awareness" - URLs focused on providing information and brand recognition
2. "Call to Action" - URLs designed to drive immediate user actions

{CLASSIFICATION_RULES}

Here are some examples of correctly classified URLs:

{few_shot_examples}

Now classify the following URL:
URL: {url}

Provide your analysis and classification in the following JSON format:
{{
  "analysis": "Brief explanation of why this URL fits the classification",
  "classification": "Brand Awareness OR Call to Action",
  "confidence": 0.XX (a number between 0.0 and 1.0)
}}

Return only the JSON object, nothing else.
"""
        return prompt
        
    def classify_with_confidence(self, url: str) -> Dict[str, Union[str, float]]:
        """
        Classify a URL with confidence using the Gemini model.
        
        Args:
            url: The URL to classify
            
        Returns:
            Dictionary with classification and confidence
        """
        # Return fallback result if no API key is available
        if not self.api_key:
            logger.warning("⚠️ No API key available for Gemini - using fallback classification")
            return self._fallback_classification(url)
            
        try:
            # Initialize the client if not already done
            self._initialize_client()
            
            # Create the prompt
            prompt = self._create_prompt(url)
            
            # Log the prompt in debug mode
            if self.debug:
                logger.debug(f"Gemini prompt: {prompt}")
                
            # Call the Gemini API
            self.api_calls_attempted += 1
            logger.info(f"📡 Calling Gemini API for URL: {url}")
            
            # Estimate input tokens for the prompt
            input_tokens = self._estimate_tokens(prompt)
            self.token_counts["input_tokens"] += input_tokens
            
            start_time = time.time()
            response = self.client.generate_content(prompt)
            elapsed_time = time.time() - start_time
            
            # Mark API call as successful
            self.api_calls_successful += 1
            logger.info(f"✅ Gemini API call successful in {elapsed_time:.2f}s")
            
            # Extract the text response
            response_text = response.text
            
            # Estimate output tokens for the response
            output_tokens = self._estimate_tokens(response_text)
            self.token_counts["output_tokens"] += output_tokens
            self.token_counts["total_tokens"] += input_tokens + output_tokens
            
            # Log token usage in debug mode
            if self.debug:
                logger.debug(f"Estimated token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {input_tokens + output_tokens}")
                
            # Parse the JSON response
            if self.debug:
                logger.debug(f"Gemini response: {response_text}")
                
            try:
                self.json_parse_attempts += 1
                
                # Extract JSON from potential markdown code blocks (```json ... ```)
                json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
                json_match = re.search(json_pattern, response_text, re.DOTALL)
                
                if json_match:
                    # Found JSON inside markdown code block
                    clean_json_text = json_match.group(1).strip()
                    if self.debug:
                        logger.debug(f"Extracted JSON from markdown: {clean_json_text}")
                    logger.info("🔍 Found JSON inside markdown code block - extracting")
                    result = json.loads(clean_json_text)
                else:
                    # Try parsing the response directly
                    logger.info("🔍 Attempting to parse response as direct JSON")
                    result = json.loads(response_text)
                
                # JSON parsing successful
                self.json_parse_successful += 1
                logger.info("✅ JSON parsing successful")
                
                # Validate the result structure
                if "classification" not in result or "confidence" not in result:
                    logger.warning(f"⚠️ Invalid response format from Gemini: missing required fields")
                    return self._fallback_classification(url)
                    
                # Ensure the classification is one of our expected values
                classification = result["classification"]
                if classification not in [BRAND_AWARENESS, CALL_TO_ACTION]:
                    logger.warning(f"⚠️ Unexpected classification from Gemini: {classification}")
                    classification = BRAND_AWARENESS  # Default fallback
                    
                # Ensure confidence is a float between 0 and 1
                confidence = float(result["confidence"])
                confidence = max(0.3, min(1.0, confidence))  # Clamp between 0.3 and 1.0
                
                logger.info(f"✅ Gemini classification result: {classification} (confidence: {confidence:.2f})")
                
                return {
                    "classification": classification,
                    "confidence": round(confidence, 2)
                }
                
            except json.JSONDecodeError as e:
                self.json_parse_failed += 1
                logger.warning(f"❌ Failed to parse JSON from Gemini response: {e}")
                logger.warning(f"Raw response: {response_text}")
                return self._fallback_classification(url)
                
        except Exception as e:
            self.api_calls_failed += 1
            logger.error(f"❌ Error in Gemini classification: {e}")
            return self._fallback_classification(url)
            
    def _fallback_classification(self, url: str) -> Dict[str, Union[str, float]]:
        """
        Fallback classification when Gemini API is unavailable or fails.
        
        Args:
            url: The URL to classify
            
        Returns:
            Dictionary with classification and confidence
        """
        url_lower = url.lower()
        
        # Check for clear CTA indicators
        cta_indicators = [
            "/cart", "/checkout", "/buy", "/order", "/shop/", "/signup",
            "/register", "/join", "/subscribe", "?promo=", "?coupon=",
            "?discount=", "/lp/", "/landing/", "/campaign/"
        ]
        
        if any(indicator in url_lower for indicator in cta_indicators):
            return {"classification": CALL_TO_ACTION, "confidence": 0.8}
            
        # Check for URL with tracking parameters but not root domain
        has_tracking = any(param in url_lower for param in ["utm_", "gclid", "fbclid", "cid"])
        is_root = re.match(r'https?://[^/]+/?(\?|$)', url_lower) is not None
        
        if has_tracking and not is_root:
            return {"classification": CALL_TO_ACTION, "confidence": 0.7}
            
        # Check for action keywords in domain or path
        action_keywords = ["signup", "sign-up", "login", "join", "buy", "shop", "cart", "checkout"]
        if any(keyword in url_lower for keyword in action_keywords):
            return {"classification": CALL_TO_ACTION, "confidence": 0.75}
            
        # Default to Brand Awareness
        return {"classification": BRAND_AWARENESS, "confidence": 0.6}
        
    def classify_url(self, url: str) -> str:
        """
        Classify a URL using the Gemini model.
        
        Args:
            url: The URL to classify
            
        Returns:
            Classification: either "Brand Awareness" or "Call to Action"
        """
        result = self.classify_with_confidence(url)
        return result["classification"] 
        
    def get_api_stats(self) -> Dict[str, Union[int, float, str]]:
        """
        Get statistics about Gemini API calls and JSON parsing.
        
        Returns:
            Dictionary with API call statistics
        """
        stats = {
            "api_calls_attempted": self.api_calls_attempted,
            "api_calls_successful": self.api_calls_successful,
            "api_calls_failed": self.api_calls_failed,
            "json_parse_attempts": self.json_parse_attempts,
            "json_parse_successful": self.json_parse_successful,
            "json_parse_failed": self.json_parse_failed
        }
        
        # Calculate success rates if there were any calls
        if self.api_calls_attempted > 0:
            stats["api_success_rate"] = f"{(self.api_calls_successful / self.api_calls_attempted) * 100:.1f}%"
        else:
            stats["api_success_rate"] = "N/A"
            
        if self.json_parse_attempts > 0:
            stats["json_success_rate"] = f"{(self.json_parse_successful / self.json_parse_attempts) * 100:.1f}%"
        else:
            stats["json_success_rate"] = "N/A"
            
        # Add average time if we implement time tracking
        # stats["avg_api_call_time"] = ...
        
        return stats 

    def get_token_stats(self) -> Dict[str, Any]:
        """
        Get token usage statistics.
        
        Returns:
            Dictionary with token usage stats
        """
        stats = {
            "input_tokens": self.token_counts["input_tokens"],
            "output_tokens": self.token_counts["output_tokens"],
            "total_tokens": self.token_counts["total_tokens"]
        }
        
        # Calculate approximate costs based on Gemini 2.0 Flash Lite pricing
        # Prices as of March 2025: $0.075 per million input tokens, $0.30 per million output tokens
        input_cost = (self.token_counts["input_tokens"] / 1_000_000) * 0.075
        output_cost = (self.token_counts["output_tokens"] / 1_000_000) * 0.30
        
        stats["input_cost"] = input_cost
        stats["output_cost"] = output_cost
        stats["total_cost"] = input_cost + output_cost
        
        return stats

    # Helper function to estimate tokens from text
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        This is a rough approximation - Gemini doesn't provide token counts directly.
        For English text, tokens are approximately 4 characters.
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        return len(text) // 4  # Rough approximation of 1 token per 4 characters 