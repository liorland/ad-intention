"""
OpenAI GPT-4o based judge for evaluating Ad Intent classifications.

This module implements a judge using OpenAI's GPT-4o model to evaluate
whether a URL classification is correct based on the classification rules
and examples.
"""

import os
import json
import logging
from typing import Dict, Union, List, Any, Optional, Tuple
import re
import time

logger = logging.getLogger(__name__)

# Constants
BRAND_AWARENESS = "Brand Awareness"
CALL_TO_ACTION = "Call to Action"

# Default API key environment variable name
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

# Judge system prompt
JUDGE_SYSTEM_PROMPT = """
You are an expert judge evaluating URL classifications for ad intent analysis.
Follow these exact steps when evaluating a URL:

STEP 1: Determine if the URL is a root domain (like example.com or example.com/) without additional path elements.
STEP 2: Check if the URL contains tracking parameters (utm_*, gclid, fbclid, etc.).
STEP 3: Apply these rules in this exact order:
  - If it's a root domain WITHOUT tracking parameters → Brand Awareness (100%)
  - If it's a root domain WITH tracking parameters → Call to Action (100%)
  - If it contains action paths (/cart, /checkout, /buy, etc.) → Call to Action
  - If the domain/subdomain contains action words (signup.example.com) → Call to Action
  - If it contains action-oriented phrases (/become-a-host) → Call to Action
  - If it's about company info (/about, /team, etc.) → Brand Awareness
  - If it's an information page (/blog, /news, etc.) → Brand Awareness
  
STEP 4: Set is_correct to TRUE if the assigned classification matches your determination from step 3.
STEP 5: Set is_correct to FALSE if the assigned classification DOES NOT match your determination.
STEP 6: When is_correct is FALSE, correct_classification MUST be different from the assigned classification.
STEP 7: When is_correct is TRUE, correct_classification MUST match the assigned classification.
STEP 8: Make sure your explanation clearly matches your classification decision.

IMPORTANT: Never contradict yourself. If you determine a URL should be "Brand Awareness" but the assigned 
classification is "Call to Action", then is_correct MUST be false and correct_classification MUST be "Brand Awareness".

For each URL you're given, evaluate the assigned classification and provide your judgment in this JSON format:
{
  "is_correct": true/false,
  "correct_classification": "Brand Awareness" or "Call to Action",
  "explanation": "Brief explanation of your judgment",
  "confidence": 0.XX (a number between 0.0 and 1.0)
}
"""

# Examples of correctly classified URLs for few-shot learning
CLASSIFICATION_EXAMPLES = [
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


class OpenAIJudge:
    """
    Judge that uses OpenAI's GPT-4o model to evaluate URL classifications.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o",
                 examples: Optional[List[Dict[str, str]]] = None,
                 debug: bool = False):
        """
        Initialize the OpenAI judge.
        
        Args:
            api_key: OpenAI API key (optional, can be set via environment variable)
            model: OpenAI model to use (default: gpt-4o)
            examples: Classification examples (optional, default examples will be used if not provided)
            debug: Whether to enable debug logging
        """
        self.api_key = api_key or os.environ.get(OPENAI_API_KEY_ENV)
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Set via constructor or OPENAI_API_KEY env variable.")
            
        self.model = model
        self.examples = examples or CLASSIFICATION_EXAMPLES
        self.debug = debug
        self.client = None
        self.api_calls_attempted = 0
        self.api_calls_successful = 0
        self.api_calls_failed = 0
        self.total_api_call_time = 0
        
        # Add token tracking
        self.token_counts = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
    
    def _initialize_client(self):
        """Initialize the OpenAI client if not already initialized."""
        if self.client is None and self.api_key:
            try:
                from openai import OpenAI
                
                # Initialize the client
                self.client = OpenAI(api_key=self.api_key)
                
                logger.info("OpenAI client initialized successfully")
                    
            except ImportError:
                logger.error("Failed to import OpenAI. Install with: pip install openai")
                raise
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {e}")
                raise
    
    def _format_few_shot_examples(self) -> str:
        """Format few-shot examples for the prompt."""
        formatted_examples = "Here are some examples of correctly classified URLs:\n\n"
        for i, example in enumerate(self.examples, 1):
            formatted_examples += f"Example {i}:\n"
            formatted_examples += f"URL: {example['url']}\n"
            formatted_examples += f"Classification: {example['classification']}\n\n"
        return formatted_examples
    
    def _create_user_prompt(self, url: str, classification: str) -> str:
        """
        Create a user prompt for the OpenAI model.
        
        Args:
            url: The URL to evaluate
            classification: The classification to evaluate
            
        Returns:
            Formatted prompt string
        """
        few_shot_examples = self._format_few_shot_examples()
        
        prompt = f"""
{few_shot_examples}

Now evaluate the following URL classification:
URL: {url}
Assigned Classification: {classification}

Is this classification correct according to the rules and examples above? 
Provide your judgment in the required JSON format.
"""
        return prompt
        
    def _validate_explanation_consistency(self, result: Dict[str, Any], original_classification: str) -> Dict[str, Any]:
        """
        Validate that the explanation is consistent with the classification decision.
        
        Args:
            result: The classification result
            original_classification: The original classification being evaluated
            
        Returns:
            Potentially corrected result
        """
        explanation = result["explanation"].lower()
        is_correct = result["is_correct"]
        correct_classification = result["correct_classification"]
        
        # Check for contradictory language in explanation
        contradiction_detected = False
        
        # Case 1: Explanation says classification is correct but is_correct is False
        if ("is correct" in explanation or "is appropriate" in explanation or "is accurate" in explanation) and not is_correct:
            logger.warning(f"⚠️ Explanation says classification is correct but is_correct=False")
            contradiction_detected = True
            
        # Case 2: Explanation says classification is incorrect but is_correct is True
        if ("is incorrect" in explanation or "is not appropriate" in explanation or "is not accurate" in explanation) and is_correct:
            logger.warning(f"⚠️ Explanation says classification is incorrect but is_correct=True")
            contradiction_detected = True
            
        # Case 3: Explanation suggests a different classification but no change is made
        suggests_different = any(phrase in explanation for phrase in [
            "should be classified as", "should be categorized as", "should be", "would be better classified"
        ])
        
        if suggests_different and is_correct:
            logger.warning(f"⚠️ Explanation suggests different classification but is_correct=True")
            contradiction_detected = True
            
        # Case 4: Explanation mentions both classifications in a confusing way
        if "brand awareness" in explanation and "call to action" in explanation:
            # Check if the explanation clearly indicates which is correct
            brand_positive = any(f"brand awareness {phrase}" in explanation for phrase in [
                "is correct", "is appropriate", "is the right", "should be"
            ])
            cta_positive = any(f"call to action {phrase}" in explanation for phrase in [
                "is correct", "is appropriate", "is the right", "should be"
            ])
            
            # If the explanation seems to indicate both are valid or is ambiguous
            if (brand_positive and cta_positive) or (not brand_positive and not cta_positive):
                logger.warning(f"⚠️ Explanation contains ambiguous references to both classifications")
                contradiction_detected = True
        
        # Fix contradictions by ensuring consistency
        if contradiction_detected:
            # Force consistency based on the classifications
            if original_classification == correct_classification:
                result["is_correct"] = True
                result["explanation"] += " [Note: Fixed contradiction - classification is correct]"
            else:
                result["is_correct"] = False
                result["explanation"] += f" [Note: Fixed contradiction - classification should be {correct_classification}]"
            
            logger.warning(f"⚠️ Fixed contradiction in explanation: original={original_classification}, corrected={correct_classification}, is_correct={result['is_correct']}")
        
        return result

    def judge_classification(self, url: str, classification: str) -> Dict[str, Any]:
        """
        Judge whether a URL classification is correct.
        
        Args:
            url: The URL to evaluate
            classification: The classification to evaluate
            
        Returns:
            Dictionary with judgment results
        """
        # Return fallback result if no API key is available
        if not self.api_key:
            logger.info(f"Using fallback judgment for URL: {url} (no API key available)")
            return self._fallback_judgment(url, classification)
            
        try:
            # Initialize the client if not already done
            self._initialize_client()
            
            # Create the user prompt
            user_prompt = self._create_user_prompt(url, classification)
            
            # Log the prompt in debug mode
            if self.debug:
                logger.debug(f"OpenAI system prompt: {JUDGE_SYSTEM_PROMPT}")
                logger.debug(f"OpenAI user prompt: {user_prompt}")
            
            # Log start of API call
            logger.info(f"Starting OpenAI API call #{self.api_calls_attempted + 1} for URL: {url}")
            self.api_calls_attempted += 1

            start_time = time.time()

            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  # Low temperature for consistent responses
                max_tokens=500,
                response_format={"type": "json_object"}
            )

            # Calculate API call duration
            duration = time.time() - start_time
            self.total_api_call_time += duration

            # Update token counts from the API response
            self.token_counts["input_tokens"] += response.usage.prompt_tokens
            self.token_counts["output_tokens"] += response.usage.completion_tokens
            self.token_counts["total_tokens"] += response.usage.total_tokens

            # Log token usage for debugging
            if self.debug:
                logger.debug(f"Token usage - Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")

            # Extract the text response
            response_text = response.choices[0].message.content
            
            # Parse JSON from the response
            try:
                result = json.loads(response_text)
                
                # Validate result structure
                required_keys = ["is_correct", "correct_classification", "explanation", "confidence"]
                for key in required_keys:
                    if key not in result:
                        logger.warning(f"⚠️ Missing '{key}' in OpenAI response")
                        result[key] = "Unknown" if key != "is_correct" else True
                        if key == "confidence":
                            result[key] = 0.7  # Default confidence
                
                # Validate that correct_classification is one of the expected values
                valid_classifications = [BRAND_AWARENESS, CALL_TO_ACTION]
                if result["correct_classification"] not in valid_classifications:
                    logger.warning(f"⚠️ Invalid classification '{result['correct_classification']}', defaulting to original")
                    result["correct_classification"] = classification
                
                # Validate consistency of the response
                result = self._validate_explanation_consistency(result, classification)
                
                # Check for root domain special case
                result = self._validate_root_domain_classification(url, classification, result)
                
                # Fix inconsistency: if is_correct=False but classifications match, update is_correct to True
                if result["is_correct"] == False and result["correct_classification"] == classification:
                    logger.warning(f"⚠️ Inconsistent OpenAI response for URL '{url}': marked as incorrect but suggested same classification '{classification}'")
                    result["is_correct"] = True
                    result["explanation"] += " [Note: Response was automatically corrected for consistency]"
                    
                # Ensure opposite inconsistency is also fixed
                if result["is_correct"] == True and result["correct_classification"] != classification:
                    logger.warning(f"⚠️ Inconsistent OpenAI response for URL '{url}': marked as correct but suggested different classification '{result['correct_classification']}' vs '{classification}'")
                    # Either change is_correct to false or change correct_classification to match
                    result["is_correct"] = False
                    result["explanation"] += " [Note: Response was automatically corrected for consistency]"
                
                # Log the final judgment result
                logger.info(f"OpenAI judgment for URL '{url}': is_correct={result['is_correct']}, classification={result['correct_classification']}, confidence={result['confidence']}")
                
                return result
                
            except json.JSONDecodeError:
                logger.warning(f"❌ Failed to parse JSON from OpenAI response: {response_text}")
                self.api_calls_failed += 1
                return self._fallback_judgment(url, classification)
            
        except Exception as e:
            logger.error(f"❌ Error in OpenAI judgment call: {e}")
            self.api_calls_failed += 1
            logger.info(f"Using fallback judgment for URL: {url} (API call failed)")
            return self._fallback_judgment(url, classification)
            
    def _validate_root_domain_classification(self, url: str, original_classification: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and potentially correct classifications for root domains.
        
        Args:
            url: The URL being classified
            original_classification: The original classification
            result: The result dictionary from OpenAI
            
        Returns:
            Potentially corrected result dictionary
        """
        url_lower = url.lower()
        
        # Check if this is a root domain
        is_root_domain = bool(re.match(r'https?://[^/]+/?$', url_lower))
        
        # Check for tracking parameters
        has_tracking = any(param in url_lower for param in ["utm_", "gclid", "fbclid", "cid", "?"])
        
        if is_root_domain:
            if has_tracking:
                # Root domain WITH tracking should be CALL_TO_ACTION
                correct_class = CALL_TO_ACTION
                confidence = 0.9
            else:
                # Root domain WITHOUT tracking should be BRAND_AWARENESS
                correct_class = BRAND_AWARENESS
                confidence = 0.9
                
            # If the OpenAI judgment contradicts our root domain rules, override it
            if result["correct_classification"] != correct_class:
                logger.warning(f"⚠️ Overriding OpenAI classification for root domain '{url}': {result['correct_classification']} → {correct_class}")
                result["correct_classification"] = correct_class
                result["is_correct"] = (original_classification == correct_class)
                result["confidence"] = confidence
                result["explanation"] += f" [Note: Classification was corrected based on root domain rules. This is a {'root domain with tracking parameters' if has_tracking else 'clean root domain without tracking'}.]"
                
        return result
    
    def _fallback_judgment(self, url: str, classification: str) -> Dict[str, Any]:
        """
        Fallback judgment when OpenAI API is unavailable or fails.
        
        Args:
            url: The URL to evaluate
            classification: The classification to evaluate
            
        Returns:
            Dictionary with judgment results
        """
        logger.info(f"Using fallback judgment mechanism for URL: {url}")
        url_lower = url.lower()
        
        # Define scoring for more nuanced classification
        brand_score = 0
        cta_score = 0
        
        # 1. Root domain check (strongest signal)
        is_root_domain = bool(re.match(r'https?://[^/]+/?$', url_lower))
        
        # 2. Additional path components
        has_path = "/" in url_lower[8:] and not url_lower.endswith("/")
        
        # 3. Tracking parameters check
        has_tracking = any(param in url_lower for param in ["utm_", "gclid", "fbclid", "cid", "?"])
        
        # Apply hard rules for root domains (highest confidence)
        if is_root_domain:
            if has_tracking:
                logger.info(f"Fallback: {url} is a root domain WITH tracking parameters -> CALL TO ACTION")
                correct_classification = CALL_TO_ACTION
                confidence = 0.95
                explanation = "This is a root domain with tracking parameters, which should be classified as Call to Action based on the classification rules."
            else:
                logger.info(f"Fallback: {url} is a clean root domain WITHOUT tracking -> BRAND AWARENESS")
                correct_classification = BRAND_AWARENESS
                confidence = 0.95
                explanation = "This is a clean root domain without tracking parameters, which should be classified as Brand Awareness based on the classification rules."
        else:
            # For non-root domains, use scoring approach
            
            # CTA indicators (each indicator adds to CTA score)
            cta_indicators = [
                "/cart", "/checkout", "/buy", "/order", "/shop/", "/signup", 
                "/register", "/join", "/subscribe", "?promo=", "?coupon=",
                "?discount=", "/lp/", "/landing/", "/campaign/"
            ]
            
            for indicator in cta_indicators:
                if indicator in url_lower:
                    cta_score += 2
                    logger.debug(f"CTA indicator found: {indicator} in {url}")
            
            # Action keywords
            action_keywords = ["signup", "sign-up", "login", "join", "buy", "shop", "cart", "checkout", "become-a-"]
            for keyword in action_keywords:
                if keyword in url_lower:
                    cta_score += 2
                    logger.debug(f"Action keyword found: {keyword} in {url}")
            
            # Brand awareness indicators (each indicator adds to brand score)
            brand_indicators = [
                "/about", "/team", "/company", "/careers", "/story", "/mission",
                "/blog", "/news", "/press", "/research", "/privacy", "/terms",
                "/contact", "/support", "/help", "/faq"
            ]
            
            for indicator in brand_indicators:
                if indicator in url_lower:
                    brand_score += 2
                    logger.debug(f"Brand indicator found: {indicator} in {url}")
            
            # Tracking parameters strongly suggest CTA for non-root domains
            if has_tracking:
                cta_score += 3
                logger.debug(f"Tracking parameters found in {url}")
            
            # Default for root domains is brand awareness
            if not has_path:
                brand_score += 1
                logger.debug(f"No path components in {url}")
            
            # Determine final classification based on scores
            if cta_score > brand_score:
                correct_classification = CALL_TO_ACTION
                confidence = min(0.95, max(0.6, (cta_score - brand_score) / (cta_score + brand_score + 1) + 0.5))
                explanation = f"This URL contains indicators of Call to Action intent ({cta_score} CTA signals vs {brand_score} Brand signals)."
            else:
                correct_classification = BRAND_AWARENESS
                confidence = min(0.95, max(0.6, (brand_score - cta_score) / (brand_score + cta_score + 1) + 0.5))
                explanation = f"This URL contains indicators of Brand Awareness intent ({brand_score} Brand signals vs {cta_score} CTA signals)."
        
        # Check if the provided classification matches the determined one
        is_correct = classification == correct_classification
        
        result = {
            "is_correct": is_correct,
            "correct_classification": correct_classification,
            "explanation": explanation,
            "confidence": round(confidence, 2)
        }
        
        # Extra verification to ensure consistency
        if result["is_correct"] == False and result["correct_classification"] == classification:
            logger.warning(f"⚠️ Inconsistent fallback judgment for URL '{url}': fixing is_correct value")
            result["is_correct"] = True
            result["explanation"] += " [Note: Fixed internal consistency]"
        
        logger.debug(f"Fallback judgment for URL '{url}': is_correct={result['is_correct']}, classification={result['correct_classification']}, confidence={result['confidence']}")
        
        return result
    
    def get_api_stats(self) -> Dict[str, Any]:
        """
        Get statistics about API calls.
        
        Returns:
            Dictionary with API call statistics
        """
        stats = {
            "api_calls_attempted": self.api_calls_attempted,
            "api_calls_successful": self.api_calls_successful,
            "api_calls_failed": self.api_calls_failed,
            "success_rate": round(self.api_calls_successful / max(1, self.api_calls_attempted) * 100, 2)
        }
        
        if self.api_calls_successful > 0:
            stats["avg_call_time"] = round(self.total_api_call_time / self.api_calls_successful, 2)
            
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
        
        # Calculate approximate costs based on GPT-4o pricing
        # Prices as of March 2025: $2.50 per million input tokens, $10.00 per million output tokens
        input_cost = (self.token_counts["input_tokens"] / 1_000_000) * 2.50
        output_cost = (self.token_counts["output_tokens"] / 1_000_000) * 10.00
        
        stats["input_cost"] = input_cost
        stats["output_cost"] = output_cost
        stats["total_cost"] = input_cost + output_cost
        
        return stats 