"""
Ensemble classifier for Ad Intent classification.

This module implements an ensemble approach that combines multiple classifiers:
1. Rule-based classifier (original)
2. Sentiment-based classifier (enhanced with VADER and TextBlob)
3. BERT zero-shot classifier
4. Gemini LLM-based classifier

The ensemble uses a weighted voting mechanism to determine the final classification.
"""

import logging
import os
from typing import Dict, Union, List, Any, Optional, Tuple
import re
import importlib.util

# Import constants
from ad_intention.classification.rules import BRAND_AWARENESS, CALL_TO_ACTION

# Import configuration
from ad_intention.config import (
    BRAND_AWARENESS, CALL_TO_ACTION, 
    USE_SENTIMENT, USE_BERT, USE_GEMINI, USE_OPENAI_JUDGE,
    USE_VADER, USE_TEXTBLOB,
    SENTIMENT_WEIGHT, BERT_WEIGHT, GEMINI_WEIGHT, RULES_WEIGHT,
    MAJORITY_THRESHOLD, OPENAI_API_KEY, GOOGLE_API_KEY, DEBUG
)

# Import original classifier
from ad_intention.classification.classifier import AdIntentClassifier

logger = logging.getLogger(__name__)


class EnsembleClassifier:
    """
    Ensemble classifier that combines multiple approaches for URL classification.
    """
    
    def __init__(self, 
                 use_sentiment: bool = USE_SENTIMENT,
                 use_bert: bool = USE_BERT,
                 use_gemini: bool = USE_GEMINI,
                 use_openai_judge: bool = USE_OPENAI_JUDGE,
                 use_vader: bool = USE_VADER,
                 use_textblob: bool = USE_TEXTBLOB,
                 sentiment_weight: float = SENTIMENT_WEIGHT,
                 bert_weight: float = BERT_WEIGHT,
                 gemini_weight: float = GEMINI_WEIGHT,
                 rules_weight: float = RULES_WEIGHT,
                 majority_threshold: float = MAJORITY_THRESHOLD,
                 openai_api_key: Optional[str] = None,
                 google_api_key: Optional[str] = None,
                 debug: bool = DEBUG):
        """
        Initialize the ensemble classifier.
        
        Args:
            use_sentiment: Whether to use sentiment analysis
            use_bert: Whether to use BERT zero-shot classification
            use_gemini: Whether to use Gemini LLM-based classification
            use_openai_judge: Whether to use OpenAI GPT-4o as a judge
            use_vader: Whether to use VADER for sentiment analysis
            use_textblob: Whether to use TextBlob for sentiment analysis
            sentiment_weight: Weight for sentiment classifier
            bert_weight: Weight for BERT classifier
            gemini_weight: Weight for Gemini classifier
            rules_weight: Weight for rule-based classifier
            majority_threshold: Threshold for majority voting (0.5 = simple majority)
            openai_api_key: OpenAI API key (optional)
            google_api_key: Google API key for Gemini (optional)
            debug: Whether to enable debug logging
        """
        self.use_sentiment = use_sentiment
        self.use_bert = use_bert
        self.use_gemini = use_gemini
        self.use_openai_judge = use_openai_judge
        self.use_vader = use_vader
        self.use_textblob = use_textblob
        
        self.sentiment_weight = sentiment_weight
        self.bert_weight = bert_weight
        self.gemini_weight = gemini_weight
        self.rules_weight = rules_weight
        
        self.majority_threshold = majority_threshold
        self.openai_api_key = openai_api_key
        self.google_api_key = google_api_key
        self.debug = debug
        
        # Initialize base rule-based classifier
        self.rule_classifier = AdIntentClassifier(debug=debug)
        
        # Initialize other classifiers as needed
        self.sentiment_classifier = None
        self.bert_classifier = None
        self.gemini_classifier = None
        self.openai_judge = None
        
        if self.use_sentiment:
            self._init_sentiment_classifier()
            
        if self.use_bert:
            self._init_bert_classifier()
            
        if self.use_gemini:
            self._init_gemini_classifier()
            
        if self.use_openai_judge:
            self._init_openai_judge()
            
        # Use environment variables for API keys if not provided
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        self.google_api_key = google_api_key or GOOGLE_API_KEY
        self.debug = debug
        
    def _init_sentiment_classifier(self):
        """Initialize the sentiment classifier."""
        try:
            from ad_intention.enhanced.sentiment_classifier import SentimentClassifier
            # Setup NLTK resources if needed
            try:
                from ad_intention.enhanced.nltk_setup import setup_nltk
                setup_nltk(quiet=True)
            except ImportError:
                pass
                
            self.sentiment_classifier = SentimentClassifier(
                use_vader=self.use_vader,
                use_textblob=self.use_textblob,
                debug=self.debug
            )
            if self.debug:
                logger.debug("Sentiment classifier initialized")
        except ImportError:
            logger.warning("Failed to import sentiment classifier. Disabling sentiment analysis.")
            self.use_sentiment = False
            
    def _init_bert_classifier(self):
        """Initialize the BERT classifier."""
        try:
            from ad_intention.enhanced.bert_classifier import get_bert_classifier
            self.bert_classifier = get_bert_classifier(debug=self.debug)
            if self.debug:
                logger.debug("BERT classifier initialized")
        except ImportError:
            logger.warning("Failed to import BERT classifier. Disabling BERT classification.")
            self.use_bert = False
            
    def _init_gemini_classifier(self):
        """Initialize the Gemini classifier."""
        try:
            from ad_intention.enhanced.gemini_classifier import GeminiClassifier
            self.gemini_classifier = GeminiClassifier(
                api_key=self.google_api_key,
                debug=self.debug
            )
            if self.debug:
                logger.debug("Gemini classifier initialized")
        except ImportError:
            logger.warning("Failed to import Gemini classifier. Disabling Gemini classification.")
            self.use_gemini = False
            
    def _init_openai_judge(self):
        """Initialize the OpenAI judge."""
        try:
            from ad_intention.enhanced.openai_judge import OpenAIJudge
            self.openai_judge = OpenAIJudge(
                api_key=self.openai_api_key,
                debug=self.debug
            )
            if self.debug:
                logger.debug("OpenAI judge initialized")
        except ImportError:
            logger.warning("Failed to import OpenAI judge. Disabling OpenAI judgment.")
            self.use_openai_judge = False
    
    def classify_url(self, url: str) -> str:
        """
        Classify a URL using the ensemble approach.
        
        Args:
            url: The URL to classify
            
        Returns:
            Classification: either "Brand Awareness" or "Call to Action"
        """
        result = self.classify_with_confidence(url)
        return result["classification"]
    
    def classify_with_confidence(self, url: str) -> Dict[str, Union[str, float, Dict[str, Any]]]:
        """
        Classify a URL with confidence using the ensemble approach.
        
        Args:
            url: The URL to classify
            
        Returns:
            Dictionary with classification, confidence, and detailed results
        """
        results = {}
        total_weight = 0.0
        cta_score = 0.0
        brand_score = 0.0
        
        # Rule-based classification
        rule_result = self.rule_classifier.classify_with_confidence(url)
        results["rule_based"] = rule_result
        
        if rule_result["classification"] == CALL_TO_ACTION:
            cta_score += self.rules_weight * rule_result["confidence"]
        else:
            brand_score += self.rules_weight * rule_result["confidence"]
            
        total_weight += self.rules_weight
        
        # Sentiment-based classification
        if self.use_sentiment and self.sentiment_classifier:
            try:
                sentiment_result = self.sentiment_classifier.classify_with_confidence(url)
                results["sentiment"] = sentiment_result
                
                if sentiment_result["classification"] == CALL_TO_ACTION:
                    cta_score += self.sentiment_weight * sentiment_result["confidence"]
                else:
                    brand_score += self.sentiment_weight * sentiment_result["confidence"]
                    
                total_weight += self.sentiment_weight
                
            except Exception as e:
                logger.error(f"Error in sentiment classification: {e}")
        
        # BERT-based classification
        if self.use_bert and self.bert_classifier:
            try:
                bert_result = self.bert_classifier.classify_with_confidence(url)
                results["bert"] = bert_result
                
                if bert_result["classification"] == CALL_TO_ACTION:
                    cta_score += self.bert_weight * bert_result["confidence"]
                else:
                    brand_score += self.bert_weight * bert_result["confidence"]
                    
                total_weight += self.bert_weight
                
            except Exception as e:
                logger.error(f"Error in BERT classification: {e}")
        
        # Gemini-based classification
        if self.use_gemini and self.gemini_classifier:
            try:
                gemini_result = self.gemini_classifier.classify_with_confidence(url)
                results["gemini"] = gemini_result
                
                if gemini_result["classification"] == CALL_TO_ACTION:
                    cta_score += self.gemini_weight * gemini_result["confidence"]
                else:
                    brand_score += self.gemini_weight * gemini_result["confidence"]
                    
                total_weight += self.gemini_weight
                
            except Exception as e:
                logger.error(f"Error in Gemini classification: {e}")
        
        # Determine final classification
        if total_weight > 0:
            # Calculate normalized scores
            cta_normalized = cta_score / total_weight
            brand_normalized = brand_score / total_weight
            
            # Determine winning classification based on threshold
            if cta_normalized >= self.majority_threshold:
                classification = CALL_TO_ACTION
                confidence = cta_normalized
            else:
                classification = BRAND_AWARENESS
                confidence = brand_normalized
        else:
            # Fallback to rule-based classification if no weights
            classification = rule_result["classification"]
            confidence = rule_result["confidence"]
        
        # Apply OpenAI judge (single implementation combining both previous blocks)
        judge_result = None
        if self.openai_judge:
            try:
                # Store pre-judge classification for logging
                pre_judge_classification = classification
                
                # Call the OpenAI judge
                judge_result = self.openai_judge.judge_classification(url, classification)
                results["openai_judge"] = judge_result
                
                # Retrieve and log API call statistics
                api_stats = self.openai_judge.get_api_stats()
                logger.info(f"OpenAI API call stats: {api_stats}")
                results["openai_api_stats"] = api_stats
                
                # Override classification if judge disagrees with high confidence
                if not judge_result["is_correct"] and judge_result["confidence"] > 0.7:  # Lowered threshold slightly
                    classification = judge_result["correct_classification"]
                    confidence = judge_result["confidence"]
                    logger.info(f"Judge corrected classification for {url} from {pre_judge_classification} to {classification}")
            except Exception as e:
                logger.error(f"Error in OpenAI judgment: {e}")
                logger.warning("Proceeding with ensemble classification without judge validation.")
        else:
            if self.use_openai_judge:
                logger.warning("OpenAI judge requested but not available. Classification will not be validated.")
        
        # Calculate weighted scores for each classifier
        weighted_scores = {}
        if self.rules_weight > 0:
            weighted_scores["rule_based"] = self.rules_weight / total_weight if total_weight > 0 else 1.0
        
        if self.use_sentiment and self.sentiment_weight > 0:
            weighted_scores["sentiment"] = self.sentiment_weight / total_weight if total_weight > 0 else 0.0
            
        if self.use_bert and self.bert_weight > 0:
            weighted_scores["bert"] = self.bert_weight / total_weight if total_weight > 0 else 0.0
            
        if self.use_gemini and self.gemini_weight > 0:
            weighted_scores["gemini"] = self.gemini_weight / total_weight if total_weight > 0 else 0.0
        
        # Build final result
        final_result = {
            "classification": classification,
            "confidence": round(confidence, 2),
            "details": {
                "rule_based": results.get("rule_based", {}),
                "sentiment": results.get("sentiment", {}),
                "bert": results.get("bert", {}),
                "gemini": results.get("gemini", {}),
                "openai_judge": results.get("openai_judge", {}),
                "openai_api_stats": results.get("openai_api_stats", {}),
                "weighted_scores": weighted_scores,
                "majority_threshold": self.majority_threshold,
                "final_score": round(confidence, 2),
                "cta_score": round(cta_normalized, 2) if total_weight > 0 else 0,
                "brand_score": round(brand_normalized, 2) if total_weight > 0 else 0
            }
        }
        
        if self.debug:
            logger.debug(f"Ensemble classification for URL: {url}")
            logger.debug(f"Final result: {final_result}")
            
        return final_result 