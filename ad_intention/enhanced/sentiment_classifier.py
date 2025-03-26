"""
Sentiment-based classifier for Ad Intent classification.

This module implements a sentiment analysis approach for
classifying URLs based on action-oriented vs. information-oriented words,
enhanced with VADER and TextBlob sentiment analysis.
"""

import re
from typing import Dict, Union, List, Tuple
import logging
import string

# Import NLTK setup utility
from ad_intention.enhanced.nltk_setup import is_vader_available

# Try to import VADER directly
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True  # Set to True if import succeeds
except ImportError:
    VADER_AVAILABLE = False

# Use the result from nltk_setup as a backup
if not VADER_AVAILABLE:
    VADER_AVAILABLE = is_vader_available()
    
# Try to import TextBlob
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

logger = logging.getLogger(__name__)

# Action-oriented words that indicate Call to Action
ACTION_WORDS = {
    'signup', 'sign-up', 'login', 'register', 'join', 'subscribe', 'buy',
    'purchase', 'order', 'shop', 'cart', 'checkout', 'get', 'download',
    'try', 'demo', 'book', 'reserve', 'apply', 'submit', 'start', 'begin',
    'add', 'activate', 'claim', 'redeem', 'save', 'discount', 'deal', 'sale',
    'special', 'offer', 'promo', 'promotion', 'coupon', 'code', 'limited',
    'exclusive', 'free', 'trial', 'install', 'enroll', 'payment', 'pay',
    'host', 'create', 'manage', 'sell', 'become', 'hire'
}

# Information-oriented words that indicate Brand Awareness
INFO_WORDS = {
    'about', 'company', 'history', 'mission', 'vision', 'values', 'team',
    'careers', 'jobs', 'press', 'news', 'blog', 'article', 'story', 'learn',
    'discover', 'explore', 'find', 'read', 'watch', 'view', 'meet', 'know',
    'understand', 'info', 'information', 'support', 'help', 'faq', 'contact',
    'locations', 'stores', 'privacy', 'terms', 'policy', 'legal', 'copyright',
    'research', 'development', 'innovation', 'technology', 'sustainability',
    'responsibility', 'community', 'events', 'partners', 'investors', 'annual',
    'report', 'foundation', 'global', 'international', 'local'
}

# Log availability of sentiment analysis tools
if VADER_AVAILABLE:
    logger.info("VADER sentiment analyzer is available")
else:
    logger.warning("VADER sentiment analyzer is not available. Install with: pip install nltk")
    
if TEXTBLOB_AVAILABLE:
    logger.info("TextBlob sentiment analyzer is available")
else:
    logger.warning("TextBlob sentiment analyzer is not available. Install with: pip install textblob")

class SentimentClassifier:
    """
    Classifier that uses sentiment analysis to determine if a URL
    is more action-oriented (Call to Action) or information-oriented
    (Brand Awareness).
    
    Enhanced with VADER and TextBlob for more sophisticated sentiment analysis.
    """
    
    def __init__(self, 
                 use_vader: bool = True, 
                 use_textblob: bool = True,
                 vader_weight: float = 1.0,
                 textblob_weight: float = 0.8,
                 lexicon_weight: float = 2.0,
                 debug: bool = False):
        """
        Initialize the sentiment classifier.
        
        Args:
            use_vader: Whether to use VADER sentiment analysis
            use_textblob: Whether to use TextBlob sentiment analysis
            vader_weight: Weight for VADER sentiment scores
            textblob_weight: Weight for TextBlob sentiment scores
            lexicon_weight: Weight for lexicon-based scores
            debug: Whether to enable debug logging
        """
        self.debug = debug
        self.use_vader = use_vader and VADER_AVAILABLE
        self.use_textblob = use_textblob and TEXTBLOB_AVAILABLE
        self.vader_weight = vader_weight
        self.textblob_weight = textblob_weight
        self.lexicon_weight = lexicon_weight
        
        # Store the last sentiment scores for easier access
        self.last_vader_scores = None
        self.last_textblob_scores = None
        
        # Initialize VADER if available
        if self.use_vader:
            try:
                self.vader = SentimentIntensityAnalyzer()
                if self.debug:
                    logger.debug("VADER sentiment analyzer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize VADER: {e}")
                self.use_vader = False
    
    def tokenize_url(self, url: str) -> List[str]:
        """
        Break a URL into meaningful tokens for sentiment analysis.
        
        Args:
            url: The URL to tokenize
            
        Returns:
            List of tokens extracted from the URL
        """
        # Convert to lowercase
        url = url.lower()
        
        # Extract domain and path
        match = re.search(r'https?://([^/]+)(.*)', url)
        if not match:
            return []
            
        domain, path = match.groups()
        
        # Process domain (split by dots and dashes)
        domain_parts = re.split(r'[.-]', domain)
        
        # Process path (split by slashes, dashes, underscores)
        path_parts = []
        if path:
            # Remove query string
            path = path.split('?')[0]
            # Split by slashes and other separators
            path_parts = re.split(r'[/\-_.]', path)
            # Remove empty parts
            path_parts = [p for p in path_parts if p]
        
        # Extract query parameters
        query_parts = []
        if '?' in url:
            query_string = url.split('?')[1]
            # Split by & to get each parameter
            params = query_string.split('&')
            for param in params:
                if '=' in param:
                    # Get parameter name and value
                    name, value = param.split('=', 1)
                    query_parts.append(name)
                    # Split value by non-alphanumeric characters
                    value_parts = re.split(r'[^a-zA-Z0-9]', value)
                    query_parts.extend([p for p in value_parts if p])
        
        # Combine all tokens
        all_tokens = domain_parts + path_parts + query_parts
        
        # Filter out common words, numbers, and short tokens
        filtered_tokens = []
        for token in all_tokens:
            # Skip numbers, common words, and short tokens
            if (token and len(token) > 2 and 
                not token.isdigit() and 
                token not in {'www', 'com', 'org', 'net', 'html', 'php', 'asp', 'jsp'}):
                filtered_tokens.append(token)
        
        if self.debug:
            logger.debug(f"Tokenized URL: {url}")
            logger.debug(f"Tokens: {filtered_tokens}")
            
        return filtered_tokens
    
    def create_readable_text(self, url: str, tokens: List[str]) -> str:
        """
        Create a readable text from URL for sentiment analysis.
        
        Args:
            url: The original URL
            tokens: Extracted tokens from the URL
            
        Returns:
            A readable text representation
        """
        # Convert tokens to a space-separated string
        token_text = " ".join(tokens)
        
        # Extract path and query components
        path_match = re.search(r'https?://[^/]+(/[^?]*)', url)
        path = path_match.group(1) if path_match else ""
        
        query = ""
        if '?' in url:
            query = url.split('?')[1]
        
        # Create a more natural language representation
        text = f"Visit website"
        
        if path and path != "/":
            # Convert path to readable format
            readable_path = path.replace('/', ' ').replace('-', ' ').replace('_', ' ').strip()
            text += f" to {readable_path}"
        
        if query:
            text += " with parameters"
        
        if tokens:
            text += f" {token_text}"
            
        return text
        
    def analyze_vader_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze text using VADER sentiment analyzer.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not self.use_vader:
            self.last_vader_scores = {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0}
            return self.last_vader_scores
            
        try:
            scores = self.vader.polarity_scores(text)
            
            if self.debug:
                logger.debug(f"VADER scores for '{text}': {scores}")
                
            # Store the scores for later access
            self.last_vader_scores = scores
            
            return scores
        except Exception as e:
            logger.error(f"Error in VADER analysis: {e}")
            self.last_vader_scores = {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0}
            return self.last_vader_scores
    
    def analyze_textblob_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Analyze text using TextBlob sentiment analyzer.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (polarity, subjectivity)
        """
        if not self.use_textblob:
            self.last_textblob_scores = (0.0, 0.0)
            return self.last_textblob_scores
            
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            if self.debug:
                logger.debug(f"TextBlob scores for '{text}': polarity={polarity}, subjectivity={subjectivity}")
                
            # Store the scores for later access
            self.last_textblob_scores = (polarity, subjectivity)
            
            return self.last_textblob_scores
        except Exception as e:
            logger.error(f"Error in TextBlob analysis: {e}")
            self.last_textblob_scores = (0.0, 0.0)
            return self.last_textblob_scores
    
    def analyze_sentiment(self, url: str) -> Dict[str, Union[str, float]]:
        """
        Analyze a URL to determine if it's more action-oriented or information-oriented.
        
        Args:
            url: The URL to analyze
            
        Returns:
            Dictionary with classification and confidence score
        """
        tokens = self.tokenize_url(url)
        
        # Count action and information words
        action_count = 0
        info_count = 0
        
        for token in tokens:
            if token in ACTION_WORDS:
                action_count += 1
                if self.debug:
                    logger.debug(f"Action word found: {token}")
            elif token in INFO_WORDS:
                info_count += 1
                if self.debug:
                    logger.debug(f"Info word found: {token}")
        
        # Special patterns that strongly indicate CTA
        has_cta_pattern = False
        cta_patterns = [
            r'(sign|log)in',
            r'sign-?up',
            r'check-?out',
            r'shop.?cart',
            r'add.?to.?cart',
            r'become.?a.?',
            r'get.?started',
            r'try.?now',
            r'buy.?now',
            r'free.?trial'
        ]
        
        for pattern in cta_patterns:
            if re.search(pattern, url.lower()):
                has_cta_pattern = True
                action_count += 2  # Give extra weight to these patterns
                if self.debug:
                    logger.debug(f"CTA pattern found: {pattern}")
        
        # Check for query parameters that indicate CTA
        cta_params = ['gclid', 'utm_', 'cid', 'promo', 'coupon', 'discount', 'ref', 'affiliate']
        for param in cta_params:
            if param in url.lower():
                action_count += 1
                if self.debug:
                    logger.debug(f"CTA parameter found: {param}")
        
        # Calculate lexicon-based scores
        total = max(1, action_count + info_count)  # Avoid division by zero
        lexicon_action_score = action_count / total
        lexicon_info_score = info_count / total
        
        # Enhanced sentiment analysis using VADER and TextBlob
        readable_text = self.create_readable_text(url, tokens)
        
        # Use VADER if available
        vader_scores = self.analyze_vader_sentiment(readable_text)
        
        # Use TextBlob if available
        textblob_polarity, textblob_subjectivity = self.analyze_textblob_sentiment(readable_text)
        
        # Interpret VADER scores for CTA vs Brand Awareness
        # Positive sentiment often correlates with action-oriented content
        vader_action_score = vader_scores["pos"] * 0.7 + vader_scores["compound"] * 0.3
        vader_info_score = vader_scores["neu"] * 0.7 + (1 - abs(vader_scores["compound"])) * 0.3
        
        # Interpret TextBlob scores
        # Higher polarity (more positive) often indicates action-oriented content
        # Higher subjectivity can indicate persuasive language used in CTAs
        textblob_action_score = max(0, textblob_polarity) * 0.6 + textblob_subjectivity * 0.4
        textblob_info_score = (1 - abs(textblob_polarity)) * 0.7 + (1 - textblob_subjectivity) * 0.3
        
        # Combine scores with weights
        total_weight = self.lexicon_weight
        action_score = lexicon_action_score * self.lexicon_weight
        info_score = lexicon_info_score * self.lexicon_weight
        
        if self.use_vader:
            action_score += vader_action_score * self.vader_weight
            info_score += vader_info_score * self.vader_weight
            total_weight += self.vader_weight
            
        if self.use_textblob:
            action_score += textblob_action_score * self.textblob_weight
            info_score += textblob_info_score * self.textblob_weight
            total_weight += self.textblob_weight
        
        # Normalize scores
        if total_weight > 0:
            action_score /= total_weight
            info_score /= total_weight
        
        # Add base confidence to avoid zero scores
        action_score = max(0.1, action_score)
        info_score = max(0.1, info_score)
        
        # Strong pattern override
        if has_cta_pattern:
            action_score = max(action_score, 0.7)
        
        # Determine classification based on scores
        if action_score > info_score:
            classification = "Call to Action"
            confidence = min(1.0, 0.5 + (action_score - info_score))
        else:
            classification = "Brand Awareness"
            confidence = min(1.0, 0.5 + (info_score - action_score))
        
        if self.debug:
            logger.debug(f"URL: {url}")
            logger.debug(f"Lexicon scores: action={lexicon_action_score}, info={lexicon_info_score}")
            logger.debug(f"VADER scores: action={vader_action_score}, info={vader_info_score}")
            logger.debug(f"TextBlob scores: action={textblob_action_score}, info={textblob_info_score}")
            logger.debug(f"Final scores: action={action_score}, info={info_score}")
            logger.debug(f"Classification: {classification}, Confidence: {confidence}")
        
        result = {
            "classification": classification,
            "confidence": round(confidence, 2)
        }
        
        # Add detailed scores for debugging or analysis
        if self.debug:
            result["details"] = {
                "lexicon_scores": {
                    "action": round(lexicon_action_score, 3),
                    "info": round(lexicon_info_score, 3)
                },
                "vader_scores": {
                    "action": round(vader_action_score, 3),
                    "info": round(vader_info_score, 3),
                    "raw": vader_scores
                },
                "textblob_scores": {
                    "action": round(textblob_action_score, 3),
                    "info": round(textblob_info_score, 3),
                    "polarity": round(textblob_polarity, 3),
                    "subjectivity": round(textblob_subjectivity, 3)
                }
            }
        else:
            # Store component scores in a lighter format even when debug is off
            result["details"] = {
                "sentiment": {
                    "component_scores": {
                        "lexicon": {
                            "action_score": round(lexicon_action_score, 3),
                            "info_score": round(lexicon_info_score, 3),
                            "action_count": action_count,
                            "info_count": info_count
                        }
                    }
                }
            }
            
            # Add VADER component if used
            if self.use_vader:
                result["details"]["sentiment"]["component_scores"]["vader"] = {
                    "action_score": round(vader_action_score, 3),
                    "info_score": round(vader_info_score, 3),
                    "compound": round(vader_scores["compound"], 3),
                    "pos": round(vader_scores["pos"], 3),
                    "neg": round(vader_scores["neg"], 3),
                    "neu": round(vader_scores["neu"], 3)
                }
                
            # Add TextBlob component if used
            if self.use_textblob:
                result["details"]["sentiment"]["component_scores"]["textblob"] = {
                    "action_score": round(textblob_action_score, 3),
                    "info_score": round(textblob_info_score, 3),
                    "polarity": round(textblob_polarity, 3),
                    "subjectivity": round(textblob_subjectivity, 3)
                }
        
        return result
    
    def classify_url(self, url: str) -> str:
        """
        Classify a URL using sentiment analysis.
        
        Args:
            url: The URL to classify
            
        Returns:
            Classification: either "Brand Awareness" or "Call to Action"
        """
        result = self.analyze_sentiment(url)
        return result["classification"]
    
    def classify_with_confidence(self, url: str) -> Dict[str, Union[str, float]]:
        """
        Classify a URL with confidence score using sentiment analysis.
        
        Args:
            url: The URL to classify
            
        Returns:
            Dictionary with classification and confidence
        """
        return self.analyze_sentiment(url) 