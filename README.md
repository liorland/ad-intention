# Ad Intention Classification System

This project implements an automated system for classifying ad landing pages into "Brand Awareness" and "Call to Action (CTA)" categories based on URL structure analysis for AdClarity.

## Project Overview

The Ad Intention Classification System analyzes landing page URLs to determine the intended goal of an advertisement:

- **Brand Awareness**: Pages focused on providing information, storytelling, or brand recognition
- **Call to Action (CTA)**: Pages designed to drive immediate user actions like purchases, sign-ups, or registrations

## Features

- URL-based classification without requiring full-page content analysis
- Efficient processing of large URL datasets
- Handles edge cases like root domains with tracking parameters
- Comprehensive rule set based on URL structure patterns
- Support for batch processing and integration with data sources
- AI-enhanced classification with multiple classification methods:
  - Rule-based classification (traditional approach)
  - Sentiment analysis for URL paths and parameters (with VADER, TextBlob, and Lexicon)
  - BERT zero-shot classification (future enhancement, requires GPU)
  - Gemini LLM-based few-shot classification
  - Weighted ensemble approach combining multiple classifiers
- Transparent classification with the Judge component:
  - Natural language explanations of classification decisions
  - Customizable level of detail and analysis depth
  - Multiple output formats (natural language, structured JSON, summary)
  - Feature-specific reasoning and confidence scores

## Project Structure

```
ad_intention/
├── classification/       # Core classification logic
│   ├── __init__.py
│   ├── classifier.py     # Main classifier implementation
│   └── rules.py          # Classification rules definition
├── enhanced/             # AI-enhanced classification components
│   ├── __init__.py
│   ├── sentiment_classifier.py     # Sentiment analysis for URLs
│   ├── bert_classifier.py          # BERT zero-shot classification
│   ├── gemini_classifier.py        # Gemini LLM-based classification
│   ├── ensemble_classifier.py      # Weighted ensemble approach
│   ├── openai_judge.py             # Classification reasoning and explanation
│   └── cli.py                      # Command-line interface
├── utils/                # Utility functions and helpers
│   ├── __init__.py
│   ├── url_helpers.py    # URL parsing and manipulation utilities
│   └── batch_process.py  # Batch processing capabilities
├── config.py             # Centralized configuration management
└── __init__.py           # Package initialization
```

## Installation

Follow these steps to get started with the Ad Intention Classification System:

```bash
# Clone the repository
git clone https://github.com/your-org/ad-intention.git
cd ad-intention

# Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt

```

### API Keys (Optional)

The application has pre-configured API keys for OpenAI and Google Gemini. 
No additional setup is required to use these features.

If you need to use your own API keys:
1. Copy the example environment file: `cp .env.example .env`
2. Edit the `.env` file to add your API keys

## Command-Line Interface (CLI)

The Ad Intention Classification System provides a command-line interface for processing URLs.

### Basic Usage

```bash
# Basic command structure (rule-based classification by default)
python -m ad_intention.enhanced.cli --urls "https://example.com/shop" "https://example.org/about"

# Save results to a file
python -m ad_intention.enhanced.cli --urls "https://example.com/shop" --output results.csv

# Process URLs from a CSV file
python -m ad_intention.enhanced.cli --file data/urls.csv --output classified_urls.csv
```

### Enabling Classification Methods

You can enable any combination of classification methods by adding flags:

```bash
# Enable sentiment analysis
python -m ad_intention.enhanced.cli --urls "https://example.com/shop" --use-sentiment

# Enable Gemini LLM-based classification
python -m ad_intention.enhanced.cli --urls "https://example.com/shop" --use-gemini

# Use all classification methods (full ensemble)
python -m ad_intention.enhanced.cli --urls "https://example.com/shop" \
    --use-sentiment --use-gemini \
    --use-vader --use-textblob --use-lexicon
```

### Adjusting Classifier Weights

You can adjust the weight of each classifier in the ensemble:

```bash
# Customize classifier weights
python -m ad_intention.enhanced.cli --urls "https://example.com/shop" \
    --use-sentiment --use-gemini \
    --rules-weight 2.0 --sentiment-weight 1.0 --gemini-weight 1.5
```

### Custom weights for different classifiers

```bash
# Custom weights for different classifiers
python -m ad_intention.enhanced.cli --urls "https://example.com/signup" \
    --use-sentiment --use-gemini \
    --sentiment-weight 1.0 --gemini-weight 1.5 --rules-weight 2.0 \
    --majority-threshold 0.6
```

### Common Command Examples

Here are examples of common tasks:

```bash
# Process a single URL with all classifiers and detailed output
python -m ad_intention.enhanced.cli --urls "https://example.com/shop" \
    --use-sentiment --use-gemini --with-details

# Process URLs from a CSV file with a custom URL column name
python -m ad_intention.enhanced.cli --file campaigns.csv --url-column landing_page_url \
    --output results.csv

# Enable debug output to see how classification decisions are made
python -m ad_intention.enhanced.cli --urls "https://example.com/shop" --debug

# Use OpenAI judge for classification explanations
python -m ad_intention.enhanced.cli --urls "https://example.com/pricing" \
    --use-openai-judge --format json
```

### Available Command Arguments

```
# URL input options
--urls URL [URL ...]       One or more URLs to classify
--file FILEPATH            Path to CSV file containing URLs to classify
--url-column COLUMN_NAME   Column name in CSV containing URLs (default: "url")

# Output options
--output FILEPATH          Output file path (default: stdout)
--with-details             Include detailed classification info in output
--debug                    Enable debug output during classification
--format {csv,json,table}  Output format (default: csv)

# Classification method selection (default is rule-based only)
--use-sentiment            Enable sentiment analysis classification
--use-vader                Use VADER for sentiment analysis
--use-textblob             Use TextBlob for sentiment analysis
--use-lexicon              Use Lexicon-based sentiment analysis
--use-bert                 Enable BERT classification (future enhancement)
--use-gemini               Enable Gemini LLM-based classification
--use-openai-judge         Enable OpenAI judge for classification explanations

# Classification weights and thresholds
--rules-weight FLOAT       Weight for rule-based classifier (default: 2.0)
--sentiment-weight FLOAT   Weight for sentiment classifier (default: 1.0)
--bert-weight FLOAT        Weight for BERT classifier (default: 1.0)
--gemini-weight FLOAT      Weight for Gemini classifier (default: 1.5)
--majority-threshold FLOAT Threshold for majority voting (default: 0.6)

# Performance options
--n-jobs INT               Number of parallel jobs (-1 for all cores, default: -1)
--backend {loky,threading,multiprocessing}  Parallelism backend (default: loky)
--separate-sentiment       Show separate results for each sentiment analyzer (default: True)
```

### Windows-Specific Notes

On Windows, use these command patterns:

```powershell
# PowerShell - Using single quotes for URLs with special characters
python -m ad_intention.enhanced.cli --urls 'https://example.com/promo?utm_source=ads'

# Command Prompt - Using escaped quotes 
python -m ad_intention.enhanced.cli --urls "https://example.com/promo?utm_source=ads"

# For files with spaces in the path
python -m ad_intention.enhanced.cli --file "C:\My Documents\ad data\urls.csv"
```

## Streamlit User Interface

A graphical user interface built with Streamlit is available for easier interaction with the Ad Intention Classification System.

### Running the Streamlit UI

After following the installation steps above, simply run:

```bash
# Start the Streamlit application
streamlit run app.py
```

This will launch the web interface in your default browser. If it doesn't open automatically, you can access it at `http://localhost:8501`.

### UI Features

The Streamlit interface includes:

- Visual selection of classification methods:
  - Rule-based Classification (always available)
  - Sentiment Analysis (with VADER, TextBlob, and Lexicon options)
  - BERT Classification (future enhancement, requires GPU)
  - Gemini LLM Classification
  - OpenAI Judge for explanation of decisions
- Interactive weight adjustment sliders for all enabled classifiers
- Persistent session state that maintains results between interactions
- Performance optimizations for handling large datasets
- Two convenient input methods:
  - Direct URL input (multiple URLs, one per line)
  - CSV file upload with customizable URL column name
- Comprehensive results display:
  - Main results table with classifier-specific columns
  - Classifier Contributions visualization (bar chart and decision table)
  - Detailed classification results with expandable JSON data
- Download options for results (CSV and JSON formats)
- Pre-configured API keys (no user setup required)

### Customization Options

The Streamlit UI allows extensive customization:

1. **Classification Methods**:
   - Toggle each classifier on/off (Rule-based, Sentiment, Gemini, Judge)
   - Select which sentiment analyzers to use (VADER, TextBlob, Lexicon)

2. **Weighting System**:
   - Set weights for each enabled classifier (Rule-based, Sentiment, BERT, Gemini)
   - Fine-tune individual sentiment analyzer weights (VADER, TextBlob, Lexicon)
   - All weights are adjustable from 0.1 to 5.0 with 0.1 increments

3. **Advanced Settings**:
   - Adjust majority threshold for ensemble voting (0.1 to 1.0)
   - Enable/disable detailed results for performance optimization
   - Configure parallel processing (jobs and backend selection)
   - Set custom URL column name for CSV uploads
   - Toggle debug mode for troubleshooting

4. **Results Visualization**:
   - View classification results in an interactive table
   - Expand detailed results for each URL
   - Download results in CSV or JSON format

## Classification Methods

### Rule-Based Classification (Default)

The rule-based classifier uses pattern matching on URL paths, parameters, and domains to determine the likely intention:

```bash
# Basic rule-based classification (default if no method flags are specified)
python -m ad_intention.enhanced.cli --urls "https://example.com/signup" "https://example.org/about"

# Rule-based with debug output to see matching rules
python -m ad_intention.enhanced.cli --urls "https://example.com/signup" --debug
```

### Sentiment Analysis Classification

The sentiment classifier analyzes the emotional tone of URL components using three methods:

```bash
# Basic sentiment analysis (enables all methods by default)
python -m ad_intention.enhanced.cli --urls "https://example.com/amazing-offer" --use-sentiment

# With specific sentiment analyzers
python -m ad_intention.enhanced.cli --urls "https://example.com/special-deal" \
    --use-sentiment --use-vader --use-textblob --use-lexicon
    
# Adjust sentiment weight in ensemble
python -m ad_intention.enhanced.cli --urls "https://example.com/limited-time-offer" \
    --use-sentiment --sentiment-weight 1.5
```

### BERT Zero-Shot Classification (Future Enhancement)

BERT classification using a pre-trained transformer model is planned as a future enhancement. It will require GPU resources:

```bash
# This feature is not fully implemented yet and will require GPU
# The UI includes a placeholder for this feature

# In the future, it will be accessible via CLI as:
python -m ad_intention.enhanced.cli --urls "https://example.com/checkout" --use-bert
```

### Gemini LLM-Based Classification

Uses Google's Gemini LLM for classification (requires API key):

```bash
# Enable Gemini classification
python -m ad_intention.enhanced.cli --urls "https://example.com/register" --use-gemini

# Run with environment variable for API key
GOOGLE_API_KEY=your_api_key_here python -m ad_intention.enhanced.cli \
    --urls "https://example.com/register" --use-gemini
```

### Ensemble Classification

Combines multiple classifiers with weighted voting:

```bash
# Full ensemble with all classifiers
python -m ad_intention.enhanced.cli --urls "https://example.com/special-offer" \
    --use-sentiment --use-gemini \
    --use-vader --use-textblob --use-lexicon

# Custom weights for different classifiers
python -m ad_intention.enhanced.cli --urls "https://example.com/signup" \
    --use-sentiment --use-gemini \
    --sentiment-weight 1.0 --gemini-weight 1.5 --rules-weight 2.0 \
    --majority-threshold 0.6
```

### Batch Processing

Process multiple URLs efficiently:

```bash
# Process URLs from a CSV file
python -m ad_intention.enhanced.cli \
    --file data/campaign_urls.csv \
    --url-column landing_page_url \
    --output classified_campaigns.csv
```



The project includes a `.gitignore` file to prevent committing sensitive information like API keys and environment files. Always use `.env.example` as a template and keep your actual API keys in the local `.env` file. 

## Sentiment Analysis Techniques

The Ad Intention Classification System employs three different sentiment analysis techniques to evaluate URL components for classification. These analyzers examine parts of the URL (path segments, query parameters) to determine whether they have a positive sentiment (suggesting CTA) or negative/neutral sentiment (suggesting Brand Awareness).

### VADER Sentiment Analysis

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool specifically attuned to sentiments expressed in social media and web content. In our system:

- VADER analyzes individual URL path segments and query parameters
- It assigns compound scores ranging from -1 (most negative) to +1 (most positive)
- Higher positive scores tend to correlate with CTA content (e.g., "amazing-offer", "special-deal")
- Lower or negative scores typically indicate Brand Awareness content (e.g., "about", "information")
- VADER is particularly effective at handling intensifiers, negations, and common internet expressions

### TextBlob Sentiment Analysis

TextBlob provides a simple API for common NLP tasks including sentiment analysis. In our implementation:

- TextBlob examines URL components using its polarity scoring system
- Polarity scores range from -1.0 (negative) to 1.0 (positive)
- TextBlob considers word polarity based on its training data
- It handles negations and modifiers differently than VADER
- TextBlob can provide a complementary perspective to VADER's analysis

### Lexicon-Based Sentiment Analysis

Our custom lexicon-based analyzer employs a domain-specific dictionary tailored for ad classification:

- Uses a manually curated dictionary of terms common in advertising URLs
- Terms are categorized as CTA-indicative ("buy", "shop", "register") or Brand-Awareness-indicative ("about", "story", "mission")
- Each term has a pre-assigned weight based on its strength as an indicator
- The analyzer identifies these terms in URL components and calculates an aggregate score
- This approach performs well with advertising-specific terminology that general sentiment analyzers might miss

### How Sentiment Analysis Integrates with Classification

In the ensemble classification approach:

1. Each sentiment analyzer independently evaluates URL components
2. Results from all enabled analyzers are combined with configurable weights
3. The overall sentiment score contributes to the final classification decision
4. When enabled, individual sentiment analyzer scores are visible in the detailed results
5. The "separate_sentiment" option (default: True) allows viewing independent results from each analyzer

This multi-technique approach to sentiment analysis provides robustness against individual analyzer limitations and improves overall classification accuracy, especially for URLs with strong emotional or action-oriented language. 