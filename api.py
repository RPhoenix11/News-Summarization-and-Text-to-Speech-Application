from flask import Flask, request, jsonify
import json
from typing import Dict, Any, List
import os
import base64

# Import our modules
from utils import NewsExtractor, SentimentAnalyzer, ComparativeAnalyzer
from tts import TextToSpeechConverter

app = Flask(__name__)

# Initialize our components
news_extractor = NewsExtractor()
sentiment_analyzer = SentimentAnalyzer()
comparative_analyzer = ComparativeAnalyzer()
tts_converter = TextToSpeechConverter()

@app.route('/health', methods=['GET'])
def health_check():
    """API endpoint for health check."""
    return jsonify({"status": "ok", "message": "API is running"})

@app.route('/api/news', methods=['GET'])
def get_news():
    """API endpoint to get news for a company."""
    company = request.args.get('company')
    
    if not company:
        return jsonify({"error": "Company parameter is required"}), 400
    
    # Extract articles
    articles = news_extractor.get_company_news(company)
    
    return jsonify({"company": company, "articles": articles})

@app.route('/api/analyze', methods=['POST'])
def analyze_news():
    """API endpoint to analyze news articles."""
    data = request.json
    
    if not data or 'company' not in data or 'articles' not in data:
        return jsonify({"error": "Company and articles are required in the request body"}), 400
    
    company = data['company']
    articles = data['articles']
    
    # Process each article to add sentiment and topics
    processed_articles = []
    for article in articles:
        if 'content' in article and article['content']:
            # Analyze sentiment
            sentiment = sentiment_analyzer.analyze_sentiment(article['content'])
            
            # Extract topics
            topics = sentiment_analyzer.extract_topics(article['content'])
            
            # Ensure there's a summary
            if 'summary' not in article or not article['summary']:
                article['summary'] = sentiment_analyzer.summarize_text(article['content'])
            
            # Add to processed articles
            processed_article = {
                "title": article.get('title', 'No title'),
                "summary": article.get('summary', 'No summary available'),
                "content": article.get('content', ''),
                "url": article.get('url', ''),
                "sentiment": sentiment,
                "topics": topics
            }
            processed_articles.append(processed_article)
    
    # Generate comparative report
    report = comparative_analyzer.generate_comparative_report(company, processed_articles)
    
    return jsonify(report)

@app.route('/api/tts', methods=['POST'])
def generate_tts():
    """API endpoint to generate TTS from a report."""
    data = request.json
    
    if not data or 'report' not in data:
        return jsonify({"error": "Report is required in the request body"}), 400
    
    report = data['report']
    
    # Generate TTS
    tts_result = tts_converter.generate_summary_speech(report)
    
    if tts_result['success'] and tts_result['output_file']:
        # Read the audio file and encode to base64
        with open(tts_result['output_file'], 'rb') as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
        
        return jsonify({
            "success": True,
            "audio_data": audio_data,
            "hindi_text": tts_result['hindi_text']
        })
    else:
        return jsonify({
            "success": False,
            "message": tts_result['message']
        }), 500

@app.route('/api/full_analysis', methods=['GET'])
def full_analysis():
    """API endpoint to perform full analysis in one request."""
    company = request.args.get('company')
    
    if not company:
        return jsonify({"error": "Company parameter is required"}), 400
    
    # Step 1: Extract articles
    articles = news_extractor.get_company_news(company)
    
    if not articles:
        return jsonify({"error": "No articles found for the company"}), 404
    
    # Step 2: Process each article
    processed_articles = []
    for article in articles:
        if 'content' in article and article['content']:
            # Analyze sentiment
            sentiment = sentiment_analyzer.analyze_sentiment(article['content'])
            
            # Extract topics
            topics = sentiment_analyzer.extract_topics(article['content'])
            
            # Ensure there's a summary
            if 'summary' not in article or not article['summary']:
                article['summary'] = sentiment_analyzer.summarize_text(article['content'])
            
            # Add to processed articles
            processed_article = {
                "title": article.get('title', 'No title'),
                "summary": article.get('summary', 'No summary available'),
                "content": article.get('content', ''),
                "url": article.get('url', ''),
                "sentiment": sentiment,
                "topics": topics
            }
            processed_articles.append(processed_article)
    
    # Step 3: Generate comparative report
    report = comparative_analyzer.generate_comparative_report(company, processed_articles)
    
    # Step 4: Generate TTS
    tts_result = tts_converter.generate_summary_speech(report)
    
    if tts_result['success'] and tts_result['output_file']:
        # Read the audio file and encode to base64
        with open(tts_result['output_file'], 'rb') as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
        
        report['audio_data'] = audio_data
        report['hindi_text'] = tts_result['hindi_text']
    
    return jsonify(report)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))