import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Any
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from transformers import pipeline

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')

class NewsExtractor:
    """Class for extracting news articles related to a company."""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.search_urls = [
            "https://news.google.com/search?q={}",
            "https://www.reuters.com/search/news?blob={}",
            "https://www.bbc.co.uk/search?q={}"
        ]
    
    def get_search_results(self, company: str, num_articles: int = 15) -> List[str]:
        """Get search results for a company from multiple sources."""
        all_urls = []
        
        for search_url in self.search_urls:
            try:
                formatted_url = search_url.format(company.replace(' ', '+'))
                response = requests.get(formatted_url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract links - patterns vary by source
                    if "google" in search_url:
                        links = soup.select('a[href^="./articles/"]')
                        base_url = "https://news.google.com"
                        for link in links:
                            href = link.get('href')
                            if href and href.startswith('./articles/'):
                                all_urls.append(base_url + href[1:])
                    else:  # Generic approach for other sources
                        links = soup.find_all('a', href=True)
                        base_url = '/'.join(formatted_url.split('/')[:3])
                        for link in links:
                            href = link.get('href')
                            if href and 'http' not in href and '/article/' in href:
                                all_urls.append(base_url + href)
                            elif href and ('http' in href) and ('article' in href or 'news' in href):
                                all_urls.append(href)
                                
            except Exception as e:
                print(f"Error fetching from {search_url}: {str(e)}")
        
        # Remove duplicates and limit to num_articles
        unique_urls = list(set(all_urls))
        return unique_urls[:num_articles]
    
    def extract_article_content(self, url: str) -> Dict[str, Any]:
        """Extract content from a news article URL."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract title
                title = None
                if soup.title:
                    title = soup.title.get_text().strip()
                else:
                    title_tags = soup.find_all(['h1', 'h2'], class_=re.compile(r'title|heading'))
                    if title_tags:
                        title = title_tags[0].get_text().strip()
                
                # Extract main content
                article_content = ""
                # Try common article content selectors
                content_selectors = [
                    'article', 
                    'div.article-body', 
                    'div.story-body',
                    'div.content',
                    'div.story-content',
                    'div[itemprop="articleBody"]'
                ]
                
                for selector in content_selectors:
                    content_element = soup.select_one(selector)
                    if content_element:
                        # Get all paragraphs within the content
                        paragraphs = content_element.find_all('p')
                        article_content = " ".join([p.get_text().strip() for p in paragraphs])
                        break
                
                # Fallback to all paragraphs if no content found
                if not article_content:
                    paragraphs = soup.find_all('p')
                    article_content = " ".join([p.get_text().strip() for p in paragraphs])
                
                # Extract publication date
                pub_date = ""
                date_patterns = [
                    soup.find('meta', property='article:published_time'),
                    soup.find('meta', property='og:article:published_time'),
                    soup.find('time'),
                    soup.find('span', class_=re.compile(r'date|time'))
                ]
                
                for pattern in date_patterns:
                    if pattern:
                        if pattern.get('content'):
                            pub_date = pattern.get('content')
                            break
                        elif pattern.get('datetime'):
                            pub_date = pattern.get('datetime')
                            break
                        elif pattern.get_text():
                            pub_date = pattern.get_text().strip()
                            break
                
                # Extract summary (first paragraph or meta description)
                summary = ""
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc and meta_desc.get('content'):
                    summary = meta_desc.get('content')
                elif article_content:
                    # Take first paragraph as summary if no meta description
                    summary = article_content.split('. ')[0] + '.'
                    if len(summary) < 100 and len(article_content.split('. ')) > 1:
                        summary += ' ' + article_content.split('. ')[1] + '.'
                
                return {
                    "title": title if title else "No title found",
                    "content": article_content,
                    "summary": summary,
                    "publication_date": pub_date,
                    "url": url
                }
            
            return {"error": f"Failed to fetch article. Status code: {response.status_code}", "url": url}
        
        except Exception as e:
            return {"error": f"Error extracting article content: {str(e)}", "url": url}
    
    def get_company_news(self, company: str, num_articles: int = 10) -> List[Dict[str, Any]]:
        """Get news articles for a company."""
        urls = self.get_search_results(company, num_articles + 5)  # Get extra to account for failures
        
        articles = []
        for url in urls:
            article = self.extract_article_content(url)
            if article and 'error' not in article and article['content']:
                articles.append(article)
                if len(articles) >= num_articles:
                    break
        
        return articles[:num_articles]


class SentimentAnalyzer:
    """Class for performing sentiment analysis on news articles."""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize topic extraction
        try:
            self.summarizer = pipeline("summarization")
        except:
            self.summarizer = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        if not text:
            return {"compound": 0, "pos": 0, "neg": 0, "neu": 0, "label": "neutral"}
        
        sentiment = self.sia.polarity_scores(text)
        
        # Determine sentiment label
        if sentiment['compound'] >= 0.05:
            label = "positive"
        elif sentiment['compound'] <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "compound": sentiment['compound'],
            "pos": sentiment['pos'],
            "neg": sentiment['neg'],
            "neu": sentiment['neu'],
            "label": label
        }
    
    def extract_topics(self, text: str, num_topics: int = 3) -> List[str]:
        """Extract key topics from text."""
        if not text or len(text) < 10:
            return []
        
        # Preprocess text
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Count word frequencies
        word_freq = {}
        for token in tokens:
            if len(token) > 2:  # Only consider words with more than 2 characters
                if token in word_freq:
                    word_freq[token] += 1
                else:
                    word_freq[token] = 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Get top topics
        topics = []
        for word, freq in sorted_words[:num_topics*2]:
            if word not in topics and len(word) > 3:
                topics.append(word.capitalize())
                if len(topics) >= num_topics:
                    break
        
        return topics
    
    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """Summarize text using transformer model or fallback to extractive summarization."""
        if not text or len(text) < 50:
            return text
        
        # Try using the transformer model
        if self.summarizer and len(text) > 200:
            try:
                summary = self.summarizer(text, max_length=max_length, min_length=30, do_sample=False)
                return summary[0]['summary_text']
            except Exception as e:
                print(f"Error in transformer summarization: {str(e)}")
        
        # Fallback to simple extractive summarization
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) <= 3:
            return text
        
        # Return first 2-3 sentences as summary
        summary = ' '.join(sentences[:3])
        if len(summary) > max_length:
            summary = ' '.join(sentences[:2])
        
        return summary


class ComparativeAnalyzer:
    """Class for performing comparative analysis on news articles."""
    
    def __init__(self):
        pass
    
    def compare_sentiments(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare sentiments across articles."""
        if not articles:
            return {}
        
        sentiment_distribution = {"positive": 0, "negative": 0, "neutral": 0}
        
        for article in articles:
            if 'sentiment' in article and 'label' in article['sentiment']:
                sentiment_distribution[article['sentiment']['label']] += 1
        
        # Calculate average sentiment
        total_compound = sum(article['sentiment']['compound'] for article in articles if 'sentiment' in article)
        avg_sentiment = total_compound / len(articles) if articles else 0
        
        # Determine overall sentiment
        overall_sentiment = "neutral"
        if avg_sentiment >= 0.05:
            overall_sentiment = "positive"
        elif avg_sentiment <= -0.05:
            overall_sentiment = "negative"
        
        # Generate coverage differences
        coverage_differences = []
        pos_articles = [a for a in articles if 'sentiment' in a and a['sentiment']['label'] == 'positive']
        neg_articles = [a for a in articles if 'sentiment' in a and a['sentiment']['label'] == 'negative']
        
        if pos_articles and neg_articles:
            coverage_differences.append({
                "comparison": f"Positive articles focus on {', '.join([a['title'].split(':')[0] for a in pos_articles[:2]])}, while negative articles discuss {', '.join([a['title'].split(':')[0] for a in neg_articles[:2]])}.",
                "impact": "These contrasting perspectives may influence stakeholder perceptions differently."
            })
        
        # Find topic overlap
        all_topics = []
        for article in articles:
            if 'topics' in article:
                all_topics.extend(article['topics'])
        
        # Count topic frequencies
        topic_counts = {}
        for topic in all_topics:
            if topic in topic_counts:
                topic_counts[topic] += 1
            else:
                topic_counts[topic] = 1
        
        # Find common and unique topics
        common_topics = [topic for topic, count in topic_counts.items() if count > 1]
        unique_topics = [topic for topic, count in topic_counts.items() if count == 1]
        
        # Get final sentiment analysis
        dominant_sentiment = max(sentiment_distribution, key=sentiment_distribution.get)
        
        final_sentiment = f"The news coverage is predominantly {dominant_sentiment}."
        if dominant_sentiment == "positive":
            final_sentiment += " This suggests a favorable outlook."
        elif dominant_sentiment == "negative":
            final_sentiment += " This indicates potential concerns or challenges."
        else:
            final_sentiment += " The overall sentiment is balanced."
        
        return {
            "sentiment_distribution": sentiment_distribution,
            "average_sentiment": avg_sentiment,
            "overall_sentiment": overall_sentiment,
            "coverage_differences": coverage_differences,
            "topic_overlap": {
                "common_topics": common_topics[:5],
                "unique_topics": unique_topics[:10]
            },
            "final_sentiment_analysis": final_sentiment
        }
    
    def generate_comparative_report(self, company: str, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comparative report for articles."""
        if not articles:
            return {"error": "No articles provided for analysis"}
        
        sentiment_comparison = self.compare_sentiments(articles)
        
        # Calculate additional metrics
        article_count = len(articles)
        sentiment_counts = sentiment_comparison['sentiment_distribution']
        
        # Format report
        report = {
            "company": company,
            "articles": articles,
            "comparative_sentiment_score": sentiment_comparison,
            "article_count": article_count,
            "sentiment_counts": sentiment_counts,
            "final_sentiment_analysis": sentiment_comparison['final_sentiment_analysis']
        }
        
        return report