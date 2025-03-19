import streamlit as st
import requests
import json
import base64
import pandas as pd
import time
import os

# Set the base URL for API
API_BASE_URL = os.environ.get('API_BASE_URL', 'http://localhost:5000')

st.set_page_config(
    page_title="News Summarization & Sentiment Analysis",
    page_icon="📰",
    layout="wide"
)

def get_news(company):
    """Get news articles for a company."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/full_analysis", params={"company": company})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"API Connection Error: {str(e)}")
        return None

def display_sentiment_chart(sentiment_counts):
    """Display a chart of sentiment distribution."""
    if not sentiment_counts:
        return
    
    # Create data for chart
    chart_data = pd.DataFrame({
        'Sentiment': ['Positive', 'Negative', 'Neutral'],
        'Count': [
            sentiment_counts.get('positive', 0),
            sentiment_counts.get('negative', 0),
            sentiment_counts.get('neutral', 0)
        ]
    })
    
    # Display chart
    st.bar_chart(chart_data.set_index('Sentiment'))

def play_audio(audio_data):
    """Play audio from base64 data."""
    audio_bytes = base64.b64decode(audio_data)
    st.audio(audio_bytes, format='audio/mp3')

def main():
    """Main function for Streamlit app."""
    st.title("News Summarization & Sentiment Analysis")
    
    # Sidebar for company input
    st.sidebar.header("Input Options")
    
    # Default company options or user input
    company_options = ["Tesla", "Apple", "Google", "Microsoft", "Amazon", "Other"]
    company_choice = st.sidebar.selectbox("Select Company", company_options)
    
    if company_choice == "Other":
        company = st.sidebar.text_input("Enter Company Name")
    else:
        company = company_choice
    
    # Analysis button
    if st.sidebar.button("Analyze News") and company:
        with st.spinner(f'Fetching and analyzing news for {company}...'):
            # Show progress bar
            progress_bar = st.progress(0)
            for i in range(100):
                # Update progress bar
                progress_bar.progress(i + 1)
                time.sleep(0.01)
            
            # Get news data
            data = get_news(company)
            
            # Display results
            if data and 'error' not in data:
                st.success("Analysis Complete!")
                
                # Display sentiment summary
                st.header("Sentiment Analysis Summary")
                st.subheader(data.get('final_sentiment_analysis', 'No sentiment analysis available'))
                
                # Display audio player if available
                if 'audio_data' in data:
                    st.subheader("Audio Summary (Hindi)")
                    play_audio(data['audio_data'])
                    st.caption("Hindi Text:")
                    st.text(data.get('hindi_text', ''))
                
                # Display charts
                st.header("Sentiment Distribution")
                if 'sentiment_counts' in data:
                    display_sentiment_chart(data['sentiment_counts'])
                
                # Display articles
                st.header("News Articles")
                for idx, article in enumerate(data.get('articles', [])):
                    with st.expander(f"{idx+1}. {article.get('title', 'No Title')}"):
                        # Show sentiment badge
                        sentiment = article.get('sentiment', {}).get('label', 'neutral')
                        if sentiment == 'positive':
                            st.success(f"Sentiment: {sentiment.upper()}")
                        elif sentiment == 'negative':
                            st.error(f"Sentiment: {sentiment.upper()}")
                        else:
                            st.info(f"Sentiment: {sentiment.upper()}")
                        
                        # Show summary
                        st.subheader("Summary")
                        st.write(article.get('summary', 'No summary available'))
                        
                        # Show topics
                        if 'topics' in article and article['topics']:
                            st.subheader("Topics")
                            st.write(", ".join(article['topics']))
                        
                        # Show URL
                        st.subheader("Source")
                        st.write(f"[Read full article]({article.get('url', '#')})")
                
                # Display comparative analysis
                st.header("Comparative Analysis")
                if 'comparative_sentiment_score' in data and 'topic_overlap' in data['comparative_sentiment_score']:
                    topic_overlap = data['comparative_sentiment_score']['topic_overlap']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Common Topics")
                        if topic_overlap['common_topics']:
                            st.write(", ".join(topic_overlap['common_topics']))
                        else:
                            st.write("No common topics found")
                    
                    with col2:
                        st.subheader("Unique Topics")
                        if topic_overlap['unique_topics']:
                            st.write(", ".join(topic_overlap['unique_topics'][:5]))
                        else:
                            st.write("No unique topics found")
                
                # Display coverage differences
                if 'comparative_sentiment_score' in data and 'coverage_differences' in data['comparative_sentiment_score']:
                    coverage_diffs = data['comparative_sentiment_score']['coverage_differences']
                    
                    if coverage_diffs:
                        st.subheader("Coverage Differences")
                        for diff in coverage_diffs:
                            st.write(f"**Comparison:** {diff.get('comparison', '')}")
                            st.write(f"**Impact:** {diff.get('impact', '')}")
            else:
                st.error("No data available or error in analysis")

if __name__ == "__main__":
    main()