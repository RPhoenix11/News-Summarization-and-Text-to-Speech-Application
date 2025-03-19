import os
from gtts import gTTS
import tempfile
from typing import Dict, Any, Optional
import re
from deep_translator import GoogleTranslator  # Using deep-translator for full translation

class TextToSpeechConverter:
    """Class for converting text to Hindi speech."""
    
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
    
    def clean_text_for_tts(self, text: str) -> str:
        """Clean and prepare text for TTS conversion."""
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^\w\s.,?!;:\-\'"()]', ' ', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        return text
    
    def translate_to_hindi(self, text: str) -> str:
        """Translate full text to Hindi using Google Translator."""
        try:
            hindi_text = GoogleTranslator(source='auto', target='hi').translate(text)
        except Exception as e:
            hindi_text = "अनुवाद में त्रुटि हुई। कृपया पुनः प्रयास करें।"  # Translation error fallback
        return hindi_text
    
    def generate_speech(self, text: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate Hindi speech from full translated text."""
        try:
            cleaned_text = self.clean_text_for_tts(text)
            hindi_text = self.translate_to_hindi(cleaned_text)
            
            if not output_file:
                output_file = os.path.join(self.temp_dir, f"news_summary_{hash(hindi_text) % 10000}.mp3")
            
            tts = gTTS(text=hindi_text, lang='hi', slow=False)
            tts.save(output_file)
            
            return {
                "success": True,
                "message": "Speech generated successfully",
                "output_file": output_file,
                "hindi_text": hindi_text
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error generating speech: {str(e)}",
                "output_file": None,
                "hindi_text": None
            }
    
    def generate_summary_speech(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate speech for a translated news summary."""
        try:
            company = report.get("company", "")
            final_sentiment = report.get("final_sentiment_analysis", "")
            
            summary_text = f"News report for {company}. {final_sentiment}"
            
            if "sentiment_counts" in report:
                counts = report["sentiment_counts"]
                summary_text += f" Found {counts.get('positive', 0)} positive, {counts.get('negative', 0)} negative, and {counts.get('neutral', 0)} neutral articles."
            
            result = self.generate_speech(summary_text)
            
            return result
        except Exception as e:
            return {
                "success": False,
                "message": f"Error generating summary speech: {str(e)}",
                "output_file": None,
                "hindi_text": None
            }
