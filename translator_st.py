import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from gtts import gTTS
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
import tempfile
import playsound
import scipy.io.wavfile as wav
import argparse

@st.cache_resource
def load_translation_model(model_path=None, device="cpu"):
    
    try:
        
        base_model_name = "facebook/nllb-200-distilled-600M"
        
        
        if model_path and os.path.exists(model_path):
            st.info(f"Loading fine-tuned model from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        else:
            st.info(f"Loading base model {base_model_name}")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        
        
        model = model.to(device)
        
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

class Translator:
    def __init__(self, model_path=None, device="cpu"):
        
        self.lang_codes = {
            'English': 'eng_Latn',
            'Arabic': 'arz_Arab',
            'French': 'fra_Latn',
            'German': 'deu_Latn'
        }
        
        
        self.tts_codes = {
            'English': 'en',
            'Arabic': 'ar',
            'French': 'fr',
            'German': 'de'
        }
        
        
        self.tokenizer = None
        self.model = None
        self.speech_recognizer = None
        self.model_path = model_path
        self.device = device
    
    def _ensure_models_loaded(self):
        """Lazy loading of models"""
        if self.speech_recognizer is None:
            try:
                self.speech_recognizer = pipeline(
                    "automatic-speech-recognition", 
                    model="openai/whisper-small",
                    device=0 if self.device == "cuda" else self.device
                )
            except Exception as e:
                st.error(f"Failed to load speech recognizer: {e}")
                return False
        
        if self.tokenizer is None or self.model is None:
            self.tokenizer, self.model = load_translation_model(
                model_path=self.model_path, 
                device=self.device
            )
            if self.tokenizer is None or self.model is None:
                return False
        
        return True

    def translate_text(self, text, source_lang, target_lang):
        
        if not self._ensure_models_loaded():
            return "Failed to load translation models"
        
        try:
            
            src_code = self.lang_codes[source_lang]
            tgt_code = self.lang_codes[target_lang]
            
            
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            
            
            tgt_lang_token = self.tokenizer.convert_tokens_to_ids(tgt_code)
            
            
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_token,
                max_length=200,
                num_beams=4,
                early_stopping=True
            )
            
            
            translated_text = self.tokenizer.batch_decode(
                translated_tokens, 
                skip_special_tokens=True
            )[0]
            
            return translated_text
        except Exception as e:
            st.error(f"Translation error: {str(e)}")
            return f"Error translating text: {str(e)}"

    def record_audio(self, duration=5, sample_rate=16000):
        
        st.info(f"Recording for {duration} seconds...")
        recording = sd.rec(int(duration * sample_rate),
                         samplerate=sample_rate,
                         channels=1,
                         dtype='float32')
        sd.wait()
        return recording, sample_rate

    def speech_to_text(self, audio_data, sample_rate):
        
        if not self._ensure_models_loaded():
            return "Failed to load speech recognition model"
        
        try:
            
            temp_wav = tempfile.mktemp(suffix='.wav')
            wav.write(temp_wav, sample_rate, (audio_data * 32767).astype(np.int16))
            
            try:
                
                result = self.speech_recognizer(temp_wav)
                text = result["text"]
            except Exception as e:
                st.error(f"Speech recognition error: {e}")
                text = "Could not recognize speech"
            
            
            try:
                os.unlink(temp_wav)
            except Exception:
                pass
            
            return text
        except Exception as e:
            st.error(f"Audio processing error: {str(e)}")
            return "Error processing audio"

    def text_to_speech(self, text, language):
        """Convert text to speech using gTTS."""
        try:
            tts_lang = self.tts_codes.get(language, 'en')
            
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                tts = gTTS(text=text, lang=tts_lang)
                tts.save(temp_file.name)
                
            
            playsound.playsound(temp_file.name)
            
            
            os.unlink(temp_file.name)
                
        except Exception as e:
            st.error(f"Text-to-speech error: {str(e)}")

def main(model_path=None, device="cpu"):
    
    st.set_page_config(page_title="Multilingual Translator", page_icon="🌐")
    
    
    translator = Translator(model_path=model_path, device=device)
    
    
    st.title("🌍 Multilingual Translation App")
    st.markdown("Translate text and speech between English, Arabic, French, and German!")
    
    
    if model_path:
        st.sidebar.info(f"Using model: {model_path}")
    
    
    tab1, tab2 = st.tabs(["Text Translation", "Speech Translation"])
    
    with tab1:
        st.header("Text Translation")
        
        
        col1, col2 = st.columns(2)
        
        with col1:
            source_lang = st.selectbox("Source Language", 
                ['English', 'Arabic', 'French', 'German'], key='text_source')
        
        with col2:
            target_lang = st.selectbox("Target Language", 
                ['English', 'Arabic', 'French', 'German'], key='text_target')
        
        
        text_input = st.text_area("Enter text to translate:", height=200)
        
        
        if st.button("Translate Text"):
            if text_input:
                
                with st.spinner('Translating...'):
                   
                    translated_text = translator.translate_text(text_input, source_lang, target_lang)
                
                
                st.success("Translated Text:")
                st.write(translated_text)
                
                
                if st.button("Listen to Translation"):
                    translator.text_to_speech(translated_text, target_lang)
            else:
                st.warning("Please enter some text to translate.")
    
    with tab2:
        st.header("Speech Translation")
        
        
        col1, col2 = st.columns(2)
        
        with col1:
            source_lang = st.selectbox("Source Language", 
                ['English', 'Arabic', 'French', 'German'], key='speech_source')
        
        with col2:
            target_lang = st.selectbox("Target Language", 
                ['English', 'Arabic', 'French', 'German'], key='speech_target')
        
        
        if st.button("Record Speech (5 seconds)"):
            
            audio_data, sample_rate = translator.record_audio()
            
            
            with st.spinner('Processing speech...'):
                text = translator.speech_to_text(audio_data, sample_rate)
            st.write("Recognized Text:", text)
            
            
            with st.spinner('Translating...'):
                translated_text = translator.translate_text(text, source_lang, target_lang)
            st.success("Translated Text:")
            st.write(translated_text)
            
            
            if st.button("Listen to Translation"):
                translator.text_to_speech(translated_text, target_lang)

def parse_arguments():
    
    parser = argparse.ArgumentParser(description="Multilingual Translation Streamlit App")
    parser.add_argument(
        "--model", 
        type=str, 
        help="Path to fine-tuned model (optional)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu", 
        choices=["cpu", "cuda"], 
        help="Device to run models on"
    )
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_arguments()
    
    
    main(model_path=args.model, device=args.device)