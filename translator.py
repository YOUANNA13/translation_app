import os
import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from gtts import gTTS
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import time
from playsound import playsound

class Translator:
    def __init__(self, model_path=None, device="cpu"):

        self.base_model_name = "facebook/nllb-200-distilled-600M"
        
        
        try:
            
            if model_path and os.path.exists(model_path):
                print(f"Loading fine-tuned model from {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            else:
                print(f"Loading base model {self.base_model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model_name)
            
            
            self.model = self.model.to(device)
            
            
            self.speech_recognizer = pipeline(
                "automatic-speech-recognition", 
                model="openai/whisper-small",
                device=0 if device == "cuda" else device
            )
        except Exception as e:
            print(f"Error initializing models: {e}")
            raise
        
        
        self.lang_codes = {
            'english': 'eng_Latn',
            'arabic': 'arz_Arab',
            'french': 'fra_Latn',
            'german': 'deu_Latn'
        }
        
        
        self.tts_codes = {
            'english': 'en',
            'arabic': 'ar',
            'french': 'fr',
            'german': 'de'
        }

    def validate_language(self, language):
        
        language = language.lower()
        if language not in self.lang_codes:
            raise ValueError(f"Unsupported language. Supported languages are: {', '.join(self.lang_codes.keys())}")
        return language

    def translate_text(self, text, source_lang, target_lang):

        try:
            
            source_lang = self.validate_language(source_lang)
            target_lang = self.validate_language(target_lang)
            
            
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
            
            
            return self.tokenizer.batch_decode(
                translated_tokens, 
                skip_special_tokens=True
            )[0]
        except Exception as e:
            print(f"Translation error: {e}")
            return f"Error translating text: {e}"

    def record_audio(self, duration=5, sample_rate=16000):
        
        print(f"Recording for {duration} seconds...")
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        return recording, sample_rate

    def speech_to_text(self, audio_data, sample_rate):
        
        try:
            
            temp_audio_path = "temp_audio.wav"
            write(temp_audio_path, sample_rate, audio_data)
            
            try:
                
                result = self.speech_recognizer(temp_audio_path)
                text = result["text"]
            except Exception as e:
                print(f"Speech recognition error: {e}")
                text = "Could not recognize speech"
            
            
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass
                
            return text
        except Exception as e:
            print(f"Audio processing error: {e}")
            return "Error processing audio"

    def text_to_speech(self, text, language):
        
        try:
            
            language = self.validate_language(language)
            tts_lang = self.tts_codes.get(language, 'en')
            
            
            temp_file = "temp_speech.mp3"
            
            
            tts = gTTS(text=text, lang=tts_lang)
            tts.save(temp_file)
            
            
            playsound(temp_file)
            
           
            try:
                os.remove(temp_file)
            except Exception:
                pass
                
        except Exception as e:
            print(f"Text-to-speech error: {e}")

def main():
    
    parser = argparse.ArgumentParser(description="Multilingual Translation System")
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
    
    
    args = parser.parse_args()
    
    
    try:
        translator = Translator(
            model_path=args.model, 
            device=args.device
        )
    except Exception as e:
        print(f"Failed to initialize translator: {e}")
        return
    
    
    while True:
        print("\n--- Multilingual Translation System ---")
        print("1. Text Translation")
        print("2. Voice Translation")
        print("3. Exit")
        
        try:
            choice = input("Select an option (1-3): ")
            
            if choice == "1":
                
                source_lang = input("Enter source language (english/arabic/french/german): ")
                target_lang = input("Enter target language (english/arabic/french/german): ")
                text = input("Enter text to translate: ")
                
                
                translated_text = translator.translate_text(text, source_lang, target_lang)
                print(f"\nTranslated text: {translated_text}")
                
                
                speak = input("Would you like to hear the translation? (y/n): ")
                if speak.lower() == 'y':
                    translator.text_to_speech(translated_text, target_lang)
                
            elif choice == "2":
                
                source_lang = input("Enter source language (english/arabic/french/german): ")
                target_lang = input("Enter target language (english/arabic/french/german): ")
                
                
                print("Recording your voice...")
                audio_data, sample_rate = translator.record_audio()
                
                
                print("Converting speech to text...")
                text = translator.speech_to_text(audio_data, sample_rate)
                print(f"Recognized text: {text}")
                
                
                translated_text = translator.translate_text(text, source_lang, target_lang)
                print(f"Translated text: {translated_text}")
                
                
                translator.text_to_speech(translated_text, target_lang)
                
            elif choice == "3":
                print("Thank you for using the translation system!")
                break
            else:
                print("Invalid choice. Please try again.")
        
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()