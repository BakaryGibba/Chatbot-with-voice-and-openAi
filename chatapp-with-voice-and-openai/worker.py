import openai
import requests
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your OpenAI API key
openai.api_key = 'OpenAI-Secrete-Key'

def speech_to_text(audio_binary):
    base_url = "https://sn-watson-stt.labs.skills.network"
    api_url = base_url + '/speech-to-text/api/v1/recognize'
    params = {'model': 'en-US_Multimedia'}
    body = audio_binary
    try:
        response = requests.post(api_url, params=params, data=audio_binary).json()
        text = 'null'
        while bool(response.get('results')):
            logger.info('speech to text response: %s', response)
            text = response.get('results').pop().get('alternatives').pop().get('transcript')
            logger.info('recognized text: %s', text)
            return text
    except requests.exceptions.RequestException as e:
        logger.error("Error during speech to text conversion: %s", str(e))
        return "Error during speech to text conversion"

def text_to_speech(text, voice=""):
    base_url = "https://sn-watson-tts.labs.skills.network"
    api_url = base_url + '/text-to-speech/api/v1/synthesize?output=output_text.wav'
    if voice != "" and voice != "default":
        api_url += "&voice=" + voice
    headers = {'Accept': 'audio/wav', 'content-Type': 'application/json'}
    json_data = {'text': text}
    try:
        response = requests.post(api_url, headers=headers, json=json_data)
        logger.info('text to speech response: %s', response)
        return response.content
    except requests.exceptions.RequestException as e:
        logger.error("Error during text to speech conversion: %s", str(e))
        return b""

def openai_process_message(user_message):
    prompt = "Act like a personal assistant. You can respond to questions, translate sentences, mathematics, summarize news, and give recommendations."
    max_retries = 5
    for attempt in range(max_retries):
        try:
            openai_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=4000
            )
            logger.info("Openai response: %s", openai_response)
            response_text = openai_response.choices[0].message['content']
            return response_text
        except openai.error.RateLimitError as e:
            logger.error("Rate limit exceeded: %s. Attempt %d/%d", str(e), attempt + 1, max_retries)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return "Rate limit exceeded. Please try again later."
        except openai.error.OpenAIError as e:
            logger.error("OpenAI error: %s", str(e))
            return f"An error occurred with the OpenAI API: {str(e)}"
        except Exception as e:
            logger.error("An unexpected error occurred: %s", str(e))
            return f"An unexpected error occurred: {str(e)}"
