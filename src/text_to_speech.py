from gtts import gTTS
import io


def text_to_speech(text, language='en',):
    language = 'en'

    # Create a BytesIO object to hold the audio file in memory
    audio_bytes = io.BytesIO()

    # Generate the audio and save it to the BytesIO object
    myobj = gTTS(text=text, lang=language, tld='co.in', slow=False)
    myobj.write_to_fp(audio_bytes)

    # Play the audio directly in the notebook from the BytesIO object
    # Audio(audio_bytes.getvalue(), autoplay=True)
    return audio_bytes.getvalue()

if __name__ == '__main__':
    mytext = "Remember, its okay to feel sad sometimes. Emotions are a natural part of life, and it's important to acknowledge and process them. Try to be gentle with yourself and give yourself the time and space you need to heal. If you're struggling to cope, know that it's okay to reach out for support. You're not alone in this journey, and together we can work through these feelings."

    print(text_to_speech(mytext))
