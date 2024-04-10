import streamlit as st

from text_to_speech import *

st.title('Text to Speech')

def clear_text():
    st.session_state.my_text = ""

languages = {
    "English": "en",
    "Vietnamese": "vi",
    "French": "fr",
    "Chinese": "zh-cn",
    "German": "de",
    "japanese": "ja",
    "Korean": "ko",
}

# Select box: Language
selected_language = st.selectbox('Choose an output language:', list(languages.keys()))

st.session_state.selected_language = languages[selected_language]

if st.button('Submit'):
    selected_code = languages[selected_language]
    st.write(f"You selected: {selected_code}")
    # st.session_state.selected_language = languages[selected_language]

input = st.text_input("Enter English text here:", on_change=clear_text)
if input:
    output = translate_text(input, 'en', st.session_state.selected_language)
    st.text_area(label="Output:", value=output, height=200)
    
    audio_button = st.button("Audio")
    if audio_button:
        input_text = translate_text(input, "en", st.session_state.selected_language)
        # print(type(st.session_state.selected_language))
        input_audio = speak_from_text(input_text, language = st.session_state.selected_language)
        
        st.audio("sound.mp3", format="audio/mpeg", loop=False)

