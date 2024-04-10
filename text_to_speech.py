import os
import playsound
import time
import sys
import ctypes
import datetime
import json
import re
import threading
import queue
from time import strftime
from gtts import gTTS

import pyttsx3
engine = pyttsx3.init()

from googletrans import Translator
translator = Translator()

def speak_from_text(text, language, directory=None):
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("sound.mp3")
    file_dir = os.path.dirname('./sound.mp3')
    file_path = file_dir + '/sound.mp3'

def speak_vietnamese(text, language, directory=None):

    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("sound.mp3")
    file_dir = os.path.dirname('./sound.mp3')
    file_path = file_dir + '/sound.mp3'
    # playsound.playsound(file_path)
    # os.remove("sound.mp3")
    
def speak_english(text):
    """ RATE"""
    rate = engine.getProperty('rate')   # getting details of current speaking rate
    # print (rate)                        #printing current voice rate
    engine.setProperty('rate', 125)     # setting up new voice rate


    """VOLUME"""
    volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
    # print (volume)                          #printing current volume level
    engine.setProperty('volume',1.0)    # setting up volume level  between 0 and 1

    """VOICE"""
    voices = engine.getProperty('voices')       #getting details of current voice
    #engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
    engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for female

    engine.say(text)
    # engine.say('My current speaking rate is ' + str(rate))
    
    # Save to file
    engine.save_to_file(text, 'speech.mp3')
    
    engine.runAndWait()
    engine.stop()

def translate_text(text="", input='en', output='vi'):
    translated_text = translator.translate(text, src=input, dest=output).text
    return translated_text

def translate_to_vn(text):
    translated_text = translator.translate(text, src='en', dest='vi').text
    return translated_text

def reading_thread(text, mode):
    if mode == 3:
        threading.Thread(target=speak_vietnamese, args=(text,)).start()
    return 