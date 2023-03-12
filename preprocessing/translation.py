import pandas as pd
import numpy as np
import nltk
import re
from langdetect import detect
from googletrans import Translator
from deep_translator import GoogleTranslator
from collections import Counter
import gc


def split_text(text):
    nltk.download('punkt')
    
    sentences = nltk.sent_tokenize(text)
    new_text = []
    current_text = ''
    for sentence in sentences:
        if len(current_text + sentence) <= 4500:
            current_text += sentence
        else:
            if current_text[-1] != '.':
                last_period = current_text.rfind('.')
                if last_period != -1:
                    new_text.append(current_text[:last_period+1])
                    current_text = current_text[last_period+1:]
            new_text.append(current_text)
            current_text = sentence
    if current_text != '':
        new_text.append(current_text)
    
    
    if len(new_text[-1]) < 200 and len(new_text) > 1:
        new_text[-2] = new_text[-2] + new_text[-1]
        new_text.pop(-1)
    
    return new_text


def df_with_split_text(df):
    nltk.download('punkt')
    df_split_text = df.copy()

    for index in range(df_split_text.shape[0]):
        new_text = split_text(df_split_text.iloc[index]['text_preprocessed'])
        df_split_text.iloc[index]['text_preprocessed'] = new_text

    return df_split_text


def translate_text(text, index, df):
    translated_text = text
    
    italian_index = df[df["lang"] == "it"]["text_preprocessed"].index
    english_index = df[df["lang"] == "en"]["text_preprocessed"].index
    spanish_index = df[df["lang"] == "es"]["text_preprocessed"].index
    french_index = df[df["lang"] == "fr"]["text_preprocessed"].index
    deutsch_index = df[df["lang"] == "de"]["text_preprocessed"].index
    
    if index in deutsch_index : 
        lang_text = 'german'
        translated_text = GoogleTranslator(source=lang_text, target='en').translate(text=text)
    
    if index in italian_index : 
        lang_text = 'it'
        translated_text = GoogleTranslator(source=lang_text, target='en').translate(text=text)
    
    if index in spanish_index : 
        lang_text = 'es'
        translated_text = GoogleTranslator(source=lang_text, target='en').translate(text=text)
        
    if index in french_index : 
        lang_text = 'fr'
        translated_text = GoogleTranslator(source=lang_text, target='en').translate(text=text)
    
    if index in english_index :
        translated_text = text
    
    return translated_text


def translate_df(df):
    
    df_translated = df.copy()

    for index in range(df_translated.shape[0]):
        list_translated_text = []
        for i in range(len(df_translated.iloc[index]['text_preprocessed'])):
            translated_text_i = translate_text(df_translated.iloc[index]['text_preprocessed'][i], index)
            list_translated_text.append(translated_text_i)
        df_translated.iloc[index]['text_preprocessed'] = list_translated_text
        
    df_translated['text_preprocessed'] = df_translated['text_preprocessed'].apply(lambda x: ''.join(x))
    
    return df_translated