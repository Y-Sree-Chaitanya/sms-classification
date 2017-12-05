# --- API Controllers ---
from bo.Base import BO_Base

# --- Logger ---
from logger import logger_error, logger_info

import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import os.path as osp

class BO_Prediction(BO_Base):

    def __init__(self):
        pass

    def sms_classifier(self, prediction_params, model):

        # ----> Machine Learning Code goes here
        logger_info("Code block at Product Prediction fn.")
        smsText = prediction_params["text"][0]
        models_dir = "tensorflow_model/"
        # model_name = "fasttext_keras.sav"
        class_mappings = {0: 'ham', 1: 'info', 2: 'spam'}
        # classifier = load_model(osp.join(models_dir, model_name))
        sms_class = "Gibberish"
        try:
            sms_text = self.preprocess(smsText)
            with open(osp.join(models_dir, 'tokenizer.pickle'), 'rb') as handle:
                tokenizer = pickle.load(handle)
            sms_encoded = tokenizer.texts_to_sequences([sms_text])
            sms_encoded = pad_sequences(sequences=sms_encoded, maxlen=140)
            predicted_probabilities = model.predict(sms_encoded)
            sms_class = class_mappings[np.argmax(predicted_probabilities[0])]
        except Exception as e:
            logger_error(str(e))
        finally:
            return sms_class

    def preprocess(self, text):
        text = text.replace("' ", " ' ")
        signs = set(',.:;"?!#!;@#$%^+=_&*()<>//\\-[]|{}.`~')
        prods = set(text) & signs
        if not prods:
            return text
        for sign in prods:
            text = text.replace(sign, ' {} '.format(sign))
        return text