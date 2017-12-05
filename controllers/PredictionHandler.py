import json
import tornado
# --- API Controllers ---
from controllers.RestClass import RestClass
from bo.Prediction import BO_Prediction

# --- Logger ---
from logger import logger_error, logger_info, logger_warning


class PredictionHandler(RestClass):
    def __init(self):
        self.name = 'PredictionHandler'
        self.model = self.config['keras_model']
        self.bPrediction = BO_Prediction()

    def get(self):
        self.__init()
        self.write("Code block reached to Prediction Handler")
        logger_info("Code @ " + self.name + " GET")

    def post(self):
        self.__init()
        logger_info("Code @ " + self.name + " POST")
        a = self.request
        body = tornado.escape.json_decode(self.request.body)
        prediction_result = self.bPrediction.sms_classifier(body,  self.model)
        json_response = json.dumps({'result': str(prediction_result)})
        self.write(json_response)
