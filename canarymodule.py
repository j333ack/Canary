import xgboost as xgb
from datadog import *
import joblib

class Canary:
    def __init__(self):
        self.model = xgb.Booster()
        self.model.load_model("./models/modelnamehere")
        self.prevprediction = None
        self.previouspercent = None
        self.normalizer = joblib.load('.scalers/normalizer.pk1') 
        self.threshold = 0.3

    def predicton(self, filename):
        dailyinput = pandas.read_csv(filename)
        normalizedinput = normalizeinput(dailyinput, self.normalizer)
        probability = self.model.predict(normalizedinput)[0,1]
        #these thresholds were taken from testing the model
        if (probability < 0.30):
            print("Low chance of rockburst")
        else if probability < 0.40:
            print("Increased chance of rockburst")
        else if probability < 0.50:
            print("High chance of rockburst")
        else:
            print("Very high chance of a rockburst")
            
    def getpreviouspredict():
        if self.previouspercent = None
        if (self.previouspredict < 0.30):
            print("Low chance of rockburst")
        else if self.previouspercent < 0.40:
            print("Increased chance of rockburst")
        else if self. < 0.50:
            print("High chance of rockburst")
        else:
            print("Very high chance of a rockburst")
         





