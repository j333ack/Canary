import xgboost as xgb
from datadog import *
import joblib

class Canary:
    def __init__(self):
        self.model = xgb.Booster()
        self.model.load_model("./models/treemodel.json")
        self.prevprediction = None
        self.previouspercent = None
        self.normalizer = joblib.load('./scalers/normalizer.pk1') 
        self.threshold = 0.3

    def prediction(self, filename):
        dailyinput = pd.read_csv(filename)
        normalizedinput = normalizeinput(dailyinput, self.normalizer)
        dmatrix = xgb.DMatrix(normalizedinput)
        probability = self.model.predict(dmatrix)[0]
        #these thresholds were taken from testing the model
        if (probability < 0.30):
            print("Low chance of rockburst next shift")
        elif probability < 0.40:
            print("Increased chance of rockburst next shift")
        elif probability < 0.50:
            print("High chance of rockburst next shift")
        else:
            print("Very high chance of a rockburst next shift")
            
         





