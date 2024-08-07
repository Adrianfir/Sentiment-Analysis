import joblib

model = joblib.load('saved_pipeline.pkl')
pred = model.predict(["this is the bad experience"])
print(pred)
