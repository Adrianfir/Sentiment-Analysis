import joblib

model = joblib.load('saved_pipeline.pkl')
pred = model.predict("this is the best experience I could wish for my enemy")
