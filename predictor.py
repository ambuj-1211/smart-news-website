import sys
import torch
from model import Model
from data import PreProcessor
import joblib

text = sys.argv[1]

preproc = PreProcessor()
text = preproc.forward(text)

vectorizer = joblib.load('tfidf_for_fakenews.pkl')
text = vectorizer.transform([text])

print(text.toarray())

text = torch.Tensor(text.toarray())

model = Model()
model.load_state_dict(torch.load('fake_model_state_dict.pt'))
model.eval()

y_pred = []

for i, data in enumerate(text):
    y_pred.append(model(data))

print(y_pred)