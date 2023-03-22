import pickle
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from mangum import Mangum
import pandas as pd
from pydantic import BaseModel
from typing import List, Dict

app=FastAPI()
handler=Mangum(app)
traits = pd.read_csv(r'traits/clusterswithtraits.csv')
SET = 4

#####
class QuestionResp(BaseModel):
  bigfive: List[str]
  question: List[str]
  question_code: List[str]

class candidateresponse(BaseModel):
  EXT5: float
  EXT2: float
  EXT4: float
  EXT9: float
  EST9: float
  EST2: float
  EST6: float
  EST4: float
  AGR9: float
  AGR4: float
  AGR7: float
  AGR10: float
  CSN3: float
  CSN9: float
  CSN6: float
  CSN7: float
  OPN2: float
  OPN10: float
  OPN7: float
  OPN1: float

class predict_response(BaseModel):
  cluster: int
  personality: str
  traits: str
  overallscore: int
  results:  Dict[str, float]

# ,response_model=List[QuestionResp]

@app.get("/questions",response_model=List[QuestionResp])
async def read_item():
  # GET your questions here
  # get your 20 question
  # interate over twenty question

  resp = [{'BigFive': 'Extraversion',
  'Question Code': 'EXT2',
  'Question': "I don't talk a lot."},
 {'BigFive': 'Extraversion',
  'Question Code': 'EXT4',
  'Question': 'I keep in the background.'},
 {'BigFive': 'Extraversion',
  'Question Code': 'EXT5',
  'Question': 'I start conversations.'},
 {'BigFive': 'Extraversion',
  'Question Code': 'EXT9',
  'Question': "I don't mind being the center of attention."},
 {'BigFive': 'Neuroticism',
  'Question Code': 'EST2',
  'Question': 'I am relaxed most of the time.'},
 {'BigFive': 'Neuroticism',
  'Question Code': 'EST4',
  'Question': 'I seldom feel blue.'},
 {'BigFive': 'Neuroticism',
  'Question Code': 'EST6',
  'Question': 'I get upset easily.'},
 {'BigFive': 'Neuroticism',
  'Question Code': 'EST9',
  'Question': 'I get irritated easily.'},
 {'BigFive': 'Agreeableness',
  'Question Code': 'AGR4',
  'Question': "I sympathize with others' feelings."},
 {'BigFive': 'Agreeableness',
  'Question Code': 'AGR7',
  'Question': 'I am not really interested in others.'},
 {'BigFive': 'Agreeableness',
  'Question Code': 'AGR9',
  'Question': "I feel others' emotions."},
 {'BigFive': 'Agreeableness',
  'Question Code': 'AGR10',
  'Question': 'I make people feel at ease.'},
 {'BigFive': 'Conscientiousness',
  'Question Code': 'CSN3',
  'Question': 'I pay attention to details.'},
 {'BigFive': 'Conscientiousness',
  'Question Code': 'CSN6',
  'Question': 'I often forget to put things back in their proper place.'},
 {'BigFive': 'Conscientiousness',
  'Question Code': 'CSN7',
  'Question': 'I like order.'},
 {'BigFive': 'Conscientiousness',
  'Question Code': 'CSN9',
  'Question': 'I follow a schedule.'},
 {'BigFive': 'Openness',
  'Question Code': 'OPN1',
  'Question': 'I have a rich vocabulary.'},
 {'BigFive': 'Openness',
  'Question Code': 'OPN2',
  'Question': 'I have difficulty understanding abstract ideas.'},
 {'BigFive': 'Openness',
  'Question Code': 'OPN7',
  'Question': 'I am quick to understand things.'},
 {'BigFive': 'Openness',
  'Question Code': 'OPN10',
  'Question': 'I am full of ideas.'}]

  # resp.append(QuestionResp(bigfive='str',question="str", question_code="str"))
  bigfive_values = [i['BigFive'] for i in resp]
  question_values = [i['Question'] for i in resp]
  question_codes = [i['Question Code'] for i in resp]
  resp_return = QuestionResp(bigfive=bigfive_values,question=question_values,question_code=question_codes)
  return resp_return
#####

# Load the saved model
clf_path =  "models/pp19.4.pkl"
with open(clf_path, 'rb') as file:
    classifier = pickle.load(file)

@app.post("/predict",response_model=predict_response)
def predict_cluster(data:candidateresponse):
  data = data.dict()
  print(data)
  df = pd.DataFrame([data], columns=data.keys())
  print("dataframe created",df)
  y_pred = classifier.predict(df)
  my_cluster = y_pred[0]
  print("My Cluster",my_cluster)
  my_df = traits[traits['SET']==SET]
  my_df = my_df[my_df['Clusters'] == my_cluster]
  my_df = my_df[['SET', 'Clusters', 'Extroversion', 'Neurotic', 'Agreeable',
                 'Conscientious', 'Openness', 'Label', 'Traits']]
  personality = my_df['Label'].to_string(index=False)
  my_df = my_df.reset_index(drop=True)
  overall_score = (my_df['Openness'] +
                   my_df['Conscientious'] +
                   my_df['Extroversion'] +
                   my_df['Agreeable'] +
                   my_df['Neurotic']) / 5
  my_sums = pd.DataFrame()
  my_sums['Extroversion'] = my_df['Extroversion']
  my_sums['Agreeable'] = my_df['Agreeable']
  my_sums['Conscientious'] = my_df['Conscientious']
  my_sums['Openness'] = my_df['Openness']
  my_sums['Neurotic'] = my_df['Neurotic']
  my_sums = my_sums[['Agreeable', 'Conscientious', 'Openness', 'Extroversion', 'Neurotic']]
  my_results = my_sums.T
  my_results = my_results.reset_index().rename(columns={'index': 'Personality', int(0): "Score"})
  my_results = my_results.set_index('Personality')['Score'].to_dict()
  mytraits = " ".join(str(i) for i in my_df['Traits'])
  predict_resp = predict_response(cluster=my_cluster,personality=personality,traits=mytraits,overallscore=overall_score,results=my_results)
  return predict_resp

if __name__=="__main__":
  uvicorn.run(app,host="127.0.0.1",port=9999,reload=True)

