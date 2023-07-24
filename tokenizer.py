import sentencepiece as spm
import pandas as pd

dataset = 'dataset/ChatbotData.csv' # Dataset location
chatbot_data = pd.read_csv(dataset, sep=',')