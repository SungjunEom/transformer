import sentencepiece as spm
import pandas as pd

dataset = 'dataset/' # Dataset location
chatbot_data = pd.read_csv(dataset+'ChatbotData.csv', sep=',')
chatbot_data['Q'].to_csv(dataset+'Q.txt',index=False,header=False)
chatbot_data['A'].to_csv(dataset+'A.txt',index=False,header=False)

spm.SentencePieceTrainer.Train('--input=dataset/Q.txt,dataset/A.txt --model_prefix=test --vocab_size=5000 \
                               --model_type=bpe --max_sentence_length=9999')