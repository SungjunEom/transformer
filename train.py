from model import *
from dataset import *

def main():
    chat_data = ChatbotDataset('dataset/Q.txt', 'dataset/A.txt')
    

if __name__ == '__main__':
    main()