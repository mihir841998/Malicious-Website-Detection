import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pandas as pd
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def getTokens(input):
    tokensBySlash = str(input.encode('utf-8')).split('/')	#get tokens after splitting by slash
    allTokens = []
    for i in tokensBySlash:
        tokens = str(i).split('-')	#get tokens after splitting by dash
        tokensByDot = []
        for j in range(0,len(tokens)):
            tempTokens = str(tokens[j]).split('.')	#get tokens after splitting by dot
            tokensByDot = tokensByDot + tempTokens
        allTokens = allTokens + tokens + tokensByDot
    allTokens = list(set(allTokens))	#remove redundant tokens
    if 'com' in allTokens:
        allTokens.remove('com')	#removing .com since it occurs a lot of times and it should not be included in our features
        #print(allTokens)
    return allTokens

def init_process(fin,fout):
    outfile = open(fout,'a')    
    with open(fin) as f:
        try:
            for line in f:
                line = line.replace('"','')               
                line = line.split(',')                
                if line[1]=='bad\n':
                    x = [1,0]
                elif line[1] == 'good\n':
                    x = [0,1]                
                url = line[0]                				
                outline = str(x)+':::'+url
                #outline = str(outline.encode("utf-8"))                
                outfile.write(outline+"\n")             
        except Exception as e:
            print(str(e))    
    outfile.close()

def create_lexicon(fin):
    lexicon = []
    with open(fin, 'r', buffering=100000, encoding='utf-8') as f:
        try:
            counter = 1
            content = ''
            for line in f:
                counter += 1
                if (counter/10).is_integer():
                    tweet = line.split(':::')[1]
                    content += ' '+tweet
                    words = getTokens(content)
                    words = [lemmatizer.lemmatize(i) for i in words]
                    lexicon = list(set(lexicon + words))					
        except Exception as e:
            print(str(e))
    print(len(lexicon))
    with open('lexicon.pickle','wb') as f:
        pickle.dump(lexicon,f)




def convert_to_vec(fin,fout,lexicon_pickle):
    with open(lexicon_pickle,'rb') as f:
        lexicon = pickle.load(f)
    outfile = open(fout,'a')
    with open(fin, buffering=20000) as f:
        counter = 0
        for line in f:
            counter +=1
            label = line.split(':::')[0]
            tweet = line.split(':::')[1]
            current_words = getTokens(tweet.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]

            features = np.zeros(len(lexicon))

            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    # OR DO +=1, test both
                    features[index_value] += 1

            features = list(features)
            outline = str(features)+'::'+str(label)
            #outline = str(outline.encode("utf-8"))
            outfile.write(outline+'\n')
        print(counter)

def shuffle_data(fin,fout):
	df = pd.read_csv(fin, error_bad_lines=False)
	df = df.iloc[np.random.permutation(len(df))]
	print(df.head())
	df.to_csv(fout, index=False)
shuffle_data('train_data.csv','train_data_shuffled.csv')
shuffle_data('test_data.csv','test_data_shuffled.csv')
init_process('train_data_shuffled.csv','train_data_set.txt')
init_process('test_data_shuffled.csv','test_data_set.txt')
create_lexicon('train_data_set.txt')
convert_to_vec('train_data_set.txt','train_vec.txt','lexicon.pickle')
convert_to_vec('test_data_set.txt','test_vec.txt','lexicon.pickle')




        