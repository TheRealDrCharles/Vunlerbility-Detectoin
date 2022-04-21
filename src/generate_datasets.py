import sys
sys.path.append('..')
from src.includes import *

model_name = 'plstm'
data_set_name = 'inj'
home_path = '/home/swj/VD/'
embedding_path = home_path + 'embedding/'

#default mode / type of vulnerability
mode = "command_injection"
data_path = home_path + 'data/python_data_set/' + mode + '/'

#get the vulnerability from the command line argument


progress = 0
count = 0


### paramters for the filtering and creation of samples
restriction = [10000,5,6,10] #which samples to filter out
step = 5 #step lenght n in the description
fulllength = 200 #context length m in the description

mode2 = str(step)+"_"+str(fulllength) 

### hyperparameters for the w2v model
mincount = 10 #minimum times a word has to appear in the corpus to be in the word2vec model
iterationen = 300 #training iterations for the word2vec model
s = 200 #dimensions of the word2vec model
w = "withString" #word2vec model is not replacing strings but keeping them

#get word2vec model
w2v = embedding_path + "word2vec_"+w+str(mincount) + "-" + str(iterationen) +"-" + str(s)
w2vmodel = w2v + ".model"

#load word2vec model
if not (os.path.isfile(w2vmodel)):
    print("word2vec model is still being created...")
    sys.exit()


w2v_model = Word2Vec.load(w2vmodel)
print(w2v_model)
word_vectors = w2v_model.wv

#load data
with open(data_path + 'plain_' + mode, 'r') as infile:
    data = json.load(infile)
    
now = datetime.now() # current date and time
nowformat = now.strftime("%H:%M")
print("finished loading. ", nowformat)

allblocks = []

for r in data:
    progress = progress + 1

    for c in data[r]:

        if "files" in data[r][c]:                      
#             if len(data[r][c]["files"]) > restriction[3]:
#                 #too many files
#                 continue

            for f in data[r][c]["files"]:

#                 if len(data[r][c]["files"][f]["changes"]) >= restriction[2]:
#                     #too many changes in a single file
#                     continue

                if not "source" in data[r][c]["files"][f]:
                    #no sourcecode
                    continue

                if "source" in data[r][c]["files"][f]:
                    sourcecode = data[r][c]["files"][f]["source"]                          
                    if len(sourcecode) > restriction[0]:
                        #sourcecode is too long
                        continue

                allbadparts = []

                for change in data[r][c]["files"][f]["changes"]:

                    #get the modified or removed parts from each change that happened in the commit                  
                    badparts = change["badparts"]
                    count = count + len(badparts)

                #     if len(badparts) > restriction[1]:
                      #too many modifications in one change
                #       break

                    for bad in badparts:
                        #check if they can be found within the file
                        pos = myutils.findposition(bad,sourcecode)
                        if not -1 in pos:
                            allbadparts.append(bad)

                 #   if (len(allbadparts) > restriction[2]):
                      #too many bad positions in the file
                 #     break

                if(len(allbadparts) > 0):
                #   if len(allbadparts) < restriction[2]:
                  #find the positions of all modified parts
                    positions = myutils.findpositions(allbadparts,sourcecode)

                    #get the file split up in samples
                    blocks = myutils.getblocks(sourcecode, positions, step, fulllength)

                    for b in blocks:
                      #each is a tuple of code and label
                      allblocks.append(b)


keys = []

#randomize the sample and split into train, validate and final test set
for i in range(len(allblocks)):
    keys.append(i)
random.shuffle(keys)

cutoff = round(0.7 * len(keys)) #     70% for the training set
cutoff2 = round(0.85 * len(keys)) #   15% for the validation set and 15% for the final test set

keystrain = keys[:cutoff]
keystest = keys[cutoff:cutoff2]
keysfinaltest = keys[cutoff2:]

print("cutoff " + str(cutoff))
print("cutoff2 " + str(cutoff2))


with open(data_path + mode + '_dataset_keystrain', 'wb') as fp:
    pickle.dump(keystrain, fp)
with open(data_path + mode + '_dataset_keystest', 'wb') as fp:
    pickle.dump(keystest, fp)
with open(data_path + mode + '_dataset_keysfinaltest', 'wb') as fp:
    pickle.dump(keysfinaltest, fp)

TrainX = []
TrainY = []
ValidateX = []
ValidateY = []
FinaltestX = []
FinaltestY = []


words = list(w2v_model.wv.index_to_key)

print("Creating training dataset... (" + mode + ")")
for k in keystrain:
    block = allblocks[k]    
    code = block[0]
    token = myutils.getTokens(code) #get all single tokens from the snippet of code
    vectorlist = []
    for t in token: #convert all tokens into their word2vec vector representation
        if t in words and t != " ":
            vector = word_vectors[t]
            vectorlist.append(vector.tolist()) 
    TrainX.append(vectorlist) #append the list of vectors to the X (independent variable)
    TrainY.append(block[1]) #append the label to the Y (dependent variable)

print("Creating validation dataset...")
for k in keystest:
    block = allblocks[k]
    code = block[0]
    token = myutils.getTokens(code) #get all single tokens from the snippet of code
    vectorlist = []
    for t in token: #convert all tokens into their word2vec vector representation
        if t in words and t != " ":
            vector = word_vectors[t]
            vectorlist.append(vector.tolist()) 
    ValidateX.append(vectorlist) #append the list of vectors to the X (independent variable)
    ValidateY.append(block[1]) #append the label to the Y (dependent variable)

print("Creating finaltest dataset...")
for k in keysfinaltest:
    block = allblocks[k]  
    code = block[0]
    token = myutils.getTokens(code) #get all single tokens from the snippet of code
    vectorlist = []
    for t in token: #convert all tokens into their word2vec vector representation
        if t in words and t != " ":
            vector = word_vectors[t]
            vectorlist.append(vector.tolist()) 
    FinaltestX.append(vectorlist) #append the list of vectors to the X (independent variable)
    FinaltestY.append(block[1]) #append the label to the Y (dependent variable)

print("Train length: " + str(len(TrainX)))
print("Test length: " + str(len(ValidateX)))
print("Finaltesting length: " + str(len(FinaltestX)))
now = datetime.now() # current date and time
nowformat = now.strftime("%H:%M")
print("time: ", nowformat)


# saving samples
print('saving datasets')
with open(data_path + 'plain_' + mode + '_dataset-train-X_'+w2v + "__" + mode2, 'wb') as fp:
    pickle.dump(TrainX, fp)
with open(data_path + 'plain_' + mode + '_dataset-train-Y_'+w2v + "__" + mode2, 'wb') as fp:
    pickle.dump(TrainY, fp)
with open(data_path + 'plain_' + mode + '_dataset-validate-X_'+w2v + "__" + mode2, 'wb') as fp:
    pickle.dump(ValidateX, fp)
with open(data_path + 'plain_' + mode + '_dataset-validate-Y_'+w2v + "__" + mode2, 'wb') as fp:
    pickle.dump(ValidateY, fp)
with open(data_path + mode + '_dataset_finaltest_X', 'wb') as fp:
    pickle.dump(FinaltestX, fp)
with open(data_path + mode + '_dataset_finaltest_Y', 'wb') as fp:
    pickle.dump(FinaltestY, fp)
print("saved dataset")
