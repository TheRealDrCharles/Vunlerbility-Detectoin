import pickle
import csv
import os
import pandas as pd

def verbose(self, msg):
    ''' Verbose function for print information to stdout'''
    print('[INFO]', msg)
        
def SavedPickle(path, file_to_save):
    with open(path, 'wb') as handle:
        pickle.dump(file_to_save, handle)

def Save3DList(save_path, list_to_save):
    with open(save_path, 'w', encoding='latin1') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerows(list_to_save)
        

def Save2DList(save_path, list_to_save):
    with open(save_path, 'w', encoding='latin1') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(list_to_save)
        
def ListToCSV(list_to_csv, path):
    df = pd.DataFrame(list_to_csv)
    df.to_csv(path, index=False)

def LoadPickleData(path):
    with open(path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data
   
def LoadToken(path):
    file_list = []
    file_id_list = []
    if os.path.isdir(path):
        for fpath,dirs,fs in os.walk(path):
            for f in fs:
                if os.path.splitext(f)[1] == '.txt':
                    with open(fpath + os.sep + f, encoding='latin1') as file:
                        lines = file.readlines()
                        for line in lines:
                            if line != ' ' and line != '\n':
                                sub_line = line.split(',')
                                file_list.append(sub_line[1:])
                                file_id_list.append(sub_line[0])
        return file_list, file_id_list  
    
def LoadData(data_path):
    ''' Load data for training/validation'''
    verbose('Loading data from '+ os.getcwd() + os.sep + data_path + '....')
    total_list, total_list_id = LoadToken(data_path)
    verbose("The length of the loaded data list is : " + str(len(total_list)))
    return total_list, total_list_id


# Data labels are generated based on the sample IDs. All the vulnerable function samples are named with CVE IDs.    
def GenerateLabels(input_arr):
    temp_arr = []
    for func_id in input_arr:
        temp_sub_arr = []
        if "cve" in func_id or "CVE" in func_id:
            temp_sub_arr.append(1)
        else:
            temp_sub_arr.append(0)
        temp_arr.append(temp_sub_arr)
    return temp_arr