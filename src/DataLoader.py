import pickle
import csv
import os
import pandas as pd


# Separate '(', ')', '{', '}', '*', '/', '+', '-', '=', ';', '[', ']' characters.
def SplitCharacters(str_to_split):
    #Character_sets = ['(', ')', '{', '}', '*', '/', '+', '-', '=', ';', ',']
    str_list_str = ''
    
    if '(' in str_to_split:
        str_to_split = str_to_split.replace('(', ' ( ') # Add the space before and after the '(', so that it can be split by space.
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if ')' in str_to_split:
        str_to_split = str_to_split.replace(')', ' ) ') # Add the space before and after the ')', so that it can be split by space.
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
        
    if '{' in str_to_split:
        str_to_split = str_to_split.replace('{', ' { ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if '}' in str_to_split:
        str_to_split = str_to_split.replace('}', ' } ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if '*' in str_to_split:
        str_to_split = str_to_split.replace('*', ' * ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if '/' in str_to_split:
        str_to_split = str_to_split.replace('/', ' / ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
        
    if '+' in str_to_split:
        str_to_split = str_to_split.replace('+', ' + ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if '-' in str_to_split:
        str_to_split = str_to_split.replace('-', ' - ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
        
    if '=' in str_to_split:
        str_to_split = str_to_split.replace('=', ' = ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if ';' in str_to_split:
        str_to_split = str_to_split.replace(';', ' ; ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if '[' in str_to_split:
        str_to_split = str_to_split.replace('[', ' [ ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if ']' in str_to_split:
        str_to_split = str_to_split.replace(']', ' ] ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
        
    if '>' in str_to_split:
        str_to_split = str_to_split.replace('>', ' > ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
        
    if '<' in str_to_split:
        str_to_split = str_to_split.replace('<', ' < ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if '"' in str_to_split:
        str_to_split = str_to_split.replace('"', ' " ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
        
    if '->' in str_to_split:
        str_to_split = str_to_split.replace('->', ' -> ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if '>>' in str_to_split:
        str_to_split = str_to_split.replace('>>', ' >> ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if '<<' in str_to_split:
        str_to_split = str_to_split.replace('<<', ' << ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
    
    if ',' in str_to_split:
        str_to_split = str_to_split.replace(',', ' , ')
        str_list = str_to_split.split(' ')
        str_list_str = ' '.join(str_list)
        
        
    
    if str_list_str != '':
        return str_list_str
    else:
        return str_to_split
    
    
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
   
    
    
def getCFilesFromText(path):
    files_list = []
    file_id_list = []
    if os.path.isdir(path):
        for fpath,dirs,fs in os.walk(path):
            for f in fs:
                if os.path.splitext(f)[1] == '.c':
                    file_id_list.append(f)  
                if os.path.splitext(f)[1] == '.c':
                    with open(fpath + os.sep + f, encoding='latin1') as file:
                        lines = file.readlines()
                        file_list = []
                        for line in lines:
                            if '/*' in line: continue                # Remove the editing
                            if line != ' ' and line != '\n': # Remove sapce and line-change characters
                                sub_line = line.split()
                                new_sub_line = []
                                for element in sub_line:
                                    new_element = SplitCharacters(element)
                                    new_sub_line.append(new_element)
                                new_line = ' '.join(new_sub_line)
                                file_list.append(new_line)
                        new_file_list = ' '.join(file_list)
                        split_by_space = new_file_list.split()
                    files_list.append(split_by_space)
        return files_list, file_id_list
    

def getPythonFilesFromText(path):
    files_list = []
    file_id_list = []
    if os.path.isdir(path):
        for fpath,dirs,fs in os.walk(path):
            for f in fs:
                if os.path.splitext(f)[1] == '.c':
                    file_id_list.append(f)  
                if os.path.splitext(f)[1] == '.c':
                    with open(fpath + os.sep + f, encoding='latin1') as file:
                        lines = file.readlines()
                        file_list = []
                        for line in lines:
                            if '/*' in line: continue                # Remove the editing
                            if line != ' ' and line != '\n': # Remove sapce and line-change characters
                                sub_line = line.split()
                                new_sub_line = []
                                for element in sub_line:
                                    new_element = SplitCharacters(element)
                                    new_sub_line.append(new_element)
                                new_line = ' '.join(new_sub_line)
                                file_list.append(new_line)
                        new_file_list = ' '.join(file_list)
                        split_by_space = new_file_list.split()
                    files_list.append(split_by_space)
        return files_list, file_id_list
def LoadToken(path):
    file_list = []
    file_id_list = []
    if os.path.isdir(path):
        for fpath,dirs,fs in os.walk(path):
            for f in fs:
                if 'checkpoint' in f: continue
                if os.path.splitext(f)[1] == '.txt':
                    with open(fpath + os.sep + f, encoding='latin1') as file:
                        lines = file.readlines()
                        for line in lines:
                            if line != ' ' and line != '\n':
                                sub_line = line.split(',')
                                file_list.append(sub_line[1:])
                                file_id_list.append(sub_line[0])
        return file_list, file_id_list  
    


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