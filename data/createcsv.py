import os
import csv
import xml.etree.ElementTree as ET

title = set()

def getKeyList():
    tree = ET.parse(r'.\values_new\string_ar.xml')
    root = tree.getroot()
    file_dict = {}
    key_list = []
    for element in root.iter('string'):
        key = element.get('name')
        key_list.append(key)
    return key_list

# Gets a dictionary for each single file
def parseOneFile(file_name):
    tree = ET.parse(file_name)
    root = tree.getroot()
    file_dict = {}
    key_set = set()
    file_dict['file_name_'] = file_name
    for element in root.iter('string'):
        key = element.get('name')
        value = element.text
        if value == None:
            value = "#None#"
        elif value == "":
            value = "#Empty#"
        file_dict[key]=value
        key_set.add(key)
    global title
    title = title | key_set
    return file_dict

def processFiles():
    excel = open("translation_reverse.csv", 'w', newline = '', encoding='utf-8')  # This is the generated txt file path
    csv_file = csv.writer(excel)
    head = []
    all_keys = []   
    csv_lines = [] # This is the rows of of excel

    rootdir = r".\values_new"  # This is the path to place string_xx.xml files
    totle_keys = set() # This variable try to get union of all key sets of each file
    dirlist = os.listdir(rootdir)

    dictionary_list = []

    for i in range(0, len(dirlist)): # Traverse each file in the file folder
        path = os.path.join(rootdir, dirlist[i])
        head.append(dirlist[i])
        csv_row = []
        if os.path.isfile(path):
            csv_row.append(dirlist[i]) # This is file name
            dictionary_list.append(parseOneFile(path)) #  a list: [{dict of en}, {dict of zh}, ...]

    other_item = title - set(getKeyList()) # A small set
    all_keys = getKeyList() + list(other_item) # The complete list'

    head.insert(0, 'N/A')
    csv_lines.append(head) # Writes the first line of cvs, which should be: app_name, etc

    for key in all_keys:
        print("log read key: ", key)
        csv_row = []
        csv_row.append(key)
        #print("keyyyy: ", key)
        for i in range(1, len(head)):  # for each file, do the same thing
            if key in dictionary_list[i-1]:
                #print(dictionary_list[i-1][key])
                csv_row.append(dictionary_list[i-1][key])
            else:
                csv_row.append("## N/A ##")
        csv_lines.append(csv_row)

    csv_file.writerows(csv_lines)
    excel.close()

if __name__ == "__main__":
    processFiles()
    print("************* All Done! *************")