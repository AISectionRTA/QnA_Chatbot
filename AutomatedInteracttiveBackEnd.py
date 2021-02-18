
'''
#########
#### command line arguments 
1. qestion set in excel file line separated
2. model location ie npz file
3. db location

'''

"""Interactive mode for the tfidf retriever module."""

import glob
import logging
import os
import sqlite3
import sys

import pandas as pd
import prettytable

######################################################

currentDirectory = os.getcwd()
sys.path.append(currentDirectory)
from mahboub import retriever
######################################################
# default npz and database
npzFolderLocation_default = 'npz/*'
mylist = [f for f in glob.glob(npzFolderLocation_default)]
npzLocation_default = mylist[0]
dbLocation_default = 'db/consolidated.db'
CSVName = "csv/consolidated_services.xlsx"
######################################################
encodinfType1 = "utf-8"
data = pd.read_excel(CSVName)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


logger.info('Initializing ranker...')
ranker = retriever.get_class('tfidf')(tfidf_path=npzLocation_default)


# ------------------------------------------------------------------------------
# Drop in to interactive
# ------------------------------------------------------------------------------

# this function get the data from db bases on doc id
def getDataFromDB(doc_names, dbFile):
    conn = sqlite3.connect(dbFile)
    cursor = conn.cursor()
    # cursor.execute("SELECT %s FROM documents where %s=?" % (text, id), (doc_names,))
    # cursor.execute("SELECT count(*) FROM documents")
    cursor.execute("SELECT %s FROM documents where %s=?" % ('text', 'id'), (doc_names,))

    result = cursor.fetchone();
    conn.close()
    # print(str(result))
    return str(result)


# this function get the data from csv or excel
def getDataFromCsv(documentId):
    # encodinfType1="iso8859_16"
    # encodinfType1="iso8859_15"  #cp866
    # encodinfType1="cp866"
    result = data[data['ID'] == documentId]
    # resultHtml = createAdocument(result["Text"].to_string(index=False))
    if len(result) > 0:
        return result["Text"].iloc[0], True
    else:
        return getDataFromDB(documentId, dbLocation_default), False
    # return resultHtml


# arguments query, number of results tfidf path which is npz file database file
# it return single  or k results
def process_and_getData(query, k):
    doc_names, doc_scores = ranker.closest_docs(query, k)
    table = prettytable.PrettyTable(
        ['Rank', 'Doc Id', 'Doc Score']
    )
    documentText = ""
    resultList = []
    flag = True
    for i in range(len(doc_names)):        
        documentText, flag = getDataFromCsv(doc_names[i])        
        resultList.append([doc_names[i], doc_scores[i], documentText])

    return resultList, flag


###############################
# assumption is  one line
def process(query, k=1):
    doc_names, doc_scores = ranker.closest_docs(query, k)
    table = prettytable.PrettyTable(

        ['Rank', 'Doc Id', 'Doc Score']
    )
    documentText = ""
    for i in range(len(doc_names)):
        documentText = getDataFromDB(doc_names[i])
####################################################