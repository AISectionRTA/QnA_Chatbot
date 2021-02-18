# Importing Relevant Libraries
import random
import re
import string
import warnings
from string import punctuation

import nltk
import numpy as np
from autocorrect import spell
from bs4 import BeautifulSoup
from flask import Flask, request
from flask_cors import CORS
from fuzzywuzzy import fuzz
# !sudo pip install gensim
from gensim.models import KeyedVectors
from googletrans import Translator
from nltk.stem import WordNetLemmatizer
from scipy import spatial
import pandas as pd

from AutomatedInteracttiveBackEnd import *

# Initializing NLTK Lemmatizer
lemmatizer = WordNetLemmatizer()

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

CSVName = "csv/consolidated_services.xlsx"
data = pd.read_excel(CSVName)
train_df = pd.read_excel(CSVName)  # or default params

encodinfType1 = "utf-8"

df_service_names = pd.DataFrame(data["ServiceName"], columns=["ServiceName"])
no_of_predictions = 3

# Lemmatizer Initialization. We should avoid stemming as stemming sometimes changes the meaning of the stemmed word
lemmer = nltk.stem.WordNetLemmatizer()

# Loading word2vec WordEmbeddings Models to form feature vectors
word2vec_file = "./glove2word2vec.txt"
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_file)

# Some of the recommended categories from the services to give a brief idea about the topics
common_category_names = ['Process', 'Service Fees', 'Description', 'Required Documents', 'Terms and Conditions',
                         'Validity', 'Eligibility', 'Process Time', 'Average Waiting Time']

# Data on which the model will be evaluated and similarity scores will be generated.
# This data has to be clean to avoid irrelevancies in the data.
sent_tokens_q = train_df['CleanedText']

word_tokens_q = nltk.word_tokenize(' '.join(sent_tokens_q))

# Some of the Greetings text which User can Input. Can be removed or added more.
# Just need to add the inputs and reponse messages in the list and algorithm will randomly pick any response for the input
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


def clean_data_before_training(text):
    """
    This function perform the data cleaning operations. Althrough most of the data cleaning is already done in the Step 1,
    If something is still left and needs to be cleaned. This functions takes uncleaned text and perform all the cleaning operations
    and returns cleaned text (String)
    Args:
        text (): Uncleaned Text (data)

    Returns: Cleaned text (String)
    """
    text = re.sub(r"[Qq]uestion(.*?)[Aa]nswer.*?[:,]", '', text).strip()
    text = re.sub(r"[Qq]uestion(.*?)\[?,]", '', text).strip()
    text = re.sub(r"[Aa]nswer.*?[:,]", '', text).strip()
    text = re.sub(r"[0-9]\.\s", "", text)
    text = re.sub(r"[\u2022|\u2023|\u25E6|\u2043|\u2219]", '', str(text)).strip()
    text = re.sub(r"\s\s+", ' ', text).strip()
    return text


# Checking for greetings (optional)
def greeting(sentence):
    """
    If user's input is a greeting, return a greeting response
    Args:
        sentence (): USer Input
    """
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# This function lemmatize tokens (words)
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


# create punctuation dictionary to faster retrieval operations and avoid looping again and again
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def clean_model_output(doc_id):
    """
    This functions cleans the output we are getting from the model and structure that to display on the webpage
    Args:
        doc_id (): Unique Document ID for which the content has to be shown back to the user

    Returns:
    Most Relevant Formatted Content/Answer which will be returned to the User/Customer.
    """
    filtered_data = data[data.ID == doc_id]
    service_name = filtered_data.ServiceName.iloc[0]
    category = filtered_data.Category.iloc[0]
    response = filtered_data.Text.iloc[0]

    category = re.sub(r"[Qq]uestion(.*?)\?", '', category).strip()
    category = re.sub(r"[Aa]nswer.*?[:,]", '', category).strip()
    category = category.strip(punctuation)

    """
    If the Identified Category of the Answer is FAQs then format of the output will be slightly changed.
    Most relevant Question from the FAQs will also be displayed with the identified Answer.
    """
    if category != "FAQs":
        response = re.sub(r"[Qq]uestion(.*?)[Aa]nswer.*?[:,]", '', response).strip()
        response = re.sub(r"[Qq]uestion(.*?)[\?,]", '', response).strip()
        response = re.sub(r"[Aa]nswer.*?[:,]", '', response).strip()
        response = response.strip()

    if not bool(BeautifulSoup(response, "html.parser").find()):
        response = response.strip(punctuation)
        regexp_1 = re.compile(r'[0-9]\.\s')
        if regexp_1.search(response):
            response = "" + re.sub(r"[0-9]\.\s", "<br>&bull;&nbsp;", response) + ""
        regexp_2 = re.compile(r"[\u2022|\u2023|\u25E6|\u2043|\u2219]")
        if regexp_2.search(response):
            response = re.sub(r"[\u2022|\u2023|\u25E6|\u2043|\u2219]", '<br>&bull;&nbsp;', str(response)).strip()

    if category == "FAQs":
        response = response.replace("<ul>", "").replace("</ul>", "").replace("<li>", "").replace("</li>", "")
        response = re.sub(r"[Qq]uestion\s[0-9]*", "Question", response)
        response = re.sub(r"[Aa]nswer\s[0-9]*", "Answer", response)

        parsed_resp = "<br><b>Service Name:</b> {}<br><b>Category:</b> {}<br> {}" \
            .format(service_name, category, response)
    else:
        parsed_resp = "<br><b>Service Name:</b> {}<br><b>Category:</b> {}<br><b>Answer:</b> {}" \
            .format(service_name, category, response)

    return parsed_resp


def LemNormalize(text):
    """
    This functions performs lemmatization on text fields by first breaking the text into tokens
    Args:
        text (): Text data (String)

    Returns:
    Lemmatized Sentence
    """
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


def NormalizeData(score_list):
    """
    This functions normalizes iterable Object using MinMax Scaler Technique to avoid unwanted boosting to predicted values.
    Args:
        score_list (): Iterable Object (List, Series)

    Returns:
    Normalized Iterable Object within range [0 - 1]
    """
    return (score_list - np.min(score_list)) / (np.max(score_list) - np.min(score_list))


def response_word2vec(user_response, specific_sent_tokens):
    """
    This function finds the similarity between user search string and the content of the filtered service
    using word2vec embeddings model and return the top matches along with the similarity score
    Args:
        user_response (): User Search String
        specific_sent_tokens (): Filtered Sentences/Data of Specific Service

    Returns:
    List of Sentences in sorted form with their similarity score
    """
    # using the glove2word2vec model created earlier!
    # (similarity between glove2word2vec embeddings)

    STOP_WORDS = nltk.corpus.stopwords.words()

    index2word_set = set(word2vec_model.wv.index2word)

    def avg_feature_vector(sentence, model, num_features, index2word_set):
        words = [i for i in sentence.lower().split() if i not in STOP_WORDS]
        feature_vec = np.zeros((num_features,), dtype='float32')
        n_words = 0
        for word in words:
            if word in index2word_set:
                n_words += 1
                feature_vec = np.add(feature_vec, word2vec_model[word])
        if (n_words > 0):
            feature_vec = np.divide(feature_vec, n_words)
        # mean used as statistical measure
        return feature_vec

    s1_afv = avg_feature_vector(user_response.lower(), model=word2vec_model,
                                num_features=50, index2word_set=index2word_set)  # resp

    global similarities
    similarities = []

    for sent in specific_sent_tokens:
        s2_afv = avg_feature_vector(sent.lower(), model=word2vec_model,
                                    num_features=50, index2word_set=index2word_set)

        sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)  # 1 - cosine DISTANCE !!

        similarities.append((sent, sim))

    # if sim > most_similar[1]:
    #    most_similar = (sent, sim)

    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    most_similar = similarities[0]
    # may set threshold arbitrarily
    if pd.isnull(most_similar[1]) or most_similar[1] < 0.4:
        return np.nan

    return similarities


def introduction(name):
    flag = True

    intro = "Hi there! I am Mahboub v2.0 and I'll be your customer service representative for RTA. Can I know your name please?"

    name = ' '.join([i.capitalize() for i in name.split()])
    follow_up = "Welcome to the RTA website, <b>" + name + "</b>! I will answer your queries \
    related to RTA. Feel free to ask a question and/or hover over to any of our information categories. \
    If you want to exit, type thanks / Bye! </br> </br>"

    return follow_up


def get_model_ids_dict(nearestQ_word2model, data_df, column_name):
    """
    This function prepare dictionary from the list of tuples
    Args:
        nearestQ_word2model (): List of Tuple  - Prediction from Model
        data_df (): Data from which filtration and Ids have to be fetched
        column_name (): Name of Column on which filtration have to be applied

    Returns:
        ids_dict with key="document_id" and value="doc_score"
    """
    ids_dict = {}
    for each in nearestQ_word2model:
        tmp_id = data_df[data_df[column_name] == each[0]].ID.iloc[0]
        if tmp_id:
            if tmp_id not in ids_dict:
                ids_dict[str(tmp_id)] = each[1]
            else:
                continue
        else:
            continue
    return ids_dict


# Main Logic of Chatbot for running the complete algorithm and finding the most relevant answer corresponding to the search string.
# This function is designed using Factory Design Pattern - which changes the logic and calling of next functions according to the current results.
def chatbot(user_input, servicename=""):
    """
    This function handles the main logic of the complete chatbot.
    Every Request of the user is passed to this function and it automatically decides the flow of the user journey.
    This function is designed using Factory design pattern and tried to incorporate every single possibility to avoid breaking of user journey.
    Args:
        user_input (): User Search String
        servicename (): Service Name from which query has to be served

    Returns:
    Output which user is going to see on front-end
    """
    global helpful_rank
    helpful_rank = 0
    # INCLUDE FAVORABLE RESPONSE ?!
    # change answer forever

    user_response = user_input  # in any language

    translator = Translator()

    # src = translator.detect(user_response).lang
    src = "en"

    if src != 'en':
        trans = True
    else:
        trans = False

    user_response = translator.translate(user_response, src=src, dest='en').text
    user_response = user_response.lower()

    # auto-correction !
    user_response = ' '.join([spell(i) for i in user_response.split()])

    trans = False
    if (user_response != 'bye'):
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False

            return translator.translate('You are welcome!', src='en', dest=src).text
        else:
            g = ""
            if greeting(user_response) != None:
                g = translator.translate(greeting(user_response), src='en', dest=src).text + '\n'

            if g != "" and len(user_response.split()) == 1:
                return g

            """
            Calling multiple models to get the results corresponding to user query and later combining 
            them for better precision and accuracy
            """

            # Calling the TFIDF Matrix Model to get the relevant results and top match results corresponding to the User Search String
            nearestQ, flag = process_and_getData(user_response, 350)
            model_filtered_dict = {}

            # Getting predicted results from word2vec Embeddings model
            if servicename and servicename == "":
                nearestQ_word2model = response_word2vec(user_response, sent_tokens_q)[0][0]
            else:
                if str(servicename).lower() in list(train_df["ServiceName"].apply(lambda x: str(x).lower())):
                    sent_tokens_filtered = train_df[train_df["ServiceName"] == servicename]["CleanedText"]
                    nearestQ_word2model_all = response_word2vec(user_response, sent_tokens_filtered)
                    model_filtered_dict = get_model_ids_dict(nearestQ_word2model_all, train_df, "CleanedText")
                else:
                    nearestQ_word2model = response_word2vec(user_response, sent_tokens_q)[0][0]

            filtered_results = []
            # Filtering results of the identified Service Name coming from the model.
            if servicename != "":
                for each in nearestQ:
                    if servicename == str(each[0]).split("_")[0]:
                        filtered_results.append((each[0], each[1]))

            """
            The code below is responsible of combining the predictions of both the models
            1:- word2vec Embeddings model
            2:- TFIDF Matrix Model
            """
            tfidf_filtered_dict = {}
            if len(filtered_results) > 0:
                unnormalized_scores = [res[1] for res in filtered_results]
                normalized_filtered_results = NormalizeData(unnormalized_scores)
                for i in range(len(filtered_results)):
                    filtered_results[i] = (filtered_results[i][0], normalized_filtered_results[i])
                tfidf_filtered_dict = get_model_ids_dict(filtered_results, data, "ID")

            cumulative_dict = {k: model_filtered_dict.get(k, 0) + tfidf_filtered_dict.get(k, 0) for k in
                               set(model_filtered_dict) | set(tfidf_filtered_dict)}

            cumulative_id = ""
            # Getting Unique ID Details from word2vec Embeddings Model and TFIDF Matrix
            if len(cumulative_dict) > 0:
                cumulative_id = sorted(cumulative_dict, key=cumulative_dict.get, reverse=True)[0]

            translate_button = "</a></br></br> <button class=\"button\" type=\"button\" onclick=\"translater()\">" \
                               "Translate to English</button> </br></br>"

            if len(nearestQ) < 1:
                if trans:
                    return translator.translate("Sorry, I don't understand you. Please try again!", src='en',
                                                dest=src).text + translate_button
                return translator.translate("Sorry, I don't understand you. Please try again!", src='en', dest=src).text

            try:

                """
                The code below decides which 
                """

                # Getting Unique ID Details from TFIDF Matrix Model
                if len(filtered_results) > 0:
                    if cumulative_id != "":
                        results_id = cumulative_id
                    else:
                        temp_df = data[data['ID'] == filtered_results[0]]
                        results_id = list(temp_df["ID"])[0]
                else:
                    if servicename != "":
                        message, service_flag, number_of_matches = find_service_name(user_input)
                        if service_flag:
                            return "<br>Sorry, I couldn't not find an answer within selected service name. " \
                                   "<br>I have found more than <b>" + str(
                                number_of_matches) + "</b> services related to your search. " \
                                                     "It may help to refine your query with some more details. " \
                                                     "Here are some of the top matches.<br>" + message

                        if trans:
                            return translator.translate("Sorry, I don't understand you. Please try again!",
                                                        src='en',
                                                        dest=src).text + translate_button
                        return translator.translate("Sorry, I don't understand you. Please try again!", src='en',
                                                    dest=src).text
                    else:
                        temp_df = data[data['ID'] == nearestQ[0][0]]
                        results_id = list(temp_df["ID"])[0]

                # Calling function to get the answer of the identified doc_id and formatting the answer in structured format.
                parsed_resp = clean_model_output(results_id)

                ans = g + parsed_resp

                # EXTRA STRING FOR TRANSLATOR FUNCTIONALITY, which is removed during translate_func API fire
                if trans:
                    return ans + translate_button
                return ans
            except:
                if trans:
                    return translator.translate("Sorry, I don't understand you. Please try again!", src='en',
                                                dest=src).text + translate_button
                return translator.translate("Sorry, I don't understand you. Please try again!", src='en', dest=src).text
    else:
        flag = False
        return "Mahboub v2.0: " + translator.translate("Bye! take care..", src='en', dest=src).text


STOP_WORDS = nltk.corpus.stopwords.words()


def find_service_name(userinput):
    """
    This function finds the relevant service name from the user input in the Step 1 by doing fuzzylogic searching
    and finding the token set ratio
    Args:
        userinput (): User Search Query

    Returns:
    Top matches in Services
    """
    # userinput = lemmatizer.lemmatize(userinput)
    # auto-correction !
    userinput = ' '.join([spell(i) for i in userinput.split()])
    words_without_stopwords = [i for i in userinput.lower().split() if i not in STOP_WORDS]
    userinput = " ".join(words_without_stopwords)

    def get_ratio(row_val):
        return fuzz.token_set_ratio(row_val, userinput)

    df_service_names['final_score'] = data.ServiceName.apply(get_ratio) * 1.5 + data.Category.apply(
        get_ratio) * 1.2 + data.Text.apply(get_ratio)
    df_service_names.sort_values(by=['final_score'], ascending=False, inplace=True)
    percentile_df = df_service_names[df_service_names.final_score > df_service_names.final_score.quantile(.99)]
    # df_value_counts = pd.DataFrame(percentile_df.ServiceName.value_counts())
    df_value_counts = list(percentile_df.drop_duplicates(subset=["ServiceName"], keep="first").ServiceName)
    if len(df_value_counts) > 0:
        unique_services = []
        for each in df_value_counts:
            unique_services.append(
                "<a href='#' class='generic_service_response' value='" + each + "'>" + each + "</a>")
        unique_services = list(unique_services)
        if len(unique_services) > 10:
            limited_services = unique_services[:10]
            response = "<ul><li>" + "</li><li>".join(limited_services) + "</li></ul>"
        else:
            response = "<ul><li>" + "</li><li>".join(unique_services) + "</li></ul>"
        return response, True, len(unique_services)

    return "<br>Sorry, I don't understand you. Please try again!", False, 0


@app.route('/intro')
def introduce():
    """
    Rest API for handling the Introduction request (First Request) of the user by asking his/her name
    Returns:
    Greetings and welcome response (String)
    """
    userinput = request.args.get("userinput")
    return introduction(userinput)


@app.route('/serviceName')
def get_service_name():
    """
    Rest API  - This API is responsible of handling and finding the most relevant ServiceName from the user search string
    using smart Search (Fuzzy Logic Matching with "Phrase Match, Token Ratio Match, Partial Match")

    Returns:
    Top matched services corresponding to user search string
    """
    userinput = request.args.get("userinput")
    message, flag, total_results = find_service_name(userinput)
    if flag:
        return "<br>I have found more than <b>" + str(total_results) + "</b> services related to your search. " \
                                                                       "It may help to refine your query with some more details. " \
                                                                       "Here are some of the top matches.<br>" + message
    else:
        return message


@app.route('/chatbot_service')
def chatbot_service():
    userinput = request.args.get("userinput")
    servicename = request.args.get("service_name")
    return chatbot(userinput, servicename)


@app.route('/service_categories')
def chatbot_service_categories():
    """
    This function finds recommended Categories for the sleected service name
    Returns:
    Clickable links for the category names to give breif idea of what all categories are available for that service name
    """
    servicename = request.args.get("service_name")
    service_category_names = data[data.ServiceName == servicename].Category
    unique_service_names = set()
    for each in list(service_category_names):
        if each in common_category_names:
            unique_service_names.add(each)

    clickable_categories = []
    for each in list(unique_service_names):
        clickable_categories.append(
            "<a href='#' class='service_categories' value='" + each + "'>" + each + "</a>")
    if len(clickable_categories) > 0:
        return "Here are some of the things I know about <b>{}</b> or you can ask any other question.<br>".format(
            servicename) \
               + "&bull;&nbsp;" + "&nbsp;&bull;&nbsp;".join(list(clickable_categories))
    else:
        return ""


@app.route('/translate_func')
def translate_func():
    userinput = request.args.get("userinput")
    userinput = userinput[:-115]  # HARD-CODED TO REMOVE LENGTH OF TRANSLATOR BUTTON
    translator = Translator()
    src = translator.detect(userinput).lang
    return translator.translate(userinput, src=src, dest='en').text


# @app.route('/feedback_func')
# def feedback_service():
#        userinput = userinput = request.args.get("userinput")
#        helpful_rank = 1 # CHECK AND UPDATE !!!
#        return response_word2vec(userinput)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5007)
