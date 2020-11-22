from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import string
from nltk.tokenize import word_tokenize
import os
from nltk.corpus import stopwords
import nltk
from datetime import datetime

class Cleaner:
    def __init__(self,stop_words_file:str,language:str,
                        perform_stop_words_removal:bool,perform_accents_removal:bool,
                        perform_stemming:bool):
        #self.set_stop_words = self.read_stop_words(stop_words_file)
        nltk.download('stopwords')
        nltk.download('punkt')
        self.set_stop_words = set(stopwords.words('portuguese'))
        self.stemmer = SnowballStemmer(language)
        in_table =  "áéíóúâêôçãẽõü!?.:;,"
        out_table = "aeiouaeocaeou      "
        #altere a linha abaixo para remoção de acentos (Atividade 11)
        self.accents_translation_table = in_table.maketrans(in_table,out_table)
        self.set_punctuation = set(string.punctuation)

        #flags
        self.perform_stop_words_removal = perform_stop_words_removal
        self.perform_accents_removal = perform_accents_removal
        self.perform_stemming = perform_stemming

    def html_to_plain_text(self,html_doc:str) ->str:
        soup = BeautifulSoup(html_doc, 'html.parser')
        return soup.get_text()

    def read_stop_words(self,str_file):
        set_stop_words = set()
        with open(str_file, "r") as stop_words_file:
            for line in stop_words_file:
                arr_words = line.split(",")
                [set_stop_words.add(word) for word in arr_words]
        return set_stop_words
    
    def is_stop_word(self,term:str):
        
        if term in self.set_stop_words:
            return True
        return False

    def word_stem(self,term:str):
        return self.stemmer.stem(term)


    def remove_accents(self,term:str) ->str:
        return term.translate(self.accents_translation_table)


    def preprocess_word(self,term:str) -> str:
        term = term.lower()
        if self.perform_stop_words_removal is True and self.is_stop_word(term):
            term = ""
            
        if self.perform_accents_removal is True:
            term = self.remove_accents(term)
        
        if self.perform_stemming is True:
            term = self.word_stem(term)
        
        return term

class HTMLIndexer:
    
    cleaner = Cleaner(stop_words_file="stopwords.txt",
                        language="portuguese",
                        perform_stop_words_removal=False,
                        perform_accents_removal=True,
                        perform_stemming=True)
    
    def __init__(self,index):
        self.index = index

    def text_word_count(self,plain_text:str):
        text_tokenized = nltk.word_tokenize(plain_text)
        dic_word_count = {}
        for word in text_tokenized:
            word = self.cleaner.preprocess_word(word)
            if word != "" and word != " ":
                if word in dic_word_count:
                    dic_word_count[word] = dic_word_count[word] + 1
                else:
                    dic_word_count[word] = 1
        return dic_word_count
    
    def index_text(self,doc_id:int, text_html:str):
        text_plain = HTMLIndexer.cleaner.html_to_plain_text(text_html)
        dic_text = self.text_word_count(text_plain)
        for term in dic_text:
            self.index.index(term,doc_id,dic_text[term])
        

    def index_text_dir(self,path:str):
        for str_sub_dir in os.listdir(path):
            path_sub_dir = f"{path}/{str_sub_dir}"
            for str_file in os.listdir(path_sub_dir):
                filename = f"{path_sub_dir}/{str_file}"
                with open(filename,"rb") as file:
                    time_first = datetime.now()
                    self.index_text(int((str_file.split("."))[0]),file)
                    time_end = datetime.now()
                    tempo_gasto = time_end-time_first
                with open("tempos.txt","a",encoding="utf-8") as file:
                    file.write(str_file+":")
                    file.write(f"{tempo_gasto.total_seconds()}\n")
                

                    
