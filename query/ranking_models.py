from typing import List
from abc import abstractmethod
from typing import List, Set,Mapping
from index.structure import TermOccurrence
import math
from enum import Enum

class IndexPreComputedVals():
    def __init__(self,index):
        self.index = index
        self.precompute_vals()

    def precompute_vals(self):
        """
        Inicializa os atributos por meio do indice (idx):
            doc_count: o numero de documentos que o indice possui
            document_norm: A norma por documento (cada termo é presentado pelo seu peso (tfxidf))
        """
        self.document_norm = {}
        self.doc_count = self.index.document_count
        self.term_idf = dict()
        
        sum_doc  = dict()
        for term in list(self.index.dic_index.keys()):
            occurrence_list = self.index.get_occurrence_list(term)
            for item in occurrence_list:
                tf_idf = VectorRankingModel.tf_idf(self.doc_count,item.term_freq,len(occurrence_list))
                print(tf_idf)
                if term not in self.term_idf.keys():
                    self.term_idf[term] = list()
                    self.term_idf[term].append((item.doc_id,tf_idf))
                else:
                    self.term_idf[term].append((item.doc_id,tf_idf))
        for term in self.term_idf.keys():
            for occurrence in self.term_idf[term]:
                if occurrence[0] in sum_doc:
                    sum_doc[occurrence[0]] =  sum_doc[occurrence[0]] + math.pow(occurrence[1],2)
                else: 
                    sum_doc[occurrence[0]] = math.pow(occurrence[1],2)
        for x in range(1,self.doc_count+1):
                self.document_norm[x] = round(math.sqrt(sum_doc[x]),2)
        
class RankingModel():
    @abstractmethod
    def get_ordered_docs(self,query:Mapping[str,TermOccurrence],
                              docs_occur_per_term:Mapping[str,List[TermOccurrence]]) -> (List[int], Mapping[int,float]):
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    def rank_document_ids(self,documents_weight):
        doc_ids = list(documents_weight.keys())
        doc_ids.sort(key= lambda x:-documents_weight[x])
        return doc_ids

class OPERATOR(Enum):
  AND = 1
  OR = 2
    
#Atividade 1
class BooleanRankingModel(RankingModel):
    def __init__(self,operator:OPERATOR):
        self.operator = operator

    def intersection_all(self,map_lst_occurrences:Mapping[str,List[TermOccurrence]]) -> List[int]:
        set_ids = set()
        set_discover_ids = set()
        for  term, lst_occurrences in map_lst_occurrences.items():
             for occurrenc in lst_occurrences:
                if occurrenc.doc_id not in set_discover_ids:
                    set_discover_ids.add(occurrenc.doc_id)
                elif occurrenc.doc_id in set_discover_ids and occurrenc.doc_id not in set_ids:
                    set_ids.add(occurrenc.doc_id)
        return list(set_ids)
    def union_all(self,map_lst_occurrences:Mapping[str,List[TermOccurrence]]) -> List[int]:
        set_ids = set()
        for term, lst_occurrences in map_lst_occurrences.items():
            for occurrenc in lst_occurrences:
                if occurrenc.doc_id not in set_ids:
                    set_ids.add(occurrenc.doc_id)
        return list(set_ids)

    def get_ordered_docs(self,query:Mapping[str,TermOccurrence],
                              map_lst_occurrences:Mapping[str,List[TermOccurrence]]) -> (List[int], Mapping[int,float]):
        """Considere que map_lst_occurrences possui as ocorrencias apenas dos termos que existem na consulta"""
        if self.operator == OPERATOR.AND:
            return self.intersection_all(map_lst_occurrences),None
        else:
            return self.union_all(map_lst_occurrences),None

#Atividade 2
class VectorRankingModel(RankingModel):

    def __init__(self,idx_pre_comp_vals:IndexPreComputedVals):
        self.idx_pre_comp_vals = idx_pre_comp_vals

    @staticmethod
    def tf(freq_term:int) -> float:
        return 1 + math.log(freq_term,2)

    @staticmethod
    def idf(doc_count:int, num_docs_with_term:int )->float:
        return math.log((doc_count/num_docs_with_term),2)

    @staticmethod
    def tf_idf(doc_count:int, freq_term:int, num_docs_with_term) -> float:
        tf = VectorRankingModel.tf(freq_term)
        idf = VectorRankingModel.idf(doc_count,num_docs_with_term)
        #print(f"TF:{tf} IDF:{idf} f: {freq_term} n_i: {num_docs_with_term} N: {doc_count}")
        return tf*idf

    def get_ordered_docs(self,query:Mapping[str,TermOccurrence],
                              docs_occur_per_term:Mapping[str,List[TermOccurrence]]) -> (List[int], Mapping[int,float]):
            documents_weight = {}
            term_idf = dict()
            sum_doc  = dict()
            docs_find = []
            doc_count=0 
            print(self.idx_pre_comp_vals.term_idf)
            for item in docs_occur_per_term:
                for occurrenc in docs_occur_per_term[item]:
                    if occurrenc.doc_id not in docs_find:
                        docs_find.append(occurrenc.doc_id)
                    if occurrenc.doc_id > doc_count:
                        doc_count = occurrenc.doc_id
            for term in docs_occur_per_term.keys():
                for item in docs_occur_per_term[term]:
                    tf_idf = VectorRankingModel.tf_idf(doc_count,query[term].term_freq,len(docs_occur_per_term[term]))
                    if term not in term_idf.keys():
                        term_idf[term] = list()
                        term_idf[term].append((item.doc_id,tf_idf))
                    else:
                        term_idf[term].append((item.doc_id,tf_idf))
           
            for term in term_idf.keys():
                for occurrence in term_idf[term]:
                    for ocs in self.idx_pre_comp_vals.term_idf[term]:
                        if occurrence[0] in sum_doc:
                            sum_doc[occurrence[0]] =  sum_doc[occurrence[0]] + (occurrence[1] * ocs[1])
                        else: 
                            sum_doc[occurrence[0]] = (occurrence[1] * ocs[1] )
            for x in docs_find:
                if x in sum_doc:
                    documents_weight[x] = sum_doc[x]/self.idx_pre_comp_vals.document_norm[x]
            #retona a lista de doc ids ordenados de acordo com o TF IDF
            return self.rank_document_ids(documents_weight),documents_weight

