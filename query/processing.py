from typing import List, Set,Mapping
from nltk.tokenize import word_tokenize
from util.time import CheckTime
from query.ranking_models import *
from index.structure import Index, TermOccurrence, FileIndex
from index.indexer import Cleaner
import os
from datetime import datetime

class QueryRunner:
	def __init__(self,ranking_model:RankingModel,index:Index, cleaner:Cleaner):
		self.ranking_model = ranking_model
		self.index = index
		self.cleaner = cleaner


	def get_relevance_per_query(self) -> Mapping[str,Set[int]]:
		"""
		Adiciona a lista de documentos relevantes para um determinada query (os documentos relevantes foram
		fornecidos no ".dat" correspondente. Por ex, belo_horizonte.dat possui os documentos relevantes da consulta "Belo Horizonte"

		"""
		dic_relevance_docs = {}
		for str_file in os.listdir("relevant_docs"):
			filename = f"relevant_docs/{str_file}"
			with open(filename) as arq:
				arquiv = str_file.split(".")[0]
				dic_relevance_docs[arquiv] = set(map(lambda s: str(s).replace("\n",""),arq.readline().split(",")))
				
		return dic_relevance_docs

	def count_topn_relevant(self,n,respostas:List[int],doc_relevantes:Set[int]) -> int:
		"""
		Calcula a quantidade de documentos relevantes na top n posições da lista lstResposta que é a resposta a uma consulta
		Considere que respostas já é a lista de respostas ordenadas por um método de processamento de consulta (BM25, Modelo vetorial).
		Os documentos relevantes estão no parametro docRelevantes
		"""
		#print(f"Respostas: {respostas} doc_relevantes: {doc_relevantes}")
		relevance_count = 0
		for position in respostas[0:n]:
			if str(position) in doc_relevantes:
				relevance_count = relevance_count + 1
		return relevance_count

	def get_query_term_occurence(self, query:str) -> Mapping[str,TermOccurrence]:
		"""
			Preprocesse a consulta da mesma forma que foi preprocessado o texto do documento (use a classe Cleaner para isso).
			E transforme a consulta em um dicionario em que a chave é o termo que ocorreu
			e o valor é uma instancia da classe TermOccurrence (feita no trabalho prático passado).
			Coloque o docId como None.
			Caso o termo nao exista no indic, ele será desconsiderado.
		"""
		#print(self.index)
		map_term_occur = {}
		cleaner_query = self.cleaner.preprocess_word(query)
		
		
		for term in cleaner_query.split(" "):
			term_id = self.index.get_term_id(term)
			if term not in map_term_occur.keys():
				term_occur = TermOccurrence(None,term_id,1)
			else:
				term_occur = map_term_occur[term]
				term_occur = TermOccurrence(None,term_id,(term_occur.term_freq +1))
			if term_id != None:
				map_term_occur[term] = term_occur
		return map_term_occur

	def get_occurrence_list_per_term(self, terms:List) -> Mapping[str, List[TermOccurrence]]:
		"""
			Retorna dicionario a lista de ocorrencia no indice de cada termo passado como parametro.
			Caso o termo nao exista, este termo possuirá uma lista vazia
		"""
		dic_terms = {}
		for term in terms:
			term_id = self.index.get_term_id(term)
			if term_id == None:
				dic_terms[term] = []
			else:
				dic_terms[term] = self.index.get_occurrence_list(term)		

		return dic_terms
	def get_docs_term(self, query:str) -> List[int]:
		"""
			A partir do indice, retorna a lista de ids de documentos desta consulta
			usando o modelo especificado pelo atributo ranking_model
		"""
		cleaner_query = self.cleaner.preprocess_word(query)
		
		#Obtenha, para cada termo da consulta, sua ocorrencia por meio do método get_query_term_occurence
		dic_query_occur = self.get_query_term_occurence(cleaner_query)
		#obtenha a lista de ocorrencia dos termos da consulta
		lista = cleaner_query.split(" ")
		dic_occur_per_term_query = self.get_occurrence_list_per_term(lista)
		#utilize o ranking_model para retornar o documentos ordenados considrando dic_query_occur e dic_occur_per_term_query
		return self.ranking_model.get_ordered_docs(dic_query_occur,dic_occur_per_term_query)
		

	@staticmethod
	def runQuery(query:str, indice:Index, indice_pre_computado: IndexPreComputedVals, map_relevantes):
		time_checker = CheckTime()

		#PEça para usuario selecionar entre Booleano ou modelo vetorial para intanciar o QueryRunner
		#apropriadamente. NO caso do booleano, vc deve pedir ao usuario se será um "and" ou "or" entre os termos.
		#abaixo, existem exemplos fixos.
        
		mod = RankingModel()
		clean = Cleaner(stop_words_file="stopwords.txt",language="portuguese",
                        perform_stop_words_removal=False,perform_accents_removal=False,
                        perform_stemming=False)
		modelo = int(input("Selecione um dos modelos:\n1)Modelo Booleano\n2)Modelo Vetorial"))
		if modelo == 1:
			operador = int(input("Qual dos termos deseja utilizar?1)AND\n2)OR"))
			if operador==1:
				mod = BooleanRankingModel(OPERATOR.AND)
			elif operador==2:
				mod = BooleanRankingModel(OPERATOR.OR)
			else:
				raise Exception("Opção inválida. A execução foi interrompida.")
		elif modelo==2:
			mod = VectorRankingModel(indice_pre_computado)
		else:
			raise Exception("Opção inválida. A execução foi interrompida.")
		
		#ranking_model:RankingModel,index:Index, cleaner:Cleaner
		qr = QueryRunner(mod,indice,clean)
		time_checker.printDelta("Query Creation")

		#Utilize o método get_docs_term para obter a lista de documentos que responde esta consulta
		respostas = qr.get_docs_term(query)
		time_checker.printDelta(f"anwered with {len(respostas)} docs")

		#nesse if, vc irá verificar se o termo possui documentos relevantes associados a ele
		#se possuir, vc deverá calcular a Precisao e revocação nos top 5, 10, 20, 50.
		#O for que fiz abaixo é só uma sugestao e o metododo countTopNRelevants podera auxiliar no calculo da revocacao e precisao
	
		if(query in map_relevantes.keys()):
			arr_top = [5,10,20,50]
			revocacao = 0 
			precisao = 0
			for n in arr_top:
				revocacao = qr.count_topn_relevant(n,respostas[0],set(map_relevantes[query])) / len(map_relevantes[query]) #substitua aqui pelo calculo da revocacao topN
				precisao = qr.count_topn_relevant(n,respostas[0],set(map_relevantes[query])) / len(respostas[0]) #substitua aqui pelo calculo da revocacao topN
				filter(map_relevantes, respostas[0])
				print(f"Precisao {n}: {precisao}")
				print(f"Recall {n}: {revocacao} \n")
		else:
			print("termo nao ta na pesquisa")
		#imprima aas top 10 respostas

	@staticmethod
	def main():
		#leia o indice (base da dados fornecida)
		index = FileIndex()
		
		index.index("irlanda",37632,1)
		index.index("irlanda",39300,3)
		index.index("espero",39300,1)
		index.index("que",11953,1)
		index.index("irlanda",11953,1)
		index.index("estejam",11953,1)
		index.index("se",11953,1)
		index.index("irlanda",37632,4)
		
		index.index("que",44259,1)
		index.index("irlanda",44259,1)
		index.index("estejam",44259,1)
		index.index("se",44259,1)
		index.index("irlanda",111966,4)

		index.index("que",51714,1)
		index.index("irlanda",51714,1)
		index.index("estejam",51714,1)
		index.index("se",51714,1)
		
		index.finish_indexing()
		#Checagem se existe um documento (apenas para teste, deveria existir)
		print(f"Existe o doc? index.hasDocId(105047)")
		
		#Instancie o IndicePreCompModelo para pr ecomputar os valores necessarios para a query
		idxPreCom = IndexPreComputedVals(index)
		print("Precomputando valores atraves do indice...")
		check_time = CheckTime()
        
		check_time.printDelta("Precomputou valores")
		#encontra os docs relevantes
		map_relevance = QueryRunner.get_relevance_per_query(QueryRunner)
		print("Fazendo query...")
		#aquui, peça para o usuário uma query (voce pode deixar isso num while ou fazer um interface grafica se estiver bastante animado ;)
		while(True):
			query = str(input("Insira um termo que deseja pesquisar: \n ou Digite Hasan para sair"))
			if query != 'Hasan' or query != 'hasan':
				QueryRunner.runQuery(query, index, idxPreCom, map_relevance)
			else:
				return
