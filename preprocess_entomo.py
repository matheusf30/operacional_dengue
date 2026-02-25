# -*- coding: latin-1 -*-
#################################################################################
## Roteiro adaptado para pré-processar dados     ## PALAVRAS-CHAVE:            ##
## Dados: Focos de _Aedes_ sp. e                 ## > Pré-processamento;       ##
##        Casos Prováveis de Dengue (DIVE/SC)    ## > Dados de Saúde;          ##
## Demanda: FAPESC edital nº 37/2024             ## > Estruturação;            ##
## Adaptado por: Matheus Ferreira de Souza       ## > Semana Epidemiológica;   ##
##               e Everton Weber Galliani        ## > Série Temporal;          ##
## Data: 03/01/2026                              ## > Dicionário de Dados.     ##
#################################################################################

##### Bibliotecas correlatas ####################################################
#import dbf
import pandas as pd
import geopandas as gpd
import numpy as np
import os, sys
from datetime import date, datetime, timedelta
from epiweeks import Week, Year

##### Padrão ANSI ###############################################################
bold = "\033[1m"
red = "\033[91m"
green = "\033[92m"
yellow = "\033[33m"
blue = "\033[34m"
magenta = "\033[35m"
cyan = "\033[36m"
white = "\033[37m"
reset = "\033[0m"

##### CAMINHOS E ARQUIVOS ########################################################
caminho_dados = "/home/meteoro/scripts/operacional_dengue/dados_operacao/"
caminho_shape = "/media/dados/shapefiles/SC/"
focos1 = "relatorio.xlsx" # Araranguá
focos2 = "relatorio(1).xlsx" # Xanxerê
focos3 = "relatorio(2).xlsx" # Videira
focos4 = "relatorio(3).xlsx" # Tubarão
focos5 = "relatorio(4).xlsx" # São Miguel do Oeste
focos6 = "relatorio(5).xlsx" # São José
focos7 = "relatorio(6).xlsx" # Rio do Sul
focos8 = "relatorio(7).xlsx" # Mafra
focos9 = "relatorio(8).xlsx" # Lages
focos10 = "relatorio(9).xlsx" # Joinville
focos11 = "relatorio(10).xlsx" # Joaçaba
focos12 = "relatorio(11).xlsx" # Jaraguá do Sul
focos13 = "relatorio(12).xlsx" # Itajaí
focos14 = "relatorio(13).xlsx" # Criciúma
focos15 = "relatorio(14).xlsx" # Concórdia
focos16 = "relatorio(15).xlsx" # Chapecó
focos17 = "relatorio(16).xlsx" # Blumenau
municipios = "SC_Municipios_2022.shp"

##### ABRINDO ARQUIVOS ###########################################################
try:
	focos1 = pd.read_excel(f"{caminho_dados}{focos1}", engine = "openpyxl", skiprows = 2)
	focos2 = pd.read_excel(f"{caminho_dados}{focos2}", engine = "openpyxl", skiprows = 2)
	focos3 = pd.read_excel(f"{caminho_dados}{focos3}", engine = "openpyxl", skiprows = 2)
	focos4 = pd.read_excel(f"{caminho_dados}{focos4}", engine = "openpyxl", skiprows = 2)
	focos5 = pd.read_excel(f"{caminho_dados}{focos5}", engine = "openpyxl", skiprows = 2)
	focos6 = pd.read_excel(f"{caminho_dados}{focos6}", engine = "openpyxl", skiprows = 2)
	focos7 = pd.read_excel(f"{caminho_dados}{focos7}", engine = "openpyxl", skiprows = 2)
	focos8 = pd.read_excel(f"{caminho_dados}{focos8}", engine = "openpyxl", skiprows = 2)
	focos9 = pd.read_excel(f"{caminho_dados}{focos9}", engine = "openpyxl", skiprows = 2)
	focos10 = pd.read_excel(f"{caminho_dados}{focos10}", engine = "openpyxl", skiprows = 2)
	focos11 = pd.read_excel(f"{caminho_dados}{focos11}", engine = "openpyxl", skiprows = 2)
	focos12 = pd.read_excel(f"{caminho_dados}{focos12}", engine = "openpyxl", skiprows = 2)
	focos13 = pd.read_excel(f"{caminho_dados}{focos13}", engine = "openpyxl", skiprows = 2)
	focos14 = pd.read_excel(f"{caminho_dados}{focos14}", engine = "openpyxl", skiprows = 2)
	focos15 = pd.read_excel(f"{caminho_dados}{focos15}", engine = "openpyxl", skiprows = 2)
	focos16 = pd.read_excel(f"{caminho_dados}{focos16}", engine = "openpyxl", skiprows = 2)
	focos17 = pd.read_excel(f"{caminho_dados}{focos17}", engine = "openpyxl", skiprows = 2)
except:
	focos1 = pd.read_excel(f"{caminho_dados}{focos1}", engine = "xlrd", skiprows = 2)
	focos2 = pd.read_excel(f"{caminho_dados}{focos2}", engine = "xlrd", skiprows = 2)
	focos3 = pd.read_excel(f"{caminho_dados}{focos3}", engine = "xlrd", skiprows = 2)
	focos4 = pd.read_excel(f"{caminho_dados}{focos4}", engine = "xlrd", skiprows = 2)
	focos5 = pd.read_excel(f"{caminho_dados}{focos5}", engine = "xlrd", skiprows = 2)
	focos6 = pd.read_excel(f"{caminho_dados}{focos6}", engine = "xlrd", skiprows = 2)
	focos7 = pd.read_excel(f"{caminho_dados}{focos7}", engine = "xlrd", skiprows = 2)
	focos8 = pd.read_excel(f"{caminho_dados}{focos8}", engine = "xlrd", skiprows = 2)
	focos9 = pd.read_excel(f"{caminho_dados}{focos9}", engine = "xlrd", skiprows = 2)
	focos10 = pd.read_excel(f"{caminho_dados}{focos10}", engine = "xlrd", skiprows = 2)
	focos11 = pd.read_excel(f"{caminho_dados}{focos11}", engine = "xlrd", skiprows = 2)
	focos12 = pd.read_excel(f"{caminho_dados}{focos12}", engine = "xlrd", skiprows = 2)
	focos13 = pd.read_excel(f"{caminho_dados}{focos13}", engine = "xlrd", skiprows = 2)
	focos14 = pd.read_excel(f"{caminho_dados}{focos14}", engine = "xlrd", skiprows = 2)
	focos15 = pd.read_excel(f"{caminho_dados}{focos15}", engine = "xlrd", skiprows = 2)
	focos16 = pd.read_excel(f"{caminho_dados}{focos16}", engine = "xlrd", skiprows = 2)
	focos17 = pd.read_excel(f"{caminho_dados}{focos17}", engine = "xlrd", skiprows = 2)

municipios = gpd.read_file(f"{caminho_shape}{municipios}")

###### FUNÇÕES ###################################################################
def tempo_epidemiologico(df_original):
	tempo = pd.DataFrame()
	tempo["Semana"] = df_original["Semana"]
	tempo["Semana"] = pd.to_datetime(tempo["Semana"])
	tempo["SE"] = tempo["Semana"].apply(lambda data: Week.fromdate(data).week)
	tempo["ano_epi"] = tempo["Semana"].dt.year
	tempo.loc[(tempo["Semana"].dt.month == 1) & (tempo["SE"] > 50), "ano_epi"] -= 1
	tempo.loc[(tempo["Semana"].dt.month == 12) & (tempo["SE"] == 1), "ano_epi"] += 1
	print(f"\n{green}TEMPO CRONOLÓGICO (epidemiológico):\n{reset}{tempo}\n")
	return tempo
	
def tratar_focos(focos):
	"""
	"""	
	print(f"\n{green}FOCOS:\n{reset}{focos}\n")
	print(f"\n{green}FOCOS:\n{reset}{focos.columns}\n")
	colunas = list(focos.columns)
	colunas_renomear = {colunas[0]:"EXCLUIR",
						"Regional":"regional",
						"Município":"municipio",
						"A. aegypti formas aquáticas":"A.aegypti",
						"A. albopictus formas aquáticas":"A.albopictus",
						"Data da Coleta":"data"}
	focos = focos.rename(columns = colunas_renomear)
	print(f"\n{green}FOCOS (parte1):\n{reset}{focos}\n")
	print(f"\n{green}FOCOS.columns (parte1):\n{reset}{focos.columns}\n")
	focos.drop(columns = ["EXCLUIR"], inplace = True)
	focos["data"] = pd.to_datetime(focos["data"], format = "%d/%m/%Y", errors = "coerce")
	focos.dropna(subset = ["data", "municipio"], axis = 0, inplace = True)
	focos.sort_values(by = ["data"], inplace = True)
	focos.set_index("data", inplace=True)
	#focos = focos.drop(pd.to_datetime("NaT"))
	colunas = ["A.aegypti", "A.albopictus"]
	for col in colunas:
		focos[col] = focos[col].fillna(0).astype(int)
	focos["Aedes"] = focos["A.aegypti"] + focos["A.albopictus"]
	focos["focos"] = np.ones(len(focos)).astype(int)
	condicao_albopictus = (focos["A.aegypti"] == 0) & (focos["A.albopictus"] > 0)
	apenas_albopictus = focos[condicao_albopictus]
	print(f"\n{green}APENAS FOCOS A. albopictus:\n{reset}{apenas_albopictus}\n")
	condicao_aegypti = (focos["A.aegypti"] > 0) & (focos["A.albopictus"] == 0)
	apenas_aegypti = focos[condicao_aegypti]
	print(f"\n{green}APENAS FOCOS A. aegypti:\n{reset}{apenas_aegypti}\n")
	condicao_ambos = (focos["A.aegypti"] > 0) & (focos["A.albopictus"] > 0)
	apenas_ambos = focos[condicao_ambos]
	print(f"\n{green}FOCOS Aedes spp.:\n{reset}{apenas_ambos}\n")
	condicao_albopictus_maior = (focos["A.albopictus"] > focos["A.aegypti"])
	maior_albopictus = focos[condicao_albopictus_maior]
	print(f"\n{green}FOCOS A. albopictus > FOCOS A. aegypti:\n{reset}{maior_albopictus}\n")
	print(f"\n{green}FOCOS (parte2):\n{reset}{focos}\n")
	print(f"\n{green}FOCOS.columns (parte2):\n{reset}{focos.columns}\n")
	fator_agregacao = {"A.aegypti":"sum", "A.albopictus":"sum", "Aedes":"sum", "focos":"sum"}
	focos_semanal = focos.groupby("municipio").resample("W-SUN", label = "left").agg(fator_agregacao)
	focos_semanal.reset_index(inplace = True)
	focos_semanal.sort_values(by = ["data"], inplace = True)
	print(f"\n{green}FOCOS (parte3):\n{reset}{focos_semanal}\n")
	print(f"\n{green}FOCOS.columns (parte3):\n{reset}{focos_semanal.columns}\n")
	focos_semanal_pivot = pd.pivot_table(focos_semanal, values = "focos", fill_value = 0,
										columns = "municipio", index = "data")
	colunas = focos_semanal_pivot.columns
	for c in colunas:
		focos_semanal_pivot[c] = focos_semanal_pivot[c].astype(int)
	focos_semanal_pivot.reset_index(inplace = True)
	focos_semanal_pivot = focos_semanal_pivot.rename(columns = {"data":"Semana"})
	#focos_semanal_pivot = focos_semanal_pivot.iloc[:-2,:]
	print(f"\n{green}FOCOS (parte4):\n{reset}{focos_semanal_pivot}\n")
	print(f"\n{green}FOCOS.columns (parte4):\n{reset}{focos_semanal_pivot.columns}\n")
	return focos_semanal_pivot

### PRÉ-PROCESSAMENTO ############################################################
### FOCOS
focos1 = tratar_focos(focos1)
focos2 = tratar_focos(focos2)
focos3 = tratar_focos(focos3)
focos4 = tratar_focos(focos4)
focos5 = tratar_focos(focos5)
focos6 = tratar_focos(focos6)
focos7 = tratar_focos(focos7)
focos8 = tratar_focos(focos8)
focos9 = tratar_focos(focos9)
focos10 = tratar_focos(focos10)
focos11 = tratar_focos(focos11)
focos12 = tratar_focos(focos12)
focos13 = tratar_focos(focos13)
focos14 = tratar_focos(focos14)
focos15 = tratar_focos(focos15)
focos16 = tratar_focos(focos16)
focos17 = tratar_focos(focos17)
lista_focos = [focos1, focos2, focos3, focos4, focos5, focos6, 
               focos7, focos8, focos9, focos10, focos11, focos12, 
               focos13, focos14, focos15, focos16, focos17]
lista_focos = [df.set_index("Semana") for df in lista_focos]
focos_semanal_pivot = pd.concat(lista_focos, axis = 1)
focos_semanal_pivot = focos_semanal_pivot.fillna(0)
#focos_semanal_pivot = focos_semanal_pivot.groupby(level = 0, axis = 1).sum()
for col in [focos_semanal_pivot.columns]:
		focos_semanal_pivot[col] = focos_semanal_pivot[col].astype(int)
focos_semanal_pivot.reset_index(inplace = True)
focos_semanal_pivot.sort_values("Semana", inplace = True)
focos_semanal_pivot["semana"] = pd.to_datetime(focos_semanal_pivot["Semana"])
focos_semanal_pivot = focos_semanal_pivot[focos_semanal_pivot["Semana"] <= datetime.now()]
tempo_epidemiologico(focos_semanal_pivot)
print(f"\n{green}FOCOS CONSOLIDADOS (17 Regionais):\n{reset}{focos_semanal_pivot}\n")
print(f"\n{green}FOCOS CONSOLIDADOS (17 Regionais - Colunas):\n{reset}{focos_semanal_pivot.columns}\n")
focos_semanal_pivot.to_csv(f"{caminho_dados}focos_semanal_pivot.csv", index = False)
print(f"\n{green}SALVANDO FOCOS (semanal):\n{reset}{caminho_dados}{focos_semanal_pivot}\n")
#sys.exit()
