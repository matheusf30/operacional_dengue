#################################################################################
## Roteiro adaptado para pr�-processar dados     ## PALAVRAS-CHAVE:            ##
## Dados: Focos de _Aedes_ sp. e                 ## > Pré-processamento;      ##
##        Casos Prov�veis de Dengue (DIVE/SC)    ## > Dados de Saúde;         ##
## Demanda: FAPESC edital n� 37/2024             ## > Estruturação;          ##
## Adaptado por: Matheus Ferreira de Souza       ## > Semana Epidemiol[ogica;  ##
##               e Everton Weber Galliani        ## > Série Temporal;         ##
## Data: 31/07/2025                              ## > Dicionário de Dados.    ##
#################################################################################

##### Bibliotecas correlatas ####################################################
#import dbf
import pandas as pd
import geopandas as gpd
import numpy as np
import os, sys

##### Padr�o ANSI ###############################################################
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
caminho_dados = "/home/meteoro/scripts/matheus/operacional_dengue/dados_operacao/"
caminho_shape = "/media/dados/shapefiles/SC/"
casos = "dengue-2025.xlsx" #"casos_2025.xlsx" # Sincronizar nomes
focos = "focos_2025.xlsx"
municipios = "SC_Municipios_2022.shp"

##### ABRINDO ARQUIVOS ###########################################################
try: # vers�o nova do excel (.xlsx)
	casos = pd.read_excel(f"{caminho_dados}{casos}", engine = "openpyxl")
except: # vers�o antiga do excel (.xls)
	casos = pd.read_excel(f"{caminho_dados}{casos}", engine = "xlrd")
try:
	focos = pd.read_excel(f"{caminho_dados}{focos}", engine = "openpyxl", skiprows = 2)
except:
	focos = pd.read_excel(f"{caminho_dados}{focos}", engine = "xlrd", skiprows = 2)
municipios = gpd.read_file(f"{caminho_shape}{municipios}")

### PR�-PROCESSAMENTO ############################################################
### CASOS
print(f"\n{green}CASOS (ORIGINAL):\n{reset}{casos}\n")
print(f"\n{green}CASOS (ORIGINAL):\n{reset}{casos.columns[0:50]}\n")
print(f"\n{green}CASOS (ORIGINAL):\n{reset}{casos.columns[50:100]}\n")
print(f"\n{green}CASOS (ORIGINAL):\n{reset}{casos.columns[100:]}\n")
print(f"\n{green}CASOS (ORIGINAL):\n{reset}{casos.columns}\n")
"""
#try: # Colunas com nome na vers�o original do banco de dados
colunas_renomear = {"ID_AGRAVO":"doenca",
					"DT_NOTIFIC":"data_notificacao",
					"DT_SIN_PRI":"data_sintoma",
					"SEM_PRI":"semana_sintoma",
					"ID_MUNICIP":"municipio_id",
					"ID_REGIONA":"regional",
					"CLASSI_FIN":"classificacao",
					"CRITERIO":"criterio ",
					"SOROTIPO":"sorotipo"}
"""
#except KeyError: # Colunas com padr�o anomolo de nomenclatura
colunas_renomear = {"ID_AGRAVO,C,5":"doenca",
					"DT_NOTIFIC,D":"data_notificacao",
					"DT_SIN_PRI,D":"data_sintoma",
					"SEM_PRI,C,6":"semana_sintoma",
					"ID_MN_RESI,C,6":"municipio_id",
					"ID_RG_RESI,C,8":"regional",
					"CLASSI_FIN,C,2":"classificacao",
					"CRITERIO,C,1":"criterio ",
					"SOROTIPO,C,1":"sorotipo"}
casos = casos.rename(columns = colunas_renomear)
casos = casos[["data_sintoma", "semana_sintoma", "data_notificacao",
				"municipio_id", "regional",	"doenca",
				"classificacao", "criterio ", "sorotipo"]]
casos.columns = casos.columns.str.strip()

print(f"\n{green}CLASSIFICA��O:\n{reset}{casos['classificacao'].unique()}\n")
print(f"\n{green}INFO:\n{reset}{casos.info()}\n")
filtro_numerico = casos["classificacao"].isin([10., 8., 11., 12.])
filtro_nan = casos["classificacao"].isna()
casos = casos[filtro_numerico | filtro_nan]
#casos = casos[casos["classificacao"].isin(filtro_classificacao)] # "5." = Descartado
print(f"\n{green}CLASSIFICA��O:\n{reset}{casos['classificacao'].unique()}\n")
print(f"\n{green}CASOS (SELE��O):\n{reset}{casos}\n")
print(f"\n{green}CASOS (SELE��O):\n{reset}{casos.columns}\n")
#sys.exit()
casos["data_sintoma"] = pd.to_datetime(casos["data_sintoma"])#, unit = "D", origin = "1899-12-30")
casos["data_notificacao"] = pd.to_datetime(casos["data_notificacao"])#, unit = "D", origin = "1899-12-30")
casos = casos[casos["data_sintoma"].dt.year == 2025]
municipio_faltando = casos["municipio_id"].isnull().sum()
print(f"\n{red}Quantidade de registros faltando munic�pio: {municipio_faltando}!{reset}\n")
casos["casos"] = np.ones(len(casos)).astype(int)
casos = casos[["semana_sintoma", "data_sintoma", "municipio_id", "casos"]]
casos.dropna(subset=["municipio_id"], inplace=True)
casos["municipio_id"] = casos["municipio_id"].astype(int)
casos.sort_values(by = ["data_sintoma"], inplace = True)
print(f"\n{green}CASOS:\n{reset}{casos}\n")
print(f"\n{green}MUNIC�PIOS:\n{reset}{municipios}\n")
pre_dict = municipios[["CD_MUN", "NM_MUN"]]
pre_dict.loc[:, "CD_MUN"] = pre_dict["CD_MUN"].str.slice(0, -1)
pre_dict["NM_MUN"] = pre_dict["NM_MUN"].str.upper()
mun_dict = pd.Series(pre_dict["NM_MUN"].values, index = pre_dict["CD_MUN"]).to_dict()
print(f"\n{green}MUNIC�PIOS (DIC.):\n{reset}{mun_dict}\n")
casos["municipio_id"] = casos["municipio_id"].astype(str)
casos["municipio"] = casos["municipio_id"].map(mun_dict)
casos.sort_values(by = ["data_sintoma"], inplace = True)
print(f"\n{green}CASOS:\n{reset}{casos}\n")
print(f"\n{green}CASOS (info):\n{reset}{casos.info()}\n")
print(f"\n{green}CASOS:\n{reset}{casos.columns}\n")
casos = casos.groupby(["data_sintoma", "municipio"])["casos"].sum(numeric_only = True)
casos_df = casos.reset_index()
casos_df.columns = ["data_sintoma", "municipio", "casos"]
casos = casos_df
casos = casos.rename(columns = {"data_sintoma":"data"})
casos["data"] = pd.to_datetime(casos["data"], format = "%Y-%m-%d", errors = "coerce")
anos = casos["data"].dt.year
print(f"\n{green}ANOS PRESENTES NO CONJUNTO DE DADOS:\n{reset}{anos.unique()}\n")
casos.set_index("data", inplace = True)
casos = casos[casos.index.year == 2025]
fator_agregacao = {"casos":"sum"}
casos_semanal = casos.groupby("municipio").resample("W-SUN", label = "left").agg(fator_agregacao)
casos_semanal.reset_index(inplace = True) #"W-SAT", label = "left"
casos_semanal.sort_values(by = ["data"], inplace = True)
casos_semanal_pivot = pd.pivot_table(casos_semanal, values = "casos", fill_value = 0,
									columns = "municipio", index = "data")
colunas = casos_semanal_pivot.columns
for c in colunas:
	casos_semanal_pivot[c] = casos_semanal_pivot[c].astype(int)
casos_semanal_pivot.reset_index(inplace = True)
casos_semanal_pivot = casos_semanal_pivot.rename(columns = {"data":"Semana"})
#casos_semanal_pivot = casos_semanal_pivot.iloc[:-1,:]
print(f"\n{green}CASOS:\n{reset}{casos_semanal_pivot}\n")
print(f"\n{green}CASOS (info):\n{reset}{casos_semanal_pivot.info()}\n")
print(f"\n{green}CASOS:\n{reset}{casos_semanal_pivot.columns}\n")

# FINALIZAR DEPURA��O DAS DATAS/MUNIC�PIOS/COLUNAS ANTES DE SALVAR
casos_semanal_pivot.to_csv(f"{caminho_dados}casos_semanal_pivot.csv", index = False)
print(f"\n{green}SALVANDO CASOS (semanal):\n{reset}{caminho_dados}{casos_semanal_pivot}\n")


#sys.exit()

### FOCOS
print(f"\n{green}FOCOS:\n{reset}{focos}\n")
print(f"\n{green}FOCOS:\n{reset}{focos.columns}\n")
data = "30/09/2025"
colunas_renomear = {f"Atualizado em {data}":"EXCLUIR",
					"Regional":"regional",
					"Munic�pio":"municipio",
					"A. aegypti formas aqu�ticas":"A.aegypti",
					"A. albopictus formas aqu�ticas":"A.albopictus",
					"Data da Coleta":"data"}
focos = focos.rename(columns = colunas_renomear)
print(f"\n{green}FOCOS:\n{reset}{focos}\n")
print(f"\n{green}FOCOS:\n{reset}{focos.columns}\n")
focos.drop(columns = ["EXCLUIR"], inplace = True)
focos.dropna(axis = 0, inplace = True)
focos["data"] = pd.to_datetime(focos["data"], format = "%d/%m/%Y", errors = "coerce")
focos.sort_values(by = ["data"], inplace = True)
focos.set_index("data", inplace=True)
#focos = focos.drop(pd.to_datetime("NaT"))
colunas = [["A.aegypti", "A.albopictus"]]
for col in colunas:
	focos[col] = focos[col].astype(int)
focos["focos"] = focos["A.aegypti"] + focos["A.albopictus"]
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
print(f"\n{green}FOCOS:\n{reset}{focos}\n")
print(f"\n{green}FOCOS:\n{reset}{focos.columns}\n")
fator_agregacao = {"A.aegypti":"sum", "A.albopictus":"sum", "focos":"sum"}
focos_semanal = focos.groupby("municipio").resample("W-SUN", label = "left").agg(fator_agregacao)
focos_semanal.reset_index(inplace = True)
focos_semanal.sort_values(by = ["data"], inplace = True)
print(f"\n{green}FOCOS:\n{reset}{focos_semanal}\n")
print(f"\n{green}FOCOS:\n{reset}{focos_semanal.columns}\n")
focos_semanal_pivot = pd.pivot_table(focos_semanal, values = "focos", fill_value = 0,
									columns = "municipio", index = "data")
colunas = focos_semanal_pivot.columns
for c in colunas:
	focos_semanal_pivot[c] = focos_semanal_pivot[c].astype(int)
focos_semanal_pivot.reset_index(inplace = True)
focos_semanal_pivot = focos_semanal_pivot.rename(columns = {"data":"Semana"})
#focos_semanal_pivot = focos_semanal_pivot.iloc[:-2,:]
print(f"\n{green}FOCOS:\n{reset}{focos_semanal_pivot}\n")
print(f"\n{green}FOCOS:\n{reset}{focos_semanal_pivot.columns}\n")
focos_semanal_pivot.to_csv(f"{caminho_dados}focos_semanal_pivot.csv", index = False)
print(f"\n{green}SALVANDO FOCOS (semanal):\n{reset}{caminho_dados}{focos_semanal_pivot}\n")
"""
print(f"\n{green}FOCOS:\n{reset}{focos}\n")
print(f"\n{green}FOCOS:\n{reset}{focos.columns}\n")
colunas_renomear = {"Unnamed: 0":"EXCLUIR",
					"Dengue - Quadro de focos":"regional",
					"Unnamed: 2":"municipio",
					"Unnamed: 3":"A.aegypti",
					"Unnamed: 4":"A.albopictus",
					"Unnamed: 5":"data"}
focos = focos.rename(columns = colunas_renomear)
print(f"\n{green}FOCOS:\n{reset}{focos}\n")
print(f"\n{green}FOCOS:\n{reset}{focos.columns}\n")
focos.drop(columns = ["EXCLUIR"], inplace = True)
focos.dropna(axis = 0, inplace = True)
focos["data"] = pd.to_datetime(focos["data"], format = "%d/%m/%Y", errors = "coerce")
focos.sort_values(by = ["data"], inplace = True)
focos.set_index("data", inplace=True)
focos = focos.drop(pd.to_datetime("NaT"))
colunas = [["A.aegypti", "A.albopictus"]]
for col in colunas:
	focos[col] = focos[col].astype(int)
focos["focos"] = focos["A.aegypti"] + focos["A.albopictus"]
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
print(f"\n{green}FOCOS:\n{reset}{focos}\n")
print(f"\n{green}FOCOS:\n{reset}{focos.columns}\n")
fator_agregacao = {"A.aegypti":"sum", "A.albopictus":"sum", "focos":"sum"}
focos_semanal = focos.groupby("municipio").resample("W-SAT", label = "left").agg(fator_agregacao)
focos_semanal.reset_index(inplace = True)
focos_semanal.sort_values(by = ["data"], inplace = True)
print(f"\n{green}FOCOS:\n{reset}{focos_semanal}\n")
print(f"\n{green}FOCOS:\n{reset}{focos_semanal.columns}\n")
focos_semanal_pivot = pd.pivot_table(focos_semanal, values = "focos", fill_value = 0,
									columns = "municipio", index = "data")
colunas = focos_semanal_pivot.columns
for c in colunas:
	focos_semanal_pivot[c] = focos_semanal_pivot[c].astype(int)
focos_semanal_pivot.reset_index(inplace = True)
focos_semanal_pivot = focos_semanal_pivot.rename(columns = {"data":"Semana"})
focos_semanal_pivot = focos_semanal_pivot.iloc[:-2,:]
print(f"\n{green}FOCOS:\n{reset}{focos_semanal_pivot}\n")
print(f"\n{green}FOCOS:\n{reset}{focos_semanal_pivot.columns}\n")
focos_semanal_pivot.to_csv(f"{caminho_dados}focos_semanal_pivot.csv", index = False)
print(f"\n{green}SALVANDO FOCOS (semanal):\n{reset}{caminho_dados}{focos_semanal_pivot}\n")
"""
