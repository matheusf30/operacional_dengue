####################################################################################
## Facilitar a visualização dos resultados do Boletim ## PALAVRAS-CHAVE:               ##
## Dados: Arquivos Preditivos de Casos de Dengue     ## > Modelagem Computacional;    ##
##                       e de Focos de _Aedes_ spp.  ## > Modelo Preditivo;           ##
## Demanda: FAPESC edital nº 37/2024                 ## > Transferência de Tecnologia;##
## Adaptado por: Everton Weber Galliani,             ## > Santa Catarina;             ##
##            Matheus Ferreira de Souza,             ## > Gerência de Zoonoses;     ##     
## Data: 09/04/2026                                  ## > Boletim Epidemiológico.     ##
########################################################################################
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from epiweeks import Week, Year
import sys

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

t0 = datetime.now()
SE = int(str(Week.fromdate(t0))[-2:])
EPI_YEAR = int(str(Week.fromdate(t0))[:-2])
caminho = f"/home/meteoro/scripts/operacional_dengue/resultados/{EPI_YEAR}/SE{SE}"
caminho_dados = "/home/meteoro/scripts/operacional_dengue/dados_operacao/"
censo = "censo_sc_xy.csv"
censo = pd.read_csv(f"{caminho_dados}{censo}")
censo = censo[["Municipio", "Populacao"]].set_index("Municipio")
censo.index = censo.index.str.upper()

def printa(str_var, var):
	STR_VAR = str_var.upper()
	print(f"\n{green}{STR_VAR}: \n{reset}{var}\n")
	
printa("se", SE)
printa("ano_epi", EPI_YEAR)
printa("censo", censo)

def previsao(analise, arquivo, colunas):
	arquivo_csv = pd.read_csv(arquivo).iloc[0:3]
	printa(f"{analise}", arquivo_csv)
	#input(f"{green}Selecione um município: {reset}")
	arquivo_csv_municipio = arquivo_csv[["index", "Semana", "FLORIANÓPOLIS"]]
	printa("focos", arquivo_csv)
	S0 = arquivo_csv.drop(columns = colunas).loc[0].sort_values(ascending = False).head(20)
	S1 = arquivo_csv.drop(columns = colunas).loc[1].sort_values(ascending = False).head(20)
	S2 = arquivo_csv.drop(columns = colunas).loc[2].sort_values(ascending = False).head(20)
	printa(f"PREVISAO DE {analise} PARA SE{SE}", S0)
	printa(f"PREVISAO DE {analise} PARA SE{SE + 1}", S1)
	printa(f"PREVISAO DE {analise} PARA SE{SE + 2}", S2)
	print(arquivo_csv_municipio)
	
def monitoramento(analise, arquivo, colunas = None, visu = 30):
	arquivo_csv = pd.read_csv(arquivo)
	arquivo_csv["Semana"]
	printa(f"{analise}", arquivo_csv)
	arquivo_csv = arquivo_csv.drop(columns = colunas)
	arquivo_csv = arquivo_csv.sum()
	printa(f"{analise}", arquivo_csv)
	if analise in ["soma_incidencia", "soma_ponderados"]:
		printa("censo", censo)
		arquivo_csv = (arquivo_csv / censo["Populacao"]) * 100000
		arquivo_csv = arquivo_csv.dropna().round(2)
		printa(f"Final: {analise}", arquivo_csv)
	arquivo_csv = arquivo_csv.sort_values(ascending = False).head(visu)
	printa(f"{analise}", arquivo_csv)
	#input(f"{green}Selecione um município: {reset}")


print(f"{green}Insira o comportamento a ser analisado.{reset}")
print(f"{cyan}incidencia\ncasos\nfocos\n{reset}")
analise = input(f"{green}Insira o comportamento a ser analisado... {reset}")

match analise:
	case "casos":
		colunas = ["index", "Semana"]
		arquivo = f"{caminho}/epidemiologia/ultimas_previsoes_vSE{SE}_h0_r2.csv"
		previsao(analise, arquivo, colunas)
	case "focos":
		colunas = ["level_0", "index", "Semana"]
		arquivo = f"{caminho}/entomologia/ultimas_previsoes_focos_vSE{SE}_h2_r4.csv"
		previsao(analise, arquivo, colunas)
	case "incidencia":
		colunas = ["index", "Semana"]
		arquivo = f"{caminho}/epidemiologia/ultimas_previsoes_incidencia_vSE{SE}_h0_r2.csv"
		previsao(analise, arquivo, colunas)
	case "soma_casos":
		arquivo = f"{caminho_dados}/casos_semanal_pivot.csv"
		colunas = ["Semana"]
		monitoramento(analise, arquivo, colunas)
	case "soma_focos":
		arquivo = f"{caminho_dados}/focos_semanal_pivot.csv"
		colunas = ["Semana", "semana"]
		monitoramento(analise, arquivo, colunas)
	case "soma_incidencia":
		arquivo = f"{caminho_dados}/casos_semanal_pivot.csv"
		colunas = ["Semana"]
		monitoramento(analise, arquivo, colunas)
	case "soma_ponderados":
		arquivo = f"{caminho_dados}/focos_semanal_pivot.csv"
		colunas = ["Semana", "semana"]
		monitoramento(analise, arquivo, colunas, 50)



#CASOS: ultimas_previsoes_vSE14_h0_r2.csv
#INCIDENCIA: ultimas_previsoes_incidencia_vSE14_h0_r2.csv
#FOCOS: ultimas_previsoes_focos_vSE14_h2_r4.csv
#PONDERADOS: 
