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

t0 = datetime.now()# - timedelta(days = 7)
SE = int(str(Week.fromdate(t0))[-2:])
EPI_YEAR = int(str(Week.fromdate(t0))[:-2])
EPI_YY_START = pd.to_datetime(Week(EPI_YEAR, 1).startdate())
caminho = f"/home/meteoro/scripts/operacional_dengue/resultados/{EPI_YEAR}/SE{SE}"
caminho_dados = "/home/meteoro/scripts/operacional_dengue/dados_operacao/"
censo = "censo_sc_xy.csv"
censo = pd.read_csv(f"{caminho_dados}{censo}")
censo = censo[["Municipio", "Populacao"]].set_index("Municipio")
censo.index = censo.index.str.upper()
regional = "censo_sc_regional.csv"
regional = pd.read_csv(f"{caminho_dados}{regional}")
mapeamento = regional.drop_duplicates(subset = ["Municipio"]).set_index("Municipio")["regional"]

def printa(str_var, var):
	STR_VAR = str_var.upper()
	print(f"\n{green}{STR_VAR}: \n{reset}{var}\n")
	
printa("se", SE)
printa("ano_epi", EPI_YEAR)
#printa("censo", censo)
#printa("regional", regional)
#printa("mapeamento", mapeamento)


def visualiza(analise, arquivo, visu = 50, monitora = None):
	arquivo_csv = pd.read_csv(arquivo)	
	#printa(f"{analise}", arquivo_csv)
	if analise in ["casos_previsao", "incidencia_previsao", "focos_previsao"]:
		#arquivo_csv["regional"] = arquivo_csv["Municipio"].map(mapeamento)
		print("VISUALIZANDO PREVISÃO")
		S0 = arquivo_csv.loc[0].sort_values(ascending = False).head(visu)
		S0 = S0.to_frame(name = "total")
		S0["regional"] = S0.index.map(mapeamento)
		S1 = arquivo_csv.loc[1].sort_values(ascending = False).head(visu)
		S1 = S1.to_frame(name = "total")
		S1["regional"] = S1.index.map(mapeamento)
		S2 = arquivo_csv.loc[2].sort_values(ascending = False).head(visu)
		S2 = S2.to_frame(name = "total")
		S2["regional"] = S2.index.map(mapeamento)
		printa(f"PREVISAO DE {analise} PARA SE{SE}", S0)
		printa(f"PREVISAO DE {analise} PARA SE{SE + 1}", S1)
		printa(f"PREVISAO DE {analise} PARA SE{SE + 2}", S2)
	elif analise in ["casos_monitoramento", "incidencia_monitoramento", "focos_monitoramento", "ponderados_monitoramento"]:
		variavel = {"incidencia_monitoramento": "incidencia",
		"casos_monitoramento": "casos",
		"focos_monitoramento": "focos",
		"ponderados_monitoramento": "focoponderados"}
		arquivo_csv = arquivo_csv[["Municipio", monitora]]
		arquivo_csv = arquivo_csv.sort_values(by = monitora, ascending = False).head(visu).reset_index(drop=True)
		arquivo_csv["regional"] = arquivo_csv["Municipio"].map(mapeamento)
		#arquivo_csv = arquivo_csv[["Municipio", "regional", "total"]]
		#arquivo_csv = arquivo_csv.sort_values(by = monitora, ascending = False).head(visu)
		printa(f"monitoramento de {variavel[analise]} até a SE{SE-1}", arquivo_csv)

print(f"{green}Insira o comportamento a ser analisado.{reset}")
print(f"{cyan}incidencia_monitoramento\ncasos_monitoramento\nfocos_monitoramento\nponderados_monitoramento\ncasos_previsao\nincidencia_previsao\nfocos_previsao{reset}")
analise = input(f"{green}Insira o comportamento a ser analisado... {reset}")

match analise:
	case "casos_previsao":
		arquivo = f"{caminho}/epidemiologia/visualiza_preditivo_casos_vSE{SE}.csv"
		visualiza(analise, arquivo, visu = 30)
	case "focos_previsao":
		arquivo = f"{caminho}/entomologia/visualiza_preditivo_focos_vSE{SE}.csv"
		visualiza(analise, arquivo, visu = 30)
	case "incidencia_previsao":
		arquivo = f"{caminho}/epidemiologia/visualiza_preditivo_incidencia_vSE{SE}.csv"
		visualiza(analise, arquivo, visu = 30)
	case "casos_monitoramento":
		arquivo = f"{caminho}/epidemiologia/visualiza_monitoramento_epidemio_vSE{SE}.csv"
		visualiza(analise, arquivo, visu = 30, monitora = "total")
	case "focos_monitoramento":
		arquivo = f"{caminho}/entomologia/visualiza_monitoramento_entomo_vSE{SE}.csv"
		visualiza(analise, arquivo, visu = 30, monitora = "total")
	case "incidencia_monitoramento":
		arquivo = f"{caminho}/epidemiologia/visualiza_monitoramento_epidemio_vSE{SE}.csv"
		visualiza(analise, arquivo, visu = 30, monitora = "incidencia")
	case "ponderados_monitoramento":
		arquivo = f"{caminho}/entomologia/visualiza_monitoramento_entomo_vSE{SE}.csv"
		visualiza(analise, arquivo, visu = 30, monitora = "incidencia")
sys.exit()
print("=="*50)
previsao("casos", f"{caminho}/epidemiologia/previsao_pivot_total_vSE{SE}_h0_r2.csv", ["index", "Semana"])
print("=="*50)
previsao("casos_1", f"{caminho}/epidemiologia/ultimas_previsoes_vSE{SE}_h0_r2.csv", ["index", "Semana"])


#CASOS: ultimas_previsoes_vSE14_h0_r2.csv
#INCIDENCIA: ultimas_previsoes_incidencia_vSE14_h0_r2.csv
#FOCOS: ultimas_previsoes_focos_vSE14_h2_r4.csv
#PONDERADOS: 
