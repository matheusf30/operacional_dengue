### Bibliotecas Correlatas
# Suporte
import sys, os
# Básicas
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt               
import seaborn as sns


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
#################################################################################

### Encaminhamento aos Diretórios
caminho_github = "https://raw.githubusercontent.com/matheusf30/dados_dengue/refs/heads/main/" # WEB
caminho_dados = "/home/meteoro/scripts/matheus/operacional_dengue/dados_operacao/" # CLUSTER
caminho_operacional = "/home/meteoro/scripts/matheus/operacional_dengue/"
caminho_shape = "/media/dados/shapefiles/SC/" #SC_Municipios_2022.shp


### Renomeação variáveis pelos arquivos
censo = "censo_IBGE22.xls"
municipios = "SC_Municipios_2022.shp"

### Abrindo Arquivos
censo = pd.read_excel(f"{caminho_dados}{censo}", skiprows = 1)
municipios = gpd.read_file(f"{caminho_shape}{municipios}")

#sys.exit()
###Visualizando arquivos e variáveis
print(f"\n{green}CENSO IBGE 2022:\n{reset}{censo}\n")
print(f"\n{green}municipios\n{reset}{municipios}\n")

### Pré-processamento e Definição de Função
censo_sc = censo[censo["UF"] == "SC"]
censo_sc = censo_sc[["NOME DO MUNICÍPIO", "POPULAÇÃO"]]
censo_sc = censo_sc.rename(columns = {"NOME DO MUNICÍPIO" : "Municipio", "POPULAÇÃO": "Populacao"})
print(f"\n{green}CENSO IBGE 2022\nSANTA CATARINA:\n{reset}{censo_sc}\n")
municipios = municipios[["NM_MUN", "geometry"]]
municipios = municipios.rename(columns = {"NM_MUN": "Municipio"})
municipios["lat"] = municipios.centroid.y
municipios["lon"] = municipios.centroid.x
municipios = municipios[["Municipio", "lat", "lon"]]
print(f"\n{green}municipios\n{reset}{municipios}\n")
censo_sc_xy = censo_sc.merge(municipios, how = "inner", on = "Municipio")
censo_sc_xy.to_csv(f"{caminho_dados}censo_sc_xy.csv", index = False)
print(f"""
{green}SALVANDO em: {cyan}{caminho_dados}censo_sc_xy.csv

{green}CENSO IBGE 2022
Latitude e Longitude
SANTA CATARINA

{reset}{censo_sc_xy}
""")

