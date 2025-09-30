#################################################################################
###
### DIVE/SC DADOS Epidemiologia e Índices Epidemiológicos
### COMO RODAR: executar o arquivo python e em seguida inserir 3 argumentos booleanos:
### (Automatiza) (Visualiza) (Salva)
###  Ex.:
###  cartografia_monitoramento_epidemio.py True False True 
###                  Atualizado: Matheus Ferreira de Souza  - 2025/09/15
#################################################################################
### Bibliotecas Correlatas
# Básicas e Gráficas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cls  
import seaborn as sns 
from datetime import date, datetime, timedelta
# Suporte
import os
import sys
import joblib
import webbrowser
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category = ShapelyDeprecationWarning)
# Mapas
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.patches as mpatches
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

#### Condições para Variar ####################################

##################### Valores Booleanos ############ # sys.argv[0] is the script name itself and can be ignored!
_AUTOMATIZAR = sys.argv[1]   # True|False                    #####
_AUTOMATIZA = True if _AUTOMATIZAR == "True" else False      #####
_VISUALIZAR = sys.argv[2]    # True|False                    #####
_VISUALIZAR = True if _VISUALIZAR == "True" else False       #####
_SALVAR = sys.argv[3]        # True|False                    #####
_SALVAR = True if _SALVAR == "True" else False               #####
##################################################################

_AGORA = datetime.now()
_ANO_ATUAL = str(datetime.today().year)
_MES_ATUAL = _AGORA.strftime("%m")
_DIA_ATUAL = _AGORA.strftime("%d")
_ANO_MES = f"{_ANO_ATUAL}{_MES_ATUAL}"
_ANO_MES_DIA = f"{_ANO_ATUAL}{_MES_ATUAL}{_DIA_ATUAL}"
_ONTEM = datetime.today() - timedelta(days = 1)
_ANO_ONTEM = str(_ONTEM.year)
_MES_ONTEM = _ONTEM.strftime("%m")
_DIA_ONTEM = _ONTEM.strftime("%d")
_ANO_MES_ONTEM = f"{_ANO_ONTEM}{_MES_ONTEM}"
_ANO_MES_DIA_ONTEM = f"{_ANO_ONTEM}{_MES_ONTEM}{_DIA_ONTEM}"

##################################################################################

### Encaminhamento aos Diretórios
caminho_dados = "/home/meteoro/scripts/matheus/operacional_dengue/dados_operacao/" # CLUSTER
caminho_operacional = "/home/meteoro/scripts/matheus/operacional_dengue/"
caminho_shape = "/media/dados/shapefiles/" #SC/SC_Municipios_2022.shp #BR/BR_UF_2022.shp
caminho_modelos = f"/home/meteoro/scripts/matheus/operacional_dengue/modelagem/casos/{_ANO_ATUAL}/{_ANO_MES_DIA}/"
caminho_resultados = f"modelagem/resultados/{_ANO_ATUAL}/{_ANO_MES}/"
print(f"\nOS DADOS UTILIZADOS ESTÃO ALOCADOS NOS SEGUINTES CAMINHOS:\n\n{caminho_dados}\n\n")


##################################################################################
### Renomeação das Variáveis pelos Arquivos
casos = "casos_dive_pivot_total.csv"  # TabNet/DiveSC
# focos?
unicos = "casos_primeiros.csv"
municipios = "SC/SC_Municipios_2024.shp"
br = "BR/BR_UF_2022.shp"
censo = "censo_sc_xy.csv"

##################################################################################
### Abrindo Arquivo
casos = pd.read_csv(f"{caminho_dados}{casos}", low_memory = False)
#focos = pd.read_csv(f"{caminho_dados}{focos}", low_memory = False)
censo = pd.read_csv(f"{caminho_dados}{censo}", low_memory = False)
unicos = pd.read_csv(f"{caminho_dados}{unicos}")
municipios = gpd.read_file(f"{caminho_shape}{municipios}")
br = gpd.read_file(f"{caminho_shape}{br}")
troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A',
         'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E',
         'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I',
         'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O',
         'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U',
         'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
cidades = unicos["Município"].copy()

##################################################################################
### Pré-processamento
print(f"\n{green}CASOS:\n{reset}{casos}\n")
casos["Semana"] = pd.to_datetime(casos["Semana"], format = "%Y-%m-%d")
print(f"\n{green}CASOS.info():\n{reset}{casos.info()}\n")
casos = casos[casos["Semana"].dt.year == datetime.today().year]
semana_epidemio = casos.loc[casos.index[-1], "Semana"]
print(f"\n{green}SEMANA EPIDEMIOLÓGICA:\n{reset}{semana_epidemio}\n")
casos.set_index("Semana", inplace = True)
print(f"\n{green}CASOS.columns:\n{reset}{casos.columns}\n")
print(f"\n{green}CASOS:\n{reset}{casos}\n")
censo["Municipio"] = censo["Municipio"].str.upper()
municipios["NM_MUN"] = municipios["NM_MUN"].str.upper()
print(f"\n{green}MUNICÍPIOS shapefile:\n{reset}{municipios}\n")
print(f"\n{green}CENSO:\n{reset}{censo}\n")
print(f"\n{green}CASOS:\n{reset}{casos}\n")
total = casos.sum()
total_dict = total.to_dict()
print(f"\n{green}TOTAL:\n{reset}{total}\n")
base_carto = censo[["Municipio", "Populacao"]]
base_carto["total"] = base_carto["Municipio"].map(total_dict)
base_carto["incidencia"] = (base_carto["total"] / base_carto["Populacao"]) * 100000
base_carto["incidencia"] = base_carto["incidencia"].round(2)
geom_dict = municipios.set_index("NM_MUN")["geometry"].to_dict()
base_carto["geometry"] = base_carto["Municipio"].map(geom_dict)
base_carto = gpd.GeoDataFrame(base_carto, geometry = "geometry", crs = "EPSG:4674")
print(f"\n{green}DICIONÁRIO geometry:\n{reset}{geom_dict}\n")
print(f"\n{green}BASE CARTOGRÁFICA:\n{reset}{base_carto}\n")



#sys.exit()

##################################################################################
### Cartografia
### Casos
fig, ax = plt.subplots(figsize = (20, 12), layout = "constrained", frameon = True)
municipios.plot(ax = ax, color = "lightgray", edgecolor = "black", linewidth = 0.5)
v_max = base_carto["total"].max()
v_min = base_carto["total"].min()
intervalo = 250
levels = np.arange(v_min, v_max + intervalo, intervalo)
print(f"\n{green}v_min\n{reset}{v_min}\n")
print(f"\n{green}v_max\n{reset}{v_max}\n")
print(f"\n{green}levels\n{reset}{levels}\n")
base_carto.plot(ax = ax, column = "total",  legend = True,
				edgecolor = "black", label = "Casos",
				cmap = "YlOrRd", linewidth = 0.5,
				norm = cls.Normalize(vmin = v_min, vmax = v_max, clip = True))
cbar_ax = ax.get_figure().get_axes()[-1]
cbar_ax.tick_params(labelsize = 20)
plt.xlim(-54, -48)
plt.ylim(-29.5, -25.75)
x_tail = -48.5
y_tail = -29.25
x_head = -48.5
y_head = -28.75
ax.text(-52.5, -29, "Sistema de Referência de Coordenadas\nDATUM: SIRGAS 2000/22S.\nBase Cartográfica: IBGE, 2024.",
	    color = "white", backgroundcolor = "darkgray", ha = "center", va = "center", fontsize = 14)
plt.xlabel("Longitude", fontsize = 18)
plt.ylabel("Latitude", fontsize = 18)
plt.title(f"Soma de Casos de Dengue em Santa Catarina\nSemana Epidemiológica: {semana_epidemio.strftime('%Y-%m-%d')}.", fontsize = 28)

nome_arquivo = f"CASOS_mapa_monitoramento_{_ANO_MES_DIA}.png"
if _AUTOMATIZA == True and _SALVAR == True:
	os.makedirs(caminho_resultados, exist_ok = True)
	plt.savefig(f"{caminho_resultados}{nome_arquivo}", format = "png", dpi = 300)
	print(f"\n\n{green}{caminho_resultados}\n{nome_arquivo}\nSALVO COM SUCESSO!{reset}\n\n")
if _AUTOMATIZA == True and _VISUALIZAR == True:	
	print(f"{cyan}\nVISUALIZANDO:\n{caminho_resultados}\n{nome_arquivo}\n{reset}\n\n")
	plt.show()
	print(f"{cyan}\nENCERRADO:\n{caminho_resultados}\n{nome_arquivo}\n{reset}\n\n")

### Incidência	
fig, ax = plt.subplots(figsize = (20, 12), layout = "constrained", frameon = True)
municipios.plot(ax = ax, color = "lightgray", edgecolor = "black", linewidth = 0.5)
v_max = base_carto["incidencia"].max()
v_min = base_carto["incidencia"].min()
intervalo = 250
levels = np.arange(v_min, v_max + intervalo, intervalo)
print(f"\n{green}v_min\n{reset}{v_min}\n")
print(f"\n{green}v_max\n{reset}{v_max}\n")
print(f"\n{green}levels\n{reset}{levels}\n")
base_carto.plot(ax = ax, column = "incidencia",  legend = True,
				edgecolor = "black", label = "Incidência",
				cmap = "YlOrRd", linewidth = 0.5,
				norm = cls.Normalize(vmin = v_min, vmax = v_max, clip = True))
cbar_ax = ax.get_figure().get_axes()[-1]
cbar_ax.tick_params(labelsize = 20)
plt.xlim(-54, -48)
plt.ylim(-29.5, -25.75)
x_tail = -48.5
y_tail = -29.25
x_head = -48.5
y_head = -28.75
ax.text(-52.5, -29, "Sistema de Referência de Coordenadas\nDATUM: SIRGAS 2000/22S.\nBase Cartográfica: IBGE, 2024.",
	    color = "white", backgroundcolor = "darkgray", ha = "center", va = "center", fontsize = 14)
plt.xlabel("Longitude", fontsize = 18)
plt.ylabel("Latitude", fontsize = 18)
plt.title(f"Incidência da Soma de Casos de Dengue em Santa Catarina.\nSemana Epidemiológica: {semana_epidemio.strftime('%Y-%m-%d')}.", fontsize = 28)

nome_arquivo = f"INCIDENCIA_mapa_monitoramento_{_ANO_MES_DIA}.png"
if _AUTOMATIZA == True and _SALVAR == True:
	os.makedirs(caminho_resultados, exist_ok = True)
	plt.savefig(f"{caminho_resultados}{nome_arquivo}", format = "png", dpi = 300)
	print(f"\n\n{green}{caminho_resultados}\n{nome_arquivo}\nSALVO COM SUCESSO!{reset}\n\n")
if _AUTOMATIZA == True and _VISUALIZAR == True:	
	print(f"{cyan}\nVISUALIZANDO:\n{caminho_resultados}\n{nome_arquivo}\n{reset}\n\n")
	plt.show()
	print(f"{cyan}\nENCERRADO:\n{caminho_resultados}\n{nome_arquivo}\n{reset}\n\n")

