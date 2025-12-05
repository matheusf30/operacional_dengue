#!/usr/bin/env python
# coding: utf-8
#
#################################################################################
###
### IFSC DADOS Temperatura SAMET - MAPAS
### COMO RODAR: executar o arquivo python e em seguida inserir ano-mês
###  Ex.:
###  python tmed_mensal_samet.py 2024-05
###                  AUTORA: Roseli de Oliveira            - 2024/07/11
###                  Adaptado: Mario Quadro                - 2024/09/05
###                  Atualizado: Everton Weber Galliani    - 2025/09/02
###                              Matheus Ferreira de Souza - 2025/09/15
#################################################################################

#################################################################################
# IMPORTANDO BIBLIOTECAS
import xarray as xr
from netCDF4 import Dataset
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as cls
import pandas as pd
import geopandas as gpd
import numpy as np
import cartopy, cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.crs as crs
from cartopy.feature import ShapelyFeature
import cartopy.feature as cfeature
import cmocean
import regionmask
import sys
import os
from datetime import date, datetime, timedelta

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

##################### Valores Booleanos ############ # sys.argv[0] is the script name itself and can be ignored!
_VISUALIZAR = sys.argv[1]    # True|False                    #####
_VISUALIZAR = True if _VISUALIZAR == "True" else False       #####
_SALVAR = sys.argv[2]        # True|False                    #####
_SALVAR = True if _SALVAR == "True" else False               #####
##################################################################

print(f"\n{red}Argumentos:")#\n{reset}Visualizar = {_VISUALIZAR}\n")
print(f"Visualizar = {_VISUALIZAR}")
print(f"Salvar = {_SALVAR}\n{reset}")

#################################################################################
# DATA DO SISTEMA (POSSIBILIDADE DE ACESSAR ARQUIVO DE 1 DIA)
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

#################################################################################
# CAMINHOS E ARQUIVOS
caminho_samet = "/media/dados/operacao/samet/daily/"
caminho_shapefile = "/media/dados/shapefiles/BR/"
caminho_resultado = f"/home/meteoro/scripts/operacional_dengue/meteorologia/{_ANO_ATUAL}/"

# Climatologia de semanas epidemiológicas (by Everton)
def calcula_numero_se(dia):
	N_SE = 0
	dia = pd.to_datetime(dia)
	# Retornando ao domingo da semana
	wkd = dia.weekday()
	domingo = dia - timedelta(days = wkd + 1)
	ano_func = domingo.strftime("%Y")
	dia_inicio = pd.to_datetime(f"{ano_func}-01-01")
	while domingo >= dia_inicio:
		N_SE += 1
		domingo -= timedelta(days = 7)
	if (dia_inicio - domingo).days <= 3:
		N_SE += 1
	return N_SE
SE = calcula_numero_se(_ANO_MES_DIA) - 1
print(SE)

tmed_climatologia = xr.open_dataset("/home/meteoro/scripts/operacional_dengue/meteorologia/climatologia/tmed_climatologia_epidemiosemanal.nc").sel(week = SE)['tmed']

os.makedirs(f"{caminho_resultado}", mode = 0o777, exist_ok = True)
municipios = "/media/dados/shapefiles/SC/SC_Municipios_2024.shp"
regionais = "/home/meteoro/scripts/operacional_dengue/dados_operacao/censo_sc_regional.csv"
municipios = gpd.read_file(municipios, low_memory = False)
sc_shape = gpd.read_file(f"/media/dados/shapefiles/SC/SC_UF_2024.shp")
regionais = pd.read_csv(regionais, low_memory = False)
try:
	arquivo_tmin = f"SAMeT_CPTEC_DAILY_SB_TMIN_{_ANO_ATUAL}.nc"
	arquivo_tmed = f"SAMeT_CPTEC_DAILY_SB_TMED_{_ANO_ATUAL}.nc" 
	arquivo_tmax = f"SAMeT_CPTEC_DAILY_SB_TMAX_{_ANO_ATUAL}.nc" 
	ds_tmin = xr.open_dataset(f"{caminho_samet}/TMIN/{_ANO_ATUAL}/{arquivo_tmin}")
	ds_tmed = xr.open_dataset(f"{caminho_samet}/TMED/{_ANO_ATUAL}/{arquivo_tmed}")
	ds_tmax = xr.open_dataset(f"{caminho_samet}/TMAX/{_ANO_ATUAL}/{arquivo_tmax}")
except FileNotFoundError:
	arquivo_tmin = f"SAMeT_CPTEC_DAILY_SB_TMIN_{_ANO_ONTEM}.nc"
	arquivo_tmed = f"SAMeT_CPTEC_DAILY_SB_TMED_{_ANO_ONTEM}.nc" 
	arquivo_tmax = f"SAMeT_CPTEC_DAILY_SB_TMAX_{_ANO_ONTEM}.nc" 
	ds_tmin = xr.open_dataset(f"{caminho_samet}/TMIN/{_ANO_ONTEM}/{arquivo_tmin}")
	ds_tmed = xr.open_dataset(f"{caminho_samet}/TMED/{_ANO_ONTEM}/{arquivo_tmed}")
	ds_tmax = xr.open_dataset(f"{caminho_samet}/TMAX/{_ANO_ONTEM}/{arquivo_tmax}")

#################################################################################
municipios["NM_MUN"] = municipios["NM_MUN"].str.upper()
municipios = municipios.merge(regionais[["Municipio", "regional"]],
								left_on = "NM_MUN", right_on = "Municipio",
								how = "left")
regionais = municipios.dissolve(by = "regional")

#################################################################################
### DEFININDO FUNÇÕES

def selecionar_tempo_espaco(dataset, tempo, str_var):
	"""
	"""
	lat_min = -29.45#-29.5 # -34.00 # -29.5
	lat_max = -25.65#-25.75 # -21.75 # -25.75
	lon_min = -54.15#-54 # -58.25 # -54
	lon_max = -47.85#-48 # -47.50 # -48
	match str_var:
		case "tmin":
			dataset_espaco = dataset.sel(time = tempo,
								lat = slice(lat_min, lat_max),
								lon = slice(lon_min, lon_max)).tmin.squeeze()
		case "tmed":
			dataset_espaco = dataset.sel(time = tempo,
							lat = slice(lat_min, lat_max),
							lon = slice(lon_min, lon_max)).tmed.squeeze()
		case "tmax":
			dataset_espaco = dataset.sel(time = tempo,
								lat = slice(lat_min, lat_max),
								lon = slice(lon_min, lon_max)).tmax.squeeze()
	dataset_espaco = dataset_espaco.resample(time = "W-SAT").mean()
	dataset_tempo_espaco = dataset_espaco.isel(time = -2)
	return dataset_tempo_espaco
	
def info_dataset(dataset):
	"""
	"""
	print(f"\n{green}ARQUIVO:\n{reset}{dataset}\n")
	print(f"{green}*{reset}*"*30)
	print(f"\n{green}LATITUDES:\n{reset}{dataset.lat}\n")
	print(f"{green}*{reset}*"*30)
	print(f"\n{green}LONGITUDES:\n{reset}{dataset.lon}\n")
	print(f"{green}*{reset}*"*30)
	print(f"\n{green}TEMPO:\n{reset}{dataset.time}\n")
	print(f"{green}*{reset}*"*30)
	print(f"\n{green}VARIÁVEIS:\n{reset}{dataset.values}\n")
	print(f"{green}*{reset}*"*30)
	print(f"\n{green}ARQUIVO:\n{reset}{dataset}\n")
	print(f"{green}*{reset}*"*30)
	
def limite_minmax_anomalia(dataset):	
	abs_value = int(max(abs(dataset.min()), abs(dataset.max())) + 1)
	int_min = -abs_value
	int_max = abs_value
	if (abs_value <= 3):
		#levels = np.linspace(-abs_value, abs_value, 4*abs_value + 1)
		levels = np.arange(-abs_value, abs_value + 0.5, 0.5)
	else:
		levels = np.linspace(-abs_value, abs_value, 2*abs_value + 1)
	norm = cls.Normalize(-abs_value, abs_value)
	cmap = plt.get_cmap("RdBu_r")
	return levels, norm, int_min, int_max, cmap
	
def limite_colobar(regiao_tmin, regiao_tmax):
	max_tmax = regiao_tmax.max().item()
	int_max = int(max_tmax) - 2
	min_tmin = regiao_tmin.min().item()
	int_min = int(min_tmin) + 2
	if ((int_max - int_min)//2 != (int_max-int_min)/2):
		int_max += 1
	levels = range(int_min, int_max + 2, 2)
	norm = cls.Normalize(vmin = int_min, vmax = int_max)
	cmap = plt.get_cmap("RdYlBu_r")
	print(f"\n{green}Valor máximo da temperatura máxima: {reset}{round(max_tmax, 2)} °C\n")
	print(f"\n{green}Valor mínimo da temperatura mínima: {reset}{round(min_tmin, 2)} °C\n")
	return levels, norm, int_min, int_max, cmap

def mascara(dataset):
	shape_estado = sc_shape
	var = dataset
	mask = regionmask.mask_geopandas(shape_estado, var.lon, var.lat)
	dados_mascarados = var.where(mask >= 0)
	media = dados_mascarados.mean().values
	media = media.round(1)
	return media
	
def quadradinho_do_mario(media_sc, comportamento = None):
	"""
	"""
	if comportamento == "anomalia":
		plt.text(-52.5, -29, f"Anomalia de SC:\n {media_sc} °C\nFonte: SAMeT - CPTEC/INPE",
				color="black", 
				ha="center", va="center", fontsize=12, zorder=10,
				bbox=dict(boxstyle="square", facecolor="white", edgecolor="black"))
	else:
		plt.text(-52.5, -29, f"Média de SC:\n {media_sc} °C\nFonte: SAMeT - CPTEC/INPE",
				color="black", 
				ha="center", va="center", fontsize=12, zorder=10,
				bbox=dict(boxstyle="square", facecolor="white", edgecolor="black"))
	
def gerar_mapa(dataset, str_var, comportamento):
	"""
	Função relativa à síntese de mapas temáticos de temperatura utilizando SAMeT.
	entrada:
	- arquivo = regiao_tmin, regiao_tmed ou regiao_tmax;
	- str_var = string da variável de interesse (tmin, tmed ou tmax).
	retorno:
	- mapa temático com a variável de interesse.
	"""
	plt.figure(figsize=(8, 5.3), layout = "constrained", frameon = True)
	ax = plt.axes(projection=ccrs.PlateCarree())
	shp = list(shpreader.Reader(f"{caminho_shapefile}/BR_UF_2022.shp").geometries())
	figure = dataset.plot.contourf( ax = ax, levels = levels, cmap = cmap, norm = norm,
										add_colorbar = False,  add_labels = False,
										robust = True, extend = "both",
										transform = ccrs.PlateCarree())
	linhas = dataset.plot.contour(ax = ax, levels = levels,
								colors = "black", linewidths = 0.3,	transform = ccrs.PlateCarree())
	rotulos = ax.clabel(linhas, inline = True, fmt = "%1.0f", fontsize = 8, colors = "black")
	for rotulo in rotulos:
		rotulo.set_rotation(0)
	plt.colorbar(figure, fraction = 0.035, pad = 0.03, ticks = levels,
				label = "Temperatura Semanal (°C)", orientation = "vertical", extend = "max")
	ax = plt.gca()
	regionais.plot(ax = ax, facecolor = "none",
				edgecolor = "dimgray", linewidth = 0.7)
	_d7 = datetime.today() - timedelta(days = 7)
	_d7 = _d7 - timedelta(days = _d7.weekday() + 1)
	_d8 = _d7 + timedelta(days = 6)
	_d7 = _d7.strftime("%Y-%m-%d")
	_d8 = _d8.strftime("%Y-%m-%d")
	print(f"\n{green}{str_var} - DOMINGO: {reset}{_d7}\n")
	if (comportamento == "media"):
		match str_var:
			case "tmin":
				plt.title(f"Temperatura Mínima na Semana Epidemiológica Nº {SE}\nPeríodo Observado: {_d7} a {_d8}",
							fontsize = 14, ha = "center")
			case "tmed":
				plt.title(f"Temperatura Média na Semana Epidemiológica Nº {SE}\nPeríodo Observado: {_d7} a {_d8}",
							fontsize = 14, ha = "center")
			case "tmax":
				plt.title(f"Temperatura Máxima na Semana Epidemiológica Nº {SE}\nPeríodo Observado: {_d7} a {_d8}",
							fontsize = 14, ha = "center")
			case _:
				print(f"\nVariável não encontrada!\n{str_var}\nVariável não encontrada!\n")
	elif (comportamento == "anomalia"):
		match str_var:
			case "tmin":
				plt.title(f"Anomalia de Temperatura Mínima na Semana Epidemiológica Nº {SE}\nPeríodo Observado: {_d7} a {_d8}",
							fontsize = 14, ha = "center")
			case "tmed":
				plt.title(f"Anomalia de Temperatura Média na Semana Epidemiológica Nº {SE}\nPeríodo Observado: {_d7} a {_d8}",
							fontsize = 14, ha = "center")
			case "tmax":
				plt.title(f"Anomalia de Temperatura Máxima na Semana Epidemiológica Nº {SE}\nPeríodo Observado: {_d7} a {_d8}",
							fontsize = 14, ha = "center")
			case _:
				print(f"\nVariável não encontrada!\n{str_var}\nVariável não encontrada!\n")
	media = mascara(dataset)
	if comportamento == "anomalia":
		quadradinho_do_mario(media, comportamento)	
	else:
		quadradinho_do_mario(media)			
	ax.add_geometries(shp, ccrs.PlateCarree(), edgecolor = "black",
					facecolor = "none", linewidth = 0.5)
	ax.coastlines(resolution = "10m", color = "black", linewidth = 0.8)
	ax.add_feature(cartopy.feature.BORDERS, edgecolor = "black", linewidth = 0.5)
	gl = ax.gridlines(crs=ccrs.PlateCarree(), color="white", alpha=1.0,
				linestyle="--", linewidth=0.25, draw_labels = False)
	plt.ylabel("Latitude")
	plt.xlabel("Longitude")
	plt.xlim(-54.05, -47.95)
	plt.ylim(-29.45, -25.75)
	ax.tick_params(axis = "both")
	ax.set_xticks([-54, -52, -50, -48])
	ax.set_xticklabels(["54°W", "52°W", "50°W", "48°W"])
	ax.set_yticks([-29, -28, -27, -26])
	ax.set_yticklabels(["29°S", "28°S", "27°S", "26°S"])
	if _SALVAR == True:
		print(f"\n{red}Figura salva como:")
		print(f"{caminho_resultado}{str_var}_semanal_samet_{comportamento}_{SE}.png\n{reset}")
		plt.savefig(f"{caminho_resultado}{str_var}_semanal_samet_{comportamento}_{_ANO_ATUAL}{SE}.png",
					transparent = False, dpi = 300, pad_inches = 0.02, format = "png")
	if _VISUALIZAR == True:
		plt.show()
	
#################################################################################
# EXECUTANDO FUNÇÕES
	
try:
	regiao_tmin = selecionar_tempo_espaco(ds_tmin, _ANO_ATUAL, "tmin")
	regiao_tmed = selecionar_tempo_espaco(ds_tmed, _ANO_ATUAL, "tmed")
	regiao_tmax = selecionar_tempo_espaco(ds_tmax, _ANO_ATUAL, "tmax")
except FileNotFoundError:
	regiao_tmin = selecionar_tempo_espaco(ds_tmin, _ANO_ONTEM, "tmin")
	regiao_tmed = selecionar_tempo_espaco(ds_tmed, _ANO_ONTEM, "tmed")
	regiao_tmax = selecionar_tempo_espaco(ds_tmax, _ANO_ONTEM, "tmax")

levels, norm, int_min, int_max, cmap = limite_colobar(regiao_tmin, regiao_tmax)

info_dataset(regiao_tmed)
gerar_mapa(regiao_tmed, "tmed", "media")
regiao_tmed -= tmed_climatologia
levels, norm, int_min, int_max, cmap = limite_minmax_anomalia(regiao_tmed)
info_dataset(regiao_tmed)
gerar_mapa(regiao_tmed, "tmed", "anomalia")
	



	
