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
#
###-------------------------------------------------------------
### IMPORTANDO BIBLIOTECAS ##
###-------------------------------------------------------------
#
import xarray as xr
from netCDF4 import Dataset                     # Read / Write NetCDF4 files
import matplotlib
import matplotlib.pyplot as plt                 #Figure
from matplotlib import cm                       # Colormap handling utilities
import matplotlib.colors as cls
import pandas as pd
import numpy as np
import cartopy, cartopy.crs as ccrs        # Plot maps
import cartopy.io.shapereader as shpreader # Import shapefiles
import cartopy.crs as crs
from cartopy.feature import ShapelyFeature
import cartopy.feature as cfeature
import cmocean
#
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
caminho_resultado = "/home/meteoro/scripts/matheus/operacional_dengue/"
try:
	arquivo_tmin = f"SAMeT_CPTEC_DAILY_SB_TMIN_{_ANO_ATUAL}.nc"
	arquivo_tmed = f"SAMeT_CPTEC_DAILY_SB_TMED_{_ANO_ATUAL}.nc" 
	arquivo_tmax = f"SAMeT_CPTEC_DAILY_SB_TMAX_{_ANO_ATUAL}.nc" 
	os.makedirs(f"{caminho_resultado}/{_ANO_ATUAL}", mode = 0o777, exist_ok = True) 
	ds_tmin = xr.open_dataset(f"{caminho_samet}/TMIN/{_ANO_ATUAL}/{arquivo_tmin}")
	ds_tmed = xr.open_dataset(f"{caminho_samet}/TMED/{_ANO_ATUAL}/{arquivo_tmed}")
	ds_tmax = xr.open_dataset(f"{caminho_samet}/TMAX/{_ANO_ATUAL}/{arquivo_tmax}")
except FileNotFoundError:
	arquivo_tmin = f"SAMeT_CPTEC_DAILY_SB_TMIN_{_ANO_ONTEM}.nc"
	arquivo_tmed = f"SAMeT_CPTEC_DAILY_SB_TMED_{_ANO_ONTEM}.nc" 
	arquivo_tmax = f"SAMeT_CPTEC_DAILY_SB_TMAX_{_ANO_ONTEM}.nc" 
	os.makedirs(f"{caminho_resultado}/{_ANO_ONTEM}", mode = 0o777, exist_ok = True) 
	ds_tmin = xr.open_dataset(f"{caminho_samet}/TMIN/{_ANO_ONTEM}/{arquivo_tmin}")
	ds_tmed = xr.open_dataset(f"{caminho_samet}/TMED/{_ANO_ONTEM}/{arquivo_tmed}")
	ds_tmax = xr.open_dataset(f"{caminho_samet}/TMAX/{_ANO_ONTEM}/{arquivo_tmax}")

#################################################################################
# DEFININDO FUNÇÕES

def selecionar_tempo_espaco(dataset, tempo, str_var):
	"""
	"""
	lat_min = -34.00
	lat_max = -21.75
	lon_min = -58.25
	lon_max = -47.50
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
	dataset_espaco = dataset_espaco.resample(time = "W-SUN").mean()
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
	
def limite_minmax(dataset):	
	abs_value = int(max(abs(dataset.min()), abs(dataset.max())) + 1)
	if (abs_value <= 3):
		levels = np.linspace(-abs_value, abs_value, 4*abs_value + 1)
		levels = np.arange(-abs_value, abs_value + 0.5, 0.5)
	else:
		levels = np.linspace(-abs_value, abs_value, 2*abs_value + 1)
		norm = cls.Normalize(-abs_value, abs_value)
	return levels, norm
	
def limite_colobar(regiao_tmin, regiao_tmax):
	max_tmax = regiao_tmax.max().item()
	int_max = int(max_tmax) + 1
	min_tmin = regiao_tmin.min().item()
	int_min = int(min_tmin)
	if ((int_max - int_min)//2 != (int_max-int_min)/2):
		int_max += 1
	levels = range(int_min, int_max + 2, 2)
	norm = cls.Normalize(vmin = int_min, vmax = int_max)
	cmap = plt.get_cmap("RdYlBu_r")
	print(f"\n{green}Valor máximo da temperatura máxima: {reset}{round(max_tmax, 2)} °C\n")
	print(f"\n{green}Valor mínimo da temperatura mínima: {reset}{round(min_tmin, 2)} °C\n")
	return levels, norm, cmap
	
def gerar_mapa(dataset, str_var):
	"""
	Função relativa à síntese de mapas temáticos de temperatura utilizando SAMeT.
	entrada:
	- arquivo = regiao_tmin, regiao_tmed ou regiao_tmax;
	- str_var = string da variável de interesse (tmin, tmed ou tmax).
	retorno:
	- mapa temático com a variável de interesse.
	"""
	plt.figure(figsize=(8, 6))#, layout = "tight", frameon = False)
	ax = plt.axes(projection=ccrs.PlateCarree())
	shp = list(shpreader.Reader(f"{caminho_shapefile}/BR_UF_2019.shp").geometries())
	figure = regiao_tmin.plot.pcolormesh(levels = levels, cmap = cmap, norm = norm,
										add_colorbar = False,  add_labels = False,
										robust = True, extend = "both") 
	_d7 = datetime.today() - timedelta(days = 7)
	_d7 = _d7 - timedelta(days = _d7.weekday() + 1)
	_d7 = _d7.strftime("%Y-%m-%d")
	print(f"\n{green}DOMINGO: {reset}{_d7} °C\n")
	match str_var:
		case "tmin":
			#figure = regiao_tmin.plot.pcolormesh(robust=True, cmap=cmap, add_colorbar=False, levels=levels, add_labels=False, extend='both', norm = norm)
			plt.title(f"Temperatura Mínima Média Semanal (°C)\nPeríodo observado: {_d7}",
						fontsize =14, ha = "center")
		case "tmed":
			figure = regiao_tmed.plot.pcolormesh(robust=True, norm=norm, cmap=cmap, add_colorbar=False, levels=levels, add_labels=False, extend='both')
			plt.title(f'Temperatura Média Mensal (°C)\nSul do Brasil, Período observado: {regiao_tmed.time}', fontsize=14, ha='center')
		case "tmax":
			figure = regiao_tmax.plot.pcolormesh(robust=True, norm=norm, cmap=cmap, add_colorbar=False, levels=levels, add_labels=False, extend='both')
			plt.title(f'Temperatura Máxima Média Mensal (°C)\nSul do Brasil, Período observado: {regiao_tmax.time}', fontsize=14, ha='center')
		case _:
			print(f"\nVariável não encontrada!\n{str_var}\nVariável não encontrada!\n")
			
	ax.add_geometries(shp, ccrs.PlateCarree(), edgecolor='gray', facecolor='none', linewidth=0.3)
	ax.coastlines(resolution='10m', color='black', linewidth=0.8)
	ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5)
	gl = ax.gridlines(crs=ccrs.PlateCarree(), color='white', alpha=1.0, linestyle='--',
			linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=True)
	gl.top_labels = False
	gl.right_labels = False
	plt.colorbar(figure, pad=0.05, fraction=0.05, label='Temperatura Semanal (°C)',ticks=levels, orientation='vertical')
	#ticks=np.arange(int_min, int_max+1, (int_max-int_min)/10), orientation='vertical')	
	# Adicionar a fonte no rodapé
	plt.figtext(0.55, 0.05, 'Fonte: SAMeT - CPTEC', ha='center', fontsize=10)
	# Salvar a figura no formato ".jpg" com dpi=300.
	#plt.savefig(f"{caminho_resultado}/{ano}/{str_var}_mensal_samet_media_{ano}{mes}.png", transparent=False, dpi=300, bbox_inches="tight", pad_inches=0.02)
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

levels, norm, cmap = limite_colobar(regiao_tmin, regiao_tmax)
info_dataset(regiao_tmin)
gerar_mapa(regiao_tmin, "tmin")

sys.exit()

# Encontrar o valor máximo da temperatura máxima e mínimo da temperatura mínima na região selecionada
max_tmax = regiao_tmax.max().item()
int_max = int(max_tmax) + 1
min_tmin = regiao_tmin.min().item()
int_min = int(min_tmin)

#Variáveis para o plot do mapa e da colorbar
if ((int_max - int_min)//2 != (int_max-int_min)/2):
	int_max += 1
levels = range(int_min, int_max + 2, 2)
#levels = np.linspace(int_min, int_max, 10)
#print(levels)
#levels = levels.astype(int)
#print(levels)
norm = cls.Normalize(vmin = int_min, vmax = int_max)
cmap = plt.get_cmap("RdYlBu_r")
#cmap.set_over('#800026')
#cmap.set_under('#040273')
print(f"Valor máximo da temperatura na região selecionada: {round(max_tmax, 2)} °C")
print(f"Valor mínimo da temperatura na região selecionada: {round(min_tmin, 2)} °C")
# Plotar temperatura de um dia específico



gerar_mapa("tmin")
gerar_mapa("tmed")
gerar_mapa("tmax")
print(f"Figuras de médias de {ano}-{mes} completas")
print("=="*50)
	
