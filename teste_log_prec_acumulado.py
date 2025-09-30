#!/usr/bin/env python
# coding: utf-8
#
#################################################################################
###
### IFSC DADOS Temperatura MERGE - MAPAS
### COMO RODAR: executar o arquivo python e em seguida inserir ano-mês
###  Ex.:
###  python tmed_mensal_samet.py 2024-05
###                  AUTORA: Roseli de Oliveira             - 2024/07/11
###                  Adaptado: Mario Quadro                 - 2024/09/05
###                  Atualizado: Matheus Ferreira de Souza  - 2025/09/15
###                              Everton Weber Galliani     - 2025/09/16
###                              Murilo Ferreira dos Santos - 2025/09/16
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
import numpy as np
import cartopy, cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.crs as crs
from cartopy.feature import ShapelyFeature
import cartopy.feature as cfeature
import cmocean
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
caminho_shapefile = "/media/dados/shapefiles/BR/"
caminho_resultado = f"/home/meteoro/scripts/matheus/operacional_dengue/meteorologia/{_ANO_ATUAL}/"
os.makedirs(f"{caminho_resultado}", mode = 0o777, exist_ok = True) 
try:
	caminho_merge = f"/media/dados/operacao/merge/daily/{_ANO_ATUAL}/"
	arquivo_merge = f"MERGE_CPTEC_DAILY_SB_{_ANO_ATUAL}.nc"
	ds_prec = xr.open_dataset(f"{caminho_merge}{arquivo_merge}")
except FileNotFoundError:
	caminho_merge = f"/media/dados/operacao/merge/daily/{_ANO_ONTEM}/"
	arquivo_merge = f"MERGE_CPTEC_DAILY_SB_{_ANO_ONTEM}.nc"
	ds_prec = xr.open_dataset(f"{caminho_merge}{arquivo_merge}")
	
#################################################################################
# DEFININDO FUNÇÕES

def selecionar_tempo_espaco(dataset, tempo):
	"""
	"""
	lat_min = -29.5 # -34.00 # -29.5
	lat_max = -25.75 # -21.75 # -25.75
	lon_min = -54 # -58.25 # -54
	lon_max = -48 # -47.50 # -48
	dataset_espaco = dataset.sel(time = tempo,
						lat = slice(lat_min, lat_max),
						lon = slice(lon_min, lon_max)).prec.squeeze()
	dataset_espaco = dataset_espaco.resample(time = "W-SAT").sum()
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
	if (abs_value <= 3):
		levels = np.linspace(-abs_value, abs_value, 4*abs_value + 1)
		levels = np.arange(-abs_value, abs_value + 0.5, 0.5)
	else:
		levels = np.linspace(-abs_value, abs_value, 2*abs_value + 1)
		norm = cls.Normalize(-abs_value, abs_value)
	return levels, norm
	
def limite_colobar(regiao_prec):
	max_tmax = regiao_prec.max().item()
	int_max = int(max_tmax) - 10
	min_tmin = 0#regiao_prec.min().item()
	int_min = int(min_tmin)# + 10
	if ((int_max - int_min)//2 != (int_max-int_min)/2):
		int_max += 1
	levels = np.array(range(int_min, int_max + 1, 5))
	levels2 = range(int_min, int_max + 1, 10)
	#norm = cls.Normalize(vmin = int_min, vmax = int_max)
	norm = cls.LogNorm(vmin = int_min + 0.01, vmax = int_max)
	print(f"\n{green}Valor máximo da precipitação: {reset}{round(max_tmax, 2)} mm\n")
	print(f"\n{green}Valor mínimo da precipitação: {reset}{round(min_tmin, 2)} mm\n")
	return levels, levels2, norm
	
def gerar_mapa(dataset):
	"""
	Função relativa à síntese de mapas temáticos de precipitação utilizando MERGE.
	entrada:
	- arquivo = regiao_prec;
	retorno:
	- mapa temático com a variável de interesse.
	"""
	plt.figure(figsize=(8, 6), layout = "constrained", frameon = True)
	ax = plt.axes(projection=ccrs.PlateCarree())
	shp = list(shpreader.Reader(f"{caminho_shapefile}/BR_UF_2022.shp").geometries())
	cmap = plt.get_cmap("YlGnBu")
	#RdYlBu gist_earth_r terrain_r winter_r summer_r YlGnBu
	#viridis_r cividis_r Blues turbo_r jet_r gnuplot2_r gist_ncar_r
	figure = dataset.plot.contourf(cmap = cmap, norm = norm, robust = True,
									add_colorbar = False,  add_labels = False,
									transform = ccrs.PlateCarree(),  levels = levels)
	linhas = dataset.plot.contour(ax = ax, levels = levels2,
								colors = "black", linewidths = 0.5,	transform = ccrs.PlateCarree())
	rotulos = ax.clabel(linhas, inline = True, fmt = "%1.0f", fontsize = 8, colors = "black")
	for rotulo in rotulos:
		rotulo.set_rotation(0)
	plt.colorbar(figure, fraction = 0.031, pad = 0.03, ticks = levels,
				label = "Precipitação (mm)", orientation = "vertical", extend = "max")
	_d7 = datetime.today() - timedelta(days = 7)
	_d7 = _d7 - timedelta(days = _d7.weekday() + 1)
	_d7 = _d7.strftime("%Y-%m-%d")
	print(f"\n{green}prec - DOMINGO: {reset}{_d7}\n")
	plt.title(f"Precipitação Acumulada Semanal\nPeríodo observado: {_d7}",
				fontsize = 14, ha = "center")
	ax.add_geometries(shp, ccrs.PlateCarree(), edgecolor = "black",
					facecolor = "none", linewidth = 0.5)
	ax.coastlines(resolution = "10m", color = "black", linewidth = 0.8)
	ax.add_feature(cartopy.feature.BORDERS, edgecolor = "black", linewidth = 0.5)
	gl = ax.gridlines(crs = ccrs.PlateCarree(), color = "white", alpha = 1.0,
					linestyle = "--", linewidth = 0.25, xlocs = np.arange(-180, 180, 1),
					ylocs = np.arange(-90, 90, 1), draw_labels = True)
	gl.top_labels = False
	gl.right_labels = False
	plt.figtext(0.55, 0.045, "Fonte: MERGE - CPTEC/INPE", ha = "center", fontsize = 10)
	#plt.savefig(f"{caminho_resultado}prec_semanal_merge_acumulada_{_d7}.png",
	#			transparent = False, dpi = 300, bbox_inches = "tight", pad_inches = 0.02)
	plt.show()
	
#################################################################################
# EXECUTANDO FUNÇÕES
	
try:
	regiao_prec = selecionar_tempo_espaco(ds_prec, _ANO_ATUAL)
except FileNotFoundError:
	regiao_prec = selecionar_tempo_espaco(ds_prec, _ANO_ONTEM)

levels, levels2, norm = limite_colobar(regiao_prec)
info_dataset(regiao_prec)
gerar_mapa(regiao_prec)

	
