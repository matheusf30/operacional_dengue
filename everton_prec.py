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

# CLimatologia de semanas epidemiológicas (by Everton)
SE = 42
prec_climatologia = xr.open_dataset("/home/meteoro/scripts/matheus/operacional_dengue/meteorologia/climatologia/prec_climatologia_epidemiosemanal.nc").sel(week = SE)['prec']

municipios = "/media/dados/shapefiles/SC/SC_Municipios_2024.shp"
sc_shape = gpd.read_file(f"/media/dados/shapefiles/SC/SC_UF_2024.shp")
regionais = "/home/meteoro/scripts/matheus/operacional_dengue/dados_operacao/censo_sc_regional.csv"
municipios = gpd.read_file(municipios, low_memory = False)
regionais = pd.read_csv(regionais, low_memory = False)
try:
	caminho_merge = f"/media/dados/operacao/merge/daily/{_ANO_ATUAL}/"
	arquivo_merge = f"MERGE_CPTEC_DAILY_SB_{_ANO_ATUAL}.nc"
	ds_prec = xr.open_dataset(f"{caminho_merge}{arquivo_merge}")
except FileNotFoundError:
	caminho_merge = f"/media/dados/operacao/merge/daily/{_ANO_ONTEM}/"
	arquivo_merge = f"MERGE_CPTEC_DAILY_SB_{_ANO_ONTEM}.nc"
	ds_prec = xr.open_dataset(f"{caminho_merge}{arquivo_merge}")
	
#################################################################################
municipios["NM_MUN"] = municipios["NM_MUN"].str.upper()
municipios = municipios.merge(regionais[["Municipio", "regional"]],
								left_on = "NM_MUN", right_on = "Municipio",
								how = "left")
regionais = municipios.dissolve(by = "regional")
# DEFININDO FUNÇÕES
def selecionar_tempo_espaco(dataset, tempo):
	"""
	"""
	lat_min = -29.45#-29.5 # -34.00 # -29.5
	lat_max = -25.65#-25.75 # -21.75 # -25.75
	lon_min = -54.15#-54 # -58.25 # -54
	lon_max = -47.85#-48 # -47.50 # -48
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
	
def limite_minmax_anomalia(regiao_prec):
	print(mascara(regiao_prec)[2], mascara(regiao_prec)[1])	
	#sys.exit()
	abs_value = int(max(abs(mascara(regiao_prec)[2]), abs(mascara(regiao_prec)[1])))
	#print(abs_value)
	#abs_value = max(abs(int(mascara(regiao_prec)[2], mascara(regiao_prec)[1])))
	#abs_value = max(abs(int(regiao_prec.min())), abs(int(regiao_prec.max() + 1)))
	# Calculate appropriate step size
	if abs_value <= 15:
		step = 3
	elif abs_value <= 30:
		step = 5
	elif abs_value <= 60:
		step = 10
	elif abs_value <= 120:
		step = 20
	else:
		step = 30

	# Round abs_value to nearest multiple of step
	rounded_abs = ((abs_value + step - 1) // step) * step

	# Create ranges that cover the full extent and include zero
	levels = list(range(-rounded_abs, rounded_abs + 1, step))
	levels2 = levels
	#levels2 = list(range(-rounded_abs, rounded_abs + 1, 2 * step))

	# Ensure zero is included (it should be with these ranges)
	if 0 not in levels:
		levels.append(0)
		levels.sort()
	if 0 not in levels2:
		levels2.append(0)
		levels2.sort()

	norm = cls.Normalize(vmin=-abs_value, vmax=abs_value)
	cmap = plt.get_cmap("RdBu")
	return levels, levels2, norm, cmap
	
def limite_colobar(regiao_prec):
	max_tmax = regiao_prec.max().item()
	int_max = int(max_tmax) - 10
	min_tmin = 0#regiao_prec.min().item()
	int_min = int(min_tmin)# + 10
	if ((int_max - int_min)//2 != (int_max-int_min)/2):
		int_max += 1
	levels = range(int_min, int_max + 1, 20)
	levels2 = range(int_min, int_max + 1, 20)
	norm = cls.Normalize(vmin = int_min, vmax = int_max)
	cmap = plt.get_cmap("YlGnBu")
	print(f"\n{green}Valor máximo da precipitação: {reset}{round(max_tmax, 2)} mm\n")
	print(f"\n{green}Valor mínimo da precipitação: {reset}{round(min_tmin, 2)} mm\n")
	return levels, levels2, norm, cmap

def mascara(dataset):
	shape_estado = sc_shape
	var = dataset
	mask = regionmask.mask_geopandas(shape_estado, var.lon, var.lat)
	dados_mascarados = var.where(mask >= 0)  # Valores dentro do polígono
	maximo = np.ceil(dados_mascarados.max().values)
	minimo = np.floor(dados_mascarados.min().values)
	media = dados_mascarados.mean().values
	media = media.round(2)
	return media, maximo, minimo
	
def quadradinho_do_mario(media_sc):
	plt.text(-48.75, -29.15, f"Média de SC\n {media_sc} mm",
			color = "black", backgroundcolor = "lightgray",
			ha = "center", va = "center", fontsize = 12, zorder = 10)
	
def gerar_mapa(dataset, str_var):
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
	#cmap = plt.get_cmap("YlGnBu")
	#RdYlBu gist_earth_r terrain_r winter_r summer_r YlGnBu
	#viridis_r cividis_r Blues turbo_r jet_r gnuplot2_r gist_ncar_r
	figure = dataset.plot.contourf(cmap = cmap, norm = norm, robust = True,
									add_colorbar = False,  add_labels = False,
									transform = ccrs.PlateCarree(),  levels = levels)
	linhas = dataset.plot.contour(ax = ax, levels = levels2,
								colors = "black", linewidths = 0.3,	transform = ccrs.PlateCarree())
	rotulos = ax.clabel(linhas, inline = True, fmt = "%1.0f", fontsize = 8, colors = "black")
	for rotulo in rotulos:
		rotulo.set_rotation(0)
	plt.colorbar(figure, fraction = 0.035, pad = 0.03, ticks = levels,
				label = "Precipitação (mm)", orientation = "vertical", extend = "max")
	ax = plt.gca()
	regionais.plot(ax = ax, facecolor = "none",# linestyle = "--",
				edgecolor = "dimgray", linewidth = 0.7)
	_d7 = datetime.today() - timedelta(days = 7)
	_d7 = _d7 - timedelta(days = _d7.weekday() + 1)
	_d7 = _d7.strftime("%Y-%m-%d")
	print(f"\n{green}prec - DOMINGO: {reset}{_d7}\n")
	if (str_var == "acumulado"):
		plt.title(f"Precipitação Acumulada Para a Semana {SE}\nPeríodo observado: {_d7}", fontsize = 14, ha = "center")
	elif (str_var == "anomalia"):
		plt.title(f"Anomalia de Precipitação Para a Semana {SE}\nPeríodo observado: {_d7}", fontsize = 14, ha = "center")
	media = mascara(dataset)[0]
	quadradinho_do_mario(media)
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
	plt.savefig(f"{caminho_resultado}prec_semanal_merge_{str_var}_{_d7}.png",
				transparent = False, dpi = 300, bbox_inches = "tight", pad_inches = 0.02)
	#plt.show()
	
#################################################################################
# EXECUTANDO FUNÇÕES
	
try:
	regiao_prec = selecionar_tempo_espaco(ds_prec, _ANO_ATUAL)
except FileNotFoundError:
	regiao_prec = selecionar_tempo_espaco(ds_prec, _ANO_ONTEM)

levels, levels2, norm, cmap = limite_colobar(regiao_prec)
#info_dataset(regiao_prec)
gerar_mapa(regiao_prec, "acumulado")
regiao_prec -= prec_climatologia
levels, levels2, norm, cmap = limite_minmax_anomalia(regiao_prec)
gerar_mapa(regiao_prec, "anomalia")
#gerar_mapa(regiao_prec - prec_climatologia)

	
