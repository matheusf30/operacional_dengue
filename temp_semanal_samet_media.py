#!/usr/bin/env python
# coding: utf-8
#
##########################################################################
###
### IFSC DADOS Temperatura SAMET - MAPAS
### COMO RODAR: executar o arquivo python e em seguida inserir ano-mês
###  Ex.:
###  python tmed_mensal_samet.py 2024-05
###                  AUTORA: Roseli de Oliveira         - 2024/07/11
###                  Adaptado: Mario Quadro             - 2024/09/05
###                  Atualizado: Everton Weber Galliani - 2025/09/02
###
##########################################################################
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
#
###-------------------------------------------------------------
### Import System Variables (argv)
###-------------------------------------------------------------
#
#print 'Argument 1 -> ', argv[1:]
data = sys.argv[1] #data = "2024-06"
print(sys.argv[1])
print(f'\nData: {data}\n')
#
# Explodir a variável em ano e mês
ano, mes = data.split('-')
print(f'Ano: {ano}')
print(f'Mês: {mes}')
#
###-------------------------------------------------------------
### Define paths and variables
###-------------------------------------------------------------
#
#path_sam = "/media/dados/operacao/samet/CDO.SAMET"
path_sam = "/media/dados/operacao/samet/daily/"
path_shp = "/media/dados/shapefiles/BR"
path_pro = "/media/produtos/samet/monthly"
file_name1 = f"SAMeT_CPTEC_MONTHLY_TMIN_MEAN_{ano}.nc"
file_name2 = f"SAMeT_CPTEC_MONTHLY_TMED_MEAN_{ano}.nc" 
file_name3 = f"SAMeT_CPTEC_MONTHLY_TMAX_MEAN_{ano}.nc" 
os.makedirs(f"{path_pro}/{ano}", mode = 0o777, exist_ok = True) 
#
###-------------------------------------------------------------
### ABRE ARQUIVO NETCDF DATASET (ds)sss
###-------------------------------------------------------------
#
# 
#ds = xr.open_dataset('/media/dados/operacao/samet/CDO.SAMET/SAMeT_CPTEC_MONTHLY_TMED_MEAN_2024.nc')
ds1 = xr.open_dataset(f"{path_sam}/TMIN/{ano}/{file_name1}")
ds2 = xr.open_dataset(f"{path_sam}/TMED/{ano}/{file_name2}")
ds3 = xr.open_dataset(f"{path_sam}/TMAX/{ano}/{file_name3}")

'''
print('Arquivo original\nTMIN:\n{ds1}\n')
print('***************************************\n')
# VISUALIZAR dimensões do aquivo (Latitude, Longitude e Time)
print('Coordenadas em Latitude:', ds1.lat, '\n')
print('***************************************\n')
print('Coordenadas em Longitude:', ds1.lon, '\n')
print('***************************************\n')
print('Datas:', ds1.time, '\n')
print('***************************************\n')
# Variável de interesse
print('Data variable:', ds1.tmin, '\n')
print('***************************************\n')

print('Arquivo original\nTMED:\n{ds2}\n')
print('***************************************\n')
# VISUALIZAR dimensões do aquivo (Latitude, Longitude e Time)
print('Coordenadas em Latitude:', ds2.lat, '\n')
print('***************************************\n')
print('Coordenadas em Longitude:', ds2.lon, '\n')
print('***************************************\n')
print('Datas:', ds2.time, '\n')
print('***************************************\n')
# Variável de interesse
print('Data variable:', ds2.tmed, '\n')
print('***************************************\n')

print('Arquivo original\nTMAX:\n{ds3}\n')
print('***************************************\n')
# VISUALIZAR dimensões do aquivo (Latitude, Longitude e Time)
print('Coordenadas em Latitude:', ds3.lat, '\n')
print('***************************************\n')
print('Coordenadas em Longitude:', ds3.lon, '\n')
print('***************************************\n')
print('Datas:', ds3.time, '\n')
print('***************************************\n')
# Variável de interesse
print('Data variable:', ds3.tmax, '\n')
print('***************************************\n')
'''

# Definir a extensão da região que você deseja plotar
lat_min = -34.00
lat_max = -21.75
lon_min = -58.25
lon_max = -47.50
# Definir data (ano-mês)
#data = "2024-06"
data = sys.argv[1] #data = "2024-06"

# Selecionar os dados de temperatura para a região e o dia específico
data_region1 = ds1.sel(time = f'{data}', lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).tmin.squeeze()
data_region2 = ds2.sel(time = f'{data}', lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).tmed.squeeze()
data_region3 = ds3.sel(time = f'{data}', lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).tmax.squeeze()

# Encontrar o valor máximo da temperatura máxima e mínimo da temperatura mínima na região selecionada
max_tmax = data_region3.max().item()
int_max = int(max_tmax) + 1
min_tmin = data_region1.min().item()
int_min = int(min_tmin)

#Variáveis para o plot do mapa e da colorbar
if ((int_max - int_min)//2 != (int_max-int_min)/2):
	int_max = int_max + 1
levels = range(int_min, int_max + 2, 2)
#levels = np.linspace(int_min, int_max, 10)
#print(levels)
#levels = levels.astype(int)
#print(levels)
norm = cls.Normalize(vmin=int_min, vmax=int_max)
cmap = plt.get_cmap("RdYlBu_r")
#cmap.set_over('#800026')
#cmap.set_under('#040273')
print(f'Valor máximo da temperatura na região selecionada: {round(max_tmax, 2)} °C')
print(f'Valor mínimo da temperatura na região selecionada: {round(min_tmin, 2)} °C')
# Plotar temperatura de um dia específico

def gerar_mapa(str_var):
	"""
	Função relativa à síntese de mapas temáticos de temperatura utilizando SAMeT.
	entrada:
	- arquivo = data_region1, data_region2 ou data_region3;
	- str_var = string da variável de interesse (tmin, tmed ou tmax).
	retorno:
	- mapa temático com a variável de interesse.
	"""
	plt.figure(figsize=(8, 6))#, layout = "tight", frameon = False)
	ax = plt.axes(projection=ccrs.PlateCarree())
	shp = list(shpreader.Reader(f"{path_shp}/BR_UF_2019.shp").geometries())
	
	#Plotando os dados, condicional para cada variáve
	match str_var:
		case "tmin":
			figure = data_region1.plot.pcolormesh(robust=True, cmap=cmap, add_colorbar=False, levels=levels, add_labels=False, extend='both', norm = norm)
			plt.title(f'Temperatura Mínima Média Mensal (°C)\nSul do Brasil, Período observado: {mes}/{ano}', fontsize=14, ha='center')
		case "tmed":
			figure = data_region2.plot.pcolormesh(robust=True, norm=norm, cmap=cmap, add_colorbar=False, levels=levels, add_labels=False, extend='both')
			plt.title(f'Temperatura Média Mensal (°C)\nSul do Brasil, Período observado: {mes}/{ano}', fontsize=14, ha='center')
		case "tmax":
			figure = data_region3.plot.pcolormesh(robust=True, norm=norm, cmap=cmap, add_colorbar=False, levels=levels, add_labels=False, extend='both')
			plt.title(f'Temperatura Máxima Média Mensal (°C)\nSul do Brasil, Período observado: {mes}/{ano}', fontsize=14, ha='center')
		case _:
			print(f"\nVariável não encontrada!\n{str_var}\nVariável não encontrada!\n")
			
	ax.add_geometries(shp, ccrs.PlateCarree(), edgecolor='gray', facecolor='none', linewidth=0.3)
	ax.coastlines(resolution='10m', color='black', linewidth=0.8)
	ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5)
	gl = ax.gridlines(crs=ccrs.PlateCarree(), color='white', alpha=1.0, linestyle='--',
			linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=True)
	gl.top_labels = False
	gl.right_labels = False
	plt.colorbar(figure, pad=0.05, fraction=0.05, label='Temperatura Mensal (°C)',ticks=levels, orientation='vertical')
	#ticks=np.arange(int_min, int_max+1, (int_max-int_min)/10), orientation='vertical')	
	# Adicionar a fonte no rodapé
	plt.figtext(0.55, 0.05, 'Fonte: SAMeT - CPTEC', ha='center', fontsize=10)
	# Salvar a figura no formato ".jpg" com dpi=300.
	plt.savefig(f"{path_pro}/{ano}/{str_var}_mensal_samet_media_{ano}{mes}.png", transparent=False, dpi=300, bbox_inches="tight", pad_inches=0.02)
	print(f'\nFigura gerada:\n{path_pro}/{ano}/{str_var}_mensal_samet_media_{ano}{mes}.png\n')
	#plt.show()

gerar_mapa("tmin")
gerar_mapa("tmed")
gerar_mapa("tmax")
print(f"Figuras de médias de {ano}-{mes} completas")
print("=="*50)
	
