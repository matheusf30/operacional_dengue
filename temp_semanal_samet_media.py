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
###                              Matheus Ferreira de Souza - 2025/09/15
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
from datetime import date, datetime, timedelta
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

'''
print('Arquivo original\nTMIN:\n{ds_tmin}\n')
print('***************************************\n')
# VISUALIZAR dimensões do aquivo (Latitude, Longitude e Time)
print('Coordenadas em Latitude:', ds_tmin.lat, '\n')
print('***************************************\n')
print('Coordenadas em Longitude:', ds_tmin.lon, '\n')
print('***************************************\n')
print('Datas:', ds_tmin.time, '\n')
print('***************************************\n')
# Variável de interesse
print('Data variable:', ds_tmin.tmin, '\n')
print('***************************************\n')

print('Arquivo original\nTMED:\n{ds_tmed}\n')
print('***************************************\n')
# VISUALIZAR dimensões do aquivo (Latitude, Longitude e Time)
print('Coordenadas em Latitude:', ds_tmed.lat, '\n')
print('***************************************\n')
print('Coordenadas em Longitude:', ds_tmed.lon, '\n')
print('***************************************\n')
print('Datas:', ds_tmed.time, '\n')
print('***************************************\n')
# Variável de interesse
print('Data variable:', ds_tmed.tmed, '\n')
print('***************************************\n')

print('Arquivo original\nTMAX:\n{ds_tmax}\n')
print('***************************************\n')
# VISUALIZAR dimensões do aquivo (Latitude, Longitude e Time)
print('Coordenadas em Latitude:', ds_tmax.lat, '\n')
print('***************************************\n')
print('Coordenadas em Longitude:', ds_tmax.lon, '\n')
print('***************************************\n')
print('Datas:', ds_tmax.time, '\n')
print('***************************************\n')
# Variável de interesse
print('Data variable:', ds_tmax.tmax, '\n')
print('***************************************\n')
'''

# Definir a extensão da região que você deseja plotar
lat_min = -34.00
lat_max = -21.75
lon_min = -58.25
lon_max = -47.50
# Definir data (ano)
data = f"{_ANO_ATUAL}"

# Selecionar os dados de temperatura para a região e o dia específico
regiao_tmin = ds_tmin.sel(time = f"{data}",
						lat = slice(lat_min, lat_max), 
						lon = slice(lon_min, lon_max)).tmin.squeeze()
regiao_tmin = regiao_tmin.resample(time = "W-SUN").mean()
regiao_tmed = ds_tmed.sel(time = f"{data}",
						lat = slice(lat_min, lat_max),
						lon = slice(lon_min, lon_max)).tmed.squeeze()
regiao_tmed = regiao_tmed.resample(time = "W-SUN").mean()
regiao_tmax = ds_tmax.sel(time = f"{data}",
						lat = slice(lat_min, lat_max),
						lon = slice(lon_min, lon_max)).tmax.squeeze()
# --- INÍCIO DO CÓDIGO DE CORREÇÃO ---

# 1. Diagnóstico: Imprime as dimensões da coordenada 'time' para confirmar o problema
print("Dimensões originais da coordenada 'time':", regiao_tmax.time.dims)

# 2. Correção: Se a coordenada 'time' tiver mais de uma dimensão, a corrige.
if regiao_tmax.time.ndim > 1:
    print("Coordenada 'time' é multidimensional. Convertendo para 1D...")
    
    # Pega o nome da primeira dimensão (que deve ser 'time')
    time_dim_name = regiao_tmax.time.dims[0]
    
    # Seleciona os valores de tempo ao longo dessa dimensão principal.
    # Isso efetivamente remove as dimensões extras, criando uma coordenada 1D.
    one_d_time = regiao_tmax.time.isel({dim: 0 for dim in regiao_tmax.time.dims if dim != time_dim_name}).values
    
    # Atribui a nova coordenada 1D de volta à dimensão 'time' do DataArray
    regiao_tmax[time_dim_name] = one_d_time
    
    print("Novas dimensões da coordenada 'time':", regiao_tmax.time.dims)

# --- FIM DO CÓDIGO DE CORREÇÃO ---
#egiao_tmax = regiao_tmax.resample(time = "W-SUN").mean()

print(f'Arquivo original\nTMAX:\n{regiao_tmax}\n')
print('***************************************\n')
# VISUALIZAR dimensões do aquivo (Latitude, Longitude e Time)
print('Coordenadas em Latitude:', regiao_tmax.lat, '\n')
print('***************************************\n')
print('Coordenadas em Longitude:', regiao_tmax.lon, '\n')
print('***************************************\n')
print('Datas:', regiao_tmax.time, '\n')
print('***************************************\n')
# Variável de interesse
print('Data variable:', regiao_tmax.values, '\n')
print('***************************************\n')
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

def gerar_mapa(str_var):
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
	
	#Plotando os dados, condicional para cada variáve
	match str_var:
		case "tmin":
			figure = regiao_tmin.plot.pcolormesh(robust=True, cmap=cmap, add_colorbar=False, levels=levels, add_labels=False, extend='both', norm = norm)
			plt.title(f'Temperatura Mínima Média Mensal (°C)\nSul do Brasil, Período observado: {mes}/{ano}', fontsize=14, ha='center')
		case "tmed":
			figure = regiao_tmed.plot.pcolormesh(robust=True, norm=norm, cmap=cmap, add_colorbar=False, levels=levels, add_labels=False, extend='both')
			plt.title(f'Temperatura Média Mensal (°C)\nSul do Brasil, Período observado: {mes}/{ano}', fontsize=14, ha='center')
		case "tmax":
			figure = regiao_tmax.plot.pcolormesh(robust=True, norm=norm, cmap=cmap, add_colorbar=False, levels=levels, add_labels=False, extend='both')
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
	plt.savefig(f"{caminho_resultado}/{ano}/{str_var}_mensal_samet_media_{ano}{mes}.png", transparent=False, dpi=300, bbox_inches="tight", pad_inches=0.02)
	print(f'\nFigura gerada:\n{caminho_resultado}/{ano}/{str_var}_mensal_samet_media_{ano}{mes}.png\n')
	#plt.show()

gerar_mapa("tmin")
gerar_mapa("tmed")
gerar_mapa("tmax")
print(f"Figuras de médias de {ano}-{mes} completas")
print("=="*50)
	
