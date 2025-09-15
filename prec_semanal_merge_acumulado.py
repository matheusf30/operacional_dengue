#!/usr/bin/env python
# coding: utf-8
#
##########################################################################
###
### IFSC DADOS Precipitaçao MERGE
### COMO RODAR: executar o arquivo python e em seguida inserir ano-mês
###  Ex.:
###  python prec_mensal_merge.py 2024-05
###                  AUTORA: Roseli de Oliveira - 2024/07/11
###                  Adaptado: Mario Quadro     - 2024/09/05
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
print(sys.argv[1])
data=sys.argv[1] 
print('Data:', data, '\n')
#
# Explodir a variável em ano e mês
ano, mes = data.split('-')
#
print(f'Ano: {ano}')
print(f'Mês: {mes}')

#
###-------------------------------------------------------------
### Define paths and variables
###-------------------------------------------------------------
#
#path_mer = "/media/dados/operacao/merge/CDO.MERGE"
path_mer = "/media/dados/operacao/merge/daily"
path_shp = "/media/dados/shapefiles/BR"
path_pro = "/media/produtos/merge/monthly"
file_name = f"MERGE_CPTEC_MONTHLY_{ano}.nc" 
os.makedirs(f"{path_pro}/{ano}", mode = 0o777, exist_ok = True) 
#
###-------------------------------------------------------------
### ABRE ARQUIVO NETCDF DATASET (ds)sss
###-------------------------------------------------------------
#
# 
#ds = xr.open_dataset('/media/dados/operacao/merge/CDO.MERGE/MERGE_CPTEC_MONTHLY_2024.nc')
#ds = xr.open_dataset(f"/media/dados/operacao/merge/CDO.MERGE/MERGE_CPTEC_MONTHLY_{ano}.nc")
ds = xr.open_dataset(f"{path_mer}/{ano}/{file_name}")

print('Arquivo original:', ds, '\n')
print('***************************************\n')

#sys.exit()

# VISUALIZAR dimensões do aquivo (Latitude, Longitude e Time)
print('Coordenadas em Latitude:', ds.lat, '\n')
print('***************************************\n')
print('Coordenadas em Longitude:', ds.lon, '\n')
print('***************************************\n')
print('Datas:', ds.time, '\n')
print('***************************************\n')

# Variável de interesse
print('Data variable:', ds.prec, '\n')
print('***************************************\n')

# Definir a extensão da região que você deseja plotar
#lat_min = -36.95
#lat_max = -19.05
#lon_min = -62.95
#lon_max = -45.05
lat_min = -34.00
lat_max = -21.75
lon_min = -58.25
lon_max = -47.50
# Definir data (ano-mês)
#data = "2024-06"
#data=sys.argv[1] 

# Selecionar os dados de precipitação para a região e o dia específico

data_region = ds.sel(time = f'{data}', lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).prec.squeeze()

# Encontrar o valor máximo da precipitacao na região selecionada
max_prec = data_region.max().item()
int_max = int(max_prec) + 10
print(f'Valor máximo da precipitação na região selecionada: {int_max} mm')
int_min = 0

# Plotar Precipitação de um dia específico
plt.figure(figsize=(8, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
shp = list(shpreader.Reader(f"{path_shp}/BR_UF_2019.shp").geometries())

# Criar uma paleta de cores personalizada
colors = ["#b4f0f0", "#96d2fa", "#78b9fa", "#3c95f5", "#1e6deb", "#1463d2", 
          "#0fa00f", "#28be28", "#50f050", "#72f06e", "#b3faaa", "#fff9aa", 
          "#ffe978", "#ffc13c", "#ffa200", "#ff6200", "#ff3300", "#ff1500", 
          "#c00100", "#a50200", "#870000", "#653b32"]
cmap = matplotlib.colors.ListedColormap(colors)
cmap.set_over('#000000')
cmap.set_under('#ffffff')

# Definir o intervalo de contorno
#data_min = int_min
#data_max = int_max
data_min = int_min
data_max = int_max
interval = 1
levels = np.linspace(data_min, data_max, num=256)
#levels = np.arange(data_min, data_max, (int_max-int_min)/10)

# Plotar o dado 2D da região selecionada
figure = data_region.plot.pcolormesh(robust=True, norm=cls.Normalize(vmin=int_min, vmax=int_max),
                                     cmap=cmap, add_colorbar=False, levels=levels, add_labels=False)


ax.add_geometries(shp, ccrs.PlateCarree(), edgecolor='gray', facecolor='none', linewidth=0.3)
ax.coastlines(resolution='10m', color='black', linewidth=0.8)
ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linewidth=0.5)
gl = ax.gridlines(crs=ccrs.PlateCarree(), color='white', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=True)
gl.top_labels = False
gl.right_labels = False
        
plt.colorbar(figure, pad=0.05, fraction=0.05, extend='max', ticks=np.arange(int_min, int_max+1, (int_max-int_min)/10), orientation='vertical', label='Precipitação Mensal (mm)')

plt.title(f'Precipitação Média Mensal (mm) - Sul do Brasil\nPeríodo observado: {mes}/{ano} ', fontsize=14, ha='center')

# Adicionar a fonte no rodapé
plt.figtext(0.55, 0.05, 'Fonte: MERGE - CPTEC', ha='center', fontsize=10)

# Salvar a figura no formato ".jpg" com dpi=300.
plt.savefig(f"{path_pro}/{ano}/prec_acum_mensal_merge_{ano}{mes}.png", transparent=True, dpi=300, bbox_inches="tight", pad_inches=0.02)
print(f'Figura gerada: {path_pro}/{ano}/prec_acum_mensal_merge_{ano}{mes}.png')

#plt.show()
