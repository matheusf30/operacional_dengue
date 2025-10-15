from datetime import datetime, timedelta
t0 = datetime.now()
print(f"Início da execução: {t0}")
import pandas as pd
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib import colors as cls

import cartopy, cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.crs as crs
from cartopy.feature import ShapelyFeature
import cartopy.feature as cfeature

import os
import sys

caminho_shapefile = "/media/dados/shapefiles/BR/"
shp = list(shpreader.Reader(f"{caminho_shapefile}/BR_UF_2022.shp").geometries())
nc_prec = xr.open_dataset("prec_climatologia_epidemiosemanal.nc")
nc_tmed = xr.open_dataset("tmed_climatologia_epidemiosemanal.nc")
caminho_figuras = "/home/meteoro/scripts/matheus/operacional_dengue/meteorologia/climatologia/figuras"

lat_ticks = [-29, -28, -27, -26]
lon_ticks = [-54, -53, -52, -51, -50, -49, -48]
print(nc_tmed, nc_prec)

def get_limits(dataset):
	sup_lim = dataset.max().values
	inf_lim = dataset.min().values
	return sup_lim, inf_lim

tmed_max = int(nc_tmed['tmed'].max().item()) + 1
tmed_min = int(nc_tmed['tmed'].min().item())
tmed_norm = cls.Normalize(vmin = tmed_min, vmax = tmed_max)
tmed_cmap = plt.get_cmap("RdYlBu_r")

prec_max = np.ceil(nc_prec['prec'].max().item()/50)*50
prec_min = 0
prec_norm = cls.Normalize(vmin = prec_min, vmax = prec_max)
prec_cmap = plt.get_cmap("YlGnBu")

def add_cartopy_features(ax, shapefile):
    """Add cartopy features to an axis"""
    ax.add_geometries(shapefile, ccrs.PlateCarree(), 
                     edgecolor="black", facecolor="none", linewidth=0.5)
    ax.coastlines(resolution="10m", color="black", linewidth=0.8)
    ax.add_feature(cartopy.feature.BORDERS, edgecolor="black", linewidth=0.5)
    #gl = ax.gridlines(crs=ccrs.PlateCarree(), color="white", alpha=1.0, 
    #                 linestyle="--", linewidth=0.25, 
    #                 xlocs=np.arange(-180, 180, 1), 
    #                 ylocs=np.arange(-90, 90, 1), draw_labels=True)
    #gl.top_labels = False
    #gl.right_labels = False
    return ax

for i in range(1, 54):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    fig.suptitle(f"Climatologia da semana N° {i}")
    
    # Temperature plot
    tmed = nc_tmed.sel(week = i)["tmed"].squeeze()
    ax[0].set_title("Temperatura Média")
    ax[0].set_xlabel("Longitude")
    ax[0].set_xticks(lon_ticks)
    ax[0].set_ylabel("Latitude")
    ax[0].set_yticks(lat_ticks)
    img_tmed = tmed.plot.pcolormesh(ax=ax[0], vmin = tmed_min, vmax = tmed_max, norm = tmed_norm, cmap = tmed_cmap,  transform=ccrs.PlateCarree(), add_colorbar = False,  add_labels = False)
    add_cartopy_features(ax[0], shp)
    cbar_tmed = plt.colorbar(img_tmed, ax=ax[0], fraction=0.030, pad=0.03,  orientation="vertical", extend="both")
    cbar_tmed.set_label("Temperatura Semanal (°C)")
    
    # Precipitation plot  
    prec = nc_prec.sel(week = i)["prec"]
    ax[1].set_title("Acumulado de Precipitação")
    ax[1].set_xlabel("Longitude")
    ax[1].set_xticks(lon_ticks)
    ax[1].set_ylabel("Latitude")
    ax[1].set_yticks(lat_ticks)
    img_prec = prec.plot.pcolormesh(ax=ax[1], vmin = prec_min, vmax = prec_max, norm = prec_norm, cmap = prec_cmap,  transform=ccrs.PlateCarree(), add_colorbar = False,  add_labels = False,)
    add_cartopy_features(ax[1], shp)
    cbar_prec = plt.colorbar(img_prec, ax=ax[1], fraction=0.030, pad=0.03,  orientation="vertical", extend="max")
    cbar_prec.set_label("Precipitação Semanal (mm)")
    
    ax[0].set_title(f"Temperatura Média (SE {i})")
    ax[1].set_title(f"Acumulado de Precipitação (SE {i})")
    plt.tight_layout()
    plt.savefig(f"{caminho_figuras}/climatologia_se_{i}", dpi=300, bbox_inches='tight')
    print(f"Feita climatologia da semana {i}")
    #
    #plt.show()
t1 = datetime.now() - t0
print(f"Tempo de execução do programa: {t1}")

sys.exit()
ax.add_geometries(shp, ccrs.PlateCarree(), edgecolor = "black", facecolor = "none", linewidth = 0.5)
ax.coastlines(resolution = "10m", color = "black", linewidth = 0.8)
ax.add_feature(cartopy.feature.BORDERS, edgecolor = "black", linewidth = 0.5)
gl = ax.gridlines(crs = ccrs.PlateCarree(), color = "white", alpha = 1.0, linestyle = "--", linewidth = 0.25, xlocs = np.arange(-180, 180, 1), ylocs = np.arange(-90, 90, 1), draw_labels = True)
gl.top_labels = False
gl.right_labels = False
