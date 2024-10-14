### Bibliotecas Correlatas
# Suporte
import sys, os
# Básicas
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
"""
import matplotlib.pyplot as plt               
import seaborn as sns
import statsmodels as sm
"""
# Manipulação de netCDF4 e shapefiles
import  xarray as xr
import geopandas as gpd
from shapely.geometry import Point

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

"""
diretorio_atual = os.getcwd()
diretorios = diretorio_atual.split(os.path.sep)
diretorio_dados = os.path.sep.join(diretorios[:-6])
print(diretorio_dados)
"""

### Encaminhamento aos Diretórios
caminho_github = "https://raw.githubusercontent.com/matheusf30/dados_dengue/refs/heads/main/" # WEB
caminho_dados = "/home/meteoro/scripts/matheus/operacional_dengue/dados_operacao/" # CLUSTER
caminho_operacional = "/home/meteoro/scripts/matheus/operacional_dengue/"
caminho_shape = "/media/dados/shapefiles/SC/" #SC_Municipios_2022.shp
caminho_gfs = "/media/dados/operacao/gfs/0p25/" #202410/20241012/ #prec_daily_gfs_2024101212.nc
caminho_merge = "/media/dados/operacao/merge/daily/2024/" #MERGE_CPTEC_DAILY_SB_2024.nc
caminho_mergeCDO = "/media/dados/operacao/merge/CDO.MERGE/" #MERGE_CPTEC_DAILY_2024.nc@
caminho_samet = "/media/dados/operacao/samet/daily/" #/TMAX/2024/ #SAMeT_CPTEC_DAILY_SB_TMAX_2024.nc
caminho_sametCDO = "/media/dados/operacao/samet/CDO.SAMET/" #SAMeT_CPTEC_DAILY_SB_TMAX_2024.nc@

_ANO_FINAL = str(datetime.today().year)
_ONTEM = datetime.today() - timedelta(days = 1)
_ANO_ONTEM = str(_ONTEM.year)
print(_ANO_FINAL, _ONTEM, _ANO_ONTEM)

### Renomeação variáveis pelos arquivos
merge = f"MERGE_CPTEC_DAILY_SB_{_ANO_FINAL}.nc"
samet_tmax = f"SAMeT_CPTEC_DAILY_TMAX_{_ANO_FINAL}.nc"
samet_tmed = f"SAMeT_CPTEC_DAILY_TMED_{_ANO_FINAL}.nc"
samet_tmin = f"SAMeT_CPTEC_DAILY_TMIN_{_ANO_FINAL}.nc"
_ANO_ONTEM2 = 2023
serie_prec = f"prec_diario_ate_{_ANO_ONTEM2}.csv"
serie_tmax = f"tmax_diario_ate_{_ANO_ONTEM2}.csv"
serie_tmed = f"tmed_diario_ate_{_ANO_ONTEM2}.csv"
serie_tmin = f"tmin_diario_ate_{_ANO_ONTEM2}.csv"
serie_prec_se = f"prec_semana_ate_{_ANO_ONTEM2}.csv"
serie_tmax_se = f"tmax_semana_ate_{_ANO_ONTEM2}.csv"
serie_tmed_se = f"tmed_semana_ate_{_ANO_ONTEM2}.csv"
serie_tmin_se = f"tmin_semana_ate_{_ANO_ONTEM2}.csv"
"""
serie_prec = f"prec_semana_ate_{_ANO_ONTEM}.nc"
serie_tmax = f"tmax_semana_ate_{_ANO_ONTEM}.nc"
serie_tmed = f"tmed_semana_ate_{_ANO_ONTEM}.nc"
serie_tmin = f"tmin_semana_ate_{_ANO_ONTEM}.nc"
"""

municipios = "SC_Municipios_2022.shp"

### Abrindo Arquivos
prec = xr.open_dataset(f"{caminho_mergeCDO}{merge}")
tmax = xr.open_dataset(f"{caminho_sametCDO}{samet_tmax}")
tmed = xr.open_dataset(f"{caminho_sametCDO}{samet_tmed}")
tmin = xr.open_dataset(f"{caminho_sametCDO}{samet_tmin}")
municipios = gpd.read_file(f"{caminho_shape}{municipios}")
serie_prec = pd.read_csv(f"{caminho_dados}{serie_prec}")
serie_tmax = pd.read_csv(f"{caminho_dados}{serie_tmax}")
serie_tmed = pd.read_csv(f"{caminho_dados}{serie_tmed}")
serie_tmin = pd.read_csv(f"{caminho_dados}{serie_tmin}")
serie_prec_se = pd.read_csv(f"{caminho_dados}{serie_prec_se}")
serie_tmax_se = pd.read_csv(f"{caminho_dados}{serie_tmax_se}")
serie_tmed_se = pd.read_csv(f"{caminho_dados}{serie_tmed_se}")
serie_tmin_se = pd.read_csv(f"{caminho_dados}{serie_tmin_se}")

print(f'\n{green}tmin.variables["tmin"][:]\n{reset}{tmin.variables["tmin"][:]}\n')
print(f'\n{green}tmin.variables["time"][:]\n{reset}{tmin.variables["tmin"][:]}\n')
print(f'\n{green}tmin.variables["nobs"][:]\n{reset}{tmin.variables["nobs"][:]}\n')
print(f"\n{green}tmin\n{reset}{tmin}\n")
print(f"\n{green}tmin\n{reset}{tmin}\n")
print(f"\n{green}serie_tmin\n{reset}{serie_tmin[['Data', 'BALNEÁRIO CAMBORIÚ', 'BOMBINHAS', 'PORTO BELO']]}\n")
print(f"\n{green}serie_tmin_se\n{reset}{serie_tmin_se}\n")
print(f"\n{green}serie_tmin_se\n{reset}{serie_tmin_se[['Semana', 'BALNEÁRIO CAMBORIÚ', 'BOMBINHAS', 'PORTO BELO']]}\n")

print(f"\n{green}municipios\n{reset}{municipios}\n")
#sys.exit()
### Pré-processamento e Definição de Função

def verifica_nan(valores_centroides):
	"""
	Função relativa a verificação de Valores Não-números (NaN) no arquivo.csv gerado!
	Argumento: próprio dataframe criado na função extrair_centroides()
	Retorno: Exibição de mensagens para casos de haver ou não valores NaN.
	"""
	print(f"\n{bold}VERIFICAÇÃO DE DADOS FALTANTES{bold}\n")
	print(f"\nQuantidade de valores {red}{bold}NaN: {valores_centroides['BOMBINHAS'].isnull().sum()}{bold}{reset}")
	if valores_centroides["BOMBINHAS"].isnull().sum() == 0:
		print(f"\n{green}{bold}NÃO há valores {red}NaN{reset}\n")
	else:
		print(f"\nOs dias com valores {red}{bold}NaN{reset} são:")
		print(f"{valores_centroides[valores_centroides['BOMBINHAS'].isna()]['Data']}\n")
	print("="*80)

def semana_epidemiologica(csv, str_var):
	"""
	Função relativa ao agrupamento de dados em semanas epidemiológicas.
	Os arquivos.csv são provenientes deo roteiro 'extrai_clima.py': colunas com datas e municípios + todas as linhas são dados diários.
	Estes Arquivos estão alocados no SifapSC ou GitHub.
	Argumento:
	- Variável com arquivo.csv;
	- String da variável referente ao arquivo.csv.
	Retorno:
	- Retorno próprio de DataFrame com Municípios (centróides) em Colunas e Tempo (semanas epidemiológicas) em Linhas, preenchidos com valores climáticos.
	- Salvando Arquivo.csv
	"""
	csv.drop(columns = str_var, inplace = True)
	csv["Data"] = pd.to_datetime(csv["Data"])
	csv = csv.sort_values(by = ["Data"])
	csv_se = csv.copy()
	csv_se["Semana"] = csv_se["Data"].dt.to_period("W-SAT").dt.to_timestamp()
	if str_var == "prec":
		csv_se = csv_se.groupby(["Semana"]).sum(numeric_only = True)
	else:
		csv_se = csv_se.groupby(["Semana"]).mean(numeric_only = True)
	csv_se.reset_index(inplace = True)
	csv_se.drop([0], axis = 0, inplace = True)
	csv_se.to_csv(f"{caminho_dados}{str_var}_semana_{_ANO_FINAL}.csv", index = False)
	print(f"\n{green}ARQUIVO SALVO COM SUCESSO!\n\nSemana Epidemiológica - {str_var.upper()}{reset}\n\n{csv_se}\n")
	print(f"\n{red}As variáveis do arquivo ({str_var.upper()}), em semanas epidemiológicas, são:{reset}\n{csv_se.dtypes}\n")
	return csv_se

def extrair_centroides(shapefile, netcdf4, str_var):
	"""
	Função relativa a extração de valores dos arquivos NetCDF4 utilizando os centróides de arquivos shapefile como filtro.
	Os arquivos NetCDF4 são provenientes de Rozante et.al. e apresentam 2 variáveis: 1 climática + 1 nest ou 1 Nobs (estações de observação).
	Os arquivos Shapefile são provenientes do IBGE (2022) e apresentam CRS = (epsg = 4674)
	Estes Arquivos estão alocados no SifapSC, dois diretórios antes do /home.
	Argumento:
	- Variável com arquivo NetCDF4;
	- Variável com arquivo Shapefile;
	- String da variável referente ao NetCDF4.
	Retorno:
	- Retorno próprio de DataFrame com Municípios (centróides) em Colunas e Tempo (dias) em Linhas, preenchidos com valores climáticos.
	- Salvando Arquivo.csv (dados diários)
	- Retorno próprio de DataFrame com Municípios (centróides) em Colunas e Tempo (semanas epidemiológicas) em Linhas, preenchidos com valores climáticos.
	- Salvando Arquivo.csv (dados semanais)
	"""
	shapefile["centroide"] = shapefile["geometry"].centroid
	shapefile["centroide"] = shapefile["centroide"].to_crs(epsg = 4674)
	valores_centroides = []
	for idx, linha in shapefile.iterrows():
		lon, lat = linha["centroide"].x, linha["centroide"].y
		if shapefile["NM_MUN"].isin(["Balneário Camboriú", "Bombinhas", "Porto Belo"]).any():
			if str_var == "tmax" or "tmed" or "tmin":
				lon -= 0.25
		valor = netcdf4.sel(lon = lon, lat = lat, method = "nearest")
		valores_centroides.append(valor)
	valores_centroides = pd.DataFrame(data = valores_centroides)
	valores_centroides["Municipio"] = shapefile["NM_MUN"].str.upper().copy()
	if str_var == "prec":
		valores_centroides.drop(columns = ["nest"], inplace = True)
	#else:
	#	valores_centroides.drop(columns = ["nobs"], inplace = True)
	valores_centroides = valores_centroides[["Municipio", str_var]]
	valores_tempo = netcdf4[str_var].time.values
	valores_variavel = netcdf4[str_var].values
	var_valores = []
	for i, linha in valores_centroides.iterrows():
		if isinstance(linha[str_var], xr.DataArray):
			var_valor = [x.item() if not np.isnan(x.item()) else np.nan for x in linha[str_var]]
			var_valores.append(var_valor)
			print(f"\n{green}---{str_var}---\n\n{reset}{valores_centroides['Municipio'][i]}: Finalizado!\n")
			print(f"\n{red}{i + 1} de {len(valores_centroides['Municipio'])}.{reset}\n")
		else:
			var_valores.append([np.nan] * len(valores_tempo))
			print(f"\n{red}{valores_centroides['Municipio'][i]}: NaN... {magenta}Finalizado!\n")
			print(f"\n{red}{i + 1} de {len(valores_centroides['Municipio'])}.\n{reset}")
	var_valores_df = pd.DataFrame(var_valores, columns = valores_tempo)
	valores_centroides = pd.concat([valores_centroides, var_valores_df], axis = 1)
	valores_centroides.drop(columns = [str_var], inplace = True)
	valores_centroides.set_index("Municipio", inplace = True)
	valores_centroides = valores_centroides.T
	valores_centroides["Data"] = valores_centroides.index
	valores_centroides.reset_index(inplace = True)
	colunas_restantes = valores_centroides.columns.drop("Data")
	valores_centroides = valores_centroides[["Data"] + colunas_restantes.tolist()]
	valores_centroides.columns.name = str_var
	valores_centroides.rename(columns = {"index" : str_var}, inplace = True)
	valores_centroides.to_csv(f"{caminho_dados}{str_var}_diario_{_ANO_FINAL}.csv", index = False)
	print("="*80)
	print(f"\n{green}{caminho_shape}{str_var}_diario_{_ANO_FINAL}.csv{reset}\n")
	print(f"\n{green}ARQUIVO SALVO COM SUCESSO!{reset}\n")
	print("="*80)
	print(netcdf4.variables[str_var][:])
	print(netcdf4.variables["time"][:])
	print("="*80)
	print(valores_tempo)
	print(valores_tempo.shape)
	print("="*80)
	print(valores_variavel)
	print(valores_variavel.shape)
	print("="*80)
	print(valores_centroides)
	print(valores_centroides.info())
	print(valores_centroides.dtypes)
	print("="*80)
	verifica_nan(valores_centroides)
	csv_se = semana_epidemiologica(valores_centroides, str_var)
	return valores_centroides, csv_se

tmin, tmin_se = extrair_centroides(municipios, tmin, "tmin")
tmintotal = pd.concat([serie_tmin, tmin], ignore_index = True)
tmintotal["Data"] = pd.to_datetime(tmintotal["Data"]).dt.strftime("%Y-%m-%d")
tmintotal.to_csv(f"{caminho_dados}tmin_diario_ate_{_ANO_FINAL}.csv", index = False)
tmintotal_se = pd.concat([serie_tmin_se, tmin_se], ignore_index = True)
tmintotal_se["Semana"] = pd.to_datetime(tmintotal_se["Semana"]).dt.strftime("%Y-%m-%d")
tmintotal_se.to_csv(f"{caminho_dados}tmin_semana_ate_{_ANO_FINAL}.csv", index = False)
print(f"\n{green}tmintotal\n{reset}{tmintotal}\n")
print(f"\n{green}tmintotal_se\n{reset}{tmintotal_se}\n")
print(tmintotal_se[["BALNEÁRIO CAMBORIÚ", "BOMBINHAS", "PORTO BELO"]])

tmed, tmed_se = extrair_centroides(municipios, tmed, "tmed")
tmedtotal = pd.concat([serie_tmed, tmed], ignore_index = True)
tmedtotal["Data"] = pd.to_datetime(tmedtotal["Data"]).dt.strftime("%Y-%m-%d")
tmedtotal.to_csv(f"{caminho_dados}tmed_diario_ate_{_ANO_FINAL}.csv", index = False)
tmedtotal_se = pd.concat([serie_tmed_se, tmed_se], ignore_index = True)
tmedtotal_se["Semana"] = pd.to_datetime(tmedtotal_se["Semana"]).dt.strftime("%Y-%m-%d")
tmedtotal_se.to_csv(f"{caminho_dados}tmed_semana_ate_{_ANO_FINAL}.csv", index = False)
print(f"\n{green}tmedtotal\n{reset}{tmedtotal}\n")
print(f"\n{green}tmedtotal_se\n{reset}{tmedtotal_se}\n")
print(tmedtotal_se[["BALNEÁRIO CAMBORIÚ", "BOMBINHAS", "PORTO BELO"]])

tmax, tmax_se = extrair_centroides(municipios, tmax, "tmax")
tmedtotal = pd.concat([serie_tmed, tmed], ignore_index = True)
tmedtotal["Data"] = pd.to_datetime(tmedtotal["Data"]).dt.strftime("%Y-%m-%d")
tmedtotal.to_csv(f"{caminho_dados}tmed_diario_ate_{_ANO_FINAL}.csv", index = False)
tmedtotal_se = pd.concat([serie_tmed_se, tmed_se], ignore_index = True)
tmedtotal_se["Semana"] = pd.to_datetime(tmedtotal_se["Semana"]).dt.strftime("%Y-%m-%d")
tmedtotal_se.to_csv(f"{caminho_dados}tmed_semana_ate_{_ANO_FINAL}.csv", index = False)
print(f"\n{green}tmedtotal\n{reset}{tmedtotal}\n")
print(f"\n{green}tmedtotal_se\n{reset}{tmedtotal_se}\n")
print(tmedtotal_se[["BALNEÁRIO CAMBORIÚ", "BOMBINHAS", "PORTO BELO"]])

prec, prec_se = extrair_centroides(municipios, prec, "prec")
tmedtotal = pd.concat([serie_tmed, tmed], ignore_index = True)
tmedtotal["Data"] = pd.to_datetime(tmedtotal["Data"]).dt.strftime("%Y-%m-%d")
tmedtotal.to_csv(f"{caminho_dados}tmed_diario_ate_{_ANO_FINAL}.csv", index = False)
tmedtotal_se = pd.concat([serie_tmed_se, tmed_se], ignore_index = True)
tmedtotal_se["Semana"] = pd.to_datetime(tmedtotal_se["Semana"]).dt.strftime("%Y-%m-%d")
tmedtotal_se.to_csv(f"{caminho_dados}tmed_semana_ate_{_ANO_FINAL}.csv", index = False)
print(f"\n{green}tmedtotal\n{reset}{tmedtotal}\n")
print(f"\n{green}tmedtotal_se\n{reset}{tmedtotal_se}\n")
print(tmedtotal_se[["BALNEÁRIO CAMBORIÚ", "BOMBINHAS", "PORTO BELO"]])

print("!!"*80)
print(f"\n{green}{bold}FINALIZADA ATUALIZAÇÃO{reset}\n")
print(f"\n{green}Atualização feita em produtos de reanálise até {red}{_ANO_FINAL}{reset}!\n")
print(f"{bold}(MERGE e SAMeT - tmin, tmed, tmax){reset}")
print("!!"*80)

