#################################################################################
## Roteiro adaptado para analisar padrões/perfis ## PALAVRAS-CHAVE:            ##  
## Dados: meteorológicos (SAMeT e MERGE),        ## > Semana Epidemiológica;   ##
##        entomológicos (Focos de _Aedes_ sp.) e ## > Ciclo Anual;             ##
##        epidemiológicos (Casos de Dengue).     ## > Perfil Epidemiológico;   ##
## Demanda: FAPESC edital nº 37/2024             ## > Média Semanal;           ##
## Adaptado por: Matheus Ferreira de Souza       ## > Distribuição Temporal;   ##
##               e Everton Weber Galliani        ## > Seminário Hospitais;     ##
## Data: 16/10/2025                              ## > Saúde é Clima.           ##
#################################################################################

##### Bibliotecas correlatas ####################################################
import matplotlib.pyplot as plt 
import matplotlib as mpl             
import pandas as pd
from matplotlib import cm
import matplotlib.colors as cls
import matplotlib.dates as mdates  
import cmocean
from datetime import timedelta
from epiweeks import Week, Year
import numpy as np
import seaborn as sns
#import statsmodels as sm
#import pymannkendall as mk
import xarray as xr
### Suporte
import sys
import os
### Tratando avisos
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

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

### Encaminhamento aos Diretórios ###############################################
caminho_dados = "/home/meteoro/scripts/matheus/operacional_dengue/dados_operacao/"
caminho_shp = "/home/sifapsc/scripts/matheus/dados_dengue/shapefiles/"
caminho_modelos = "/home/sifapsc/scripts/matheus/dados_dengue/modelos/"
caminho_resultados = "/home/sifapsc/scripts/matheus/dengue/resultados/modelagem/"
caminho_correlacao = "/home/sifapsc/scripts/matheus/dengue/resultados/correlacao/"
caminho_cartografia = "/home/sifapsc/scripts/matheus/dengue/resultados/cartografia/"
caminho_gh = "https://raw.githubusercontent.com/matheusf30/dados_dengue/refs/heads/main/"

print(f"\n{green}OS DADOS UTILIZADOS ESTÃO ALOCADOS NOS SEGUINTES CAMINHOS:\n{reset}\n{caminho_dados}\n\n")

### Renomeação das Variáveis pelos Arquivos #####################################
serie_casos = "casos_dive_pivot_total.csv"
serie_focos = "focos_pivot.csv"
serie_prec = "2025/prec_semana_ate_2025.csv"
serie_tmin = "2025/tmin_semana_ate_2025.csv"
serie_tmed = "2025/tmed_semana_ate_2025.csv"
serie_tmax = "2025/tmax_semana_ate_2025.csv"

### Abrindo Arquivo #############################################################
serie_casos = pd.read_csv(f"{caminho_dados}{serie_casos}", low_memory = False)
serie_focos = pd.read_csv(f"{caminho_dados}{serie_focos}", low_memory = False)
serie_prec = pd.read_csv(f"{caminho_dados}{serie_prec}", low_memory = False)
serie_tmin = pd.read_csv(f"{caminho_dados}{serie_tmin}", low_memory = False)
serie_tmed = pd.read_csv(f"{caminho_dados}{serie_tmed}", low_memory = False)
serie_tmax = pd.read_csv(f"{caminho_dados}{serie_tmax}", low_memory = False)

print(f"\n{green}CASOS (Série Temporal)\n{reset}{serie_casos}\n")
print(f"\n{green}FOCOS (Série Temporal)\n{reset}{serie_focos}\n")
print(f"\n{green}PRECIPITAÇÃO (Série Temporal)\n{reset}{serie_prec}\n")
print(f"\n{green}TEMPERATURA MÍNIMA (Série Temporal)\n{reset}{serie_tmin}\n")
print(f"\n{green}TEMPERATURA MÉDIA (Série Temporal)\n{reset}{serie_tmed}\n")
print(f"\n{green}TEMPERATURA MÁXIMA (Série Temporal)\n{reset}{serie_tmax}\n")

### Funções #####################################################################
def ciclo_epi_anual(df, str_var, municipio, a_partir_de, ate):
	"""
	"""
	print(f"\n{green}{str_var.upper()} (Original)\n{reset}{df}\n")
	municipio = municipio.upper()
	df["Semana"] = pd.to_datetime(df["Semana"])
	#df = df[df["Semana"].dt.year >= a_partir_de]
	#df = df[df["Semana"].dt.year <= ate]
	print(f"\n{green}{str_var.upper()} (Recorte Temporal)\n{reset}{df}\n")
	df["SE"] = df["Semana"].apply(lambda data: Week.fromdate(data).week)
	df_melt = df.melt(id_vars = ["Semana", "SE"],
						var_name = "Municipio",
						value_name = str_var)
	df_melt["ano_epi"] = df_melt["Semana"].dt.year
	df_melt.loc[(df_melt["Semana"].dt.month == 1) & (df_melt["SE"] > 50), "ano_epi"] -= 1
	df_melt.loc[(df_melt["Semana"].dt.month == 12) & (df_melt["SE"] == 1), "ano_epi"] += 1
	df_melt = df_melt[df_melt["ano_epi"] >= a_partir_de]
	df_melt = df_melt[df_melt["ano_epi"] <= ate]
	print(f"\n{green}{str_var.upper()} (Tabulação)\n{reset}{df_melt}\n")
	df_melt = df_melt[df_melt["Municipio"] == municipio]
	ciclo_anual = df_melt.groupby("SE")[str_var].mean().reset_index()
	print(f"\n{green}{str_var.upper()} (Média Anual - {municipio})\n{reset}{ciclo_anual}\n")
	return ciclo_anual

###### Alterar ### Recorte Espacial (Município) e Temporal (Anos Epidemiológicos)
ano_i = 2020
ano_f = 2024
nome_cidade = "Florianópolis"
nome_cidade = nome_cidade.upper()
nome_cidade_arquivo = nome_cidade
troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A', 'Ä': 'A',
		'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E', 'Ë': 'E',
		'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I', 'Ï': 'I',
		'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O', 'Ö': 'O',
		'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U', 'Ü': 'U',
		'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
for velho, novo in troca.items():
	nome_cidade_arquivo = nome_cidade_arquivo.replace(velho, novo)
nome_cidade_arquivo = nome_cidade_arquivo.lower()

### Pré-Processamento (Climatologia) ###########################################
casos = ciclo_epi_anual(serie_casos, "casos", nome_cidade, ano_i, ano_f)
focos = ciclo_epi_anual(serie_focos, "focos", nome_cidade, ano_i, ano_f)
prec = ciclo_epi_anual(serie_prec, "prec", nome_cidade, ano_i, ano_f)
tmin = ciclo_epi_anual(serie_tmin, "tmin", nome_cidade, ano_i, ano_f)
tmed = ciclo_epi_anual(serie_tmed, "tmed", nome_cidade, ano_i, ano_f)
tmax = ciclo_epi_anual(serie_tmax, "tmax", nome_cidade, ano_i, ano_f)
#sys.exit()
cidade = pd.DataFrame()
cidade["semana"] = focos["SE"]
cidade["casos"] = casos["casos"]
cidade["focos"] = focos["focos"]
cidade["prec"] = prec["prec"]
cidade["tmin"] = tmin["tmin"]
cidade["tmed"] = tmed["tmed"]
cidade["tmax"] = tmax["tmax"]
print(f"\n{green}{nome_cidade}\n{reset}{cidade}\n")
#sys.exit()

### Visualização Gráfica (CICLO ANUAL) #########################################
fig, axs = plt.subplots(2, 1, figsize = (12, 6), layout = "tight", frameon = False,  sharex = True)
axs[0].set_facecolor("honeydew") #.gcf()
ax2 = axs[0].twinx()
sns.lineplot(x = cidade.index, y = cidade["casos"], ax = axs[0],
				color = "purple", linewidth = 1, linestyle = "--", label = "Casos de Dengue")
axs[0].fill_between(cidade.index, cidade["casos"], color = "purple", alpha = 0.3)
axs[0].set_ylabel("Casos de Dengue")
axs[0].legend(loc = "upper center")
sns.lineplot(x = cidade.index, y = cidade["focos"],  ax = ax2,
				color = "darkgreen", linewidth = 1, linestyle = ":", label = "Focos de _Aedes_ sp.")
ax2.fill_between(cidade.index, cidade["focos"], color = "darkgreen", alpha = 0.35)
ax2.set_ylabel("Focos de _Aedes_ sp.")
ax2.legend(loc = "upper right")
axs[1].set_facecolor("honeydew") #.gcf()
ax3 = axs[1].twinx()#.set_facecolor("honeydew")
sns.barplot(x = cidade["semana"], y = cidade["prec"],  ax = ax3,
				color = "royalblue", linewidth = 1.5, alpha = 0.8, label = "Precipitação")
ax3.set_ylabel("Precipitação (mm)")
ax3.legend(loc = "lower right")
sns.lineplot(x = cidade.index, y = cidade["tmax"],  ax = axs[1],
				color = "red", linewidth = 1.5, label = "Temperatura Máxima")
sns.lineplot(x = cidade.index, y = cidade["tmed"],  ax = axs[1],
				color = "orange", linewidth = 1.5, label = "Temperatura Média")
sns.lineplot(x = cidade.index, y = cidade["tmin"],  ax = axs[1],
				color = "darkblue", linewidth = 1.5, label = "Temperatura Mínima") #alpha = 0.7, linewidth = 3
axs[1].set_ylabel("Temperaturas (ºC)")
axs[1].legend(loc = "upper center")
axs[1].grid(False)
axs[1].set_xlabel("Semanas Epidemiológicas")
fig.suptitle(f"CASOS DE DENGUE, FOCOS DE _Aedes_ sp., TEMPERATURAS (MÍNIMA, MÉDIA E MÁXIMA) E PRECIPITAÇÃO.\nCICLO ANUAL POR MÉDIAS EM SEMANAS EPIDEMIOLÓGICAS PARA O MUNICÍPIO DE {nome_cidade}, SANTA CATARINA ({ano_i} - {ano_f}).")
nome_arquivo = f"shs25_ciclo_anual_SE_subplots_{nome_cidade_arquivo}_{ano_i}_{ano_f}.png"
caminho_estatistica = "/home/sifapsc/scripts/matheus/dengue/resultados/estatistica/sazonalidade/"
caminho_shs = "/home/meteoro/scripts/matheus/operacional_dengue/modelagem/resultados/2025/"
#if _SALVAR == True:
os.makedirs(caminho_shs, exist_ok = True)
plt.savefig(f'{caminho_shs}{nome_arquivo}', format = "png", dpi = 300,  bbox_inches = "tight", pad_inches = 0.0)
print(f"""\n{green}SALVO COM SUCESSO!\n
{cyan}ENCAMINHAMENTO: {caminho_shs}\n
NOME DO ARQUIVO: {nome_arquivo}{reset}\n""")
#if _VISUALIZAR == True:
print(f"\n{cyan}Visualizando:\n{caminho_shs}{nome_arquivo}\n{reset}")
plt.show()

#sys.exit()
### Pré-Processamento (Série Temporal) ###############################################

serie_cidade = pd.DataFrame()
semanas = serie_casos["Semana"]
serie_cidade["Semana"] = semanas
serie_casos_x = serie_casos[["Semana", nome_cidade]]
serie_cidade = serie_cidade.merge(serie_casos_x, how = "inner", on = "Semana")
serie_cidade = serie_cidade.rename(columns = {nome_cidade : "casos"})
serie_focos_x = serie_focos[["Semana", nome_cidade]]
serie_cidade = serie_cidade.merge(serie_focos_x, how = "inner", on = "Semana")
serie_cidade = serie_cidade.rename(columns = {nome_cidade : "focos"})
serie_tmin_x = serie_tmin[["Semana", nome_cidade]]
serie_cidade = serie_cidade.merge(serie_tmin_x, how = "inner", on = "Semana")
serie_cidade = serie_cidade.rename(columns = {nome_cidade : "tmin"})
serie_tmed_x = serie_tmed[["Semana", nome_cidade]]
serie_cidade = serie_cidade.merge(serie_tmed_x, how = "inner", on = "Semana")
serie_cidade = serie_cidade.rename(columns = {nome_cidade : "tmed"})
serie_tmax_x = serie_tmax[["Semana", nome_cidade]]
serie_cidade = serie_cidade.merge(serie_tmax_x, how = "inner", on = "Semana")
serie_cidade = serie_cidade.rename(columns = {nome_cidade : "tmax"})
serie_prec_x = serie_prec[["Semana", nome_cidade]]
serie_cidade = serie_cidade.merge(serie_prec_x, how = "inner", on = "Semana")
serie_cidade = serie_cidade.rename(columns = {nome_cidade : "prec"}) 
serie_cidade.to_csv(f"{caminho_dados}serie_temporal_{nome_cidade}.csv", index = False)
print(f"""\n{green}SALVO COM SUCESSO!\n
{cyan}ENCAMINHAMENTO: {caminho_dados}\n
NOME DO ARQUIVO: serie_temporal_{nome_cidade_arquivo}.csv{reset}\n""")
serie_cidade["Semana"] = pd.to_datetime(serie_cidade["Semana"])
serie_cidade.set_index("Semana", inplace = True)

print(f"\n{green}{nome_cidade} (Série Temporal)\n{reset}{serie_cidade}\n")

#sys.exit()

### Visualização Gráfica ###############################################
fig, axs = plt.subplots(2, 1, figsize = (12, 6), layout = "tight", frameon = False,  sharex = True)
axs[0].set_facecolor("honeydew") #.gcf()
ax2 = axs[0].twinx()
sns.lineplot(x = serie_cidade.index, y = serie_cidade["casos"], ax = axs[0],
				color = "purple", linewidth = 1, linestyle = "--", label = "Casos de Dengue")
axs[0].fill_between(serie_cidade.index, serie_cidade["casos"], color = "purple", alpha = 0.3)
axs[0].set_ylabel("Casos de Dengue")
axs[0].legend(loc = "upper center")
sns.lineplot(x = serie_cidade.index, y = serie_cidade["focos"],  ax = ax2,
				color = "darkgreen", linewidth = 1, linestyle = ":", label = "Focos de _Aedes_ sp.")
ax2.fill_between(serie_cidade.index, serie_cidade["focos"], color = "darkgreen", alpha = 0.35)
ax2.set_ylabel("Focos de _Aedes_ sp.")
ax2.legend(loc = "upper left")
axs[1].set_facecolor("honeydew") #.gcf()
ax3 = axs[1].twinx()#.set_facecolor("honeydew")
ax3.bar(serie_cidade.index, serie_cidade["prec"],
        color = "royalblue", alpha = 0.8, label = "Precipitação", width = 5)
ax3.set_ylabel("Precipitação (mm)")
ax3.legend(loc = "lower left")
sns.lineplot(x = serie_cidade.index, y = serie_cidade["tmax"],  ax = axs[1],
				color = "red", linewidth = 1.5, label = "Temperatura Máxima")
sns.lineplot(x = serie_cidade.index, y = serie_cidade["tmed"],  ax = axs[1],
				color = "orange", linewidth = 1.5, label = "Temperatura Média")
sns.lineplot(x = serie_cidade.index, y = serie_cidade["tmin"],  ax = axs[1],
				color = "darkblue", linewidth = 1.5, label = "Temperatura Mínima")
axs[1].set_ylabel("Temperaturas (ºC)")
axs[1].legend(loc = "upper left")
axs[1].grid(False)
axs[1].set_xlabel("Semanas Epidemiológicas")
#xticks_por_ano = serie_cidade.groupby(serie_cidade.index.year).head(1).index
#axs[1].set_xticks(xticks_por_ano)
#axs[1].set_xticklabels([str(ano.year) for ano in xticks_por_ano])
axs[1].xaxis.set_major_locator(mdates.YearLocator())
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.suptitle(f"CASOS DE DENGUE, FOCOS DE _Aedes_ sp., TEMPERATURAS (MÍNIMA, MÉDIA E MÁXIMA) E PRECIPITAÇÃO.\nSÉRIE TEMPORAL PARA O MUNICÍPIO DE {nome_cidade}, SANTA CATARINA.")
print(f"\n{green}{nome_cidade} (Série Temporal)\n{reset}{serie_cidade}\n")
nome_arquivo = f"shs25_distribuicao_temporal_subplots_{nome_cidade_arquivo}.png"
caminho_estatistica = "/home/sifapsc/scripts/matheus/dengue/resultados/estatistica/sazonalidade/"
caminho_shs = "/home/meteoro/scripts/matheus/operacional_dengue/modelagem/resultados/2025/"
#if _SALVAR == True:
os.makedirs(caminho_shs, exist_ok = True)
plt.savefig(f"{caminho_shs}{nome_arquivo}", format = "png", dpi = 300,  bbox_inches = "tight", pad_inches = 0.0)
print(f"""\n{green}SALVO COM SUCESSO!\n
{cyan}ENCAMINHAMENTO: {caminho_shs}\n
NOME DO ARQUIVO: {nome_arquivo}{reset}\n""")
#if _VISUALIZAR == True:
print(f"\n{cyan}Visualizando:\n{caminho_shs}{nome_arquivo}\n{reset}")
plt.show()
