### Bibliotecas Correlatas
# Básicas e Gráficas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cls  
import seaborn as sns 
from datetime import date, datetime, timedelta
# Suporte
import os
import sys
import joblib
import webbrowser
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category = ShapelyDeprecationWarning)
# Pré-Processamento e Validações
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
# Modelos
from sklearn.ensemble import RandomForestRegressor
# Mapas
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.patches as mpatches
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

#### Condições para Variar ####################################

##################### Valores Booleanos ############ # sys.argv[0] is the script name itself and can be ignored!
_AUTOMATIZAR = sys.argv[1]   # True|False                    #####
_AUTOMATIZA = True if _AUTOMATIZAR == "True" else False      #####
_VISUALIZAR = sys.argv[2]    # True|False                    #####
_VISUALIZAR = True if _VISUALIZAR == "True" else False       #####
_SALVAR = sys.argv[3]        # True|False                    #####
_SALVAR = True if _SALVAR == "True" else False               #####
##################################################################

SEED = np.random.seed(0)

_CIDADE = "Joinville"#"Joinville"#"Florianópolis"
_CIDADE = _CIDADE.upper()

_RETROAGIR = 2 # Semanas Epidemiológicas
_HORIZONTE = 0 # Tempo de Previsão



#################################################################################

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

##################################################################################

### Encaminhamento aos Diretórios
caminho_dados = "/home/meteoro/scripts/matheus/operacional_dengue/dados_operacao/" # CLUSTER
caminho_operacional = "/home/meteoro/scripts/matheus/operacional_dengue/"
caminho_shape = "/media/dados/shapefiles/" #SC/SC_Municipios_2022.shp #BR/BR_UF_2022.shp
caminho_modelos = f"/home/meteoro/scripts/matheus/operacional_dengue/modelagem/casos/{_ANO_ATUAL}/{_ANO_MES_DIA}/"
caminho_resultados = f"modelagem/resultados/{_ANO_ATUAL}/{_ANO_MES}/"
print(f"\nOS DADOS UTILIZADOS ESTÃO ALOCADOS NOS SEGUINTES CAMINHOS:\n\n{caminho_dados}\n\n")


##################################################################################
### Renomeação das Variáveis pelos Arquivos
casos = "casos_dive_pivot_total.csv"  # TabNet/DiveSC
# focos?
unicos = "casos_primeiros.csv"
municipios = "SC/SC_Municipios_2024.shp"
br = "BR/BR_UF_2022.shp"

##################################################################################
### Abrindo Arquivo
casos = pd.read_csv(f"{caminho_dados}{casos}", low_memory = False)
#focos = pd.read_csv(f"{caminho_dados}{focos}", low_memory = False)

unicos = pd.read_csv(f"{caminho_dados}{unicos}")
municipios = gpd.read_file(f"{caminho_shape}{municipios}")
br = gpd.read_file(f"{caminho_shape}{br}")
troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A',
         'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E',
         'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I',
         'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O',
         'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U',
         'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
cidades = unicos["Município"].copy()


#sys.exit()

###############################################################################



######################################################MODELAGEM############################################################




######################################################################################################
################################## Cartografia #######################################################
######################################################################################################

# Semana Epidemiológica
#semana_epidemio = "2024-10-20"
semana_epidemio1 = previsao_total.loc[previsao_total.index[-3], "Semana"]
semana_epidemio2 = previsao_total.loc[previsao_total.index[-2], "Semana"]
semana_epidemio3 = previsao_total.loc[previsao_total.index[-1], "Semana"]
lista_semanas = [semana_epidemio1, semana_epidemio2, semana_epidemio3]
# "2020-04-19" "2021-04-18" "2022-04-17" "2023-04-16"
for idx, semana_epidemio in enumerate(lista_semanas):
# SC_Pontos
	"""
#previsao_melt_geo = gpd.GeoDataFrame(previsao_melt_geo)#, geometry = municipios.geometry)
	fig, ax = plt.subplots(figsize = (20, 12), layout = "constrained", frameon = False)
	coord_atlantico = [(-54, -30),(-48, -30),
						(-48, -25),(-54, -25),
						(-54, -30)]
	atlantico_poly = Polygon(coord_atlantico)
	atlantico = gpd.GeoDataFrame(geometry = [atlantico_poly])
	atlantico.plot(ax = ax, color = "lightblue") # atlantico ~ base
	ax.set_aspect("auto")
	coord_arg = [(-55, -30),(-52, -30),
				(-52, -25),(-55, -25),
				(-55, -30)]
	arg_poly = Polygon(coord_arg)
	argentina = gpd.GeoDataFrame(geometry = [arg_poly])
	argentina.plot(ax = ax, color = "tan")
	br.plot(ax = ax, color = "tan", edgecolor = "black")
	municipios.plot(ax = ax, color = "lightgreen", edgecolor = "black")
	#
	print("IS NA...", previsao_melt_geo["Casos"].isna().sum())
	#sys.exit()
	previsao_melt_geo[previsao_melt_geo["Semana"] == semana_epidemio ].plot(ax = ax, column = "Casos",  legend = True,
																			label = "Casos", cmap = "YlOrRd", markersize = 50)
	zero = previsao_melt_geo[previsao_melt_geo["Casos"] == 0]
	zero[zero["Semana"] == semana_epidemio].plot(ax = ax, column = "Casos", legend = False,	
												label = "Casos", cmap = "YlOrBr")
	plt.xlim(-54, -48)
	plt.ylim(-29.5, -25.75)
	x_tail = -48.5
	y_tail = -29.25
	x_head = -48.5
	y_head = -28.75
	arrow = mpatches.FancyArrowPatch((x_tail, y_tail), (x_head, y_head),
									mutation_scale = 50, color = "darkblue")
	ax.add_patch(arrow)
	mid_x = (x_tail + x_head) / 2
	mid_y = (y_tail + y_head) / 2
	ax.text(mid_x, mid_y, "N", color = "white", ha = "center", va = "center",
			fontsize = "large", fontweight = "bold")
	ax.text(-52.5, -29, "Sistema de Referência de Coordenadas\nDATUM: SIRGAS 2000/22S.\nBase Cartográfica: IBGE, 2022.",
			color = "white", backgroundcolor = "darkgray", ha = "center", va = "center", fontsize = 14)
	plt.xlabel("Longitude")
	plt.ylabel("Latitude")
	plt.title(f"Casos Prováveis de Dengue Previstos em Santa Catarina.\nSemana Epidemiológica: {semana_epidemio}.", fontsize = 18)
	plt.grid(True)
	nome_arquivo = f"CASOS_pontual_preditivo_{data_atual}_{idx}.pdf"
	if _AUTOMATIZA == True and _SALVAR == True:
		os.makedirs(caminho_resultados, exist_ok = True)
		plt.savefig(f"{caminho_resultados}{nome_arquivo}", format = "pdf", dpi = 300)
		print(f"\n\n{green}{caminho_resultados}\n{nome_arquivo}\nSALVO COM SUCESSO!{reset}\n\n")
	if _AUTOMATIZA == True and _VISUALIZAR == True:
		print(f"{cyan}\nVISUALIZANDO:\n{caminho_resultados}\n{nome_arquivo}\n{reset}\n\n")
		plt.show()
		print(f"{cyan}\nENCERRADO:\n{caminho_resultados}\n{nome_arquivo}\n{reset}\n\n")
	"""
	# SC_Coroplético
	xy = municipios.copy()
	xy.drop(columns = ["CD_MUN", "SIGLA_UF", "AREA_KM2"], inplace = True)
	xy = xy.rename(columns = {"NM_MUN" : "Município"})
	xy["Município"] = xy["Município"].str.upper() 
	previsao_melt_poli = pd.merge(previsao_melt, xy, on = "Município", how = "left")
	previsao_melt_poligeo = gpd.GeoDataFrame(previsao_melt_poli, geometry = "geometry", crs = "EPSG:4674")
	fig, ax = plt.subplots(figsize = (20, 12), layout = "constrained", frameon = True)
	#plt.gca().tick_params(labelsize = 20)
	"""
	coord_atlantico = [(-54, -30),(-48, -30),
		               (-48, -25),(-54, -25),
		               (-54, -30)]
	atlantico_poly = Polygon(coord_atlantico)
	atlantico = gpd.GeoDataFrame(geometry = [atlantico_poly])
	atlantico.plot(ax = ax, color = "lightblue") # atlantico ~ base
	ax.set_aspect("auto")
	coord_arg = [(-55, -30),(-52, -30),
		         (-52, -25),(-55, -25),
		         (-55, -30)]
	arg_poly = Polygon(coord_arg)
	argentina = gpd.GeoDataFrame(geometry = [arg_poly])
	argentina.plot(ax = ax, color = "tan")
	br.plot(ax = ax, color = "tan", edgecolor = "black")
	"""
	municipios.plot(ax = ax, color = "lightgray", edgecolor = "black", linewidth = 0.5)
	v_max = previsao_melt_poligeo.select_dtypes(include="number").max().max()
	v_min = previsao_melt_poligeo.select_dtypes(include="number").min().min()
	intervalo = 250
	levels = np.arange(v_min, v_max + intervalo, intervalo)
	print(f"\n{green}v_min\n{reset}{v_min}\n")
	print(f"\n{green}v_max\n{reset}{v_max}\n")
	print(f"\n{green}levels\n{reset}{levels}\n")
	#recorte_temporal.plot(ax = ax, column = f"{str_var}",  legend = True,
							#label = f"{str_var}", cmap = "YlOrRd")#, add_colorbar = False,
												#levels = levels, add_labels = False,
												#norm = cls.Normalize(vmin = v_min, vmax = v_max))
	previsao_melt_poligeo[previsao_melt_poligeo["Semana"] == semana_epidemio].plot(ax = ax, column = "Casos",  legend = True, edgecolor = "black",
		                                                                           label = "Casos", cmap = "YlOrRd", linewidth = 0.5,#levels = levels, 
		                                                                           norm = cls.Normalize(vmin = v_min, vmax = v_max, clip = True))
	cbar_ax = ax.get_figure().get_axes()[-1]
	cbar_ax.tick_params(labelsize = 20)
	zero = previsao_melt_poligeo[previsao_melt_poligeo["Casos"] <= 0]
	zero[zero["Semana"] == semana_epidemio].plot(ax = ax, column = "Casos", legend = False, edgecolor = "black", linewidth = 0.5,
		                                         label = "Casos", cmap = "YlOrBr")#"YlOrBr")
	plt.xlim(-54, -48)
	plt.ylim(-29.5, -25.75)
	x_tail = -48.5
	y_tail = -29.25
	x_head = -48.5
	y_head = -28.75
	"""
	arrow = mpatches.FancyArrowPatch((x_tail, y_tail), (x_head, y_head),
		                             mutation_scale = 50, color = "darkblue")
	ax.add_patch(arrow)
	mid_x = (x_tail + x_head) / 2
	mid_y = (y_tail + y_head) / 2
	ax.text(mid_x, mid_y, "N", color = "white", ha = "center", va = "center",
		    fontsize = "large", fontweight = "bold")
	"""
	ax.text(-52.5, -29, "Sistema de Referência de Coordenadas\nDATUM: SIRGAS 2000/22S.\nBase Cartográfica: IBGE, 2022.",
		    color = "white", backgroundcolor = "darkgray", ha = "center", va = "center", fontsize = 20)
	ax.text(-52.5, -28.25, """LEGENDA

▢           Sem registro*

*Não há registro oficial ou
modelagem inexistente.""",
		    color = "black", backgroundcolor = "lightgray", ha = "center", va = "center", fontsize = 20)
	plt.xlabel("Longitude", fontsize = 18)
	plt.ylabel("Latitude", fontsize = 18)
	plt.title(f"Casos Prováveis de Dengue Previstos em Santa Catarina.\nSemana Epidemiológica: {semana_epidemio.strftime('%Y-%m-%d')}.", fontsize = 28)
	#plt.grid(True)
	nome_arquivo = f"CASOS_mapa_preditivo_{data_atual}_{idx}.pdf"
	nome_arquivo_png = f"CASOS_mapa_preditivo_{data_atual}_{idx}.png"
	if _AUTOMATIZA == True and _SALVAR == True:
		os.makedirs(caminho_resultados, exist_ok = True)
		plt.savefig(f"{caminho_resultados}{nome_arquivo}", format = "pdf", dpi = 150)
		plt.savefig(f"{caminho_resultados}{nome_arquivo_png}", format = "png", dpi = 300)
		print(f"\n\n{green}{caminho_resultados}\n{nome_arquivo}\nSALVO COM SUCESSO!{reset}\n\n")
		print(f"\n\n{green}{caminho_resultados}\n{nome_arquivo_png}\nSALVO COM SUCESSO!{reset}\n\n")
	if _AUTOMATIZA == True and _VISUALIZAR == True:	
		print(f"{cyan}\nVISUALIZANDO:\n{caminho_resultados}\n{nome_arquivo}\n{reset}\n\n")
		plt.show()
		print(f"{cyan}\nENCERRADO:\n{caminho_resultados}\n{nome_arquivo}\n{reset}\n\n")

### Salvando Últimas Previsões
municipios = previsao_total.columns[1:]
ultimas_previsoes_df = pd.DataFrame(index = municipios)
ultimas_previsoes = previsao_total.iloc[-3:,:]
print(f"\n{green}previsao_total\n{reset}{previsao_total}\n")
print(f"\n{green}ultimas_previsoes_df\n{reset}{ultimas_previsoes_df}\n")
print(f"\n{green}ultimas_previsoes\n{reset}{ultimas_previsoes}\n")
#ultimas_previsoes_teste.reset_index(inplace = True)
ultimas_previsoes = ultimas_previsoes.T
print(f"\n{green}ultimas_previsoes.T\n{reset}{ultimas_previsoes}\n")
s0 = ultimas_previsoes.columns[0]
s0 = ultimas_previsoes[s0]
s1 = ultimas_previsoes.columns[1]
s1 = ultimas_previsoes[s1]
s2 = ultimas_previsoes.columns[2]
s2 = ultimas_previsoes[s2]
print(f"\n{green}s0\n{reset}{s0}\n")
print(f"\n{green}s1\n{reset}{s1}\n")
print(f"\n{green}s2\n{reset}{s2}\n")
#sys.exit()
ultimas_previsoes["S0"] = s0
ultimas_previsoes["S1"] = s1
ultimas_previsoes["dif_S1-S0"] = s1 - s0
ultimas_previsoes["S2"] = s2
ultimas_previsoes["dif_S2-S1"] = s2 - s1
ultimas_previsoes["dif_S2-S0"] = s2 - s0
ultimas_previsoes_df = ultimas_previsoes[["S0", "S1", "S2", "dif_S1-S0", "dif_S2-S1","dif_S2-S0"]]
print(f"\n{green}ultimas_previsoes.T_df\n{reset}{ultimas_previsoes_df}\n")
ultimas_previsoes_vdd = ultimas_previsoes_df.T
ultimas_previsoes_vdd.reset_index(inplace = True)
print(f"\n{green}ultimas_previsoes.T_df.T\n{reset}{ultimas_previsoes_vdd}\n")
#ultimas_previsoes_vdd = ultimas_previsoes_vdd.drop(columns = "Semana")
if _SALVAR == True:
	os.makedirs(caminho_resultados, exist_ok = True)
	ultimas_previsoes_csv = f"ultimas_previsoes_v{_ANO_MES_DIA}_h{_HORIZONTE}_r{_RETROAGIR}.csv"
	ultimas_previsoes_vdd.to_csv(f"{caminho_resultados}{ultimas_previsoes_csv}", index = False)
	print(f"\n\n{green}{caminho_resultados}\n{ultimas_previsoes_csv}\nSALVO COM SUCESSO!{reset}\n\n")
	print(f"\n\n{green}OS VALORES DAS ÚLTIMAS PREVISÕES SÃO APRESENTADOS ABAIXO:\n{reset}{ultimas_previsoes_vdd}\n\n")
	
