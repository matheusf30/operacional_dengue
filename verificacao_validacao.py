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

#SEED = np.random.seed(0)

_CIDADE = "Florianópolis"#"Joinville"#"Florianópolis"
_CIDADE = _CIDADE.upper()

_RETROAGIR = 2 # Semanas Epidemiológicas (SE)
_HORIZONTE = 0 # Tempo de Previsão em SE

#################################################################################

_AGORA = datetime.now()
_ANO_ATUAL = str(datetime.today().year)
_MES_ATUAL = _AGORA.strftime("%m")
_DIA_ATUAL = _AGORA.strftime("%d")
_ANO_MES = f"{_ANO_ATUAL}{_MES_ATUAL}"
_ANO_MES_DIA = f"{_ANO_ATUAL}{_MES_ATUAL}{_DIA_ATUAL}"
#_ANO_MES = 202510
#_ANO_MES_DIA = 20251028 #f"{_ANO_ATUAL}{_MES_ATUAL}{_DIA_ATUAL}"
_ONTEM = datetime.today() - timedelta(days = 1)
_ANO_ONTEM = str(_ONTEM.year)
_MES_ONTEM = _ONTEM.strftime("%m")
_DIA_ONTEM = _ONTEM.strftime("%d")
_ANO_MES_ONTEM = f"{_ANO_ONTEM}{_MES_ONTEM}"
_ANO_MES_DIA_ONTEM = f"{_ANO_ONTEM}{_MES_ONTEM}{_DIA_ONTEM}"

##################################################################################

### Encaminhamento aos Diretórios
caminho_dados = "/home/meteoro/scripts/matheus/teste/operacional_dengue/dados_operacao/" # CLUSTER
caminho_operacional = "/home/meteoro/scripts/matheus/teste/operacional_dengue/"
caminho_shape = "/media/dados/shapefiles/" #SC/SC_Municipios_2022.shp #BR/BR_UF_2022.shp
caminho_modelos = f"/home/meteoro/scripts/matheus/teste/operacional_dengue/modelagem/casos/{_ANO_ATUAL}/{_ANO_MES_DIA}/"
caminho_resultados = f"home/meteoro/scripts/matheus/teste/operacional_dengue/modelagem/resultados/{_ANO_ATUAL}/{_ANO_MES}/"
caminho_previsao = f"modelagem/resultados/{_ANO_ATUAL}/{_ANO_MES}/"#dados_previstos/"
print(f"\n{green}HOJE:\n{reset}{_ANO_MES_DIA}\n")
print(f"\n{green}ONTEM:\n{reset}{_ANO_MES_DIA_ONTEM}\n")
print(f"\n{green}OS DADOS UTILIZADOS ESTÃO ALOCADOS NOS SEGUINTES CAMINHOS:\n\n{caminho_dados}{reset}\n\n")


######################################################

### Renomeação das Variáveis pelos Arquivos
casos = "casos_dive_pivot_total.csv"  # TabNet/DiveSC
ultimas_previsoes = f"ultimas_previsoes_v{_ANO_MES_DIA}_h{_HORIZONTE}_r{_RETROAGIR}.csv"
previsao_pivot = f"previsao_pivot_total_v{_ANO_MES_DIA}_h{_HORIZONTE}_r{_RETROAGIR}.csv"
previsao_melt = f"previsao_melt_total_v{_ANO_MES_DIA}_h{_HORIZONTE}_r{_RETROAGIR}.csv"
ultimas_previsoes_ontem = f"ultimas_previsoes_v{_ANO_MES_DIA_ONTEM}_h{_HORIZONTE}_r{_RETROAGIR}.csv"
previsao_pivot_ontem = f"previsao_pivot_total_v{_ANO_MES_DIA_ONTEM}_h{_HORIZONTE}_r{_RETROAGIR}.csv"
previsao_melt_ontem = f"previsao_melt_total_v{_ANO_MES_DIA_ONTEM}_h{_HORIZONTE}_r{_RETROAGIR}.csv"
#focos = "focos_pivot.csv"
prec = f"{_ANO_ATUAL}/prec_semana_ate_{_ANO_ATUAL}.csv"
tmin = f"{_ANO_ATUAL}/tmin_semana_ate_{_ANO_ATUAL}.csv"
tmed = f"{_ANO_ATUAL}/tmed_semana_ate_{_ANO_ATUAL}.csv"
tmax = f"{_ANO_ATUAL}/tmax_semana_ate_{_ANO_ATUAL}.csv"
unicos = "casos_primeiros.csv"
municipios = "SC/SC_Municipios_2024.shp"
regionais = regionais = "censo_sc_regional.csv"
br = "BR/BR_UF_2022.shp"

###############################################################

### Abrindo Arquivo
casos = pd.read_csv(f"{caminho_dados}{casos}", low_memory = False)
try:
	ultimas_previsoes = pd.read_csv(f"{caminho_previsao}{ultimas_previsoes}", low_memory = False)
	previsao_pivot = pd.read_csv(f"{caminho_previsao}{previsao_pivot}", low_memory = False)
	previsao_melt = pd.read_csv(f"{caminho_previsao}{previsao_melt}", low_memory = False)
except FileNotFoundError:
	ultimas_previsoes = pd.read_csv(f"{caminho_previsao}{ultimas_previsoes_ontem}", low_memory = False)
	previsao_pivot = pd.read_csv(f"{caminho_previsao}{previsao_pivot_ontem}", low_memory = False)
	previsao_melt = pd.read_csv(f"{caminho_previsao}{previsao_melt_ontem}", low_memory = False)
#focos = pd.read_csv(f"{caminho_dados}{focos}", low_memory = False)
prec = pd.read_csv(f"{caminho_dados}{prec}", low_memory = False)
tmin = pd.read_csv(f"{caminho_dados}{tmin}", low_memory = False)
tmed = pd.read_csv(f"{caminho_dados}{tmed}", low_memory = False)
tmax = pd.read_csv(f"{caminho_dados}{tmax}", low_memory = False)
unicos = pd.read_csv(f"{caminho_dados}{unicos}")
municipios = gpd.read_file(f"{caminho_shape}{municipios}")
br = gpd.read_file(f"{caminho_shape}{br}")
regionais = pd.read_csv(f"{caminho_dados}{regionais}")
troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A',
         'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E',
         'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I',
         'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O',
         'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U',
         'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
cidades = unicos["Município"].copy()

####################################################################

### Funções Modelagem-Previsão

def abre_modelo(cidade):
	troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A', 'Ä': 'A',
		 'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E', 'Ë': 'E',
		 'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I', 'Ï': 'I',
		 'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O', 'Ö': 'O',
		 'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U', 'Ü': 'U',
		 'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
	for velho, novo in troca.items():
		cidade = cidade.replace(velho, novo)
	nome_modelo = f"RF_casos_v{_ANO_MES_DIA}_h{_HORIZONTE}_r{_RETROAGIR}_{cidade}.h5"
	modelo = joblib.load(f"{caminho_modelos}{nome_modelo}") #RF_casos_v20241017_h0_r2_URUSSANGA.h5
	print("\n" + f"{red}={cyan}={reset}"*40 + "\n")
	print(f"\n{green}MODELO RANDOM FOREST DE {bold}{cidade} ABERTO!\n{reset}")
	print(f"\n{green}Caminho e Nome:\n {bold}{caminho_modelos}{nome_modelo}\n{reset}")
	print("\n" + f"{red}~{cyan}~{reset}"*40 + "\n")
	return modelo
	
def monta_dataset(cidade):
	dataset = tmin[["Semana"]].copy()
	dataset["TMIN"] = tmin[cidade].copy()
	dataset["TMED"] = tmed[cidade].copy()
	dataset["TMAX"] = tmax[cidade].copy()
	dataset = dataset.merge(prec[["Semana", cidade]], how = "right", on = "Semana").copy()
	dataset = dataset.rename(columns = {f"{cidade}" : "PREC"})
	dataset.dropna(axis = 0, inplace = True)
	dataset["Semana"] = pd.to_datetime(dataset["Semana"])
	dataset = dataset.merge(casos[["Semana", cidade]], how = "right", on = "Semana").copy()
	dataset = dataset.rename(columns = {f"{cidade}" : "CASOS"})
	dataset.fillna(0, inplace = True)
	for r in range(_HORIZONTE + 1, _RETROAGIR + 1):
		dataset[f"TMIN_r{r}"] = dataset["TMIN"].shift(-r)
		dataset[f"TMED_r{r}"] = dataset["TMED"].shift(-r)
		dataset[f"TMAX_r{r}"] = dataset["TMAX"].shift(-r)
		dataset[f"PREC_r{r}"] = dataset["PREC"].shift(-r)
		dataset[f"CASOS_r{r}"] = dataset["CASOS"].shift(-r)
	dataset.drop(columns = ["TMIN", "TMED", "TMAX", "PREC"], inplace = True)
	dataset.dropna(inplace = True)
	dataset = dataset[dataset["Semana"].dt.year == 2025]
	dataset.set_index("Semana", inplace = True)
	dataset.columns.name = f"{_CIDADE}"
	print(f"\n{green}dataset:\n{reset}{dataset}")
	print(f"\n{green}dataset.info:\n{reset}{dataset.info()}")
	x = dataset.drop(columns = "CASOS")
	print(f"\n{green}x:\n{reset}{x}")
	print(f"\n{green}x.info:\n{reset}{x.info()}")
	y = dataset["CASOS"]
	print(f"\n{green}y:\n{reset}{y}")
	print(f"\n{green}y.info:\n{reset}{y.info()}")
	#sys.exit()
	return dataset, x, y
	
def treino_teste(x, y, cidade):
	SEED = np.random.seed(0)
	x_array = x.to_numpy()
	x_array = x_array.reshape(x_array.shape[0], -1)
	x_array = x.to_numpy().astype(int)
	y_array = y.to_numpy().astype(int)
	x_array = x_array.reshape(x_array.shape[0], -1)
	treino_x, teste_x, treino_y, teste_y = train_test_split(x_array, y_array,
															random_state = SEED,
															test_size = 0.2)
	explicativas = x.columns.tolist()
	treino_x_explicado = pd.DataFrame(treino_x, columns = explicativas)
	treino_x_explicado = treino_x_explicado.to_numpy().astype(int)
	return treino_x, teste_x, treino_y, teste_y, treino_x_explicado

def preve(modelo, x, treino_x_explicado = None):
	previsoes_x = modelo.predict(x)
	previsoes = [int(p) for p in previsoes_x]
	return previsoes
	
def metricas(dataset, previsoes, n, y):
	print(f"{red}={cyan}={reset}"*40)
	print(f"\n{green}RANDOM FOREST - {bold}{_CIDADE}\n{reset}")
	lista_op = [f"Observado: {dataset['CASOS'][i]}\nPrevisão (RF): {previsoes[i]}\n" for i in range(n)]
	print("\n".join(lista_op))
	print(f"{red}~{cyan}{reset}"*40)
	EQM = mean_squared_error(y, previsoes)
	RQ_EQM = np.sqrt(EQM)
	R_2 = round(r2_score(y, previsoes), 2)
	print(f"""
	\n MÉTRICAS - RANDOM FOREST - {_CIDADE}
	\n Erro Quadrático Médio: {EQM}
	\n Coeficiente de Determinação (R²): {R_2}
	\n Raiz Quadrada do Erro Quadrático Médio: {RQ_EQM}
	""")
	print("="*80)
	return EQM, RQ_EQM, R_2
	
##########################################################################################

### Visualizando arquivos
print(f"\n{green}CASOS:\n{reset}{casos}\n")
print(f"\n{green}PRECIPITAÇÃO:\n{reset}{prec}\n")
print(f"\n{green}TEMPERATURA MÍNIMA:\n{reset}{tmin}\n")
print(f"\n{green}TEMPERATURA MÉDIA:\n{reset}{tmed}\n")
print(f"\n{green}TEMPERATURA MÁXIMA:\n{reset}{tmax}\n")

print(f"\n{green}ÚLTIMAS PREVISÕES:\n{reset}{ultimas_previsoes}\n")
print(f"\n{green}PREVISÃO TOTAL (pivot):\n{reset}{previsao_pivot}\n")
print(f"\n{green}PREVISÃO TOTAL (melt):\n{reset}{previsao_melt}\n")

print(f"\n{green}CASOS.dtypes:\n{reset}{casos.dtypes}\n")
print(f"\n{green}ÚLTIMAS PREVISÕES.dtypes:\n{reset}{ultimas_previsoes.dtypes}\n")

print(f"\n{green}REGIONAIS:\n{reset}{regionais}\n")
print(f"\n{green}REGIONAIS (colunas):\n{reset}{regionais.columns}\n")
print(f"\n{green}REGIONAIS:\n{reset}{regionais['regional'].unique()}\n")

#sys.exit()
previstos = ultimas_previsoes.iloc[:3, :]
previsao_pivot["Semana"] = pd.to_datetime(previsao_pivot["Semana"])
previsao12 = previsao_pivot.iloc[:-2, :]
previstos["Semana"] = pd.to_datetime(previstos["Semana"])
casos["Semana"] = pd.to_datetime(casos["Semana"])
casos_atual = casos[casos["Semana"].dt.year == 2025]
casos_atual = casos_atual.iloc[:-1,:]

mapeamento = regionais.drop_duplicates(subset = ["Municipio"]).set_index("Municipio")["regional"]
previstos = previstos.set_index("Semana")
previsao12 = previsao12.set_index("Semana")
casos_atual = casos_atual.set_index("Semana")
previstos_reg = previstos.groupby(previstos.columns.map(mapeamento), axis = 1).sum()
previsao12_reg = previsao12.groupby(previsao12.columns.map(mapeamento), axis = 1).sum()
casos_atual_reg = casos_atual.groupby(casos_atual.columns.map(mapeamento), axis = 1).sum()
previstos.reset_index(inplace = True)
previsao12.reset_index(inplace = True)
casos_atual.reset_index(inplace = True)


print(f"\n{green}CASOS.dtypes:\n{reset}{casos}\n{casos.dtypes}\n")
print(f"\n{green}CASOS_ATUAL.dtypes:\n{reset}{casos_atual.dtypes}\n")
print(f"\n{green}PREVISTOS.dtypes:\n{reset}{previstos}\n{previstos.dtypes}\n")

print(f"\n{green}CASOS REGIONAIS:\n{reset}{casos_atual_reg}\n{casos_atual_reg.dtypes}\n")
print(f"\n{green}PREVISTOS REGIONAIS:\n{reset}{previstos_reg}\n{previstos_reg.dtypes}\n")
print(f"\n{green}ÚLTIMOS PREVISTOS REGIONAIS:\n{reset}{previsao12_reg}\n{previsao12_reg.dtypes}\n")
#sys.exit()

regionais = ["FOZ DO RIO ITAJAÍ", "GRANDE FLORIANÓPOLIS", "EXTREMO OESTE", "OESTE",
			"XANXERÊ", "ALTO URUGUAI CATARINENSE", "ALTO VALE DO RIO DO PEIXE",
			"MEIO OESTE", "NORDESTE", "VALE DO ITAPOCU", "PLANALTO NORTE",
			"SERRA CATARINENSE", "CARBONÍFERA", "EXTREMO SUL CATARINENSE", "LAGUNA",
			"ALTO VALE DO ITAJAÍ", "MÉDIO VALE DO ITAJAÍ"]
###

### Visualizações Gráficas


## Série (observado-previsão)
modelo = abre_modelo(_CIDADE)
dataset, x, y = monta_dataset(_CIDADE)
treino_x, teste_x, treino_y, teste_y, treino_x_explicado = treino_teste(x, y, _CIDADE)
previsoes = preve(modelo, x, treino_x_explicado)
print(f"\n\n{green}PREVISÕES:\n{reset}{previsoes}\n")
EQM, RQ_EQM, R_2 = metricas(dataset, previsoes, 5, y)

previstos2 = pd.DataFrame()
previstos2.index = y.index
previstos2[_CIDADE] = previsoes
previstos2.reset_index(inplace = True)
previstos2["Semana"] = pd.to_datetime(previstos2["Semana"])
previstos2[_CIDADE] = previstos2[_CIDADE].shift(1)
print(f"\n\n{green}PREVISÕES2:\n{reset}{previstos2}\n")
print(f"\n\n{green}CASOS_ATUAL:\n{reset}{casos_atual}\n")



plt.figure(figsize = (15, 8), layout = "constrained", frameon = False)
plt.plot(previstos["Semana"], previstos[_CIDADE],
			label = "Previsto (GFS)", color = "red", linewidth = 3, linestyle = ":")
plt.plot(previstos2["Semana"], previstos2[_CIDADE],
			label = "Previsto", color = "red", linewidth = 3)
plt.plot(casos_atual["Semana"], casos_atual[_CIDADE],
				label = "Observado", color = "blue")
#plt.plot(previsao12["Semana"], previsao12[_CIDADE], alpha = .4,
#			label = "Previsto (.csv)", color = "black", linewidth = 6)
plt.xlabel("Semanas Epidemiológicas (Série Histórica)")
plt.ylabel("Número de Casos de Dengue")
plt.title(f"COMPARAÇÃO ENTRE CASOS DE DENGUE PREVISTOS E OBSERVADOS, MUNICÍPIO DE {_CIDADE} (R² = {R_2})")
plt.legend()
plt.gca().set_facecolor("honeydew")
plt.show()

plt.figure(figsize = (15, 8), layout = "constrained", frameon = False)
plt.plot(previstos["Semana"], previstos[_CIDADE],
			label = "Previsto (GFS)", color = "red", linewidth = 3, linestyle = ":")
plt.plot(previsao12["Semana"], previsao12[_CIDADE],
			label = "Previsto", color = "red", linewidth = 3)
plt.plot(casos_atual["Semana"], casos_atual[_CIDADE],
				label = "Observado", color = "blue")
plt.xlabel("Semanas Epidemiológicas (Série Histórica)")
plt.ylabel("Número de Casos de Dengue")
plt.title(f"COMPARAÇÃO ENTRE CASOS DE DENGUE PREVISTOS E OBSERVADOS, MUNICÍPIO DE {_CIDADE} (R² = {R_2})")
plt.legend()
plt.gca().set_facecolor("honeydew")
plt.show()

regionais = ["FOZ DO RIO ITAJAÍ", "GRANDE FLORIANÓPOLIS", "EXTREMO OESTE", "OESTE",
			"XANXERÊ", "ALTO URUGUAI CATARINENSE", "ALTO VALE DO RIO DO PEIXE",
			"MEIO OESTE", "NORDESTE", "VALE DO ITAPOCU", "PLANALTO NORTE",
			"SERRA CATARINENSE", "CARBONÍFERA", "EXTREMO SUL CATARINENSE", "LAGUNA",
			"ALTO VALE DO ITAJAÍ", "MÉDIO VALE DO ITAJAÍ"]
"""			
for _REG in regionais:
	plt.figure(figsize = (15, 8), layout = "constrained", frameon = False)
	plt.plot(previstos_reg.index, previstos_reg[_REG],
				label = "Previsto (GFS)", color = "red", linewidth = 3, linestyle = ":")
	plt.plot(previsao12_reg.index, previsao12_reg[_REG],
				label = "Previsto", color = "red", linewidth = 3)
	plt.plot(casos_atual_reg.index, casos_atual_reg[_REG],
					label = "Observado", color = "blue")
	plt.xlabel("Semanas Epidemiológicas (Série Histórica)")
	plt.ylabel("Número de Casos de Dengue")
	plt.title(f"COMPARAÇÃO ENTRE CASOS PROVÁVEIS DE DENGUE PREVISTOS E OBSERVADOS\nREGIONAL: {_REG}")
	plt.legend()
	plt.gca().set_facecolor("honeydew")
	plt.show()
"""
"""
sns.lineplot(x = previstos["Semana"], y = previstos[_CIDADE],
             color = "red", alpha = 0.7, linewidth = 3, label = "Previsto")
sns.lineplot(x = previsao_pivot["Semana"], y = previsao_pivot[_CIDADE], linestyle = ":",
             color = "red", alpha = 0.7, linewidth = 3, label = "Previsto (GFS)")
sns.lineplot(x = casos_atual["Semana"], y = casos_atual[_CIDADE],
             color = "blue", alpha = 0.9, linewidth = 1, label = "Observado")
plt.title(f"MODELO RANDOM FOREST: PREVISÃO DE CASOS PROVÁVEIS DE DENGUE.\n MUNICÍPIO DE {_CIDADE}, SANTA CATARINA.\n", fontsize = 18)
plt.xlabel("Série Temporal (Semanas Epidemiológicas)", fontsize = 18)
plt.ylabel("Número de Casos de Dengue", fontsize = 18)
plt.xticks(rotation = "horizontal")
"""


if _SALVAR == True: #_AUTOMATIZA == True and 
	caminho_pdf = "modelagem/resultados/dados_previstos/graficos/"
	nome_arquivo = f"CASOS_serie_{_CIDADE}_v{_ANO_MES_DIA}_h{_HORIZONTE}_r{_RETROAGIR}.pdf"
	os.makedirs(caminho_pdf, exist_ok = True)
	plt.savefig(f"{caminho_pdf}{nome_arquivo}", format = "pdf", dpi = 300)
	print(f"\n\n{green}{caminho_pdf}\n{nome_arquivo}\nSALVO COM SUCESSO!{reset}\n\n")

## Regressão Linear (observado x previsão)


## Validação Modelo (treino x teste x previsão)

"""
eixo = fig.add_axes([0, 0, 1, 1])            # type: ignore
eixo2 = fig.add_axes([0.08, 0.6, 0.55, 0.3]) # type: ignore

eixo.plot(focos["Semana"], focos[cidade], color = "r")
eixo.set_xlim(datetime.datetime(2020,1,1), datetime.datetime(2022,12,4))
eixo.set_ylim(0, focos[cidade].max() + 50)
eixo.set_title(f"FOCOS DE _Aedes_ spp. EM {cidade}.", fontsize = 20, pad = 20)
eixo.set_ylabel("Quantidade de Focos Registrados (n)", fontsize = 16)
eixo.set_xlabel("Tempo (Semanas Epidemiológicas)", fontsize = 16)
#eixo.legend([cidade], loc = "upper left", fontsize = 14)
eixo.grid(True)

azul_esquerda = focos["Semana"] < datetime.datetime(2020,1,1)
azul_direita = focos["Semana"] > datetime.datetime(2022,12,4)

eixo2.plot(focos["Semana"], focos[cidade], color = "r")
eixo2.plot(focos[azul_esquerda]["Semana"], focos[azul_esquerda][cidade], color = "b")
eixo2.plot(focos[azul_direita]["Semana"], focos[azul_direita][cidade], color = "b")
eixo2.set_xlim(datetime.datetime(2012,1,1), datetime.datetime(2022,12,25))
eixo2.set_title(f"FOCOS DE _Aedes_ spp. EM {cidade}.", fontsize = 15)
eixo2.set_ylabel("Quantidade de Focos Registrados (n)", fontsize = 10)
eixo2.set_xlabel("Tempo (Semanas Epidemiológicas)", fontsize = 10)
eixo2.legend([cidade], loc = "best", fontsize = 8)
eixo2.grid(True)
plt.show()
"""
"""
try:
	prec_gfs = pd.read_csv(f"{caminho_dados}gfs_prec_semana_{_ANO_MES_DIA}.csv", low_memory = False)
	tmin_gfs = pd.read_csv(f"{caminho_dados}gfs_tmin_semana_{_ANO_MES_DIA}.csv", low_memory = False)
	tmed_gfs = pd.read_csv(f"{caminho_dados}gfs_tmed_semana_{_ANO_MES_DIA}.csv", low_memory = False)
	tmax_gfs = pd.read_csv(f"{caminho_dados}gfs_tmax_semana_{_ANO_MES_DIA}.csv", low_memory = False)
	print(f"\n{green}Arquivos utilizados do dia:\n{bold}{_DIA_ATUAL}/{_MES_ATUAL}/{_ANO_ATUAL}.\n{reset}")
	data_atual = _ANO_MES_DIA
except FileNotFoundError:
	prec_gfs = pd.read_csv(f"{caminho_dados}gfs_prec_semana_{_ANO_MES_DIA_ONTEM}.csv", low_memory = False)
	tmin_gfs = pd.read_csv(f"{caminho_dados}gfs_tmin_semana_{_ANO_MES_DIA_ONTEM}.csv", low_memory = False)
	tmed_gfs = pd.read_csv(f"{caminho_dados}gfs_tmed_semana_{_ANO_MES_DIA_ONTEM}.csv", low_memory = False)
	tmax_gfs = pd.read_csv(f"{caminho_dados}gfs_tmax_semana_{_ANO_MES_DIA_ONTEM}.csv", low_memory = False)
	print(f"\n{green}Arquivos utilizados do dia:\n{bold}{_DIA_ONTEM}/{_MES_ONTEM}/{_ANO_ONTEM}.\n{reset}")
	data_atual = _ANO_MES_DIA_ONTEM

prec_total = pd.concat([prec, prec_gfs])
print(f"\n{green}prec_total:\n{reset}{prec_total}")
print(f"\n{green}prec_total.info:\n{reset}{prec_total.info()}")
tmin_total = pd.concat([tmin, tmin_gfs])
print(f"\n{green}tmin_total:\n{reset}{tmin_total}")
print(f"\n{green}tmin_total.info:\n{reset}{tmin_total.info()}")
tmed_total = pd.concat([tmed, tmed_gfs])
print(f"\n{green}tmed_total:\n{reset}{tmed_total}")
print(f"\n{green}tmed_total.info:\n{reset}{tmed_total.info()}")
tmax_total = pd.concat([tmax, tmax_gfs])
print(f"\n{green}tmax_total:\n{reset}{tmax_total}")
print(f"\n{green}tmax_total.info:\n{reset}{tmax_total.info()}")
print(f"\n{green}tmax_total[['Semana', 'BOMBINHAS']]:\n{reset}{tmax_total[['Semana', 'BOMBINHAS']]}")
print(f"\n{green}tmax[['Semana', 'BOMBINHAS']]:\n{reset}{tmax[['Semana', 'BOMBINHAS']]}")
print(f"\n{green}cidades:\n{reset}{cidades.unique()}")
#sys.exit()
"""
###############################################################################
