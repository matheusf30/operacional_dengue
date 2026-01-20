### Bibliotecas Correlatas
# Básicas e Gráficas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cls  
import seaborn as sns 
from datetime import date, datetime, timedelta
from epiweeks import Week, Year
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
#for r in range(_HORIZONTE + 1, _RETROAGIR + 1):

#_AUTOMATIZA = True#False

#################################################################################

_AGORA = datetime.now()
_ANO_ATUAL = str(datetime.today().year)
_MES_ATUAL = _AGORA.strftime("%m")
_DIA_ATUAL = _AGORA.strftime("%d")
_ANO_MES = f"{_ANO_ATUAL}{_MES_ATUAL}"
_ANO_MES_DIA = f"{_ANO_ATUAL}{_MES_ATUAL}{_DIA_ATUAL}"
#_ANO_MES_DIA = "20251124" #Everton - Excluir
_ONTEM = datetime.today() - timedelta(days = 1)
_ANO_ONTEM = str(_ONTEM.year)
_MES_ONTEM = _ONTEM.strftime("%m")
_DIA_ONTEM = _ONTEM.strftime("%d")
_ANO_MES_ONTEM = f"{_ANO_ONTEM}{_MES_ONTEM}"
_ANO_MES_DIA_ONTEM = f"{_ANO_ONTEM}{_MES_ONTEM}{_DIA_ONTEM}"

##################################################################################

### Encaminhamento aos Diretórios
_LOCAL = "IFSC" # OPÇÕES>>> "GH" "CASA" "IFSC"
if _LOCAL == "GH": # _ = Variável Privada
	caminho_dados = "https://raw.githubusercontent.com/matheusf30/dados_dengue/main/"
	caminho_modelos = "https://github.com/matheusf30/dados_dengue/tree/main/modelos"
elif _LOCAL == "IFSC":
	caminho_dados = "/home/meteoro/scripts/operacional_dengue/dados_operacao/" # CLUSTER
	caminho_operacional = "/home/meteoro/scripts/operacional_dengue/"
	caminho_shape = "/media/dados/shapefiles/" #SC/SC_Municipios_2022.shp #BR/BR_UF_2022.shp
	caminho_modelos = f"/home/meteoro/scripts/operacional_dengue/modelagem/casos/{_ANO_ATUAL}/{_ANO_MES_DIA}/"
	caminho_resultados = f"modelagem/resultados/{_ANO_ATUAL}/{_ANO_MES}/"
#	caminho_resultados = "home/meteoro/scripts/operacional_dengue/modelagem/resultados/"
else:
	print("CAMINHO NÃO RECONHECIDO! VERIFICAR LOCAL!")
print(f"\n{green}HOJE:\n{reset}{_ANO_MES_DIA}\n")
print(f"\n{green}ONTEM:\n{reset}{_ANO_MES_DIA_ONTEM}\n")

print(f"\nOS DADOS UTILIZADOS ESTÃO ALOCADOS NOS SEGUINTES CAMINHOS:\n\n{caminho_dados}\n\n")


regionais = "/home/meteoro/scripts/operacional_dengue/dados_operacao/censo_sc_regional.csv"
municipios = "/media/dados/shapefiles/SC/SC_Municipios_2024.shp"
##################################################################################
municipios = gpd.read_file(municipios, low_memory = False)
regionais = pd.read_csv(regionais, low_memory = False)

### Renomeação das Variáveis pelos Arquivos
casos = "casos_dive_pivot_total.csv"  # TabNet/DiveSC
#focos = "focos_pivot.csv"
prec = f"{_ANO_ATUAL}/prec_semana_ate_{_ANO_ATUAL}.csv"
tmin = f"{_ANO_ATUAL}/tmin_semana_ate_{_ANO_ATUAL}.csv"
tmed = f"{_ANO_ATUAL}/tmed_semana_ate_{_ANO_ATUAL}.csv"
tmax = f"{_ANO_ATUAL}/tmax_semana_ate_{_ANO_ATUAL}.csv"
unicos = "casos_primeiros.csv"
#municipios = "SC/SC_Municipios_2022.shp"
br = "BR/BR_UF_2022.shp"
censo = "censo_sc_xy.csv"


###############################################################

### Abrindo Arquivo
casos = pd.read_csv(f"{caminho_dados}{casos}", low_memory = False)
#focos = pd.read_csv(f"{caminho_dados}{focos}", low_memory = False)
censo = pd.read_csv(f"{caminho_dados}{censo}", low_memory = False)
prec = pd.read_csv(f"{caminho_dados}{prec}", low_memory = False)
tmin = pd.read_csv(f"{caminho_dados}{tmin}", low_memory = False)
tmed = pd.read_csv(f"{caminho_dados}{tmed}", low_memory = False)
tmax = pd.read_csv(f"{caminho_dados}{tmax}", low_memory = False)
unicos = pd.read_csv(f"{caminho_dados}{unicos}")
#municipios = gpd.read_file(f"{caminho_shape}{municipios}")
br = gpd.read_file(f"{caminho_shape}{br}")
troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A',
         'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E',
         'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I',
         'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O',
         'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U',
         'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
cidades = unicos["Município"].copy()

try:
	prec_gfs = pd.read_csv(f"{caminho_dados}{_ANO_ATUAL}/{_MES_ATUAL}/gfs_prec_semana_{_ANO_MES_DIA}.csv", low_memory = False)
	tmin_gfs = pd.read_csv(f"{caminho_dados}{_ANO_ATUAL}/{_MES_ATUAL}/gfs_tmin_semana_{_ANO_MES_DIA}.csv", low_memory = False)
	tmed_gfs = pd.read_csv(f"{caminho_dados}{_ANO_ATUAL}/{_MES_ATUAL}/gfs_tmed_semana_{_ANO_MES_DIA}.csv", low_memory = False)
	tmax_gfs = pd.read_csv(f"{caminho_dados}{_ANO_ATUAL}/{_MES_ATUAL}/gfs_tmax_semana_{_ANO_MES_DIA}.csv", low_memory = False)
	print(f"\n{green}Arquivos utilizados do dia:\n{bold}{_DIA_ATUAL}/{_MES_ATUAL}/{_ANO_ATUAL}.\n{reset}")
	data_atual = _ANO_MES_DIA
except FileNotFoundError:
	prec_gfs = pd.read_csv(f"{caminho_dados}{_ANO_ONTEM}/{_MES_ONTEM}/gfs_prec_semana_{_ANO_MES_DIA_ONTEM}.csv", low_memory = False)
	tmin_gfs = pd.read_csv(f"{caminho_dados}{_ANO_ONTEM}/{_MES_ONTEM}/gfs_tmin_semana_{_ANO_MES_DIA_ONTEM}.csv", low_memory = False)
	tmed_gfs = pd.read_csv(f"{caminho_dados}{_ANO_ONTEM}/{_MES_ONTEM}/gfs_tmed_semana_{_ANO_MES_DIA_ONTEM}.csv", low_memory = False)
	tmax_gfs = pd.read_csv(f"{caminho_dados}{_ANO_ONTEM}/{_MES_ONTEM}/gfs_tmax_semana_{_ANO_MES_DIA_ONTEM}.csv", low_memory = False)
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

### Pré-processamento Incidência
print(f"\n{green}CASOS:\n{reset}{casos}\n")
censo["Municipio"] = censo["Municipio"].str.upper()
print(f"\n{green}CENSO:\n{reset}{censo}\n")
populacao_dict = censo.set_index("Municipio")["Populacao"].to_dict()
incidencia = casos.copy()
for municipio in incidencia.columns[1: ]:
	if municipio in populacao_dict:
		incidencia[municipio] = (incidencia[municipio] / populacao_dict[municipio]) * 100000
		#incidencia[municipio] = incidencia[municipio] * 100000

#incidencia = casos[cidade]/censo[cidade] for cidade in censo if cidade in casos.columns
print(f"\n{green}INCIDÊNCIA:\n{reset}{incidencia}\n")
print(f"\n{green}DICIONÁRIO POPULAÇÃO:\n{reset}{populacao_dict}\n")
#sys.exit()

###############################################################################

# modelo = joblib.load(f"{caminho_modelos}RF_r{_RETROAGIR}_{cidade}.h5")
def tempo_epidemiologico(df_original):
	tempo = pd.DataFrame()
	tempo["Semana"] = df_original["Semana"]
	tempo["Semana"] = pd.to_datetime(tempo["Semana"])
	tempo["SE"] = tempo["Semana"].apply(lambda data: Week.fromdate(data).week)
	tempo["ano_epi"] = tempo["Semana"].dt.year
	tempo.loc[(tempo["Semana"].dt.month == 1) & (tempo["SE"] > 50), "ano_epi"] -= 1
	tempo.loc[(tempo["Semana"].dt.month == 12) & (tempo["SE"] == 1), "ano_epi"] += 1
	print(f"\n{green}TEMPO CRONOLÓGICO (epidemiológico):\n{reset}{tempo}\n")
	return tempo

caminho_dados = "/home/meteoro/scripts/operacional_dengue/dados_operacao/"
focos = "focos_semanal_pivot.csv"
focos = pd.read_csv(f"{caminho_dados}{focos}", low_memory = False)	
tempo = tempo_epidemiologico(casos)
SE = tempo["SE"].iloc[-1]
ano_epi = tempo["ano_epi"].iloc[-1]
print(f"\n{green}DATA EPIDEMIOLÓGICA:\n{reset}{tempo['SE'].iloc[-1]}/{tempo['ano_epi'].iloc[-1]}\n")
print(f"\n{green}SEMANA EPIDEMIOLÓGICA: {reset}{SE}\n{green}ANO EPIDEMIOLÓGICO: {reset}{ano_epi}\n")
caminho_resultados = f"/home/meteoro/scripts/operacional_dengue/resultados/{ano_epi}/SE{SE}/epidemiologia/"
caminho_modelos = f"/home/meteoro/scripts/operacional_dengue/resultados/{ano_epi}/SE{SE}/epidemiologia/modelos/"
print(f"\n{green}CAMINHO DOS RESULTADOS:\n{reset}{caminho_resultados}\n")
print(f"\n{green}CAMINHO DOS MODELOS:\n{reset}{caminho_modelos}\n")
if not os.path.exists(caminho_resultados):
	os.makedirs(caminho_resultados)
	
def abre_modelo(cidade):
	troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A', 'Ä': 'A',
		 'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E', 'Ë': 'E',
		 'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I', 'Ï': 'I',
		 'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O', 'Ö': 'O',
		 'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U', 'Ü': 'U',
		 'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
	for velho, novo in troca.items():
		cidade = cidade.replace(velho, novo)
	nome_modelo = f"RF_casos_vSE{SE}_h{_HORIZONTE}_r{_RETROAGIR}_{cidade}.h5"
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
	dataset = dataset.merge(casos[["Semana", cidade]], how = "right", on = "Semana").copy()
	dataset = dataset.rename(columns = {f"{cidade}" : "CASOS"})
	dataset.fillna(0, inplace = True)
	for r in range(_HORIZONTE + 1, _RETROAGIR):
		dataset[f"TMIN_r{r}"] = dataset["TMIN"].shift(-r)
		dataset[f"TMED_r{r}"] = dataset["TMED"].shift(-r)
		dataset[f"TMAX_r{r}"] = dataset["TMAX"].shift(-r)
		dataset[f"PREC_r{r}"] = dataset["PREC"].shift(-r)
		dataset[f"CASOS_r{r}"] = dataset["CASOS"].shift(-r)
	dataset.dropna(inplace = True)
	dataset.set_index("Semana", inplace = True)
	dataset.columns.name = f"{_CIDADE}"
	print(f"\n{green}dataset:\n{reset}{dataset}")
	print(f"\n{green}dataset.info:\n{reset}{dataset.info()}")
	dataset1 = dataset.iloc[-12:-1,:]
	print(f"\n{green}dataset1:\n{reset}{dataset1}")
	print(f"\n{green}dataset1.info:\n{reset}{dataset1.info()}")
	x1 = dataset1.copy() #dataset.drop(columns = "CASOS")
	print(f"\n{green}x1:\n{reset}{x1}")
	print(f"\n{green}x1.info:\n{reset}{x1.info()}")
	y1 = dataset1["CASOS"]
	print(f"\n{green}y1:\n{reset}{y1}")
	print(f"\n{green}y1.info:\n{reset}{y1.info()}")
	#sys.exit()
	return dataset1, x1, y1
	
def monta_dataset2(cidade, previsoes):
	dataset2 = tmin_total[["Semana"]].copy()
	dataset2["TMIN"] = tmin_total[_CIDADE].copy()
	dataset2["TMED"] = tmed_total[_CIDADE].copy()
	dataset2["TMAX"] = tmax_total[_CIDADE].copy()
	dataset2 = dataset2.merge(prec_total[["Semana", _CIDADE]], how = "right", on = "Semana").copy()
	dataset2 = dataset2.rename(columns = {_CIDADE : "PREC"})
	print(f"\n{green}dataset2:\n{reset}{dataset2}")
	print(f"\n{green}dataset2.info:\n{reset}{dataset2.info()}")
	dataset_casos = pd.DataFrame()
	dataset_casos["Semana"] = dataset2["Semana"].iloc[-12:]
	print(f"\n{green}dataset_casos:\n{reset}{dataset_casos}")
	print(f"\n{green}dataset_casos.info:\n{reset}{dataset_casos.info()}")
	dataset_casos = dataset_casos.merge(casos[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
	dataset_casos = dataset_casos.rename(columns = {_CIDADE : "CASOS"})
	print(f"\n{green}dataset_casos:\n{reset}{dataset_casos}")
	print(f"\n{green}dataset_casos.info:\n{reset}{dataset_casos.info()}")
	dataset_casos.loc[dataset_casos.index[-3], "CASOS"] = previsoes[-3]
	dataset_casos.loc[dataset_casos.index[-2], "CASOS"] = previsoes[-2]
	dataset_casos.loc[dataset_casos.index[-1], "CASOS"] = previsoes[-1]
	print(f"\n{green}dataset_casos:\n{reset}{dataset_casos}")
	print(f"\n{green}dataset_casos.info:\n{reset}{dataset_casos.info()}")
	#sys.exit()
	dataset2 = dataset2.merge(dataset_casos[["Semana", "CASOS"]], how = "right", on = "Semana").copy()
	print(f"\n{green}dataset2:\n{reset}{dataset2}")
	print(f"\n{green}dataset2.info:\n{reset}{dataset2.info()}")
	for r in range(_HORIZONTE + 1, _RETROAGIR):
		dataset2[f"TMIN_r{r}"] = dataset2["TMIN"].shift(-r)
		dataset2[f"TMED_r{r}"] = dataset2["TMED"].shift(-r)
		dataset2[f"TMAX_r{r}"] = dataset2["TMAX"].shift(-r)
		dataset2[f"PREC_r{r}"] = dataset2["PREC"].shift(-r)
		dataset2[f"CASOS_r{r}"] = dataset2["CASOS"].shift(-r)
	dataset2.dropna(inplace = True)
	dataset2.set_index("Semana", inplace = True)
	dataset2.columns.name = f"{_CIDADE}"
	print(f"\n{green}dataset2:\n{reset}{dataset2}")
	print(f"\n{green}dataset2.info:\n{reset}{dataset2.info()}")
	### Dividindo Dataset em Treino e Teste
	x2 = dataset2.iloc[-12:,:]
	print(f"\n{green}x2:\n{reset}{x2}")
	print(f"\n{green}x2.info:\n{reset}{x2.info()}")
	y2 = dataset2["CASOS"]
	print(f"\n{green}y2:\n{reset}{y2}")
	print(f"\n{green}y2.info:\n{reset}{y2.info()}")
	return dataset2, x2, y2

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

def escalona(treino_x, teste_x):
	escalonador = StandardScaler()
	escalonador.fit(treino_x)
	treino_normal_x = escalonador.transform(treino_x)
	teste_normal_x = escalonador.transform(teste_x)
	return treino_normal_x, teste_normal_x

def modela_treina_preve(treino_x, treino_y, teste_x, SEED):
	modelo = RandomForestRegressor(n_estimators = 100, random_state = SEED)
	modelo.fit(treino_x_explicado, treino_y)
	y_previsto = modelo.predict(teste_x)
	previsoes = modelo.predict(x)
	previsoes = [int(p) for p in previsoes]
	return modelo, y_previsto, previsoes

def preve(modelo, x, treino_x_explicado = None):
	previsoes_x = modelo.predict(x)
	previsoes = [int(p) for p in previsoes_x]
	return previsoes
	
def preve_ultimos_12(modelo, x, treino_x_explicado):
	x = x.iloc[-12:,:]
	#treino_x_explicado = treino_x_explicado.iloc[-12:,:]
	y_previsto = modelo.predict(treino_x_explicado)
	previsoes_x = modelo.predict(x)
	previsoes = [int(p) for p in previsoes_x]
	print(f"\n{green}x:\n{reset}{x}")
	print(f"\n{green}previsoes:\n{reset}{previsoes}")
	print(f"\n{green}treino_x_explicado:\n{reset}{treino_x_explicado}")
	print(f"\n{green}y_previsto:\n{reset}{y_previsto}")
	return previsoes, y_previsto

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

def grafico(previsoes, R_2):
	final = pd.DataFrame()
	final["Semana"] = casos["Semana"]
	final["Casos"] = casos[cidade]
	final.drop(1, axis=0, inplace = True)
	final["Previstos"] = previsoes
	final["Semana"] = pd.to_datetime(final["Semana"])
	print(final)
	print("="*80)
	sns.lineplot(x = final["Semana"], y = final["Casos"], # linestyle = "--" linestyle = "-."
 				color = "darkblue", linewidth = 1, label = "Observado")
	sns.lineplot(x = final["Semana"], y = final["Previstos"],
				color = "red", alpha = 0.7, linewidth = 3, label = "Previsto")
	plt.title(f"MODELO RANDOM FOREST (R²: {R_2}): OBSERVAÇÃO E PREVISÃO.\n MUNICÍPIO DE {cidade}, SANTA CATARINA.")
	plt.xlabel("Semanas Epidemiológicas na Série de Anos")
	plt.ylabel("Número de Focos de _Aedes_ sp.")
	#plt.show()

def previsao_metricas(dataset, previsoes, n, teste_y, y_previsto):
	nome_modelo = "Random Forest"
	print("="*80)
	print(f"\n{nome_modelo.upper()} - {cidade}\n")
	lista_op = [f"Focos: {dataset['CASOS'][i]}\nPrevisão {nome_modelo}: {previsoes[i]}\n" for i in range(n)]
	print("\n".join(lista_op))
	print("~"*80)
	EQM = mean_squared_error(teste_y, y_previsto)
	RQ_EQM = np.sqrt(EQM)
	R_2 = r2_score(teste_y, y_previsto).round(2)
	print(f"""
	\n MÉTRICAS {nome_modelo.upper()} - {cidade}
	\n Erro Quadrático Médio: {EQM}
	\n Coeficiente de Determinação (R²): {R_2}
	\n Raiz Quadrada do Erro Quadrático Médio: {RQ_EQM}
	""")
	print("="*80)
	return EQM, RQ_EQM, R_2

def grafico_previsao(previsao, teste, string_modelo):
	if string_modelo not in ["RF", "NN"]:
		print("!!"*80)
		print("\n   MODELO NÃO RECONHECIDO\n   TENTE 'RF' PARA RANDOM FOREST\n   OU 'NN' PARA REDE NEURAL\n")
		print("!!"*80)
		#sys.exit()
	# Gráfico de Comparação entre Observação e Previsão dos Modelos
	nome_modelo = "Random Forest" if string_modelo == "RF" else "Rede Neural"
	final = pd.DataFrame()
	final["Semana"] = focos["Semana"]
	final["Casos"] = focos[cidade]
	final.drop([d for d in range(_RETROAGIR)], axis=0, inplace = True)
	final.drop(final.index[-_RETROAGIR:], axis=0, inplace = True)
	previsoes = previsao if string_modelo == "RF" else [np.argmax(p) for p in previsao]
	previsoes = previsoes[:len(final)]
	final["Previstos"] = previsoes
	final["Semana"] = pd.to_datetime(final["Semana"])
	print(final)
	print("="*80)
	sns.lineplot(x = final["Semana"], y = final["Casos"], # linestyle = "--" linestyle = "-."
				color = "darkblue", linewidth = 1, label = "Observado")
	sns.lineplot(x = final["Semana"], y = final["Previstos"],
				color = "red", alpha = 0.7, linewidth = 3, label = "Previsto")
	plt.title(f"MODELO {nome_modelo.upper()} (R²: {R_2}): OBSERVAÇÃO E PREVISÃO.\n MUNICÍPIO DE {cidade}, SANTA CATARINA.")
	plt.xlabel("Semanas Epidemiológicas na Série de Anos")
	plt.ylabel("Número de Casos de Dengue")
	#plt.show()

def salva_modelo(modelo, cidade):
	troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A', 'Ä': 'A',
			'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E', 'Ë': 'E',
			'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I', 'Ï': 'I',
			'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O', 'Ö': 'O',
			'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U', 'Ü': 'U',
			'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
	for velho, novo in troca.items():
		cidade = cidade.replace(velho, novo)
	nome_modelo = f"RF_casos_vSE{SE}_h{_HORIZONTE}_r{_RETROAGIR}_{cidade}.h5"
	joblib.dump(modelo, f"{caminho_modelos}{nome_modelo}")
	print("\n" + f"{red}={cyan}={reset}"*40 + "\n")
	print(f"\n{green}MODELO RANDOM FOREST DE {bold}{cidade} ABERTO!\n{reset}")
	print(f"\n{green}Caminho e Nome:\n {bold}{caminho_modelos}{nome_modelo}\n{reset}")
	print("\n" + f"{red}~{cyan}~{reset}"*40 + "\n")
	print("\n" + "="*80 + "\n")
	
def tempo_epidemiologico(df_original):
	tempo = pd.DataFrame()
	tempo["Semana"] = df_original["Semana"]
	tempo["Semana"] = pd.to_datetime(tempo["Semana"])
	tempo["SE"] = tempo["Semana"].apply(lambda data: Week.fromdate(data).week)
	tempo["ano_epi"] = tempo["Semana"].dt.year
	tempo.loc[(tempo["Semana"].dt.month == 1) & (tempo["SE"] > 50), "ano_epi"] -= 1
	tempo.loc[(tempo["Semana"].dt.month == 12) & (tempo["SE"] == 1), "ano_epi"] += 1
	print(f"\n{green}TEMPO CRONOLÓGICO (epidemiológico):\n{reset}{tempo}\n")
	return tempo

######################################################MODELAGEM############################################################
municipios["NM_MUN"] = municipios["NM_MUN"].str.upper()
municipios = municipios.merge(regionais[["Municipio", "regional"]],
								left_on = "NM_MUN", right_on = "Municipio",
								how = "left")
regionais = municipios.dissolve(by = "regional")

_d7 = datetime.today() - timedelta(days = 7)
_d7 = _d7 - timedelta(days = _d7.weekday() + 1)
_d7 = _d7.strftime("%Y-%m-%d")
### Exibindo Informações, Gráficos e Métricas
#previsao_total = []
previsao_total = pd.DataFrame()
casos_ultimos = casos.iloc[-12:,:]
previsao_total["Semana"] = casos_ultimos["Semana"].copy()
 #pd.date_range(start = "2014-01-05", end = "2022-12-25", freq = "W")
previsao_total["Semana"] = pd.to_datetime(previsao_total["Semana"])
previsao_total.reset_index(inplace = True)
#previsao_total.drop(1, axis = 0, inplace = True)
previsao_total.drop(columns = "index", inplace = True)
novas_SE = pd.date_range(start = previsao_total['Semana'].max(), periods = 3, freq = "W-SUN")[1:]
novas_linhas = pd.DataFrame({"Semana": novas_SE})
previsao_total = pd.concat([previsao_total, novas_linhas], ignore_index=True)
previsao_total.reset_index(inplace = True)
previsao_total = previsao_total.drop_duplicates(subset = ["Semana"], keep = "first")
previsao_total.drop(columns = "index", inplace = True)
print(f"\n{green}PREVISÃO TOTAL:\n{reset}{previsao_total}")
#previsao_total.drop([d for d in range(_RETROAGIR)], axis = 0, inplace = True)
#previsao_total.drop(previsao_total.index[-_RETROAGIR:], axis = 0, inplace = True)
#sys.exit()

if _AUTOMATIZA == True:
	for _CIDADE in cidades:
		try:
			modelo = abre_modelo(_CIDADE)
			dataset, x, y = monta_dataset(_CIDADE)
			treino_x, teste_x, treino_y, teste_y, treino_x_explicado = treino_teste(x, y, _CIDADE)
			previsoes = preve(modelo, x, treino_x_explicado)
			EQM, RQ_EQM, R_2 = metricas(dataset, previsoes, 5, y)
			dataset2, x2, y2 = monta_dataset2(_CIDADE, previsoes)
			previsoes2 = preve(modelo, x2)
			previsao_total[_CIDADE] = previsoes2
			#sys.exit()
		except ValueError as e:
			print(f"\n{red}Modelo para {_CIDADE} apresenta ERRO!\n{magenta}ValueError: {e}\n{reset}")
			print(f"\n{cyan}Por favor, entre em contato para {bold}resolver o problema!{reset}\n")
		except KeyError as e:
			print(f"\n{red}Modelo para {_CIDADE} apresenta ERRO!\n{magenta}KeyError: {e}\n{reset}")
			print(f"\n{cyan}Por favor, entre em contato para {bold}resolver o problema!{reset}\n")
		except FileNotFoundError as e:
			print(f"\n{red}Modelo para {_CIDADE} apresenta ERRO!\n{magenta}FileNotFoundError: {e}\n{reset}")
			print(f"\n{cyan}Por favor, entre em contato para {bold}resolver o problema!{reset}\n")     
else:
	modelo = abre_modelo(_CIDADE)
	dataset, x, y = monta_dataset(_CIDADE)
	treino_x, teste_x, treino_y, teste_y, treino_x_explicado = treino_teste(dataset, _CIDADE)
	previsoes, y_previsto = preve(modelo, x, treino_x_explicado)
	EQM, RQ_EQM, R_2 = metricas(dataset, previsoes, 5, y)
	dataset2, x2, y2 = monta_dataset2(_CIDADE, previsoes)
	previsoes2 = preve(modelo, x2)
	previsao_total[_CIDADE] = previsoes2
	grafico(previsoes2, R_2)
	
tempo = tempo_epidemiologico(previsao_total)
SE = tempo["SE"].iloc[-3]
ano_epi = tempo["ano_epi"].iloc[-3]
print(f"\n{green}DATA EPIDEMIOLÓGICA:\n{reset}{SE}/{ano_epi}\n")
print(f"\n{green}previsao_total:\n{cyan}{previsao_total}{reset}\n")
print(f"\n{green}DATA EPIDEMIOLÓGICA (previsão):\n{reset}{tempo['SE'].iloc[-1]}/{tempo['ano_epi'].iloc[-1]}\n")
print(f"\n{green}previsao_total:\n{cyan}{previsao_total}{reset}\n")
incidencia = previsao_total.copy()
for municipio in incidencia.columns[1: ]:
	if municipio in populacao_dict:
		incidencia[municipio] = (incidencia[municipio] / populacao_dict[municipio]) * 100000
previsao_melt = pd.melt(incidencia, id_vars = ["Semana"], 
                        var_name = "Município", value_name = "Incidencia")
previsao_melt = previsao_melt.sort_values(by = "Semana")
xy = unicos.drop(columns = ["Semana", "Casos"])
previsao_melt_xy = pd.merge(previsao_melt, xy, on = "Município", how = "left")
geometry = [Point(xy) for xy in zip(previsao_melt_xy["longitude"], previsao_melt_xy["latitude"])]
previsao_melt_geo = gpd.GeoDataFrame(previsao_melt_xy, geometry = geometry, crs = "EPSG:4674")
previsao_melt_geo = previsao_melt_geo[["Semana", "Município", "Incidencia", "geometry"]]
previsao_melt_geo["Semana"] = pd.to_datetime(previsao_melt_geo["Semana"])
print(f"\n{green}Caminho e Nome do arquivo:\n{reset}")
print(f"\n{green}{caminho_modelos}RF_casos_vSE{SE}_h{_HORIZONTE}_r{_RETROAGIR}_{_CIDADE}.h5\n{reset}")
if _SALVAR == True:
	os.makedirs(caminho_resultados, exist_ok = True)
	previsao_pivot_csv = f"previsao_pivot_incidencia_vSE{SE}_h{_HORIZONTE}_r{_RETROAGIR}.csv"
	previsao_total.to_csv(f"{caminho_resultados}{previsao_pivot_csv}", index = False)
	previsao_melt_csv = f"previsao_melt_incidencia_vSE{SE}_h{_HORIZONTE}_r{_RETROAGIR}.csv"
	previsao_melt.to_csv(f"{caminho_resultados}{previsao_melt_csv}", index = False)
	print(f"\n\n{green}{caminho_resultados}\n{previsao_pivot_csv}\nSALVO COM SUCESSO!{reset}\n\n")
	print(f"\n\n{green}{caminho_resultados}\n{previsao_melt_csv}\nSALVO COM SUCESSO!{reset}\n\n")
#sys.exit()

######################################################################################################
################################## Cartografia #######################################################
######################################################################################################

# Semana Epidemiológica
#semana_epidemio = "2024-10-20"
semana_epidemio1 = tempo.loc[tempo.index[-3],:]#previsao_total.loc[previsao_total.index[-3], "Semana"]
semana_epidemio2 = tempo.loc[tempo.index[-2],:]#previsao_total.loc[previsao_total.index[-2], "Semana"]
semana_epidemio3 = tempo.loc[tempo.index[-1],:]#previsao_total.loc[previsao_total.index[-1], "Semana"]
lista_semanas = [semana_epidemio1, semana_epidemio2, semana_epidemio3]
# "2020-04-19" "2021-04-18" "2022-04-17" "2023-04-16"
for idx, semana_epidemio in enumerate(lista_semanas):
	print(f"\n{green}CALENDÁRIO EPIDEMIOLÓGICO:\n{cyan}{semana_epidemio}{reset}\n")
	print(f"\n{green}SEMANA EPIDEMIOLÓGICA:\n{cyan}{semana_epidemio['SE']}{reset}\n")
	print(f"\n{green}ANO EPIDEMIOLÓGICO:\n{cyan}{semana_epidemio['ano_epi']}{reset}\n")
	# SC_Coroplético
	xy = municipios.copy()
	xy.drop(columns = ["CD_MUN", "SIGLA_UF", "AREA_KM2"], inplace = True)
	xy = xy.rename(columns = {"NM_MUN" : "Município"})
	xy["Município"] = xy["Município"].str.upper() 
	previsao_melt_poli = pd.merge(previsao_melt, xy, on = "Município", how = "left")
	previsao_melt_poligeo = gpd.GeoDataFrame(previsao_melt_poli, geometry = "geometry", crs = "EPSG:4674")
	#fig, ax = plt.subplots(figsize = (8, 6), layout = "constrained", frameon = True)
	fig, ax = plt.subplots(figsize = (8, 5.3), layout = "constrained", frameon = True)
	municipios.plot(ax = ax, color = "lightgray", edgecolor = "white", linewidth = 0.05)
	print(previsao_melt_poligeo)
	v_max = previsao_melt_poligeo.select_dtypes(include="number").max().max()
	v_min = previsao_melt_poligeo.select_dtypes(include="number").min().min()
	intervalo = 250
	levels = np.arange(v_min, v_max + intervalo, intervalo)
	print(f"\n{green}v_min\n{reset}{v_min}\n")
	print(f"\n{green}v_max\n{reset}{v_max}\n")
	print(f"\n{green}levels\n{reset}{levels}\n")
	previsao_melt_poligeo[previsao_melt_poligeo["Semana"] == semana_epidemio["Semana"]].plot(ax = ax, column = "Incidencia",  legend = True, edgecolor = "white", # fontsize = 20,
		                                                                           label = "Incidência (Casos Prováveis de Dengue)", cmap = "YlOrRd", linewidth = 0.05,   legend_kwds = {"extend": "max", "fraction": 0.035, "pad": 0.03, "label": "Número de Casos Prováveis/100 Mil Habitantes"},#levels = levels, 
		                                                                           norm = cls.Normalize(vmin = v_min, vmax = v_max, clip = True))
	cbar_ax = ax.get_figure().get_axes()[-1]
	#cbar_ax.tick_params(labelsize = 20)
	zero = previsao_melt_poligeo[previsao_melt_poligeo["Incidencia"] <= 0]
	zero[zero["Semana"] == semana_epidemio["Semana"]].plot(ax = ax, column = "Incidencia", legend = False, edgecolor = "white", linewidth = 0.05, label = "Incidência (Casos Prováveis de Dengue)", cmap = "YlOrBr")#"YlOrBr")
	regionais.plot(ax = ax, facecolor = "none",
				edgecolor = "dimgray", linewidth = 0.6)	
	#plt.xlim(-54.15, -47.95)#(-54, -48)
	#plt.ylim(-29.45, -25.64)#(-29.5, -25.75)
	plt.xlim(-54.05, -47.95)
	plt.ylim(-29.45, -25.75)
	ax.text(-52.5, -29, "Sistema de Referência de Coordenadas\nDATUM: SIRGAS 2000/22S.\nBase Cartográfica: IBGE, 2022.",
		    color = "white", backgroundcolor = "darkgray", ha = "center", va = "center", fontsize = 10)
	ax.text(-52.5, -28.25, """LEGENDA

▢           Sem registro*

*Não há registro oficial ou
modelagem inexistente.""",
		    color = "black", backgroundcolor = "lightgray", ha = "center", va = "center", fontsize = 10)
	plt.xlabel("Longitude")#, fontsize = 18)
	plt.ylabel("Latitude")#, fontsize = 18)
	ax.tick_params(axis = "both")#, labelsize = 18)
	ax.set_xticks([-54, -52, -50, -48])
	ax.set_xticklabels(["54°W", "52°W", "50°W", "48°W"])#, fontsize = 18)
	ax.set_yticks([-29, -28, -27, -26])
	ax.set_yticklabels(["29°S", "28°S", "27°S", "26°S"])#, fontsize = 18)
	plt.title(f"Incidência de Dengue Prevista em Santa Catarina.\nSemana Epidemiológica: {semana_epidemio['SE']}/{semana_epidemio['ano_epi']}.")#, fontsize = 28)
	#plt.grid(True)
	nome_arquivo = f"INCIDENCIA_mapa_preditivo_{data_atual}_{idx}.pdf"
	nome_arquivo_png = f"INCIDENCIA_mapa_preditivo_vSE{SE}_SE{semana_epidemio['SE']}-{semana_epidemio['ano_epi']}.png"
	if _AUTOMATIZA == True and _SALVAR == True:
		os.makedirs(caminho_resultados, exist_ok = True)
		#plt.savefig(f"{caminho_resultados}{nome_arquivo}", format = "pdf", dpi = 150)
		#plt.savefig(f"{caminho_resultados}{nome_arquivo_png}", format = "png", dpi = 300,
		#			transparent = False, bbox_inches = "tight", pad_inches = 0.02)
		plt.savefig(f"{caminho_resultados}{nome_arquivo_png}", format = "png", dpi = 300,
					transparent = False, pad_inches = 0.02)
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
	ultimas_previsoes_csv = f"ultimas_previsoes_incidencia_vSE{SE}_h{_HORIZONTE}_r{_RETROAGIR}.csv"
	ultimas_previsoes_vdd.to_csv(f"{caminho_resultados}{ultimas_previsoes_csv}", index = False)
	print(f"\n\n{green}{caminho_resultados}\n{ultimas_previsoes_csv}\nSALVO COM SUCESSO!{reset}\n\n")
	print(f"\n\n{green}OS VALORES DAS ÚLTIMAS PREVISÕES DE INCIDÊNCIA SÃO APRESENTADOS ABAIXO:\n{reset}{ultimas_previsoes_vdd}\n\n")
	
