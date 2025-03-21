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
	caminho_dados = "/home/meteoro/scripts/matheus/operacional_dengue/dados_operacao/" # CLUSTER
	caminho_operacional = "/home/meteoro/scripts/matheus/operacional_dengue/"
	caminho_shape = "/media/dados/shapefiles/" #SC/SC_Municipios_2022.shp #BR/BR_UF_2022.shp
	caminho_modelos = f"/home/meteoro/scripts/matheus/operacional_dengue/modelagem/casos/{_ANO_MES_DIA}/"
	caminho_resultados = "home/meteoro/scripts/matheus/operacional_dengue/modelagem/resultados/"
	#caminho_previsao = "home/meteoro/scripts/matheus/operacional_dengue/modelagem/resultados/dados_previstos/"
	caminho_previsao = "modelagem/resultados/dados_previstos/"
else:
	print("CAMINHO NÃO RECONHECIDO! VERIFICAR LOCAL!")
print(f"\n{green}HOJE:\n{reset}{_ANO_MES_DIA}\n")
print(f"\n{green}ONTEM:\n{reset}{_ANO_MES_DIA_ONTEM}\n")

print(f"\n{green}OS DADOS UTILIZADOS ESTÃO ALOCADOS NOS SEGUINTES CAMINHOS:\n\n{caminho_dados}{reset}\n\n")


######################################################

### Renomeação das Variáveis pelos Arquivos
casos = "casos_dive_pivot_total.csv"  # TabNet/DiveSC
ultimas_previsoes = f"ultimas_previsoes_v{_ANO_MES_DIA}_h{_HORIZONTE}_r{_RETROAGIR}.csv"
previsao_pivot = f"previsao_pivot_total_v{_ANO_MES_DIA}_h{_HORIZONTE}_r{_RETROAGIR}.csv"
previsao_melt = f"previsao_melt_total_v{_ANO_MES_DIA}_h{_HORIZONTE}_r{_RETROAGIR}.csv"
"""
focos = "focos_pivot.csv"
prec = f"prec_semana_ate_{_ANO_ATUAL}.csv"
tmin = f"tmin_semana_ate_{_ANO_ATUAL}.csv"
tmed = f"tmed_semana_ate_{_ANO_ATUAL}.csv"
tmax = f"tmax_semana_ate_{_ANO_ATUAL}.csv"
"""
unicos = "casos_primeiros.csv"
municipios = "SC/SC_Municipios_2022.shp"
br = "BR/BR_UF_2022.shp"

###############################################################

### Abrindo Arquivo
casos = pd.read_csv(f"{caminho_dados}{casos}", low_memory = False)
ultimas_previsoes = pd.read_csv(f"{caminho_previsao}{ultimas_previsoes}", low_memory = False)
previsao_pivot = pd.read_csv(f"{caminho_previsao}{previsao_pivot}", low_memory = False)
previsao_melt = pd.read_csv(f"{caminho_previsao}{previsao_melt}", low_memory = False)
"""
focos = pd.read_csv(f"{caminho_dados}{focos}", low_memory = False)
prec = pd.read_csv(f"{caminho_dados}{prec}", low_memory = False)
tmin = pd.read_csv(f"{caminho_dados}{tmin}", low_memory = False)
tmed = pd.read_csv(f"{caminho_dados}{tmed}", low_memory = False)
tmax = pd.read_csv(f"{caminho_dados}{tmax}", low_memory = False)
"""
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

### Visualizando arquivos
print(f"\n{green}CASOS:\n{reset}{casos}\n")
print(f"\n{green}ÚLTIMAS PREVISÕES:\n{reset}{ultimas_previsoes}\n")
print(f"\n{green}PREVISÃO TOTAL (pivot):\n{reset}{previsao_pivot}\n")
print(f"\n{green}PREVISÃO TOTAL (melt):\n{reset}{previsao_melt}\n")

print(f"\n{green}CASOS.dtypes:\n{reset}{casos.dtypes}\n")
print(f"\n{green}ÚLTIMAS PREVISÕES.dtypes:\n{reset}{ultimas_previsoes.dtypes}\n")
previstos = ultimas_previsoes.iloc[:3, :]
previstos["Semana"] = pd.to_datetime(previstos["Semana"])
casos["Semana"] = pd.to_datetime(casos["Semana"])
casos25 = casos[casos["Semana"].dt.year == 2025]
print(f"\n{green}CASOS.dtypes:\n{reset}{casos}\n{casos.dtypes}\n")
print(f"\n{green}PREVISTOS.dtypes:\n{reset}{previstos}\n{previstos.dtypes}\n")

### Visualizações Gráficas

## Série (observado-previsão)

plt.figure(figsize = (15, 8), layout = "constrained", frameon = False)
plt.plot(previstos["Semana"], previstos[_CIDADE], color = "red", label = "Previsto", linewidth = 2)
#plt.show()
#plt.figure(figsize = (15, 8))
plt.plot(casos25["Semana"], casos25[_CIDADE], color = "blue", label = "Observado")
plt.xlabel("Semanas Epidemiológicas (Série Histórica)")
plt.ylabel("Número de Casos de Dengue")
plt.title(f"COMPARAÇÃO ENTRE CASOS DE DENGUE PREVISTOS E OBSERVADOS, MUNICÍPIO DE {_CIDADE}")
plt.legend()
plt.gca().set_facecolor("honeydew")
plt.show()
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
