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
import webbrowser
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category = ShapelyDeprecationWarning)
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

SEED = np.random.seed(0)

_CIDADE = input(f"{red}\nATENTAR PARA ACENTUAÇÃO (ex.: florianópolis)\n\n{cyan}Carol, querida, {green}escreva o nome do município que desejas visualizar o gráfico de previsão: {reset}")#"Joinville"#"Florianópolis"
_CIDADE = _CIDADE.upper()
"""
troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A',
         'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E',
         'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I',
         'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O',
         'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U',
         'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
for velho, novo in troca.items():
    _CIDADE = _CIDADE.replace(velho, novo)
"""
print(f"\n{green}CIDADE ESCOLHIDA:\n{reset}{_CIDADE}\n")

##################################################################################

### Encaminhamento aos Diretórios
caminho_dados = "/home/meteoro/scripts/matheus/operacional_dengue/dados_operacao/"
caminho_previsao = "/home/meteoro/scripts/matheus/operacional_dengue/modelagem/resultados/dados_previstos/"
caminho_modelos = f"/home/meteoro/scripts/matheus/operacional_dengue/modelagem/casos/2025/20250216/"

print(f"\nOS DADOS UTILIZADOS ESTÃO ALOCADOS NOS SEGUINTES CAMINHOS:\n\n{caminho_dados}\n\n{caminho_previsao}\n\n{caminho_modelos}\n\n")

######################################################
### Renomeação das Variáveis pelos Arquivos
obs = "casos_dive_pivot_total.csv" 
prev = "previsao_pivot_total_v20250216_h0_r2.csv"
modelo = f"RF_casos_v20250216_h0_r2_{_CIDADE}.h5"

###############################################################

### Abrindo Arquivo
obs = pd.read_csv(f"{caminho_dados}{obs}", low_memory = False)
prev = pd.read_csv(f"{caminho_previsao}{prev}", low_memory = False)

print(f"\n{green}OBSERVADOS:\n{reset}{obs}\n")
print(f"\n{green}PREVISTOS:\n{reset}{prev}\n")

### Pré-processamento
dataset = pd.DataFrame()
dataset["Semana"] = obs["Semana"]
dataset["obs"] = obs[_CIDADE].astype(int)
pred = pd.DataFrame()
pred["Semana"] = prev["Semana"]
pred["prev"] = prev[_CIDADE].astype(int)
print(f"\n{green}OBS x PREV ({_CIDADE}):\n{reset}{dataset}\n")
dataset = dataset.merge(pred, on = "Semana", how = "inner")
dataset["Semana"] = pd.to_datetime(dataset["Semana"])
gfs = dataset.iloc[-3:,:]
dataset2 = dataset.iloc[:-2,:]
print(f"\n{green}PREVISTOS ({_CIDADE}):\n{reset}{pred}\n")
print(f"\n{green}OBS x PREV ({_CIDADE}):\n{reset}{dataset}\n")
print(f"\n{green}OBS x PREV-GFS ({_CIDADE}):\n{reset}{gfs}\n")

### Visualização Gráfica

plt.figure(figsize = (10, 6), layout = "constrained", frameon = False)
ax = plt.gca()
ax.set_facecolor("honeydew")
sns.lineplot(x = dataset2["Semana"], y = dataset2["prev"],
             color = "red", alpha = 0.7, linewidth = 3, label = "Previsto")
sns.lineplot(x = gfs["Semana"], y = gfs["prev"], linestyle = ":",
             color = "red", alpha = 0.7, linewidth = 3, label = "Previsto (GFS)")
sns.lineplot(x = dataset["Semana"], y = dataset["obs"],
             color = "blue", alpha = 0.9, linewidth = 1, label = "Observado")
plt.title(f"MODELO RANDOM FOREST: PREVISÃO DE CASOS DE DENGUE.\n MUNICÍPIO DE {_CIDADE}, SANTA CATARINA.\n", fontsize = 18)
plt.xlabel("Semanas Epidemiológicas", fontsize = 18)
plt.ylabel("Número de Casos de Dengue", fontsize = 18)
plt.xticks(rotation = "horizontal")
plt.show()

