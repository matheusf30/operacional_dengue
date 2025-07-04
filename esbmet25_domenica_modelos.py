### Bibliotecas Correlatas
# Básicas e Gráficas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, datetime, timedelta
#import datetime
# Suporte
import os
import sys
import joblib
# Pré-Processamento
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from imblearn.over_sampling import SMOTE
# pip install imbalanced-learn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

# Validações
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.metrics import confusion_matrix, classification_report #, RocCurveDisplay
from sklearn.inspection import permutation_importance
# Modelos e Visualizações
from sklearn.ensemble import RandomForestRegressor
#from imblearn.ensemble import BalancedRandomForestClassifier
#from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
#from sklearn.tree import export_graphviz, export_text, plot_tree
#from sklearn.utils.graph import single_source_shortest_path_lenght as short_path

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

################## Condições para Variar ####################################

#_RETROAGIR = 6 # Semanas Epidemiológicas
#_HORIZONTE = 4 # Tempo de Previsão

_JANELA_MM = 25 # Média Móvel
_K = 3 # constante para fórmulas de índices

_AJUSTADO = True #True #False # SMOTE (NÃO FUNCIONAL) #"BRFC"

z = 6
_LIMITE = "out2023"
_FIM = "nov2023"

z = 19
_LIMITE = "jul2023"
_FIM = "ago2023"

z = 32
_LIMITE = "abr2023"
_FIM = "mai2023"

z = 50
_LIMITE = "dez2022"
_FIM = "jan2023"
"""
"""
obs = f"(Treino até {_LIMITE}; Teste após {_FIM})"
obs_k = f"(Treino até {_LIMITE}; Teste após {_FIM}; k = {_K})"

######################################################

_ANO_ATUAL = "2025"
_MES_ATUAL = "02"
_DIA_ATUAL = "16"
_ANO_MES = f"{_ANO_ATUAL}{_MES_ATUAL}"
_ANO_MES_DIA = f"{_ANO_ATUAL}{_MES_ATUAL}{_DIA_ATUAL}"

print(f"\n{green}DATA:\n{reset}{_DIA_ATUAL}\n")
print(f"\n{green}DATA:\n{reset}{_ANO_MES_DIA}\n")

_CIDADE = "Joinville"#"Joinville"#"Florianópolis"
_CIDADE = _CIDADE.upper()

_RETROAGIR = 2 # Semanas Epidemiológicas
_HORIZONTE = 0 # Tempo de Previsão
#for r in range(_HORIZONTE + 1, _RETROAGIR + 1):

_AUTOMATIZAR = sys.argv[1]   # True|False                    #####
_AUTOMATIZA = True if _AUTOMATIZAR == "True" else False 
"""
##################### Valores Booleanos ############ # sys.argv[0] is the script name itself and can be ignored!
_AUTOMATIZAR = sys.argv[1]   # True|False                    #####
_AUTOMATIZA = True if _AUTOMATIZAR == "True" else False      #####
_VISUALIZAR = sys.argv[2]    # True|False                    #####
_VISUALIZAR = True if _VISUALIZAR == "True" else False       #####
_SALVAR = sys.argv[3]        # True|False                    #####
_SALVAR = True if _SALVAR == "True" else False               #####
##################################################################
"""
##################################################################################

### Encaminhamento aos Diretórios
#/home/meteoro/scripts/matheus/operacional_dengue/#CLUSTER
caminho_dados = "/home/meteoro/scripts/matheus/operacional_dengue/dados_operacao/"
#os.makedirs(caminho_dados, exist_ok = True)
caminho_modelos = "home/meteoro/scripts/matheus/operacional_dengue/modelagem/casos/"
#os.makedirs(caminho_modelos, exist_ok = True)
caminho_resultados = "home/meteoro/scripts/matheus/operacional_dengue/modelagem/resultados/"
#os.makedirs(caminho_resultados, exist_ok = True)

### Renomeação das Variáveis pelos Arquivos
casos = "casos_dive_pivot_total.csv"  # TabNet/DiveSC
focos = "focos_pivot.csv"

prec = f"{_ANO_ATUAL}/prec_semana_ate_{_ANO_ATUAL}.csv"
tmin = f"{_ANO_ATUAL}/tmin_semana_ate_{_ANO_ATUAL}.csv"
tmed = f"{_ANO_ATUAL}/tmed_semana_ate_{_ANO_ATUAL}.csv"
tmax = f"{_ANO_ATUAL}/tmax_semana_ate_{_ANO_ATUAL}.csv"
unicos = "casos_primeiros.csv"

### Abrindo Arquivo
casos = pd.read_csv(f"{caminho_dados}{casos}", low_memory = False)

#focos = pd.read_csv(f"{caminho_dados}{focos}", low_memory = False)
prec = pd.read_csv(f"{caminho_dados}{prec}", low_memory = False)
tmin = pd.read_csv(f"{caminho_dados}{tmin}", low_memory = False)
tmed = pd.read_csv(f"{caminho_dados}{tmed}", low_memory = False)
tmax = pd.read_csv(f"{caminho_dados}{tmax}", low_memory = False)
unicos = pd.read_csv(f"{caminho_dados}{unicos}", low_memory = False)


### Sanando Erros

_CIDADEs = unicos["Município"]
teste1 = _CIDADEs[_CIDADEs.isin(["GRÃO PARÁ", "SÃO CRISTOVÃO DO SUL"])]
teste2 = _CIDADEs[_CIDADEs.isin(["GRÃO-PARÁ", "SÃO CRISTÓVÃO DO SUL"])]
print(f"\n{green}_CIDADEs:\n{reset}{_CIDADEs}\n")
print(f"\n{red}teste1:\n{reset}{teste1}\n")
print(f"\n{red}teste2:\n{reset}{teste2}\n")
_CIDADEs = _CIDADEs.replace({"GRÃO PARÁ":"GRÃO-PARÁ", "SÃO CRISTOVÃO DO SUL":"SÃO CRISTÓVÃO DO SUL"})
teste3 = _CIDADEs[_CIDADEs.isin(["GRÃO-PARÁ", "SÃO CRISTÓVÃO DO SUL"])]
print(f"\n{red}teste3:\n{reset}{teste3}\n")
print(f"\n{green}_CIDADEs:\n{reset}{_CIDADEs}\n")

print(f"\n{green}prec:\n{reset}{prec}\n")
print(f"\n{green}tmin:\n{reset}{tmin}\n")
print(f"\n{green}tmed:\n{reset}{tmed}\n")
print(f"\n{green}tmax:\n{reset}{tmax}\n")
print(f"\n{green}casos:\n{reset}{casos}\n")
print(f"\n{green}unicos:\n{reset}{unicos}\n")

### Montando Dataset
dataset = tmin[["Semana"]].copy()
dataset["TMIN"] = tmin[_CIDADE].copy()
dataset["TMED"] = tmed[_CIDADE].copy()
dataset["TMAX"] = tmax[_CIDADE].copy()
print(f"\n{green}dataset:\n{reset}{dataset}\n")
dataset = dataset.merge(prec[["Semana", _CIDADE]], how = "right", on = "Semana").copy()
dataset.dropna(inplace = True)
print(f"\n{green}dataset:\n{reset}{dataset}\n")
#dataset = dataset.iloc[104:, :].copy()
dataset = dataset.merge(casos[["Semana", _CIDADE]], how = "right", on = "Semana").copy()
troca_nome = {f"{_CIDADE}_x" : "PREC", f"{_CIDADE}_y" : "CASOS"}
dataset = dataset.rename(columns = troca_nome)
dataset["CASOS"] = dataset["CASOS"].astype(int)
dataset.fillna(0, inplace = True)
print(f"\n{green}dataset:\n{reset}{dataset}\n")
#sys.exit()

#dataset["TMED"] = dataset["TMED"]#.rolling(_JANELA_MM).mean()
#dataset["PREC"] = dataset["PREC"]#.rolling(_JANELA_MM).mean()
#dataset["FOCOS"] = dataset["FOCOS"]#.rolling(_JANELA_MM).mean()
#dataset["CASOS"] = dataset["CASOS"]#.rolling(_JANELA_MM).mean()

#_RETROAGIR = 12
#_RETROAGIR = 4
#_RETROAGIR = 2 # Semanas Epidemiológicas
#_HORIZONTE = 0 # Tempo de Previsão
for r in range(_HORIZONTE + 1, _RETROAGIR + 1):
#for r in range(3, 5):
	dataset[f"TMIN_r{r}"] = dataset["TMIN"].shift(-r)
	dataset[f"TMED_r{r}"] = dataset["TMED"].shift(-r)
	dataset[f"TMAX_r{r}"] = dataset["TMAX"].shift(-r)
	dataset[f"PREC_r{r}"] = dataset["PREC"].shift(-r)
	#dataset[f"FOCOS_r{r}"] = dataset["FOCOS"].shift(-r)
#for r in range(2, 5):
	dataset[f"CASOS_r{r}"] = dataset["CASOS"].shift(-r)
"""
#_RETROAGIR = 2
#dataset[f"TMED_r{_RETROAGIR}"] = dataset["TMED"].shift(-_RETROAGIR)
#dataset[f"PREC_r{_RETROAGIR}"] = dataset["PREC"].shift(-_RETROAGIR)
#dataset[f"FOCOS_r{_RETROAGIR}"] = dataset["FOCOS"].shift(-_RETROAGIR)

for r in range(_HORIZONTE + 1, _RETROAGIR + 1):
    dataset[f"TMIN_r{r}"] = dataset["TMIN"].shift(-r)
    dataset[f"TMED_r{r}"] = dataset["TMED"].shift(-r)
    dataset[f"TMAX_r{r}"] = dataset["TMAX"].shift(-r)
    dataset[f"PREC_r{r}"] = dataset["PREC"].shift(-r)
    dataset[f"FOCOS_r{r}"] = dataset["FOCOS"].shift(-r)
    dataset[f"CASOS_r{r}"] = dataset["CASOS"].shift(-r)
"""

dataset.drop(columns = ["TMIN", "TMED", "TMAX", "PREC"], inplace = True)
dataset.dropna(inplace = True)
dataset.set_index("Semana", inplace = True)
dataset.columns.name = f"{_CIDADE}"
dataset = dataset.iloc[:-1,:]
print(f"\n{green}dataset (após retroação):\n{reset}{dataset}\n")
#sys.exit()
### Dividindo Dataset em Treino e Teste
SEED = np.random.seed(0)
x = dataset.drop(columns = "CASOS")
y = dataset["CASOS"]
x_array = x.to_numpy().astype(int)
y_array = y.to_numpy().astype(int)
x_array = x_array.reshape(x_array.shape[0], -1)


treino_x, teste_x, treino_y, teste_y = train_test_split(x_array, y_array,
                                                        random_state = SEED,
                                                        test_size = 0.2)
"""
x_ate__LIMITE = x.iloc[:-z]
y_ate__LIMITE = y.iloc[:-z]
x_LIMITE = x.iloc[-z:]
y_LIMITE = y.iloc[-z:]
treino_x = x_ate__LIMITE.copy()
teste_x = x_LIMITE.copy()
treino_y = y_ate__LIMITE.copy()
teste_y = y_LIMITE.copy()
"""
explicativas = x.columns.tolist() # feature_names = explicativas
treino_x_explicado = pd.DataFrame(treino_x, columns = explicativas)
treino_x_explicado = treino_x_explicado.to_numpy().astype(int)

#print(f"""Conjunto de Treino com as Variáveis Explicativas (<{_LIMITE}):\n{treino_x}\n
#Conjunto de Treino com as Variáveis Explicativas (>{_FIM}):\n{teste_x}\n 
#Conjunto de Teste com a Variável Dependente (<{_LIMITE}):\n{treino_y}\n 
#Conjunto de Teste com a Variável Dependente (>{_FIM}):\n{teste_y}\n
#Conjunto de Treino com as Variáveis Explicativas (Explicitamente Indicadas)(<{_LIMITE}):\n{treino_x_explicado}\n""")
#sys.exit()
"""
### Normalizando/Escalonando Dataset_x (Se Necessário)
escalonador = StandardScaler()
escalonador.fit(treino_x)
treino_normal_x = escalonador.transform(treino_x)
teste_normal_x = escalonador.transform(teste_x)

### Ajuste Ponderado
# SMOTE
contagem_classes = treino_y.value_counts()
classe_minoritaria = contagem_classes.min()
k_neighbors = 2
smote = SMOTE()
smote_treino_x_explicado, smote_treino_y = smote.fit_resample(treino_x_explicado, treino_y)
# Calcular os pesos das amostras
sample_weights = compute_sample_weight(class_weight='balanced', y = y_train_balanced) 
"""
# Ajuste Sobre/Sub
sobreajuste = RandomOverSampler(sampling_strategy = "not majority", random_state = SEED)
subajuste = RandomUnderSampler(sampling_strategy = "not minority", random_state = SEED)
# Primeiro oversampling para aumentar a classe minoritária
ajuste1_treino_x, ajusto1_treino_y = sobreajuste.fit_resample(treino_x_explicado, treino_y)
# Depois undersampling para reduzir a classe majoritária
ajustado_treino_x_explicado, ajustado_treino_y = subajuste.fit_resample(ajuste1_treino_x, ajusto1_treino_y)
print("\nTreino X ajustado\n", ajustado_treino_x_explicado, "\nTreino Y ajustado\n", ajustado_treino_y)

# Instanciar e treinar o modelo Random Forest:
# model = RandomForestClassifier(n_estimators=100, random_state=42)  
# model = RandomForestRegressor(n_estimators=100)#,random_state=42)
# model.fit(X_train_balanced, y_train_balanced)
# Treinar o modelo com os pesos das amostras


### Exibindo Informações
print("\n \n CONJUNTO DE DADOS PARA TREINO E TESTE \n")
print(dataset.info())
print("~"*80)
#print(dataset.dtypes)
#print("~"*80)
print(dataset)
#print("="*80)
#print(f"X no formato numpy.ndarray: {x_array}.")
"""
print("="*80)
print(f"Treinando com {len(treino_x)} elementos e testando com {len(teste_x)} elementos.") # Tamanho é igual para dados normalizados
print(f"Formato dos dados (X) nas divisões treino: {treino_x.shape} e teste: {teste_x.shape}.")
print(f"Formato dos dados (Y) nas divisões treino: {treino_y.shape} e teste: {teste_y.shape}.")
print("="*80)
ax = plt.gca()
ax.set_facecolor("honeydew")
sns.lineplot(x = x.index, y = previsoes,
             color = "red", alpha = 0.7, linewidth = 3, label = "Previsto")
sns.lineplot(x = x.index, y = x["CASOS"],
             color = "blue", alpha = 0.9, linewidth = 1, label = "Observado")
plt.title(f"MODELO RANDOM FOREST: PREVISÃO DE CASOS DE DENGUE.\n MUNICÍPIO DE {_CIDADE}, SANTA CATARINA.\n")
plt.xlabel("Semanas Epidemiológicas")
plt.ylabel("Número de Casos de Dengue")
plt.xticks(rotation = "horizontal")
print(f"\n{green}list(x3.index):\n{reset}{list(x3.index)}")
print(f"\n{green}list(x3['CASOS']):\n{reset}{list(x3['CASOS'])}")
#print(f"\n{green}semanas_futuras:\n{reset}{semanas_futuras}")
print(f"\n{green}previsoes3:\n{reset}{previsoes3}")
print(f"\n{green}type(previsoes3):\n{reset}{type(previsoes3)}")
plt.show()
"""
#sys.exit()
#########################################################FUNÇÕES###############################################################
### Definições
def monta_dataset(_CIDADE):
	dataset = tmin[["Semana"]].copy()
	dataset["TMIN"] = tmin[_CIDADE].copy()
	dataset["TMED"] = tmed[_CIDADE].copy()
	dataset["TMAX"] = tmax[_CIDADE].copy()
	dataset = dataset.merge(prec[["Semana", _CIDADE]], how = "right", on = "Semana").copy()
	dataset = dataset.rename(columns = {_CIDADE : "PREC"})
	dataset = dataset.merge(casos[["Semana", _CIDADE]], how = "right", on = "Semana").copy()
	dataset = dataset.rename(columns = {_CIDADE : "CASOS"})
	#troca_nome = {f"{_CIDADE}_x" : "PREC", f"{_CIDADE}_y" : "CASOS", f"{_CIDADE}" : "CASOS"}
	#dataset = dataset.rename(columns = troca_nome)
	dataset["CASOS"] = dataset["CASOS"].astype(int)
	dataset.dropna(axis = 0, inplace = True)
	dataset.fillna(0, inplace = True)
	for r in range(_HORIZONTE + 1, _RETROAGIR + 1):
		dataset[f"TMIN_r{r}"] = dataset["TMIN"].shift(-r)
		dataset[f"TMED_r{r}"] = dataset["TMED"].shift(-r)
		dataset[f"TMAX_r{r}"] = dataset["TMAX"].shift(-r)
		dataset[f"PREC_r{r}"] = dataset["PREC"].shift(-r)
		dataset[f"CASOS_r{r}"] = dataset["CASOS"].shift(-r)
	dataset.drop(columns = ["TMIN", "TMED", "TMAX", "PREC"], inplace = True)
	dataset.dropna(inplace = True)
	dataset.set_index("Semana", inplace = True)
	dataset.columns.name = f"{_CIDADE}"
	print(f"\n{green}dataset:\n{reset}{dataset}\n")
	dataset = dataset.iloc[:-1,:]
	print(f"\n{green}dataset:\n{reset}{dataset}\n")
	#sys.exit()
	return dataset

def testa_dataset(_CIDADE):
	dataset = tmin[["Semana"]].copy()
	dataset["TMIN"] = tmin[_CIDADE].copy()
	dataset["TMED"] = tmed[_CIDADE].copy()
	dataset["TMAX"] = tmax[_CIDADE].copy()
	dataset = dataset.merge(prec[["Semana", _CIDADE]], how = "right", on = "Semana").copy()
	#dataset = dataset.merge(focos[["Semana", _CIDADE]], how = "left", on = "Semana").copy()
	#dataset = dataset.iloc[104:, :].copy()
	dataset = dataset.merge(casos[["Semana", _CIDADE]], how = "right", on = "Semana").copy()
	troca_nome = {f"{_CIDADE}_x" : "PREC", f"{_CIDADE}_y" : "FOCOS", f"{_CIDADE}" : "CASOS"}
	dataset = dataset.rename(columns = troca_nome)
	dataset.dropna(axis = 0, inplace = True)
	dataset.fillna(0, inplace = True)
	for r in range(_HORIZONTE + 1, _RETROAGIR + 1):
		dataset[f"TMIN_r{r}"] = dataset["TMIN"].shift(-r)
		dataset[f"TMED_r{r}"] = dataset["TMED"].shift(-r)
		dataset[f"TMAX_r{r}"] = dataset["TMAX"].shift(-r)
		dataset[f"PREC_r{r}"] = dataset["PREC"].shift(-r)
		dataset[f"CASOS_r{r}"] = dataset["CASOS"].shift(-r)
	dataset.drop(columns = ["TMIN", "TMED", "TMAX", "PREC"], inplace = True)
	dataset.dropna(inplace = True)
	dataset.set_index("Semana", inplace = True)
	dataset.columns.name = f"{_CIDADE}"
	print(dataset)
	return dataset

def treino_teste(dataset, _CIDADE):
    SEED = np.random.seed(0)
    x = dataset.drop(columns = "CASOS").copy()
    y = dataset["CASOS"]
    if x.empty or x.isnull().all().all():
        print(f"'X' está vazio ou contém apenas valores 'NaN! Confira o dataset do município {_CIDADE}!")
        print(f"{_CIDADE} possui um conjunto com erro:\n {x}")
        return None, None, None, None, None
    x = x.dropna()
    if x.empty:
        print(f"'X' continua vazio, mesmo removendo valores 'NaN'! Confira o dataset do município {_CIDADE}!")
        print(f"{_CIDADE} possui um conjunto com erro:\n {x}")
        return None, None, None, None, None
    if y.empty or y.isnull().all().all():
        print(f"'Y' está vazio ou contém apenas valores 'NaN! Confira o dataset do município {_CIDADE}!")
        print(f"{_CIDADE} possui um conjunto com erro:\n {y}")
        return None, None, None, None, None
    y = y.dropna()
    if y.empty:
        print(f"'Y' continua vazio, mesmo removendo valores 'NaN'! Confira o dataset do município {_CIDADE}!")
        print(f"{_CIDADE} possui um conjunto com erro:\n {y}")
        return None, None, None, None, None
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
    return x, y, treino_x, teste_x, treino_y, teste_y, treino_x_explicado, explicativas

def escalona(treino_x, teste_x):
    escalonador = StandardScaler()
    escalonador.fit(treino_x)
    treino_normal_x = escalonador.transform(treino_x)
    teste_normal_x = escalonador.transform(teste_x)
    return treino_normal_x, teste_normal_x

def RF_modela_treina_preve(treino_x_explicado, treino_y, teste_x, SEED):
    modelo = RandomForestRegressor(n_estimators = 100, random_state = SEED)
    modelo.fit(treino_x_explicado, treino_y)
    y_previsto = modelo.predict(teste_x)
    previsoes = modeloRF.predict(x)
    previsoes = [int(p) for p in previsoes]
    return modelo, y_previsto, previsoes

def RF_previsao_metricas(dataset, previsoes, n, teste_y, y_previsto):
    nome_modelo = "Random Forest"
    print("="*80)
    print(f"\n{nome_modelo.upper()} - {_CIDADE}\n")
    lista_op = [f"Casos: {dataset['CASOS'][i]}\nPrevisão {nome_modelo}: {previsoes[i]}\n" for i in range(n)]
    print("\n".join(lista_op))
    print("~"*80)
    EQM = mean_squared_error(teste_y, y_previsto)
    RQ_EQM = np.sqrt(EQM)
    R_2 = round(r2_score(teste_y, y_previsto), 2)
    print(f"""
         \n MÉTRICAS {nome_modelo.upper()} - {_CIDADE}
         \n Erro Quadrático Médio: {EQM}
         \n Coeficiente de Determinação (R²): {R_2}
         \n Raiz Quadrada do Erro Quadrático Médio: {RQ_EQM}
         """)
    print("="*80)
    return EQM, RQ_EQM, R_2

def salva_modeloRF(modelo, _CIDADE):
	_cidade = _CIDADE
	troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A', 'Ä': 'A',
			'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E', 'Ë': 'E',
			'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I', 'Ï': 'I',
			'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O', 'Ö': 'O',
			'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U', 'Ü': 'U',
			'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
	for velho, novo in troca.items():
		_cidade = _cidade.replace(velho, novo)
	_ANO_MES_DIA = f"{_ANO_ATUAL}{_MES_ATUAL}{_DIA_ATUAL}"
	caminho_modelos = f"modelagem/casos/{_ANO_ATUAL}/{_ANO_MES_DIA}/"
	if not os.path.exists(caminho_modelos):
		os.makedirs(caminho_modelos)
	nome_modelo = f"RF_casos_v{_ANO_MES_DIA}_h{_HORIZONTE}_r{_RETROAGIR}_{_cidade}.h5"
	joblib.dump(modelo, f"{caminho_modelos}{nome_modelo}")
	print(f"\n{green}MODELO RANDOM FOREST DE {bold}{_cidade} SALVO!\n{reset}")
	print(f"\n{cyan}Caminho e Nome:\n {bold}{caminho_modelos}{nome_modelo}\n{reset}")
	print("\n" + f"{red}={reset}"*80 + "\n")

def lista_previsao(previsao, n, string_modelo):
    if string_modelo not in ["RF", "NN"]:
        print("!!"*80)
        print("\n   MODELO NÃO RECONHECIDO\n   TENTE 'RF' PARA RANDOM FOREST\n   OU 'NN' PARA REDE NEURAL\n")
        print("!!"*80)
        sys.exit()
    nome_modelo = "Random Forest" if string_modelo == "RF" else "Rede Neural"
    previsoes = previsao if string_modelo == "RF" else [np.argmax(p) for p in previsao]
    print("="*80)
    print(f"\n{nome_modelo.upper()} - {_CIDADE}\n")
    lista_op = [f"CASOS: {dataset['CASOS'][i]}\nPrevisão {nome_modelo}: {previsoes[i]}\n" for i in range(n)]
    print("\n".join(lista_op))
    print("="*80)

def grafico_previsao(teste, previsao, string_modelo, _CIDADE):
    if string_modelo not in ["RF", "NN"]:
        print("!!"*80)
        print("\n   MODELO NÃO RECONHECIDO\n   TENTE 'RF' PARA RANDOM FOREST\n   OU 'NN' PARA REDE NEURAL\n")
        print("!!"*80)
        sys.exit()
    # Gráfico de Comparação entre Observação e Previsão dos Modelos
    nome_modelo = "Random Forest" if string_modelo == "RF" else "Rede Neural"
    final = pd.DataFrame()
    final["Semana"] = casos["Semana"]
    final["Casos"] = casos[_CIDADE]
    #excluir_linhas = list(range(0,_JANELA_MM))
    #final.drop(excluir_linhas, axis=0, inplace = True)
    #final.drop([0,1, 2, 3], axis=0, inplace = True)
    final.drop(final.index[-_RETROAGIR:], axis=0, inplace = True)
    #final.drop([d for d in range(_RETROAGIR + _HORIZONTE)], axis=0, inplace = True)
    #final.drop(final.index[-(_RETROAGIR - _HORIZONTE):], axis=0, inplace = True)
    #final.drop([d for d in range(_RETROAGIR + _HORIZONTE + _JANELA_MM)], axis=0, inplace = True)
    previsoes = previsao if string_modelo == "RF" else [np.argmax(p) for p in previsao]
    """
    lista_previsao = [previsoes[v] for v in range(len(previsoes))]
    final["Previstos"] = lista_previsao
    """
    #previsoes = previsoes[:len(final)]
    print(f"\n{green}Tamanho de 'previsoes':\n{reset}{len(previsoes)}\n")
    print(f"\n{green}previsoes:\n{reset}{previsoes}\n")
    print(f"\n{green}Tamanho do 'final.index':\n{reset}{len(final.index)}\n")
    print(f"\n{green}final:\n{reset}{final}\n")
    
    final["Previstos"] = previsoes
    final["Semana"] = pd.to_datetime(final["Semana"])
    print(final)
    print("="*80)
    plt.figure(figsize = (10, 6), layout = "constrained", frameon = False)
    ax = plt.gca()
    ax.set_facecolor("honeydew")
    sns.lineplot(x = final["Semana"], y = final["Casos"], # linestyle = "--" linestyle = "-."
                 color = "darkblue", linewidth = 1, label = "Observado")
    sns.lineplot(x = final["Semana"], y = final["Previstos"],
                 color = "red", alpha = 0.7, linewidth = 3, label = "Previsto")
    plt.title(f"MODELO {nome_modelo.upper()} (R²: {R_2}): OBSERVAÇÃO E PREVISÃO.\n MUNICÍPIO DE {_CIDADE}, SANTA CATARINA.")#\n{obs}")
    plt.xlabel("Semanas Epidemiológicas na Série de Anos")
    plt.ylabel("Número de Casos de Dengue")
    troca = {'Á': 'A', 'Â': 'A', 'À': 'A', 'Ã': 'A', 'Ä': 'A',
           'É': 'E', 'Ê': 'E', 'È': 'E', 'Ẽ': 'E', 'Ë': 'E',
         'Í': 'I', 'Î': 'I', 'Ì': 'I', 'Ĩ': 'I', 'Ï': 'I',
         'Ó': 'O', 'Ô': 'O', 'Ò': 'O', 'Õ': 'O', 'Ö': 'O',
         'Ú': 'U', 'Û': 'U', 'Ù': 'U', 'Ũ': 'U', 'Ü': 'U', 
         'Ç': 'C', " " : "_", "'" : "_", "-" : "_"}
    _CIDADE = _CIDADE
    for velho, novo in troca.items():
        _CIDADE = _CIDADE.replace(velho, novo)
    #plt.savefig(f'{caminho_resultados}verificatualizacao_modelo_RF_casos_{_CIDADE}_{_LIMITE}-{_FIM}.pdf', format = "pdf", dpi = 1200)
    plt.show()

def histograma_erro(teste, previsao):
    final = pd.DataFrame()
    final["Semana"] = casos["Semana"]
    final["Casos"] = casos[_CIDADE]
    final.drop([d for d in range(_RETROAGIR)], axis=0, inplace = True)
    final.drop(final.index[-_RETROAGIR:], axis=0, inplace = True)
    previsoes = previsao.copy()
    previsoes = previsoes[:len(final)]
    final["Previstos"] = previsoes
    final["Semana"] = pd.to_datetime(final["Semana"])
    final["Erro"] = final["Casos"] - final["Previstos"]
    print(final)
    print("="*80)
    media = round(final["Erro"].mean(), 2)
    desvp = round(final["Erro"].std(), 2)
    plt.figure(figsize = (10, 6), layout = "constrained", frameon = False)
    ax = plt.gca()
    ax.set_facecolor("honeydew")
    sns.histplot(final["Casos"], bins = 50, kde = True, color = "blue")
    sns.histplot(final["Previstos"], bins = 20, kde = True, color = "red")
    sns.histplot(final["Erro"], bins = 50, kde = True, color = "black", label = "Erro")
    plt.legend(title = "Distribuição", loc = "upper right", labels = ["Observado", "Previsto", "Erro"])
    plt.title(f"MODELO RANDOM FOREST* (R²: {R_2}): HISTOGRAMA DO ERRO**.\n MUNICÍPIO DE {_CIDADE}, SANTA CATARINA.\n *{obs} **($\\mu = {media}; \\sigma = {desvp}$)")
    plt.xlabel("Valor")
    plt.ylabel("Quantidade")
    plt.show()

def boxplot_erro(teste, previsao):
    final = pd.DataFrame()
    final["Semana"] = casos["Semana"]
    final["Casos"] = casos[_CIDADE]
    final.drop([d for d in range(_RETROAGIR)], axis=0, inplace = True)
    final.drop(final.index[-_RETROAGIR:], axis=0, inplace = True)
    previsoes = previsao.copy()
    previsoes = previsoes[:len(final)]
    final["Previstos"] = previsoes
    final["Semana"] = pd.to_datetime(final["Semana"])
    final["Erro"] = final["Casos"] - final["Previstos"]
    print(final)
    print("="*80)
    media = round(final["Erro"].mean(), 2)
    desvp = round(final["Erro"].std(), 2)
    plt.figure(figsize = (10, 6), layout = "constrained", frameon = False)
    ax = plt.gca()
    ax.set_facecolor("honeydew")
    posicao = [1, 2, 3]
    plt.boxplot([final["Casos"], final["Previstos"], final["Erro"]], positions = posicao)
    #plt.xticks(posicao, ["Observado", "P
    plt.title(f"MODELO RANDOM FOREST* (R²: {R_2}): BOXPLOT DO ERRO**.\n MUNICÍPIO DE {_CIDADE}, SANTA CATARINA.\n *{obs} **($\\mu = {media}; \\sigma = {desvp}$)")
    plt.xlabel("Boxplot")
    plt.ylabel("Valor")
    plt.grid(axis = "y")
    plt.show()

def matriz_confusao(teste, previsao):
	matriz_confusao = confusion_matrix(teste, previsao)
	print(matriz_confusao)
	plt.figure(figsize = (10, 6), layout = "constrained", frameon = False)
	sns.heatmap(matriz_confusao, annot = True)
	return matriz_confusao

def relatorio_metricas(teste, previsao):
	relatorio = classification_report(teste, previsao)
	print(relatorio)
	return relatorio

def metricas_importancias(modeloRF, explicativas):
	importancias = modeloRF.feature_importances_
	importancias = importancias.round(4)
	indices = np.argsort(importancias)[::-1]
	variaveis_importantes = pd.DataFrame({"Variáveis": explicativas, "Importâncias": importancias})
	variaveis_importantes = variaveis_importantes.sort_values(by = "Importâncias", ascending = False)
	importancia_impureza = pd.Series(importancias, index = explicativas)
	print(variaveis_importantes)
	#1 Impurezas
	std = np.std([tree.feature_importances_ for tree in modeloRF.estimators_], axis=0)
	fig, ax = plt.subplots(figsize = (10, 6), layout = "constrained", frameon = False)
	importancia_impureza = importancia_impureza.sort_values(ascending = False)
	importancia_impureza.plot.bar(yerr = std, ax = ax)
	ax.set_title(f"VARIÁVEIS IMPORTANTES PARA MODELO RANDOM FOREST\nMUNICÍPIO DE {_CIDADE}, SANTA CATARINA.\n{obs}")
	ax.set_ylabel("Impureza Média")
	ax.set_xlabel("Variáveis Explicativas para Modelagem de Casos de Dengue")
	ax.set_facecolor("honeydew")
	plt.xticks(rotation = 70)#"horizontal")
	for i, v in enumerate(importancia_impureza.values):
		ax.text(i, v + 0.01, f"{v.round(4)}", color = "black", ha = "left")
	fig.tight_layout()
	plt.show()
	#2 Permutações
	n_permuta = 10
	resultado_permuta = permutation_importance(modeloRF, teste_x, teste_y, n_repeats = n_permuta, random_state = SEED, n_jobs = 2)
	importancia_permuta = pd.Series(resultado_permuta.importances_mean, index = explicativas)
	importancia_permuta = importancia_permuta.sort_values(ascending = False)
	fig, ax = plt.subplots(figsize = (10, 6), layout = "constrained", frameon = False)
	importancia_permuta.plot.bar(yerr = resultado_permuta.importances_std, ax = ax)
	ax.set_title(f"VARIÁVEIS IMPORTANTES UTILIZANDO PERMUTAÇÃO ({n_permuta})\nMUNICÍPIO DE {_CIDADE}, SANTA CATARINA.\n{obs}")
	ax.set_ylabel("Acurácia Média")
	ax.set_xlabel("Variáveis Explicativas para Modelagem de Casos de Dengue")
	ax.set_facecolor("honeydew")
	plt.xticks(rotation = 70)#"horizontal")
	for i, v in enumerate(importancia_permuta.values):
		ax.text(i, v + 0.01, f"{v.round(4)}", color = "black", ha = "left")
	fig.tight_layout()
	plt.show()
	print(f"\nVARIÁVEIS IMPORTANTES:\n{importancia_impureza}\n")
	print(f"\nVARIÁVEIS IMPORTANTES UTILIZANDO PERMUTAÇÃO:\n{importancia_permuta}")
	return importancias, indices, variaveis_importantes

def metricas(string_modelo, modeloNN = None):
    if string_modelo not in ["RF", "NN"]:
        print("!!"*80)
        print("\n   MODELO NÃO RECONHECIDO\n   TENTE 'RF' PARA RANDOM FOREST\n   OU 'NN' PARA REDE NEURAL\n")
        print("!!"*80)
        sys.exit()
    elif string_modelo == "NN":
        if modeloNN is None:
            print("!!"*80)
            raise ValueError("'modeloNN' não foi fornecido para a função metricas() do modelo de rede neural!")
        else:
            sumario = []
            modeloNN.summary(print_fn = lambda x: sumario.append(x))
            sumario = "\n".join(sumario)
            print(f"\n MÉTRICAS REDE NEURAL\n \n {sumario}")
    else:
        print(f"""
             \n MÉTRICAS RANDOM FOREST - {_CIDADE}
             \n Erro Quadrático Médio: {EQM_RF}
             \n Coeficiente de Determinação (R²): {R_2}
             \n Raiz Quadrada do Erro Quadrático Médio: {RQ_EQM_RF}
              """)

def salva_modelo(string_modelo, modeloNN = None):
    if string_modelo not in ["RF", "NN"]:
        print("!!"*80)
        print("\n   MODELO NÃO RECONHECIDO\n   TENTE 'RF' PARA RANDOM FOREST\n   OU 'NN' PARA REDE NEURAL\n")
        print("!!"*80)
        sys.exit()
    elif string_modelo == "NN":
        if modeloNN is None:
            print("!!"*80)
            raise ValueError("'modeloNN' não foi fornecido para a função metricas() do modelo de rede neural!")
        else:
            modeloNN.save(modeloNN, f"{caminho_modelos}NN_casos_r{_RETROAGIR}_{_CIDADE}.h5")
    else:
        joblib.dump(modeloRF, f"{caminho_modelos}RF_casos_r{_RETROAGIR}_v2_{_CIDADE}.h5")
        print(f"\n\n{caminho_modelos}RF_casos_r{_RETROAGIR}_v2_{_CIDADE}.h5\n\n")

######################################################RANDOM_FOREST############################################################
### Iniciando Dataset
#dataset = monta_dataset(_CIDADE)
#dataset = testa_dataset(_CIDADE)
#x, y, treino_x, teste_x, treino_y, teste_y, treino_x_explicado, explicativas = treino_teste(dataset, _CIDADE)

### Instanciando e Treinando Modelo Regressor Random Forest
modeloRF = RandomForestRegressor(n_estimators = 100, random_state = SEED) #n_estimators = número de árvores
if _AJUSTADO == True:
	modeloRF.fit(ajustado_treino_x_explicado, ajustado_treino_y)
elif _AJUSTADO == "SMOTE":
	modeloRF.fit(smote_treino_x_explicado, smote_treino_y)
elif _AJUSTADO == "BRFC":
	modeloBRFC = RandomForestRegressor(n_estimators = 100, random_state = SEED)
	modeloBRFC.fit(treino_x_explicado, treino_y)
	y_previsto = modeloBRFC.predict(teste_x)
	previsoes_modelo = modeloBRFC.predict(x)
else:
	modeloRF.fit(treino_x_explicado, treino_y)

### Testando e Avaliando

y_previsto = modeloRF.predict(teste_x)
previsoes_modelo = modeloRF.predict(x)
print(f"\n{green}Shape - y:\n{reset}{y.shape}\n")
print(f"\n{green}Shape - previsoes_modelo:\n{reset}{previsoes_modelo.shape}\n")

previsoes_modelo = [int(p) for p in previsoes_modelo]#[1:]#ValueError: Found input variables with inconsistent numbers of samples: [561, 560]
EQM_RF = mean_squared_error(y, previsoes_modelo)
RQ_EQM_RF = np.sqrt(EQM_RF)
R_2 = round(r2_score(y, previsoes_modelo), 2) 
metricas("RF")
### Testando e Validando Modelo

### Exibindo Informações, Gráficos e Métricas
lista_previsao(previsoes_modelo, 5, "RF")
"""
plt.figure(figsize = (10, 6), layout = "constrained", frameon = False)
ax = plt.gca()
ax.set_facecolor("honeydew")
sns.lineplot(x = x3.index, y = previsoes3,
             color = "red", alpha = 0.7, linewidth = 3, label = "Previsto")
sns.lineplot(x = x3.index, y = x3["CASOS"],
             color = "blue", alpha = 0.9, linewidth = 1, label = "Observado")
plt.title(f"MODELO RANDOM FOREST: PREVISÃO DE CASOS DE DENGUE.\n MUNICÍPIO DE {_CIDADE}, SANTA CATARINA.\n")
plt.xlabel("Semanas Epidemiológicas")
plt.ylabel("Número de Casos de Dengue")
plt.xticks(rotation = "horizontal")
print(f"\n{green}list(x3.index):\n{reset}{list(x3.index)}")
print(f"\n{green}list(x3['CASOS']):\n{reset}{list(x3['CASOS'])}")
#print(f"\n{green}semanas_futuras:\n{reset}{semanas_futuras}")
print(f"\n{green}previsoes3:\n{reset}{previsoes3}")
print(f"\n{green}type(previsoes3):\n{reset}{type(previsoes3)}")
plt.show()
"""
#########################################################AUTOMATIzANDO###############################################################
if _AUTOMATIZA == True:
	for _CIDADE in _CIDADEs:
		try:
			dataset = monta_dataset(_CIDADE)
			x, y, treino_x, teste_x, treino_y, teste_y, treino_x_explicado, explicativas = treino_teste(dataset, _CIDADE)
			modelo, y_previsto, previsoes = RF_modela_treina_preve(treino_x_explicado, treino_y, teste_x, SEED)
			EQM, RQ_EQM, R_2 = RF_previsao_metricas(dataset, previsoes, 5, teste_y, y_previsto)
			salva_modeloRF(modelo, _CIDADE)
		except KeyError as _erro: # "GRÃO PARÁ" "SÃO CRISTÓVÃO DO SUL"
			print(f"\n{red} PROBLEMAS (KeyError) COM O MUNICÍPIO:{bold}{_erro}\n{reset}")

######################################################################################################################################
