"""
#################################################################################################
REGISTRO SINAN/SC >>>(DIVESC)
Atentar aos dados provenientes do TabNet... http://200.19.223.105/cgi-bin/dh?sinan/def/dengon.def

--Seleções:
>>> LINHAS: Município de Infecção SC;
>>> COLUNAS: Semana Epidemilógica dos 1ºs sinais.

--Períodos:
>>> Haverá um arquivo por ano.

--Seleções Disponíveis(Filtros):
>>> UF F.infecção: 42 Santa Catarina (Desconsiderados todas as outras UFs, Ignorado e Exterior);

>>> Classificação Nova: Dengue Clássico, Dengue com Complicações, Febre Hemorrágica do Dengue,
                         Síndrome do Choque do Dengue, Dengue e Dengue com Sinais de alarme.
(Nesse filtro foram desconsiderados: Ignorado, Branco, Descartado e Inconclusivo);

>>> Conf.Desc pos2010: Laboratorial e Clínico-Epidemiológico.
(Nesse filtro foram desconsiderados: Ignorado, Branco e Em Investigação);

>>>(Também não foram filtrados por sorotipo, pois não há confirmação laboratorial de todos.)
#################################################################################################
"""

### Bibliotecas Correlatas
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import geopandas as gpd
import sys, os, warnings

original_filter = warnings.filters[:]
warnings.simplefilter("ignore", category = UserWarning)

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


### Encaminhamento aos Diretórios
caminho_github = "https://raw.githubusercontent.com/matheusf30/dados_dengue/main/" # WEB
caminho_dados = "/home/meteoro/scripts/matheus/operacional_dengue/dados_operacao/" # CLUSTER
caminho_operacional = "/home/meteoro/scripts/matheus/operacional_dengue/"
caminho_shape = "/media/dados/shapefiles/SC/" #SC_Municipios_2022.shp
caminho_gfs = "/media/dados/operacao/gfs/0p25/" #202410/20241012/ #prec_daily_gfs_2024101212.nc
caminho_merge = "/media/dados/operacao/merge/daily/2024/" #MERGE_CPTEC_DAILY_SB_2024.nc
caminho_merge2 = "/media/dados/operacao/merge/CDO.MERGE" #MERGE_CPTEC_DAILY_2024.nc@
caminho_samet = "/media/dados/operacao/samet/daily/" #/TMAX/2024/ #SAMeT_CPTEC_DAILY_SB_TMAX_2024.nc
caminho_samet2 = "/media/dados/operacao/samet/CDO.SAMET/" #SAMeT_CPTEC_DAILY_SB_TMAX_2024.nc@

### Renomeação variáveis pelos arquivos
ANO_ESCOLHIDO = str(datetime.today().year)
#prec_daily_gfs_2024101212.nc
#temp_min_daily_gfs_2024101200.nc
#temp_mean_daily_gfs_2024101200.nc
#temp_max_daily_gfs_2024101200.nc
#MERGE_CPTEC_DAILY_SB_2024.nc
#SAMeT_CPTEC_DAILY_SB_TMAX_2024.nc
municipios = "SC_Municipios_2022.shp"
# Fonte: TABNET/DATASUS - SINAN/SC
casos24 = "dados_atualizados_dengue.csv" #"A100523200_135_184_253.csv" # "A173120200_135_184_253.csv"
serie_casos = "casos_dive_total.csv"
### Abrindo Arquivos
municipios = gpd.read_file(f"{caminho_shape}{municipios}", low_memory = False)
serie_casos = pd.read_csv(f"{caminho_dados}{serie_casos}")
casos24 = pd.read_csv(f"{caminho_operacional}{casos24}", skiprows = 5,
                      sep = ";", encoding = "latin1", engine = "python")
                      
print(f"\n{green}serie_casos:\n{reset}{serie_casos}\n")
print(f"\n{green}casos_atuais (TabNetSinanDiveSC):\n{reset}{casos24}\n")
print(datetime.today().strftime("%Y-%m-%d"))
print(datetime.today())
print(date.today().isoformat())

## Dados "Brutos"
print(f"""{green}
 INVESTIGAÇÃO DENGUE A PARTIR DE 2014
Frequência por Mun infec SC e Sem.Epid.Sintomas{red}
Classificacao Nova: {green}Dengue com complicações, Febre Hemorrágica do Dengue, Síndrome do Choque do Dengue, Dengue, Dengue com sinais de alarme, Dengre grave{red}
Conf.Desc pos2010: {green}Laboratórial, Clínico-epidemiológico{red}
Período: {green}{ANO_ESCOLHIDO}{magenta}

    A partir de 2020 o estado do Espírito Santo passou a utilizar o sistema e-SUS Vigilância em Saúde. Portanto, para os casos de Arboviroses urbanas do Espírito Santo foram considerados apenas os dados disponibilizados pelo Sinan online (dengue e chikungunya) e Sinan Net (zika).
    Períodos Disponíveis ou período - Correspondem aos anos de notificação dos casos e semana epidemiológica, em cada período pode apresentar notificações com data de notificação do ano anterior (semana epidemiológica 52 ou 53) e posterior (semana epidemiológica 01).{red}
    Para cálculo da incidência recomenda-se utilizar locais de residência.
    Dados de 2014 atualizados em 13/07/2015.
    Dados de 2015 atualizados em 27/09/2016.
    Dados de 2016 atualizados em 06/07/2017.
    Dados de 2017 atualizados em 18/07/2018.
    Dados de 2018 atualizados em 01/10/2019.
    Dados de 2019 atualizados em 10/07/2020.
    Dados de 2020 atualizados em 23/07/2021.
    Dados de 2021 atualizados em 12/07/2022.
    Dados de 2022 atualizados em 18/07/2023.
    Dados de 2023 atualizados em 04/03/2024 à 01 hora, sujeitos à revisão.
    Dados de 2024 atualizados em 11/03/2024 às 08 horas, sujeitos à revisão.
<<<<<<< HEAD
{green}
=======
>>>>>>> 72ec65a... Iniciando com dados brutos TabNet/DiveSC.
    * Dados disponibilizados no TABNET em março de 2024. 

Legenda:
-	- Dado numérico igual a 0 não resultante de arredondamento.
0; 0,0	- Dado numérico igual a 0 resultante de arredondamento de um dado originalmente positivo.{reset}
""")

### Pré-Processamento
lista_municipio = {'ABDON BATISTA': 'ABDON BATISTA',
  'ABELARDO LUZ': 'ABELARDO LUZ',
  'AGROLANDIA': 'AGROLÂNDIA',
  'AGRONOMICA': 'AGRONÔMICA',
  'AGUA DOCE': 'ÁGUA DOCE',
  'AGUAS DE CHAPECO': 'ÁGUAS DE CHAPECÓ',
  'AGUAS FRIAS': 'ÁGUAS FRIAS',
  'AGUAS MORNAS': 'ÁGUAS MORNAS',
  'ALFREDO WAGNER': 'ALFREDO WAGNER',
  'ALTO BELA VISTA': 'ALTO BELA VISTA',
  'ANCHIETA': 'ANCHIETA',
  'ANGELINA': 'ANGELINA',
  'ANITA GARIBALDI': 'ANITA GARIBALDI',
  'ANITAPOLIS': 'ANITÁPOLIS',
  'ANTONIO CARLOS': 'ANTÔNIO CARLOS',
  'APIUNA': 'APIÚNA',
  'ARABUTA': 'ARABUTÃ',
  'ARAQUARI': 'ARAQUARI',
  'ARARANGUA': 'ARARANGUÁ',
  'ARMAZEM': 'ARMAZÉM',
  'ARROIO TRINTA': 'ARROIO TRINTA',
  'ARVOREDO': 'ARVOREDO',
  'ASCURRA': 'ASCURRA',
  'ATALANTA': 'ATALANTA',
  'AURORA': 'AURORA',
  'BALNEARIO ARROIO DO SILVA': 'BALNEÁRIO ARROIO DO SILVA',
  'BALNEARIO CAMBORIU': 'BALNEÁRIO CAMBORIÚ',
  'BALNEARIO BARRA DO SUL': 'BALNEÁRIO BARRA DO SUL',
  'BALNEARIO GAIVOTA': 'BALNEÁRIO GAIVOTA',
  'BANDEIRANTE': 'BANDEIRANTE',
  'BARRA BONITA': 'BARRA BONITA',
  'BARRA VELHA': 'BARRA VELHA',
  'BELA VISTA DO TOLDO': 'BELA VISTA DO TOLDO',
  'BELMONTE': 'BELMONTE',
  'BENEDITO NOVO': 'BENEDITO NOVO',
  'BIGUACU': 'BIGUAÇU',
  'BLUMENAU': 'BLUMENAU',
  'BOCAINA DO SUL': 'BOCAINA DO SUL',
  'BOMBINHAS': 'BOMBINHAS',
  'BOM JARDIM DA SERRA': 'BOM JARDIM DA SERRA',
  'BOM JESUS': 'BOM JESUS',
  'BOM JESUS DO OESTE': 'BOM JESUS DO OESTE',
  'BOM RETIRO': 'BOM RETIRO',
  'BOTUVERA': 'BOTUVERÁ',
  'BRACO DO NORTE': 'BRAÇO DO NORTE',
  'BRACO DO TROMBUDO': 'BRAÇO DO TROMBUDO',
  'BRUNOPOLIS': 'BRUNÓPOLIS',
  'BRUSQUE': 'BRUSQUE',
  'CACADOR': 'CAÇADOR',
  'CAIBI': 'CAIBI',
  'CALMON': 'CALMON',
  'CAMBORIU': 'CAMBORIÚ',
  'CAPAO ALTO': 'CAPÃO ALTO',
  'CAMPO ALEGRE': 'CAMPO ALEGRE',
  'CAMPO BELO DO SUL': 'CAMPO BELO DO SUL',
  'CAMPO ERE': 'CAMPO ERÊ',
  'CAMPOS NOVOS': 'CAMPOS NOVOS',
  'CANELINHA': 'CANELINHA',
  'CANOINHAS': 'CANOINHAS',
  'CAPINZAL': 'CAPINZAL',
  'CAPIVARI DE BAIXO': 'CAPIVARI DE BAIXO',
  'CATANDUVAS': 'CATANDUVAS',
  'CAXAMBU DO SUL': 'CAXAMBU DO SUL',
  'CELSO RAMOS': 'CELSO RAMOS',
  'CERRO NEGRO': 'CERRO NEGRO',
  'CHAPADAO DO LAGEADO': 'CHAPADÃO DO LAGEADO',
  'CHAPECO': 'CHAPECÓ',
  'COCAL DO SUL': 'COCAL DO SUL',
  'CONCORDIA': 'CONCÓRDIA',
  'CORDILHEIRA ALTA': 'CORDILHEIRA ALTA',
  'CORONEL FREITAS': 'CORONEL FREITAS',
  'CORONEL MARTINS': 'CORONEL MARTINS',
  'CORUPA': 'CORUPÁ',
  'CORREIA PINTO': 'CORREIA PINTO',
  'CRICIUMA': 'CRICIÚMA',
  'CUNHA PORA': 'CUNHA PORÃ',
  'CUNHATAO': 'CUNHATAÍ',
  'CURITIBANOS': 'CURITIBANOS',
  'DESCANSO': 'DESCANSO',
  'DIONOSIO CERQUEIRA': 'DIONÍSIO CERQUEIRA',
  'DONA EMMA': 'DONA EMMA',
  'DOUTOR PEDRINHO': 'DOUTOR PEDRINHO',
  'ENTRE RIOS': 'ENTRE RIOS',
  'ERMO': 'ERMO',
  'ERVAL VELHO': 'ERVAL VELHO',
  'FAXINAL DOS GUEDES': 'FAXINAL DOS GUEDES',
  'FLOR DO SERTAO': 'FLOR DO SERTÃO',
  'FLORIANOPOLIS': 'FLORIANÓPOLIS',
  'FORMOSA DO SUL': 'FORMOSA DO SUL',
  'FORQUILHINHA': 'FORQUILHINHA',
  'FRAIBURGO': 'FRAIBURGO',
  'FREI ROGERIO': 'FREI ROGÉRIO',
  'GALVAO': 'GALVÃO',
  'GAROPABA': 'GAROPABA',
  'GARUVA': 'GARUVA',
  'GASPAR': 'GASPAR',
  'GOVERNADOR CELSO RAMOS': 'GOVERNADOR CELSO RAMOS',
  'GRAO PARA': 'GRÃO-PARÁ',
  'GRAVATAL': 'GRAVATAL',
  'GUABIRUBA': 'GUABIRUBA',
  'GUARACIABA': 'GUARACIABA',
  'GUARAMIRIM': 'GUARAMIRIM',
  'GUARUJA DO SUL': 'GUARUJÁ DO SUL',
  'GUATAMBU': 'GUATAMBÚ',
  'HERVAL D OESTE': "HERVAL D'OESTE",
  'IBIAM': 'IBIAM',
  'IBICARE': 'IBICARÉ',
  'IBIRAMA': 'IBIRAMA',
  'ICARA': 'IÇARA',
  'ILHOTA': 'ILHOTA',
  'IMARUO': 'IMARUÍ',
  'IMBITUBA': 'IMBITUBA',
  'IMBUIA': 'IMBUIA',
  'INDAIAL': 'INDAIAL',
  'IOMERE': 'IOMERÊ',
  'IPIRA': 'IPIRA',
  'IPORA DO OESTE': 'IPORÃ DO OESTE',
  'IPUACU': 'IPUAÇU',
  'IPUMIRIM': 'IPUMIRIM',
  'IRACEMINHA': 'IRACEMINHA',
  'IRANI': 'IRANI',
  'IRATI': 'IRATI',
  'IRINEOPOLIS': 'IRINEÓPOLIS',
  'ITA': 'ITÁ',
  'ITAIOPOLIS': 'ITAIÓPOLIS',
  'ITAJAI': 'ITAJAÍ',
  'ITAPEMA': 'ITAPEMA',
  'ITAPIRANGA': 'ITAPIRANGA',
  'ITAPOA': 'ITAPOÁ',
  'ITUPORANGA': 'ITUPORANGA',
  'JABORA': 'JABORÁ',
  'JACINTO MACHADO': 'JACINTO MACHADO',
  'JAGUARUNA': 'JAGUARUNA',
  'JARAGUA DO SUL': 'JARAGUÁ DO SUL',
  'JARDINOPOLIS': 'JARDINÓPOLIS',
  'JOACABA': 'JOAÇABA',
  'JOINVILLE': 'JOINVILLE',
  'JOSE BOITEUX': 'JOSÉ BOITEUX',
  'JUPIA': 'JUPIÁ',
  'LACERDOPOLIS': 'LACERDÓPOLIS',
  'LAGES': 'LAGES',
  'LAGUNA': 'LAGUNA',
  'LAJEADO GRANDE': 'LAJEADO GRANDE',
  'LAURENTINO': 'LAURENTINO',
  'LAURO MULLER': 'LAURO MÜLLER',
  'LEBON REGIS': 'LEBON RÉGIS',
  'LEOBERTO LEAL': 'LEOBERTO LEAL',
  'LINDOIA DO SUL': 'LINDÓIA DO SUL',
  'LONTRAS': 'LONTRAS',
  'LUIZ ALVES': 'LUIZ ALVES',
  'LUZERNA': 'LUZERNA',
  'MACIEIRA': 'MACIEIRA',
  'MAFRA': 'MAFRA',
  'MAJOR GERCINO': 'MAJOR GERCINO',
  'MAJOR VIEIRA': 'MAJOR VIEIRA',
  'MARACAJA': 'MARACAJÁ',
  'MARAVILHA': 'MARAVILHA',
  'MAREMA': 'MAREMA',
  'MASSARANDUBA': 'MASSARANDUBA',
  'MATOS COSTA': 'MATOS COSTA',
  'MELEIRO': 'MELEIRO',
  'MIRIM DOCE': 'MIRIM DOCE',
  'MODELO': 'MODELO',
  'MONDAO': 'MONDAÍ',
  'MONTE CARLO': 'MONTE CARLO',
  'MONTE CASTELO': 'MONTE CASTELO',
  'MORRO DA FUMACA': 'MORRO DA FUMAÇA',
  'MORRO GRANDE': 'MORRO GRANDE',
  'NAVEGANTES': 'NAVEGANTES',
  'NOVA ERECHIM': 'NOVA ERECHIM',
  'NOVA ITABERABA': 'NOVA ITABERABA',
  'NOVA TRENTO': 'NOVA TRENTO',
  'NOVA VENEZA': 'NOVA VENEZA',
  'NOVO HORIZONTE': 'NOVO HORIZONTE',
  'ORLEANS': 'ORLEANS',
  'OTACOLIO COSTA': 'OTACÍLIO COSTA',
  'OURO': 'OURO',
  'OURO VERDE': 'OURO VERDE',
  'PAIAL': 'PAIAL',
  'PAINEL': 'PAINEL',
  'PALHOCA': 'PALHOÇA',
  'PALMA SOLA': 'PALMA SOLA',
  'PALMEIRA': 'PALMEIRA',
  'PALMITOS': 'PALMITOS',
  'PAPANDUVA': 'PAPANDUVA',
  'PARAOSO': 'PARAÍSO',
  'PASSO DE TORRES': 'PASSO DE TORRES',
  'PASSOS MAIA': 'PASSOS MAIA',
  'PAULO LOPES': 'PAULO LOPES',
  'PEDRAS GRANDES': 'PEDRAS GRANDES',
  'PENHA': 'PENHA',
  'PERITIBA': 'PERITIBA',
  'PESCARIA BRAVA': 'PESCARIA BRAVA',
  'PETROLANDIA': 'PETROLÂNDIA',
  'BALNEARIO PICARRAS': 'BALNEÁRIO PIÇARRAS',
  'PINHALZINHO': 'PINHALZINHO',
  'PINHEIRO PRETO': 'PINHEIRO PRETO',
  'PIRATUBA': 'PIRATUBA',
  'PLANALTO ALEGRE': 'PLANALTO ALEGRE',
  'POMERODE': 'POMERODE',
  'PONTE ALTA': 'PONTE ALTA',
  'PONTE ALTA DO NORTE': 'PONTE ALTA DO NORTE',
  'PONTE SERRADA': 'PONTE SERRADA',
  'PORTO BELO': 'PORTO BELO',
  'PORTO UNIAO': 'PORTO UNIÃO',
  'POUSO REDONDO': 'POUSO REDONDO',
  'PRAIA GRANDE': 'PRAIA GRANDE',
  'PRESIDENTE CASTELLO BRANCO': 'PRESIDENTE CASTELLO BRANCO',
  'PRESIDENTE GETULIO': 'PRESIDENTE GETÚLIO',
  'PRESIDENTE NEREU': 'PRESIDENTE NEREU',
  'PRINCESA': 'PRINCESA',
  'QUILOMBO': 'QUILOMBO',
  'RANCHO QUEIMADO': 'RANCHO QUEIMADO',
  'RIO DAS ANTAS': 'RIO DAS ANTAS',
  'RIO DO CAMPO': 'RIO DO CAMPO',
  'RIO DO OESTE': 'RIO DO OESTE',
  'RIO DOS CEDROS': 'RIO DOS CEDROS',
  'RIO DO SUL': 'RIO DO SUL',
  'RIO FORTUNA': 'RIO FORTUNA',
  'RIO NEGRINHO': 'RIO NEGRINHO',
  'RIO RUFINO': 'RIO RUFINO',
  'RIQUEZA': 'RIQUEZA',
  'RODEIO': 'RODEIO',
  'ROMELANDIA': 'ROMELÂNDIA',
  'SALETE': 'SALETE',
  'SALTINHO': 'SALTINHO',
  'SALTO VELOSO': 'SALTO VELOSO',
  'SANGAO': 'SANGÃO',
  'SANTA CECOLIA': 'SANTA CECÍLIA',
  'SANTA HELENA': 'SANTA HELENA',
  'SANTA ROSA DE LIMA': 'SANTA ROSA DE LIMA',
  'SANTA ROSA DO SUL': 'SANTA ROSA DO SUL',
  'SANTA TEREZINHA': 'SANTA TEREZINHA',
  'SANTA TEREZINHA DO PROGRESSO': 'SANTA TEREZINHA DO PROGRESSO',
  'SANTIAGO DO SUL': 'SANTIAGO DO SUL',
  'SANTO AMARO DA IMPERATRIZ': 'SANTO AMARO DA IMPERATRIZ',
  'SAO BERNARDINO': 'SÃO BERNARDINO',
  'SAO BENTO DO SUL': 'SÃO BENTO DO SUL',
  'SAO BONIFACIO': 'SÃO BONIFÁCIO',
  'SAO CARLOS': 'SÃO CARLOS',
  'SAO CRISTOVAO DO SUL': 'SÃO CRISTÓVÃO DO SUL',
  'SAO DOMINGOS': 'SÃO DOMINGOS',
  'SAO FRANCISCO DO SUL': 'SÃO FRANCISCO DO SUL',
  'SAO JOAO DO OESTE': 'SÃO JOÃO DO OESTE',
  'SAO JOAO BATISTA': 'SÃO JOÃO BATISTA',
  'SAO JOAO DO ITAPERIU': 'SÃO JOÃO DO ITAPERIÚ',
  'SAO JOAO DO SUL': 'SÃO JOÃO DO SUL',
  'SAO JOAQUIM': 'SÃO JOAQUIM',
  'SAO JOSE': 'SÃO JOSÉ',
  'SAO JOSE DO CEDRO': 'SÃO JOSÉ DO CEDRO',
  'SAO JOSE DO CERRITO': 'SÃO JOSÉ DO CERRITO',
  'SAO LOURENCO DO OESTE': 'SÃO LOURENÇO DO OESTE',
  'SAO LUDGERO': 'SÃO LUDGERO',
  'SAO MARTINHO': 'SÃO MARTINHO',
  'SAO MIGUEL DA BOA VISTA': 'SÃO MIGUEL DA BOA VISTA',
  'SAO MIGUEL DO OESTE': 'SÃO MIGUEL DO OESTE',
  'SAO PEDRO DE ALCANTARA': 'SÃO PEDRO DE ALCÂNTARA',
  'SAUDADES': 'SAUDADES',
  'SCHROEDER': 'SCHROEDER',
  'SEARA': 'SEARA',
  'SERRA ALTA': 'SERRA ALTA',
  'SIDEROPOLIS': 'SIDERÓPOLIS',
  'SOMBRIO': 'SOMBRIO',
  'SUL BRASIL': 'SUL BRASIL',
  'TAIO': 'TAIÓ',
  'TANGARA': 'TANGARÁ',
  'TIGRINHOS': 'TIGRINHOS',
  'TIJUCAS': 'TIJUCAS',
  'TIMBE DO SUL': 'TIMBÉ DO SUL',
  'TIMBO': 'TIMBÓ',
  'TIMBO GRANDE': 'TIMBÓ GRANDE',
  'TRES BARRAS': 'TRÊS BARRAS',
  'TREVISO': 'TREVISO',
  'TREZE DE MAIO': 'TREZE DE MAIO',
  'TREZE TOLIAS': 'TREZE TÍLIAS',
  'TROMBUDO CENTRAL': 'TROMBUDO CENTRAL',
  'TUBARAO': 'TUBARÃO',
  'TUNAPOLIS': 'TUNÁPOLIS',
  'TURVO': 'TURVO',
  'UNIAO DO OESTE': 'UNIÃO DO OESTE',
  'URUBICI': 'URUBICI',
  'URUPEMA': 'URUPEMA',
  'URUSSANGA': 'URUSSANGA',
  'VARGEAO': 'VARGEÃO',
  'VARGEM': 'VARGEM',
  'VARGEM BONITA': 'VARGEM BONITA',
  'VIDAL RAMOS': 'VIDAL RAMOS',
  'VIDEIRA': 'VIDEIRA',
  'VITOR MEIRELES': 'VITOR MEIRELES',
  'WITMARSUM': 'WITMARSUM',
  'XANXERE': 'XANXERÊ',
  'XAVANTINA': 'XAVANTINA',
  'XAXIM': 'XAXIM',
  'ZORTEA': 'ZORTÉA',
  'BALNEARIO RINCAO': 'BALNEÁRIO RINCÃO'}

# 2024 (Padronização)
ano = 2024
total_semana = 52
lista_str_semanas = []
for i in range(1, total_semana + 1):
    n_semana = str(i).zfill(2)
    chave_semana = f"Semana {n_semana}"
    lista_str_semanas.append(chave_semana)
inicio = datetime(ano-1, 12, 31)
fim = datetime(ano, 12, 28)
lista_semanas = []
semana_corrente = inicio
while semana_corrente <= fim:
    lista_semanas.append(semana_corrente)
    semana_corrente += timedelta(weeks = 1)
dict_semanas = dict(zip(lista_str_semanas, [date.strftime("%Y-%m-%d") for date in lista_semanas]))
casos24.rename(columns = {"Mun infec SC" : "Município"}, inplace = True)
casos24.rename(columns = {"Município infecção" : "Município"}, inplace = True)
casos24.rename(columns = dict_semanas, inplace = True)
casos24["Município"] = casos24["Município"].str.replace("\d+ ", "", regex = True)
casos24["Município"] = casos24["Município"].str.upper()
casos24.drop(columns = "Total", inplace = True)
casos24.drop(casos24.index[-1:], axis = 0, inplace = True)
casos24.set_index("Município", inplace = True)
casos24 = casos24.T
casos24.reset_index(inplace=True)
casos24 = casos24.rename(columns = {"index" : "Semana"})
casos24.rename(columns = lista_municipio, inplace = True)
colunas = casos24.columns.drop("Semana")
semanas = pd.DataFrame(lista_semanas, columns=["Semana"])
casos24["Semana"] = pd.to_datetime(casos24["Semana"])
casos24 = pd.merge(semanas, casos24, on = "Semana", how = "left").fillna(0)
casos24[colunas] = casos24[colunas].astype(int)
hoje = pd.to_datetime(datetime.today().date())
casos24 = casos24[casos24["Semana"] <= hoje]
casos24 = pd.melt(casos24, id_vars = "Semana", var_name = "Município", value_name = "Casos", ignore_index = True)
casos24.sort_values(by = "Semana",  ignore_index = True, inplace = True)
print("="*80, f"\n{ano}\n\n", casos24)
print(casos24.info())
print(casos24.columns.drop("Semana"))
print("="*80)

##$ Concatenando e Extraindo Dados
casostotal = pd.concat([serie_casos, casos24], ignore_index = True)
casostotal["Semana"] = pd.to_datetime(casostotal["Semana"]).dt.strftime("%Y-%m-%d")
casos_pivot = pd.pivot_table(casostotal, index = "Semana", columns = "Município",
								values = "Casos", fill_value = 0)
casos_pivot.reset_index(inplace = True)
casos_pivot = casos_pivot[casos_pivot["Semana"] <= str(hoje)]
unicos = casostotal[casostotal["Casos"] > 0].drop_duplicates(subset = ["Município"])
municipios["Município"] = municipios["NM_MUN"].str.upper()
cidades = municipios[["Município", "geometry"]]
cidades = cidades.to_crs(municipios.crs) # SIRGAS 2000/22S ("EPSG:31982") | IBGE.shp ("EPSG:4674")
cidades["centroide"] = cidades["geometry"].centroid
cidades["latitude"] = cidades["geometry"].centroid.y
cidades["longitude"] = cidades["geometry"].centroid.x
xy = cidades[["Município", "latitude", "longitude"]]
unicos_xy = pd.merge(unicos, xy, on = "Município", how = "left")

### Salvando Arquivos

os.makedirs(caminho_dados, exist_ok = True)
#cidades.to_csv(f"{caminho_dados}municipios_coordenadas.csv", index = False)
unicos_xy.to_csv(f"{caminho_dados}casos_primeiros.csv", index = False)
casostotal.to_csv(f"{caminho_dados}casos_dive_total.csv", index = False)
casos_pivot.to_csv(f"{caminho_dados}casos_dive_pivot_total.csv", index = False)

print(f"\n \n {green}CASOS DE DENGUE EM SANTA CATARINA - SÉRIE HISTÓRICA {reset} \n")
print(casostotal.info())
print("~"*80)
print(casostotal.dtypes)
print("~"*80)
print(casostotal)
print("="*80)

print(f"\n \n {green}CASOS DE DENGUE EM SANTA CATARINA - TABELA DINÂMICA / SÉRIE HISTÓRICA {reset} \n")
print(casos_pivot.info())
print("~"*80)
print(casos_pivot.dtypes)
print("~"*80)
print(casos_pivot)
print("="*80)

print(f"\n \n {green} CASOS DE DENGUE EM SANTA CATARINA - 1º REGISTRO NA SÉRIE HISTÓRICA  {reset} \n")
print(unicos_xy.info())
print("~"*80)
print(unicos_xy.dtypes)
print("~"*80)
print(unicos_xy)
print("="*80)

print(f"\n \n {green} CASOS DE DENGUE EM SANTA CATARINA - 1º REGISTRO NA SÉRIE HISTÓRICA {reset} \n")
print(cidades.info())
print("~"*80)
print(cidades.dtypes)
print("~"*80)
print(cidades)
print("="*80)
print(f"{green}Coordinate Reference System (CRS)\nIBGE.shp: {reset}{municipios.crs}\n{red}(Instituto Brasileiro de Geografia e Estatística){reset}")
print("="*80)
