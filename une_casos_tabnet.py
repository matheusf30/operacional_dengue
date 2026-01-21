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
caminho_dados = "/home/meteoro/scripts/operacional_dengue/dados_operacao/" # CLUSTER
caminho_operacional = "/home/meteoro/scripts/operacional_dengue/"
caminho_shape = "/media/dados/shapefiles/SC/" #SC_Municipios_2022.shp
caminho_gfs = "/media/dados/operacao/gfs/0p25/" #202410/20241012/ #prec_daily_gfs_2024101212.nc
caminho_merge = "/media/dados/operacao/merge/daily/2024/" #MERGE_CPTEC_DAILY_SB_2024.nc
caminho_merge2 = "/media/dados/operacao/merge/CDO.MERGE" #MERGE_CPTEC_DAILY_2024.nc@
caminho_samet = "/media/dados/operacao/samet/daily/" #/TMAX/2024/ #SAMeT_CPTEC_DAILY_SB_TMAX_2024.nc
caminho_samet2 = "/media/dados/operacao/samet/CDO.SAMET/" #SAMeT_CPTEC_DAILY_SB_TMAX_2024.nc@
url_gh = "https://raw.githubusercontent.com/matheusf30/"
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
casos = f"{caminho_dados}casos_semanal_pivot.csv"
#casos = f"{url_gh}fapesc_dengue/refs/heads/main/matheus/dados/casos_semanal_pivot.csv"
#focos = f"{url_gh}fapesc_dengue/refs/heads/main/matheus/dados/focos_semanal_pivot.csv"
serie_casos = f"{caminho_dados}casos_dive_pivot_total.csv"
#serie_focos = f"{url_gh}dados_dengue/blob/main/focos_pivot.csv"
#casos = "dados_atualizados25_dengue.csv" #"A100523200_135_184_253.csv" # "A173120200_135_184_253.csv"
#serie_casos = "casos_dive_pivot_total.csv"

### Abrindo Arquivos
municipios = gpd.read_file(f"{caminho_shape}{municipios}", low_memory = False)
serie_casos = pd.read_csv(serie_casos)
#serie_focos = pd.read_csv(serie_focos)
casos = pd.read_csv(casos)
#focos = pd.read_csv(focos)
"""
casos = pd.read_csv(f"{caminho_operacional}{casos}", skiprows = 4,
                      sep = ";", encoding = "latin1", engine = "python")
"""                  
print(f"\n{green}serie_casos:\n{reset}{serie_casos}\n")
#print(f"\n{green}serie_focos:\n{reset}{serie_focos}\n")
print(f"\n{green}casos_atuais (TabNetSinanDiveSC):\n{reset}{casos}\n")
print(f"\n{green}casos_atuais (DiveSC):\n{reset}{casos}\n")
#print(f"\n{green}focos_atuais (DiveSC):\n{reset}{focos}\n")
print(datetime.today().strftime("%Y-%m-%d"))
print(datetime.today())
print(date.today().isoformat())
#sys.exit()
## Dados "Brutos"


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
#

##$ Concatenando e Extraindo Dados
hoje = pd.to_datetime(datetime.today().date())
ano_atual = hoje.year
serie_casos["Semana"] = pd.to_datetime(serie_casos["Semana"])
serie_casos = serie_casos[serie_casos["Semana"].dt.year < ano_atual]
casos_atual = casos.copy()
casostotal = pd.concat([serie_casos, casos_atual], ignore_index = True)
casostotal.fillna(0, inplace = True)
colunas = casostotal.drop(columns = "Semana")
colunas = colunas.columns
casostotal[colunas] = casostotal[colunas].astype(int)
casostotal.columns = casostotal.columns.str.strip()
casostotal["Semana"] = pd.to_datetime(casostotal["Semana"]).dt.strftime("%Y-%m-%d")
casostotal = casostotal.drop_duplicates(subset = ["Semana"], keep = "first")
print(f"\n \n {green}SÉRIE TEMPORAL TOTAL{reset}\n{casostotal}\n")
print(f"\n \n {green}SÉRIE TEMPORAL TOTAL (NULOS){reset}\n{casostotal.isnull().sum()}\n")
#casos_pivot = pd.pivot_table(casostotal, index = "Semana", columns = "Município",
#								values = "Casos", fill_value = 0)
#casos_pivot.reset_index(inplace = True)
#sys.exit()
casos_pivot = casostotal.copy()
casos_pivot = casos_pivot[casos_pivot["Semana"] <= str(hoje)]
casostotal = pd.melt(casos_pivot, id_vars = "Semana", var_name = "Município", value_name = "Casos", ignore_index = True)
unicos = casostotal[casostotal["Casos"] > 0].drop_duplicates(subset = ["Município"])
municipios["Município"] = municipios["NM_MUN"].str.upper()
cidades = municipios[["Município", "geometry"]]
cidades = cidades.to_crs(municipios.crs) # SIRGAS 2000/22S ("EPSG:31982") | IBGE.shp ("EPSG:4674")
cidades["centroide"] = cidades["geometry"].centroid
cidades["latitude"] = cidades["geometry"].centroid.y
cidades["longitude"] = cidades["geometry"].centroid.x
xy = cidades[["Município", "latitude", "longitude"]]
unicos_xy = pd.merge(unicos, xy, on = "Município", how = "left")

#sys.exit()
### Salvando Arquivos

os.makedirs(caminho_dados, exist_ok = True)
#cidades.to_csv(f"{caminho_dados}municipios_coordenadas.csv", index = False)
unicos_xy.to_csv(f"{caminho_dados}casos_primeiros.csv", index = False)
casostotal.to_csv(f"{caminho_dados}casos_dive_total.csv", index = False)
casos_pivot.to_csv(f"{caminho_dados}casos_dive_pivot_total.csv", index = False)

print(f"\n \n {green}CASOS DE DENGUE EM SANTA CATARINA - TEMPORAL {reset} \n")
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
