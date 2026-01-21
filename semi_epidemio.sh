####################################################################################
## Roteiro adaptado para automatizar execução    ## PALAVRAS-CHAVE:               ##
## Dados: Focos de _Aedes_ sp. e                 ## > Modelagem Computacional;    ##
##        Casos Prováveis de Dengue (DIVE/SC)    ## > Modelo Preditivo;           ##
## Demanda: FAPESC edital nº 37/2024             ## > Transferência de Tecnologia;##
## Adaptado por: Matheus Ferreira de Souza,      ## > Santa Catarina;             ##
##               Caroline Bresciani,             ## > Secretaria de Saúde;        ##
##               Beatriz Campanharo Garcia C.,   ## > Produto Técnico-Tecnológico;##
##               Domênica Tcacenco,              ## > IFSC;                       ##
##               Everton Weber Galliani,         ## > Gerência de Zoonoses;       ##
##               Murilo Ferreira dos Santos.     ## > Monitoramento Meteorológico;##
## Data: 21/01/2026                              ## > Boletim Epidemiológico.     ##
####################################################################################

conda init
conda activate dados_nc
###
### EXTRAÇÃO E PRÉ-PROCESSAMENTO DE DADOS ENTOMO-EPIDEMIOLÓGICOS
python preprocess_epidemio.py
python une_casos_tabnet.py 
#### EXTRAÇÃO E PRÉ-PROCESSAMENTO DE DADOS METEROLÓGICOS #SAMeT, MERGE e GFS
python extrai_clima.py 
python extrai_gfs.py
### MODELAGENS
python modelagem_casos.py True
### CARTOGRAFIAS
python temp_samet.py False True
python prec_merge.py False True
python cartografia_monitoramento_epidemio.py True False True
python cartografia_preditiva_casos.py True False True
python cartografia_preditiva_incidencia.py True False True
### VALIDAÇÕES
python verificacao_validacao_regional.py True False True
###
conda init
conda deactivate
