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
## Data: 31/07/2025                              ## > Boletim Epidemiológico.     ##
####################################################################################
"""
Tentando automatizar tudo para executar de uma única vez.
Deve-se alterar as permissões deste próprio roteiro,
que originalmente são apenas leitura (w) e escrita (r).
No terminal, fazer um [ls -l] para verificar as permissões dos roteiros.
Depois executar [chmod +x {script}.sh] para incluir a permissão de executável (x).
E executar da seguinte forma [./{script}.sh]
"""

conda init
conda activate dados_nc
### EXTRAÇÃO E PRÉ-PROCESSAMENTO DE DADOS EPIDEMIOLÓGICOS
#. get_dengue.sh # necessita alterar o nome de saída do arquivo baixado # Roteiro disponibilizado por Elmo Neto
python preprocess_saude.py # sincronizar o nome do arquivo baixado
python une_casos_tabnet.py 
#### EXTRAÇÃO E PRÉ-PROCESSAMENTO DE DADOS METEROLÓGICOS
python extrai_clima.py #SAMeT e MERGE
python extrai_gfs.py
### MODELAGENS
python modelagem_casos.py True
#python modelagem_incidencia.py True
### CARTOGRAFIAS
#python cartografia_casos_incidencia.py True False True
python everton_temp.py
python everton_prec.py
#python temp_semanal_samet_media.py
#python prec_semanal_merge_acumulado.py
#python temp_semanal_samet_anomalia.py
#python prec_semanal_merge_anomalia.py
python cartografia_monitoramento_epidemio.py True False True
python cartografia_preditiva_casos.py True False True
python cartografia_preditiva_incidencia.py True False True
### VALIDAÇÕES
python verificacao_validacao.py False True False
"""
python teste_modelagem_casos.py
python teste_cartografia.py True False True
"""
conda init
conda deactivate
