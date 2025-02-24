# Automatizando roteiros de execução
"""
Tentando automatizar tudo para executar de uma única vez.
Deve-se alterar as permissões deste próprio roteiro, que originalmente são apenas leitura (w) e escrita (r).
No terminal, fazer um [ls -l] para verificar as permissões dos roteiros.
Depois executar [chmod +x {script}.sh] para incluir a permissão de executável (x).
E executar da seguinte forma [./{script}.sh]
"""
# Pré-Processamento dos Focos de _Aedes_sp. dos Municípios Catarinenses
"""
Dados brutos disponibilizados pela Diretoria de Vigilância Epidemiológica de Santa Catarina (DIVE/SC).
Baixando diretamente da plataforma online TabNet/Sinan.
"""

conda activate dados_nc

. get_dengue.sh # necessita alterar o nome de saída do arquivo baixado # Roteiro disponibilizado por Elmo Neto
python une_casos_tabnet.py # sincronizar o nome do arquivo baixado
python extrai_clima.py
#python extrai_tempo.py # necessário para não sobrecarregar
python extrai_gfs.py
python modelagem_casos.py True
python cartografia_preditiva_casos.py True False True

python teste_modelagem_casos.py
python teste_cartografia.py True False True

conda deactivate
