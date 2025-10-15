# Script que gera o CSV com informações sobre início, 
# fim e número das semanas epidemiológicas

from datetime import datetime, timedelta
t0 = datetime.now()
print(f"Início da execução: {t0}")
import pandas as pd
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import os
import sys
'''
Weekday retorna 0 para 2º-feira e 6 para domingo
Se o primeiro dia do ano cair entre domingo e quarta-feira,
a semana epidemiológica pertence ao ano que começou e passa a ser
a primeira S.E. do ano.
Se cair entre quinta-feira e sábado, pertence ao ano passado e
passa a ser a última S.E. do ano.
'''
data_zero = pd.to_datetime('2000-01-01')
# Definindo a data de começo e fim da semana epidemiológica
# do primeiro dia do ano
domingo_zero = data_zero - timedelta(days = data_zero.weekday())
sabado_zero = domingo_zero + timedelta(days = 6)

#Verificando se a SE do primeiro dia pertence ao ano anterior ou atual
def verifica_se(ano):
    data_zero = pd.to_datetime(f"{ano}-01-01")
    if data_zero.weekday() <= 2:
        print("A SE do 1° dia do ano pertence ao ano atual.")
    elif data_zero.weekday() > 2 and data_zero.weekday() <= 6:
        print("A SE do 1° dia do ano pertence ao ano anterior.")
    else:
        print("Erro na coleta do dia da SE do 1° dia do ano.")

def gera_semanas_climatologia(ano_zero, ano):
    """Generate epidemiological weeks from 2001 to specified year"""
    colunas = ["S.E.", "Domingo", "Sábado"]
    df = pd.DataFrame(columns=colunas)
    #ano_zero = 2020
    data_zero = pd.to_datetime(f"{ano_zero}-01-01")
    wkd = data_zero.weekday()
    semana = 1
    domingo = data_zero - timedelta(days = wkd + 1)
    sabado = domingo + timedelta(days = 6)
    if (data_zero - domingo).days <= 3:
    	current_epi_year = ano_zero
    else:
    	current_epi_year = ano_zero - 1
    	domingo += timedelta(days = 7)
    	sabado += timedelta(days = 7)
    """
    # Determine starting point
    if wkd in [6, 0, 1]:  # Sun, Mon, Tue
        domingo = data_zero - timedelta(days=(wkd + 1) % 7)
        semana = 1
        current_epi_year = ano_zero + 1
    else:  # Wed, Thu, Fri, Sat
        domingo = data_zero + timedelta(days=(6 - wkd))
        semana = 52 if domingo.year == 2000 else 53  # Week 52 or 53 of 2000
        current_epi_year = ano_zero
    """
    # Generate weeks
    while current_epi_year <= ano:
        sabado = domingo + timedelta(days=6)
        
        # Determine epidemiological year (week belongs to year of its Thursday)
        thursday = domingo + timedelta(days=3)
        epi_year = thursday.year
        
        # Only add if week belongs to a year we want (2001-ano)
        if epi_year >= ano_zero and epi_year <= ano:
            linha = pd.DataFrame([[semana, domingo, sabado]], columns=colunas)
            df = pd.concat([df, linha], ignore_index=True)
        
        # Move to next week
        domingo += timedelta(days=7)
        sabado += timedelta(days=7)
        
        # Update week number and year
        thursday_next = domingo + timedelta(days=3)
        epi_year_next = thursday_next.year
        
        if epi_year_next != epi_year:
            # New epidemiological year, reset week counter
            semana = 1
            current_epi_year = epi_year_next
        else:
            semana += 1
    
    return df

se_infos = gera_semanas_climatologia(2000, 2025)
print(se_infos)
#sys.exit()

lat_min = -29.5 # -34.00 # -29.5
lat_max = -25.75 # -21.75 # -25.75
lon_min = -54 # -58.25 # -54
lon_max = -48 # -47.50 # -48

def concatena_netcdf(variavel):
	print("=="*50)
	print(f"Iniciada a execução da função concatena_netcdf{variavel}")
	t0_function = datetime.now()
	recorte_list = []
	valid_weeks = []
	ano_inicio = 2000 #Ano de início da criação do netCDF
	match variavel:
		case "prec":
			_path = f"/media/dados/operacao/merge/daily"
			#_file = f"/MERGE_CPTEC_DAILY_SB_2000.nc"
			_file = f"/MERGE_CPTEC_DAILY_SB_"
			_vars = "nest"
		case "tmed":
			_path = f"/media/dados/operacao/samet/daily/TMED"
			#_file = f"/SAMeT_CPTEC_DAILY_SB_TMED_2000.nc"
			_file = f"/SAMeT_CPTEC_DAILY_SB_TMED_"
			_vars = "nobs"
		case _:
			print("Erro na variável de argumento.")
			return
	dataset = xr.open_dataset(f"{_path}/{ano_inicio}{_file}{ano_inicio}.nc").sel(lat = slice(lat_min, lat_max), lon = slice(lon_min, lon_max)).drop_vars([_vars])
	lista = [dataset]
	for i in range(ano_inicio + 1, 2025):
		dataset2 = xr.open_dataset(f"{_path}/{i}{_file}{i}.nc").sel(lat = slice(lat_min, lat_max), lon = slice(lon_min, lon_max)).drop_vars([_vars])
		#print(dataset2.data_vars.keys())
		lista.append(dataset2)
		ano_final = i #Salva o último ano do arquivo netCDF
		#print(f"Concatenado do ano 2000 até o ano {i}")
	#print("=="*50)
	#print(f"Iniciada a concatenação dos dados de {ano_inicio} a {ano_final}")
	dataset = xr.concat(lista, dim='time')
	#print("Finalizada a concatenação.")
	t1_function = datetime.now() - t0_function
	print(f"Tempo de execução da função 'concatena_netcdf({variavel})': {t1_function}")
	print("=="*50)
	return dataset

serie_samet = concatena_netcdf("tmed")
serie_merge = concatena_netcdf("prec")
#print(serie_merge, serie_samet)

#sys.exit()
def gera_climatologia_se(dataset, str_var):
	print("=="*50)
	print(f"Iniciada a execução da função gea_climatologia_se{str_var}")
	t0_function = datetime.now()	
		# Initialize lists to store data
	recorte_list = []
	valid_weeks = []

	for j in range(1, 54):
		df_se = se_infos.loc[se_infos['S.E.'] == j].reset_index()

		if len(df_se) == 0:
			continue

		for i in df_se.index:
			domingo = pd.to_datetime(df_se.loc[i]['Domingo'])
			sabado = pd.to_datetime(df_se.loc[i]['Sábado'])
			if str_var == "tmed":
				recorte = dataset.sel(time=slice(domingo, sabado)).mean(dim="time")
			elif str_var == "prec":
				recorte = dataset.sel(time=slice(domingo, sabado)).sum(dim="time")
			# Append to lists
			recorte_list.append(recorte)
			valid_weeks.append(j)

	# Create new dataset with week dimension
	if recorte_list:
		new_dataset = xr.concat(recorte_list, dim='week')
		new_dataset['week'] = valid_weeks
		new_dataset = new_dataset.groupby('week').mean()
		#if str_var == "tmed":
		#	new_dataset = new_dataset.groupby('week').mean()
		#elif(str_var == "prec"):
		#	new_dataset = new_dataset.groupby('week').mean()
	new_dataset.to_netcdf(f'{str_var}_climatologia_epidemiosemanal.nc')
	t1_function = datetime.now() - t0_function
	print(f"Tempo de execução da função 'gera_climatologia_se({str_var})': {t1_function}")
	print("=="*50)
	return new_dataset
		# Save the dataset
		#new_dataset.to_netcdf('weekly_averages.nc')
	#print(f"Saved dataset with {len(recorte_list)} weeks of data")	
	
clima_merge = gera_climatologia_se(serie_merge, "prec")
clima_samet = gera_climatologia_se(serie_samet, "tmed")
print(clima_samet, clima_merge)



t1 = datetime.now() - t0
print(f"Tempo de execução do programa: {t1}")
