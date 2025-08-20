import requests
import os
import time
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import dotenv

dotenv.load_dotenv()

X_API_KEY = os.environ['X_API_KEY']

# Define the maximum number of retries
MAX_RETRIES = 3
RETRY_DELAY = 10  # Delay between retries in seconds

# Base URL da API
BASE_URL = 'https://api-bndmet.decea.mil.br/v1/estacoes'

# Cabeçalhos necessários para a requisição
headers = {
    'accept': 'application/json',
    'x-api-key': X_API_KEY
}

# Set up the logging configuration to write errors to a file
logging.basicConfig(filename='error_log.txt', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def make_request(url, params=None):
    """Function to make the GET request and return the response, with error handling."""
    try:
        # Send the GET request
        response = requests.get(url, headers=headers, params=params)
        
        # Check if the status code is OK (200)
        if response.status_code == 200:
            return response.json()
        else:
            # Log the error if the status code is not 200
            error_message = f"Erro {response.status_code}: {response.text}"
            logging.error(f"Request failed for URL: {url} with params: {params}. {error_message}")
            return None  # Return None if the request fails
    except requests.exceptions.RequestException as e:
        # Catch any request-related errors (e.g., connection error, timeout, etc.)
        logging.error(f"Request failed for URL: {url} with params: {params}. Error: {str(e)}")
        return None  # Return None if the request fails due to an exception




def get_estacoes(tipo='todas', regiao=None, estado=None):
    """Obter todas as estações disponíveis."""
    params = {'tipo': tipo}
    if regiao:
        params['regiao'] = regiao
    if estado:
        params['estado'] = estado
    
    url = BASE_URL
    return make_request(url, params)

def get_estacao_atributos(cod_estacao, periodo='horario', agrupar_por='intervalo'):
    """Obter lista de atributos de uma estação específica."""
    url = f'{BASE_URL}/{cod_estacao}/atributos'
    params = {'periodo': periodo, 'agruparPor': agrupar_por}
    return make_request(url, params)

def get_fenomeno(cod_estacao, cod_fenomeno, data_inicio, data_final):
    """Obter medição de um fenômeno específico."""
    url = f'{BASE_URL}/{cod_estacao}/fenomenos/{cod_fenomeno}'
    params = {'dataInicio': data_inicio, 'dataFinal': data_final}
    return make_request(url, params)

def comparar_estacoes_atributos(cod_estacoes, periodo='horario', agrupar_por='classe'):
    """Obter atributos comuns entre um grupo de estações."""
    url = f'{BASE_URL}/comparar/atributos'
    params = {'codEstacoes': ','.join(cod_estacoes), 'periodo': periodo, 'agruparPor': agrupar_por}
    return make_request(url, params)

# Function to attempt the API request and retry if necessary
def fetch_fenomeno_with_retry(cod_est, codigo, nome, retries=MAX_RETRIES):
    attempt = 0
    while attempt < retries:
        try:
            # Try to fetch the data
            fenomeno_data = get_fenomeno(cod_estacao=cod_est, cod_fenomeno=codigo, data_inicio='2008-01-01', data_final='2024-12-31')
            
            if fenomeno_data is not None:
                # If the data is valid, return it
                return fenomeno_data
            else:
                # If the data is None, log the error and retry
                logging.error(f"No data returned for {cod_est} - {codigo} - {nome}. Attempt {attempt + 1} of {retries}.")
        
        except Exception as e:
            # If an exception occurs (e.g., network error, timeout), log the error and retry
            logging.error(f"Error fetching data for {cod_est} - {codigo} - {nome}. Attempt {attempt + 1} of {retries}. Error: {str(e)}")
        
        # Wait before retrying
        time.sleep(RETRY_DELAY)
        attempt += 1
    
    # After retries, if no valid data is returned, return None
    logging.error(f"Failed to fetch data for {cod_est} - {codigo} - {nome} after {retries} attempts.")
    return None

# Função para calcular a distância usando a fórmula de Haversine em vetor
def haversine_vectorized(lat1, lon1, lat2, lon2):
    # Raio da Terra em quilômetros
    R = 6371.0
    
    # Converter as latitudes e longitudes de graus para radianos
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Diferença de latitudes e longitudes
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    
    # Fórmula de Haversine em vetor
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Distância em quilômetros
    distance = R * c
    return distance



# Testes
if __name__ == '__main__':
    try:
        # Getting the station data for each state
        estacoes_SP = get_estacoes(tipo='todas', estado='SP')
        estacoes_MG = get_estacoes(tipo='todas', estado='MG')
        estacoes_RJ = get_estacoes(tipo='todas', estado='RJ')
        estacoes_SC = get_estacoes(tipo='todas', estado='SC')
        estacoes_PR = get_estacoes(tipo='todas', estado='PR')
        estacoes_GO = get_estacoes(tipo='todas', estado='GO')
        estacoes_MS = get_estacoes(tipo='todas', estado='MS')

        # Convert each state's data to DataFrames
        df_SP = pd.DataFrame(estacoes_SP['data'])
        df_MG = pd.DataFrame(estacoes_MG['data'])
        df_RJ = pd.DataFrame(estacoes_RJ['data'])
        df_SC = pd.DataFrame(estacoes_SC['data'])
        df_PR = pd.DataFrame(estacoes_PR['data'])
        df_GO = pd.DataFrame(estacoes_GO['data'])
        df_MS = pd.DataFrame(estacoes_MS['data'])

        # Concatenate all DataFrames into one
        df_all_estacoes = pd.concat([df_SP, df_MG, df_RJ, df_SC, df_PR, df_GO, df_MS], ignore_index=True)
        print(df_all_estacoes.head(3))

        df_all_estacoes = df_all_estacoes[( (df_all_estacoes['dataFimOperacao'] >= '2008-01-01') | (df_all_estacoes['dataFimOperacao'].isna()) )]

        shapeM = gpd.read_file('../data/raw/shapefiles/SP_Municipios_2022.shp')
        LON = shapeM['xcentroide'].values
        LAT = shapeM['ycentroide'].values
            
            
        # Número de estacoes
        n_stations = len(df_all_estacoes) 
        n_municipios = len(LAT)

        # Extrair latitudes e longitudes do DataFrame
        latitudes  = df_all_estacoes['latitude'].values.astype(float)
        longitudes = df_all_estacoes['longitude'].values.astype(float)

        # Criar uma matriz de distâncias x5 usando broadcasting
        distances = np.zeros((n_municipios, n_stations))
        print(LAT[0], LON[0], latitudes, longitudes)

        # Usando broadcasting para calcular todas as distâncias
        for i in range( n_municipios ):
            distances[i] = haversine_vectorized(LAT[i], LON[i], latitudes, longitudes)

        print(distances)
        
        # Precomputando 1/distances
        inv_distances = 1 / distances

        # Exibir a matriz de distâncias
        distance_df = pd.DataFrame(distances, columns=df_all_estacoes['codEstacao'], index=shapeM['CD_MUN'])


        distance_df_filtered = distance_df.loc[:, distance_df.min() <= 280]


        # Ensure that 'CODIGO(WMO)' exists as a column in df_stations_final
        filtered_df_stations = df_all_estacoes[df_all_estacoes['codEstacao'].isin(distance_df_filtered.columns)]

        print(filtered_df_stations.head(3))
        

        variables_fenomeno = {'I175': 'PRECIPITACAO TOTAL, HORARIO',
        'I106': 'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA',
        'I615': 'PRESSAO ATMOSFERICA MAX.NA HORA ANT. (AUT)',
        'I616': 'PRESSAO ATMOSFERICA MIN. NA HORA ANT. (AUT)',
        'I133': 'RADIACAO GLOBAL',
        'I101': 'TEMPERATURA DO AR - BULBO SECO, HORARIA',
        'I611': 'TEMPERATURA MAXIMA NA HORA ANT. (AUT)',
        'I612': 'TEMPERATURA MINIMA NA HORA ANT. (AUT)',
        'I617': 'UMIDADE REL. MAX. NA HORA ANT. (AUT)',
        'I618': 'UMIDADE REL. MIN. NA HORA ANT. (AUT)',
        'I105': 'UMIDADE RELATIVA DO AR, HORARIA',
        'I608': 'VENTO, RAJADA MAXIMA',
        'I111': 'VENTO, VELOCIDADE HORARIA'}


        # Set up logging for error handling
        logging.basicConfig(filename='fenomeno_error_log.txt', level=logging.ERROR,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        # Iterate over each station in the filtered list of stations
        for cod_est in filtered_df_stations['codEstacao'][287:]:
            
            # Directory where files will be saved
            output_dir = f'data/raw/{cod_est}'

            # Create the directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a list to collect the file paths for merging later
            file_paths = []

            # Iterate over the 'variables_fenomeno' dictionary to fetch data and save each phenomenon to a file
            for codigo, nome in variables_fenomeno.items():
                # Fetch the phenomenon data for the specific station and date range with retry
                fenomeno_data = fetch_fenomeno_with_retry(cod_est, codigo, nome)
                
                if fenomeno_data is not None:
                    # Convert the fetched data into a DataFrame
                    fenomeno_data_df = pd.DataFrame(fenomeno_data['data']['data'], columns=['HORA UTC', nome])

                    # Convert the 'timestamp' column from milliseconds to a proper datetime
                    fenomeno_data_df['HORA UTC'] = pd.to_datetime(fenomeno_data_df['HORA UTC'], unit='ms')

                    # Save the phenomenon data to a CSV file
                    file_path = os.path.join(output_dir, f'{cod_est}_{codigo}_{nome}.csv')
                    fenomeno_data_df.to_csv(file_path, index=False)

                    # Append the file path for later use
                    file_paths.append(file_path)
                else:
                    # If no data returned after retries, skip to the next phenomenon
                    logging.error(f"Skipping {cod_est} - {codigo} - {nome} due to failed data fetch.")
                    continue
            
            # List to hold all DataFrames
            data_frames = []

            # Read and process each file into DataFrames
            for file_path in file_paths:
                try:
                    df = pd.read_csv(file_path)

                    # Convert 'HORA UTC' column to datetime if it's not already
                    if df['HORA UTC'].dtype == 'O':  # If 'HORA UTC' is a string
                        df['HORA UTC'] = pd.to_datetime(df['HORA UTC'])

                    # Set 'HORA UTC' as the index to align by time
                    df.set_index('HORA UTC', inplace=True)
                    
                    # Add the DataFrame to the list
                    data_frames.append(df)

                except Exception as e:
                    logging.error(f"Error reading file {file_path}. Error: {str(e)}")
                    continue

            # If no data was successfully processed, skip to the next station
            if not data_frames:
                logging.error(f"No valid data for station {cod_est}. Skipping to the next station.")
                continue

            # Extract the indices (timestamps) from each DataFrame and concatenate them as lists
            all_indices = pd.to_datetime(pd.concat([df.index.to_series() for df in data_frames]).unique())

            # Reindex all DataFrames to the union of all timestamps (outer join)
            aligned_data_frames = [df.reindex(all_indices) for df in data_frames]

            # Concatenate all DataFrames along the columns (axis=1)
            fenomeno = pd.concat(aligned_data_frames, axis=1)

            # Save the final DataFrame to a CSV file
            final_file_path = os.path.join(output_dir, f'{cod_est}_final.csv')
            fenomeno.to_csv(final_file_path, index=True)

            print(f"Final data for station {cod_est} saved to {final_file_path}")

    
    except Exception as e:
        print(f"Ocorreu um erro: {e}")