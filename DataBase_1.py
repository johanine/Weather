# Clase DataBase encargada de manejar la data (creacion, descarga y demás).
import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from DataFrameSelector import DataFrameSelector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from CombinedAttributesAdder import CombinedAttributesAdder
from datetime import datetime
from dateutil import parser, rrule
from sklearn.compose import ColumnTransformer

#from dateutil.rrule import YEARLY, MONTHLY, WEEKLY, DAILY
#from dateutil.rrule import HOURLY, MINUTELY, SECONDLY

import xlrd
import csv
import requests


class DataBase():
    def __init__(self):
        # Atributo de instancia
        self.DATABASE_PATH = "datasets"
        self.IMAGES_PATH = "images"
        self.full_pipeline = None
        self.data = None
        self.data_labels = None
        self.data_num = None
        self.data_prepared = None
        self.strat_test_set = None
        self.strat_train_set = None#
        #
        self.train_set = None
        self.test_set = None
        self.fechaInicio = None
        self.fechaFinal = None
    
    def get_data(self):
        return self.data

    def get_train_set(self):
        return self.train_set

    def get_test_set(self):
        return self.test_set
    
    def get_data_labels(self):
        return self.data_labels
    
    def get_data_prepared(self):
        return self.data_prepared
    
    def get_encoder(self):
        return self.encoder
    
    def get_data_num(self):
        return self.data_num
    
    def get_full_pipeline(self):
        return self.full_pipeline
    
    def get_strat_test_set(self):
        return self.strat_test_set

    def get_strat_train_set(self):
        return self.strat_train_set
    
    # Funcion que recupera la data en formato json
    def getWeatherData(self,url,x):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:74.0) Gecko/20100101 Firefox/74.0'
        }
        try:
            r = requests.get(url, headers=headers).json()
            y = []
            for item in r['metadata']:
                df = pd.DataFrame.from_dict([item])
                y.append(df)
            if(r['observations']):
                for item in r['observations']:
                    df = pd.DataFrame.from_dict([item])
                    x.append(df)
            else:
                print(url)
        except requests.exceptions.RequestException as e:  # This is the correct syntax
            raise SystemExit(e)
        except Exception as ex:
            print(ex)
    
    # Funcion que almacena los datos en .csv
    def saveData(self,station, fechaInicio, fechaFinal):
        # Generate a list of all of the dates we want data for
        start_date = fechaInicio
        end_date = fechaFinal
        start = parser.parse(start_date)
        end = parser.parse(end_date)
        dates = list(rrule.rrule(rrule.DAILY, dtstart=start, until=end))
        weather_data=[]
        backoff_time = 10
        for date in dates:
            day = date.day
            month = date.month
            year = date.year
            url = "https://api.weather.com/v1/location/{station}/observations/historical.json?apiKey=6532d6454b8aa370768e63d6ba5a832e&units=m&startDate={year}"
            if(month<10):
                url = url +"0{month}"
            else:
                url = url +"{month}"
            if(day<10):
                url = url + "0{day}"
            else:
                url = url + "{day}"

            full_url = url.format(station=station, day=day, month=month, year=year)
            
            if date.day % 10 == 0:
                print("Working on date: {} for station {}".format(date, station))            
            done = False
            while done == False:
                try:
                    self.getWeatherData(full_url, weather_data)
                    done = True
                except ConnectionError as e:
                    # May get rate limited by Wunderground.com, backoff if so.
                    print("Got connection error on {}".format(date))
                    print("Will retry in {} seconds".format(backoff_time))
                    time.sleep(10)
        new = pd.concat(weather_data, ignore_index=True)
        new.to_csv("datasets/{}_weather.csv".format(station.replace(":","_")))
        print(" Guardado con Exito en: datasets/{}_weather.csv".format(station.replace(":","_")))

    # Descarga de un servidor por cada estacion desde la fecha de Inicio a la fecha final
    def fetch_housing_data(self, stations,fechaInicio,fechaFinal):
        database_path=self.DATABASE_PATH
        # crea la carpeta de descarga si no existe.
        if not os.path.isdir(database_path):
            os.makedirs(database_path)
        # asignacion del nombre y ruta de descarga de la database
        for station in stations:
            print("Working on {}".format(station))
            self.saveData(station,fechaInicio,fechaFinal)
        #tgz_path = os.path.join(database_path, "weather.tgz")
        # descarga del archivo de URL a nuestra pc
        #urllib.request.urlretrieve(database_url, tgz_path)
        # descomprimimos el documento .tgz
        # database_tgz = tarfile.open(tgz_path)
        # housing_tgz.extractall(path=housing_path)
        # housing_tgz.close()

    # Cargar el database.
    def load_weather_data(self,archivo):
        database_path = self.DATABASE_PATH
        csv_path = os.path.join(database_path, archivo)
        return pd.read_csv(csv_path)
    
    # Carga la data desde un archivo excel
    def load_weather_data_excel(self):
        database_path = self.DATABASE_PATH
        excel_path = os.path.join(database_path, "cusco.xlsx")
        
        wb = xlrd.open_workbook(excel_path)
        sh = wb.sheet_by_name('Hoja1')
        csv_path = os.path.join(database_path, "cusco1.csv")
        your_csv_file = open(csv_path, 'w')
        wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)
        for rownum in range(sh.nrows):
            wr.writerow(sh.row_values(rownum))        
        your_csv_file.close()
        return pd.read_csv(csv_path)
    # Funcion que carga data 
    # caso 0: La data se carga desde un archivo local csv
    # caso 1: La data se descarga de Internet y almacena en un csv
    def cargar_data(self,caso,stations,fechaInicio, fechaFinal,archivo):
        if  (caso==1):
            self.data = self.fetch_housing_data(stations, fechaInicio, fechaFinal)
            if(len(stations) > archivo):
                nombre = "{}_weather.csv".format(stations[archivo].replace(":","_"))
                self.data = self.load_weather_data(nombre)
        elif (caso ==0):
            if (len(stations) > archivo):
                nombre = "{}_weather.csv".format(stations[archivo].replace(":","_"))
                self.data = self.load_weather_data(nombre)
                self.fechaInicio = fechaInicio
                self.fechaFinal = fechaFinal
        return self.data
    
    # funcion que muestra las proporciones de la columna en la data
    def col_cat_proportions(self, data, col):
        return data[col].value_counts() / len(data)
    
    def mostrar_datos(self, caso, col):
        caso = caso.upper();
        
        if( caso == 'HEAD'):
            print(self.data.head())
        elif caso == 'INFO':
            print(self.data.info())
        elif caso == 'VALUE_COUNT':
            # muestra los diferentes valores en la columna
            print(self.data[col].value_counts())
        elif caso == 'DESCRIBE':
            print(self.data.describe())
        elif caso == 'HISTOGRAMA':
            self.data.hist(bins=50, figsize=(20,15))
            self.save_fig("attribute_histogram_plots")
            plt.show()
        elif caso=='CONJUNTOS':
            print("tamaño de los conjuntos: ",len(self.train_set), "train +", len(self.test_set), "test")
        elif caso == "TEST_HEAD":
            print(self.test_set.head())
        elif caso == "HISTOGRAMA_COL":
            self.data[col].hist()
        elif caso == "STRAT_TEST":
            print(self.strat_test_set[col].value_counts() / len(self.strat_test_set))
        elif caso == "DATA_PROPORCION":
            print(self.data[col].value_counts() / len(self.data))
        elif caso =="OVERALL":
            if col in self.data.columns:
                self.train_set, self.test_set = train_test_split(self.data, test_size=0.2, random_state=42)
                compare_props = pd.DataFrame({
                    "Overall": self.col_cat_proportions(self.data,col),
                    "Stratified": self.col_cat_proportions(self.strat_test_set,col),
                    "Random": self.col_cat_proportions(self.test_set,col),
                }).sort_index()
                compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
                compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
                print(compare_props)
            else:
                print('Columna {} no encontrada.'.format(col))
    
    #preparar test set (conjunto de prueba) simple con un split    
    def split_train_test(self, data, test_ratio):
        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return data.iloc[train_indices], data.iloc[test_indices]    
    
    def test_set_check(self, identifier, test_ratio, hash):
        return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio
    
    #preparar test set simple con un split
    def preparar_data_split(self):
        # conjunto de entrenamiento (100% data), conjunto de prueba (20%)
        self.train_set, self.test_set = self.split_train_test(self.data, 0.2)
        #print("tamaño de los conjuntos: ",len(self.train_set), "train +", len(self.test_set), "test")
    
    # preparar data con tablas e indices hash
    def split_train_test_by_id(self, data, test_ratio, id_column, hash=hashlib.md5):
        ids = data[id_column]
        in_test_set = ids.apply(lambda id_: self.test_set_check(id_, test_ratio, hash))
        return data.loc[~in_test_set], data.loc[in_test_set]
    
    def preparar_data_hash(self):
        data_with_id = self.data.reset_index() # adds an `index` column
        self.train_set, self.test_set = self.split_train_test_by_id(data_with_id, 0.2, "index")
        data_with_id["id"] = ((self.data["longitud"] * 100000) + (self.data["latitud"]*(-1)) + (self.data['year']*10000)+ (self.data['month']*1000)+(self.data['day']*100)+(self.data['hour']*10)+(self.data['minute']))
        
        #print("\n")
        #print(data_with_id.head(2))
        
        self.train_set, self.test_set = self.split_train_test_by_id(data_with_id, 0.2, "id")
        # muestra el nuevo dataframe con la data con el index y el id
        #print(test_set.head())

    def preparar_data_tts(self):
        self.train_set, self.test_set = train_test_split(self.data, test_size=0.2, random_state=42)
        #print(test_set.head())
        #print(self.data["temp"].hist())
        self.categorizar_data(1,'temp', 6.,-6,30.)
        self.categorizar_data(0,'humedad',17.5, 0., 0.)
        self.categorizar_data(0,'presion',6.6, 690., 696.3)
        #solo para un año
        #star = self.data['wind_speed'].min()-1
        #stop = self.data['wind_speed'].max()-90
        #self.categorizar_data(1,'wind_speed',3., star, stop)
        # para toda la data
        self.categorizar_data(1,'wind_speed',6.5, 0., 30.)
        print(self.data[self.data['wind_speed_cat'].isnull()])


    # funcion que categoriza una columna o (caracteristica de la data)
    # Si opcion es 0 => por defecto se establecen los datos min y max, cualquier otro se le da inicio y final
    # inc es el incremento con el categorizara
    def categorizar_data(self, opcion,col, inc, star,stop):
        if (opcion==0):
            star = self.data[col].min()-1
            stop = self.data[col].max()+1

        bins_, labels_= self.getBins(star,stop,inc)
        new_col = col +"_cat"
        self.data[new_col] = pd.cut(self.data[col], bins=bins_, labels=labels_)

        if self.data[new_col].isnull().values.any():
          print('Existen datos null en la categoria de la columna: {}, se pasara a reparar..'.format(new_col))
          last = self.data[new_col].max()
          self.data[new_col].fillna(last, inplace=True) # option 3        


    #return bins and labels for hist
    def getBins(self, star, stop, increment):
        bins = list(np.arange(star, stop, increment))
        bins.append(np.inf)
        labels = list(np.arange(1,len(bins)))
        return bins, labels
    
    def preparar_data_sss(self,col):
        if col in self.data.columns:
          split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
          for train_index, test_index in split.split(self.data, self.data[col]):
              self.strat_train_set = self.data.loc[train_index]
              self.strat_test_set = self.data.loc[test_index]
          
          #self.mostrar_datos('STRAT_TEST',col)
          #self.mostrar_datos('DATA_PROPORCION',col)
          #self.mostrar_datos('OVERALL',col)

          #restaurar la data
        else:
            print('No se encuentra la columna: {}'.format(col))
    
    #restaurar la data
    def restaurar_data(self):
        for col_name in self.strat_train_set.columns: 
            array = col_name.split('_')
            if array[len(array)-1] == 'cat' :
                col = col_name
                for set in (self.strat_train_set, self.strat_test_set):
                    set.drop([col], axis=1, inplace=True)
                self.data = self.strat_train_set.copy()
                #print('La Data ha sido restaurada')

    def preparar_data_sss_all(self):
        #self.mostrar_datos('VALUE_COUNT','temp_cat')
        #self.mostrar_datos('HISTOGRAMA_COL','temp_cat')
        self.preparar_data_sss('temp_cat')
        #self.mostrar_datos('VALUE_COUNT','humedad_cat')
        #self.mostrar_datos('HISTOGRAMA_COL','humedad_cat')
        self.preparar_data_sss('humedad_cat')
        #self.mostrar_datos('VALUE_COUNT','presion_cat')
        #self.mostrar_datos('HISTOGRAMA_COL','presion_cat')
        self.preparar_data_sss('presion_cat')
        #self.mostrar_datos('VALUE_COUNT','wind_speed_cat')
        #self.mostrar_datos('HISTOGRAMA_COL','wind_speed_cat')
        self.preparar_data_sss('wind_speed_cat')
        #restaurar la data
        self.restaurar_data()

    def show_plot(self, caso, data,nombre):
        if caso ==0:
            self.data.plot(kind="scatter", x="longitud", y="latitud")#
            self.save_fig("bad_visualization_plot")
            self.data.plot(kind="scatter", x="longitud", y="latitud", alpha=0.1)
            self.save_fig("better_visualization_plot")
        elif caso ==1:
            data.plot(kind="scatter", x="fecha", y="temp", alpha=0.4,
                s=data["wind_speed"], label="wind_speed", figsize=(10,7),
                c="presion", cmap=plt.get_cmap("jet"), colorbar=True,
                sharex=False)
            #data.plot("fecha", 'temp', linestyle='solid')

            plt.legend()
            self.save_fig(nombre)
            #plt.clf()
        elif caso ==2:
            data.plot(kind="scatter", x="fecha", y="temp", alpha=0.4,
                s=data["humedad"], label="humedad", figsize=(10,7),
                c="presion", cmap=plt.get_cmap("jet"), colorbar=True,
                sharex=False)
            #data.plot("fecha", 'temp', linestyle='solid')

            plt.legend()
            self.save_fig(nombre)


    # Muestra la data
    # muestra toda la data tipo = 0
    # muestra la data por año tipo = 1
    # muestra la data por mes tipo = 2,
    # muestra la data por semana tipo = 3,
    # muestra la data por dias tipo = 4
    # muestra la data por horas tipo = 5
    # muestra la data por minutos (media hora) tipo = 6
    def mostrar_data_preparada(self,tipo, fechaInicio, fechaFinal):
        try:
            if tipo == 0:
                nombre = "weather_{}-{}_all".format(fechaInicio, fechaFinal)
                self.show_plot(1,self.data, nombre)
            elif tipo == 1:
                start_date = fechaInicio
                end_date = fechaFinal
                start = parser.parse(start_date)
                end = parser.parse(end_date)
                dates = list(rrule.rrule(rrule.YEARLY, dtstart=start, until=end))
                for date in dates:
                    data_ = self.data.loc[(self.data['year'] == date.year)]
                    nombre = "weather_{}-{}_year".format(data_['fecha'].iloc[0].strftime("%d-%m-%Y"), data_['fecha'].iloc[-1].strftime("%d-%m-%Y"))
                    self.show_plot(1,data_,nombre)
                
            elif tipo==2:
                start_date = fechaInicio
                end_date = fechaFinal
                start = parser.parse(start_date)
                end = parser.parse(end_date)
                dates = list(rrule.rrule(rrule.MONTHLY, dtstart=start, until=end))
                for date in dates:
                    data_ = self.data.loc[(self.data['year'] == date.year) & (self.data['month'] == date.month)]
                    nombre = "weather_{}-{}_month".format(data_['fecha'].iloc[0].strftime("%d-%m-%Y"), data_['fecha'].iloc[-1].strftime("%d-%m-%Y"))
                    #self.show_plot(1,data_,nombre)
                    self.show_plot(2,data_,nombre)
            elif tipo ==3:
                start_date = fechaInicio
                end_date = fechaFinal
                start = parser.parse(start_date)
                end = parser.parse(end_date)
                dates = list(rrule.rrule(rrule.WEEKLY, dtstart=start, until=end))
                for date in dates:
                    data_ = self.data.loc[(self.data['year'] == date.year) & (self.data['month'] == date.month) & (self.data['day'] == date.day)]
                    nombre = "weather_{}-{}_semana".format(data_['fecha'].iloc[0].strftime("%d-%m-%Y"), data_['fecha'].iloc[-1].strftime("%d-%m-%Y"))
                    self.show_plot(1,data_,nombre)        
            elif tipo ==4:
                start_date = fechaInicio
                end_date = fechaFinal
                start = parser.parse(start_date)
                end = parser.parse(end_date)
                dates = list(rrule.rrule(rrule.DAILY, dtstart=start, until=end))
                for date in dates:
                    data_ = self.data.loc[(self.data['year'] == date.year) & (self.data['month'] == date.month) & (self.data['day'] == date.day)]
                    nombre = "weather_{}-{}_daily".format(data_['fecha'].iloc[0].strftime("%d-%m-%Y"), data_['fecha'].iloc[-1].strftime("%d-%m-%Y"))
                    self.show_plot(1,data_,nombre)
            elif tipo ==5:
                start_date = fechaInicio
                end_date = fechaFinal
                start = parser.parse(start_date)
                end = parser.parse(end_date)
                dates = list(rrule.rrule(rrule.HOURLY, dtstart=start, until=end))
                for date in dates:
                    data_ = self.data.loc[(self.data['year'] == date.year) & (self.data['month'] == date.month) & (self.data['day'] == date.day)  & (self.data['hour'] == date.hour)]
                    nombre = "weather_{}-{}_hour".format(data_['fecha'].iloc[0].strftime("%d-%m-%Y"), data_['fecha'].iloc[-1].strftime("%d-%m-%Y"))
                    self.show_plot(1,data_,nombre)
        except NameError:
            print("Variable x is not defined")
        except:
            print("Something else went wrong")
        
        
    # Filtramos la data evitando que se encuentren datos NaN
    # Limpiamos la data
    def filtrar_data(self,latitud,longitud,altitud):
        median = self.data["temp"].median()
        self.data["temp"].fillna(median, inplace=True)
        self.data["temp"] = [float(x) for x in self.data['temp']]
        self.data["latitud"] = float(latitud)
        self.data["longitud"] = float(longitud)
        self.data["altitud"] = float(altitud)

        self.data['fecha'] = [datetime.fromtimestamp(x) for x in self.data['valid_time_gmt']]
        self.data['day']= [x.day for x in self.data['fecha']]
        self.data['month']= [x.month for x in self.data['fecha']]
        self.data['year']= [x.year for x in self.data['fecha']]
        self.data['hour']= [x.hour for x in self.data['fecha']]
        self.data['minute']= [x.minute for x in self.data['fecha']]
        self.data["station"] = self.data["obs_id"]
        self.data["station_name"] = self.data["obs_name"]

        median = self.data["wdir_cardinal"].mode()
        for i in self.data.index:
            if pd.isnull(self.data['wdir_cardinal'][i]):
                self.data.at[i,'wdir_cardinal']= median[0]
        self.data["wind_direction"] = [str(x) for x in self.data['wdir_cardinal']] 

        median = self.data["wspd"].median()
        self.data["wspd"].fillna(median, inplace=True) # option 3        
        self.data["wind_speed"] = [float(x) for x in self.data['wspd']] 
        
        self.data["nubes"] = [str(x) for x in self.data['clds']] 
        
        median = self.data["rh"].median()
        self.data["rh"].fillna(median, inplace=True)
        self.data["humedad"] = [float(x) for x in self.data['rh']]
        #
        median = self.data["pressure"].median()
        self.data["pressure"].fillna(median, inplace=True)
        self.data["presion"] = [float(x) for x in self.data['pressure']]
        #
        self.data["total_rain"] = [float(x) for x in self.data['precip_total']]
        self.data["total_rain"].fillna(0, inplace=True) # option 3
        self.data["hour_rain"] = [float(x) for x in self.data['precip_hrly']]
        self.data["hour_rain"].fillna(0, inplace=True)
        # Indice de continentalidad Ic = TMaxima - Tmin
        #self.data["Ic"] = self.data["TM"] - self.data["Tm"]
        #start_date = self.fechaInicio
        #end_date = self.fechaFinal
        #start = parser.parse(start_date)
        #end = parser.parse(end_date)
        #dates = list(rrule.rrule(rrule.DAILY, dtstart=start, until=end))
        
        #for date in dates:
            #self.data = self.data.loc[(self.data['year'] == date.year)]
        #
        #
        median = self.data["dewPt"].median()
        self.data["dewPt"].fillna(median, inplace=True)
        self.data["rocio"] = [float(x) for x in self.data['dewPt']]
        ##
        
        self.data = self.data.loc[:, 
            ['latitud','longitud','altitud','station','station_name',
            'fecha','year', 'month', 'day','hour','minute',
            'temp','wind_direction','wind_speed','nubes','humedad',
            'presion','total_rain','hour_rain','rocio']
        ]

    # Filtra la data y reemplaza la data faltante con la media
    def clean_data(self,sample_incomplete_rows):
        imputer = SimpleImputer(strategy="median")
        # data_num no contiene las columnas no numericas
        self.data_num = self.data
        self.data_num = self.data_num.drop("fecha", axis=1)
        self.data_num = self.data_num.drop("station", axis=1)
        self.data_num = self.data_num.drop("station_name", axis=1)
        self.data_num = self.data_num.drop("wind_direction", axis=1)
        self.data_num = self.data_num.drop("nubes", axis=1)
        # Ajustar la instancia del imputador a los datos de entrenamiento usando el método fit():
        imputer.fit(self.data_num)
        #print(imputer)
        #print(imputer.statistics_)
        # calcular la media para todas las columnas
        #print('Los valores medios calculados')
        #print(self.data_num.median().values)
        # Imputer se encuentra entrenado
        # Transformamos el conjunto de entrenamiento para reemplazar los valores faltantes con los valores del Imputer(media de cada columna)
        X = imputer.transform(self.data_num)
        data_tr = pd.DataFrame(X, columns=self.data_num.columns, index=self.data.index)
        if(len(sample_incomplete_rows)>0):
            value = data_tr.loc[sample_incomplete_rows.index.values]
            print(value)
        else:
            print('No hay valores faltantes en la data')
        print('La estrategia usada es: {}'.format(imputer.strategy))
        data_tr = pd.DataFrame(X, columns=self.data_num.columns,index=self.data_num.index)
        #print(data_tr.head())
        return data_tr
    # de string a numerico
    def convertString(self,atributo):
        encoder = LabelEncoder()
        data_cat = self.data[atributo]
        #print("Los valores (head) sin mapear de {} son: \n{}".format(atributo,data_cat.head(10)))
        data_cat_encoded = encoder.fit_transform(data_cat)
        #print('Los valores numericos son:   {}'.format(data_cat_encoded))
        #print('Los valores mapeados son:    {}'.format(encoder.classes_))
        self.encoder = encoder
    
    # Convierte datos del tipo texto categorico en numerico
    def convertCategory(self, atributos):
        data_cat = self.data[atributos]
        ordinal_encoder = OrdinalEncoder()
        data_cat_encoded = ordinal_encoder.fit_transform(data_cat)
        #print('Los (10 primeros) valores numericos para {} son:\n{}'.format(atributos[0],data_cat_encoded[:10]))
        #print('Las categorias para {} son:\n{}'.format(atributos[0],ordinal_encoder.categories_))

    # Evitar que los algoritmos ML supondrán que dos valores cercanos son más similares que dos valores distantes.
    # convertir valores categóricos en un one hot vector:
    # Alternativamente, puede establecer sparse = False al crear OneHotEncoder:
    def convertVector(self, atributos, sparse):
        data_cat = self.data[atributos]
        if(sparse):
            cat_encoder = OneHotEncoder()
            data_cat_1hot = cat_encoder.fit_transform(data_cat)
            #print('La data_cat con One Hoter para "{}" es: \n{}'.format(atributos[0],data_cat_1hot))
            # De forma predeterminada, la clase OneHotEncoder devuelve una matriz dispersa, 
            # pero podemos convertirla en una matriz densa si es necesario llamando al método toarray ():
            #print(data_cat_1hot.toarray())
        else:
            cat_encoder = OneHotEncoder(sparse=False)
            data_cat_1hot = cat_encoder.fit_transform(data_cat)
            #print('La data_cat con One Hoter para "{}" es: \n{}'.format(atributos[0],data_cat_1hot))
        # get list of categories using the encoders categories_ instance variable:
        #print('Las categorias para {} son:\n{}'.format(atributos[0],cat_encoder.categories_))

    def coeficienteCorrelacion(self):
        try:
            corr_matrix = self.data.corr()
            print('Matriz correlacion de temperatura')
            print(corr_matrix["temp"].sort_values(ascending=False))
            #Seleccionamos los atributos para calcular el coeficiente
            attributes = ["temp", "presion", "humedad","wind_speed"]
            scatter_matrix(self.data[attributes], figsize=(12, 8))
            self.save_fig("scatter_matrix_plot")
            # de la figura anterior se deduce que la presion y humedad son los atributos con mayor posibilidad
            # para el calculo y le hacemos zoom
            self.data.plot(kind="scatter", x="presion", y="temp",alpha=0.1)
            plt.axis([self.data['presion'].min(), self.data['presion'].max(), self.data['temp'].min(), self.data['temp'].max()])
            self.save_fig("presion_vs_temp_scatterplot")
            
            self.data.plot(kind="scatter", x="humedad", y="temp",alpha=0.1)
            plt.axis([self.data['humedad'].min(), self.data['humedad'].max(), self.data['temp'].min(), self.data['temp'].max()])
            self.save_fig("humedad_vs_temp_scatterplot")

            self.data.plot(kind="scatter", x="wind_speed", y="temp",alpha=0.1)
            plt.axis([self.data['wind_speed'].min(), self.data['wind_speed'].max(), self.data['temp'].min(), self.data['temp'].max()])
            self.save_fig("wind_speed_vs_temp_scatterplot")

            # Combinando atributos para mejor el coeficiente
            #self.data["population_per_household"]=self.data["population"]/self.data["households"]
            #corr_matrix = housing.corr()
            #corr_matrix["median_house_value"].sort_values(ascending=False)
        except NameError:
            print("Variable x is not defined")
        except:
            print("Something else went wrong")

    def preparar_data(self, latitud,longitud, altitud):
        # para que la salida de este portátil sea idéntica en cada ejecución
        np.random.seed(42)
        self.preparar_data_split()
        self.filtrar_data(latitud,longitud,altitud)
        self.preparar_data_hash()       
        self.preparar_data_tts()
        self.preparar_data_sss_all()
        #self.show_plot(0,None)
        #self.mostrar_data_preparada(0,"2018-01-01","2018-12-31")
        #self.mostrar_data_preparada(2,"2018-01-01","2018-04-30")
        #self.mostrar_data_preparada(3,"2018-01-01","2018-01-10")
        #self.mostrar_data_preparada(4,"2018-01-01 00:00:00","2018-01-01 12:00:00")
        # coeficiente de correlacion o Pearson's r
        #self.coeficienteCorrelacion()
        self.data = self.strat_train_set.drop("temp", axis=1)
        self.data_labels = self.strat_train_set["temp"].copy()
        #verifimanos si existe algun null en la data
        sample_incomplete_rows = self.data[self.data.isnull().any(axis=1)].head()
        #print(sample_incomplete_rows)
        #sample_incomplete_rows.dropna(subset=["wind_direction"])
        #print(sample_incomplete_rows)
        #sample_incomplete_rows.drop("wind_direction", axis=1)       # option 2
        #print(sample_incomplete_rows)
        #median = self.data["wind_direction"].median()
        #sample_incomplete_rows["wind_direction"].fillna(median, inplace=True) # option 3
        #print(sample_incomplete_rows['wind_direction'])
        # Si no se cuenta con datos para algun atributo
        #• Deshágase de los distritos correspondientes.
        #• Deshágase de todo el atributo.
        #• Establezca los valores en algún valor (cero, la media, la mediana, etc.).
        data_tr = self.clean_data(sample_incomplete_rows)
        # Convertir las columnas string to number
        atributos = [['fecha'],['wind_direction'],['nubes']]
        self.convertString(atributos[0][0])
        self.convertString(atributos[1][0])
        self.convertString(atributos[2][0])
        # usando Ordial Encoder
        print("\n")
        self.convertCategory(atributos[0])
        self.convertCategory(atributos[1])
        self.convertCategory(atributos[2])

        # usando One Hot Vector
        print("\n")
        self.convertVector(atributos[0], False)
        self.convertVector(atributos[1], False)
        self.convertVector(atributos[2], False)
        
    # Transformaciones Pipelines
    # Escalado de características: Una de las transformaciones más importantes que necesita aplicar a loss datos 
    # es el escalado de características.
    def trans_pipeline(self):       
        # small pipeline for the numerical attributes:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler()),
        ])
        data_num_tr = num_pipeline.fit_transform(self.data_num)
        #print('\n')
        #print(data_num_tr)
        
        # Aplicamos un solo transformador capaz de manejar todas las columnas, aplicando las transformaciones apropiadas a cada columna.
        num_attribs = list(self.data_num)
        cat_attribs = ['fecha','wind_direction','nubes']

        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])
        
        self.data_prepared = full_pipeline.fit_transform(self.data)
        self.full_pipeline = full_pipeline

        #print('\n Data Preparada')
        #print(self.data_prepared)
        #print(self.data_prepared.shape)

        
        
        
    # Guardar imagenes
    def save_fig(self, fig_id, tight_layout=True, fig_extension="png", resolution=300):
        ruta_imagenes = self.IMAGES_PATH
        path = os.path.join(ruta_imagenes, fig_id + "." + fig_extension)
        print("Saving figure", fig_id)
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)



    def get_part_data(self,columnas=[], rows=10, tipo='all', desde=''):
      data = self.data
      if desde !='':
        fila = data.loc[(data['fecha'] == desde)]
        indice = next(iter(data[data['fecha']==desde].index), 'no match')
        if indice !="no match":
          data = data[indice:len(data)]
          data = data.reset_index(drop=True)
        else: 
          print('Error: Fecha no encontrada')

      if tipo == "day":
        last = data.iloc[-1]['fecha']
        fin = datetime.strptime(str(last), '%Y-%m-%d %H:%M:%S').date()
        
        start_date = desde
        end_date = fin.strftime('%Y/%m/%d')
        dates = self.get_lista_fechas(start_date,end_date,'day')
        
        if rows <= len(dates):
          del dates[rows:]
        else:
          print('El numero de filas solicitadas {} es mayor al tamaño del dataset {}'.format(row, len(dates)))

      if len(columnas)>0:
        if not "fecha" in columnas:
          columnas.insert(0,"fecha")
        data = data[columnas]

      lista = []
      for col in columnas:
        if col != "fecha":
          lista = []
          data['check']= 0
          new_col = "mean"+col
          data[new_col] = data[col].median()

          for date in dates:
            date1 = datetime(date.year, date.month, date.day)
            list_indices =[]
            recorrer = True
            for i in data.index:
              if data.iloc[i]['check']==0:
                timestamp = data.iloc[i]['fecha']
                date2 = datetime(timestamp.year, timestamp.month, timestamp.day)
                if date1 == date2:
                  list_indices.insert(0,i)
                  data.at[i,"check"]= 1
                elif date1 < date2:
                  recorrer = False
              if not recorrer:
                break

            inicio = list_indices[-1]
            fin = list_indices[0]
            lista.append([inicio,fin])
            data_day = data[inicio:fin]
            median = data_day[col].median()
            #print('para la fecha: {} la {} media es {}'.format(date,col,median))

            for i in range(inicio,fin):
              data.at[i,new_col]= median

      data = data[0:lista[-1][1]+1]
      for ele in lista:
        a = ele[0]+1
        b = ele[1]+1
        for k in range(a,b):
          data.drop([k], inplace = True )

      temp = data.head(rows)
      temp = temp.reset_index(drop=True)

      for col in columnas:
        if col != "fecha":
          del temp[col]
      del temp['check']
      return temp

    def get_lista_fechas(self,start_date, end_date, tipo):
      start = parser.parse(start_date)
      end = parser.parse(end_date)
      if tipo == "day":
        dates = list(rrule.rrule(rrule.DAILY, dtstart=start, until=end))

      return dates

    def get_fecha(self,fecha,date=True, time = False):
      value =""
      if date and time:
        value_ = datetime.strptime(str(fecha), '%Y-%m-%d %H:%M:%S')
        value = value_.strftime('%Y-%m-%d %H:%M:%S')
      elif date and not time:
        value_ = datetime.strptime(str(fecha), '%Y-%m-%d %H:%M:%S').date()
        value = value_.strftime('%Y-%m-%d')
      
      return value
    
    def get_n_dia_before(self,data,features,n=3):
      # target measurement of mean temperature
      for feature in features:
        if feature != 'fecha':
            # 1 day prior
            for N in range(1, n+1):
                data = self.derive_nth_day_feature(data, feature, N)
      print(data.columns)
      return data

    def derive_nth_day_feature(self,df, feature, N):
      # total number of rows
      rows = df.shape[0]
      # a list representing Nth prior measurements of feature
      # notice that the front of the list needs to be padded with N
      # None values to maintain the constistent rows length for each N
      nth_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]
      # make a new column name of feature_N and add to DataFrame
      col_name = "{}_{}".format(feature, N)
      df[col_name] = nth_prior_measurements
      return df
      
