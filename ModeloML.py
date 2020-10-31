# Clase Modelo de ML que engloba todos los modelos de ML (Regresion Lineal, Random Forest, ...)
from DataBase import DataBase
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVR

class ModeloML():
    def __init__(self, dataBase):
        if(dataBase):
            self.data_preparada = dataBase
        else:
            self.data_preparada = DataBase()
        
    # Funcion para mostrar los datos del modelo predictivo
    def display_scores(self, scores):
        print("Scores:", scores)
        print("Mean:", scores.mean())
        print("Standard deviation:", scores.std())
    
    #regresion lineal: entrenamos el modelo
    def RegresionLineal(self):
        #self.data_preparada = database
        lin_reg = LinearRegression()
        lin_reg.fit(self.data_preparada.get_data_prepared(), self.data_preparada.get_data_labels())
        #Prediccion del modelo de Regresion Lineal
        some_data = self.data_preparada.get_data().iloc[:5]
        some_labels = self.data_preparada.get_data_labels().iloc[:5]
        some_data_prepared = self.data_preparada.get_full_pipeline().transform(some_data)
        print("Predictions:\t", lin_reg.predict(some_data_prepared))
        print('Compare against the actual values:')
        print("Labels:\t\t", list(some_labels))
        # Medir el RMSE(medida de rendimiento preferida para tareas de regresión)
        # en todo el conjunto de entrenamiento 
        data_predictions = lin_reg.predict(self.data_preparada.get_data_prepared())
        lin_mse = mean_squared_error(self.data_preparada.get_data_labels(), data_predictions)
        # función de costo medida en el conjunto de ejemplos usando su hipótesis h(data prediction).
        lin_rmse = np.sqrt(lin_mse)
        print('RMSE para LR es: {}'.format(lin_rmse))

        lin_mae = mean_absolute_error(self.data_preparada.get_data_labels(), data_predictions)
        print('El valor mean absolut error es: {}'.format(lin_mae))
        return lin_rmse, lin_mae, lin_reg
    
    # DecisionTreeRegressor: 
    def DecisionTreeRegressor(self):
        #self.data_preparada = database            
        tree_reg = DecisionTreeRegressor()
        tree_reg.fit(self.data_preparada.get_data_prepared(), self.data_preparada.get_data_labels())
        #Prediccion del modelo de Decision Tree Regressor
        some_data = self.data_preparada.get_data().iloc[:5]
        some_labels = self.data_preparada.get_data_labels().iloc[:5]
        some_data_prepared = self.data_preparada.get_full_pipeline().transform(some_data)
        print('\n')
        print("Predictions:\t", tree_reg.predict(some_data_prepared))
        print("Labels:\t\t", list(some_labels))
        # Medir el RMSE(medida de rendimiento preferida para tareas de regresión)
        # en todo el conjunto de entrenamiento 
        data_predictions = tree_reg.predict(self.data_preparada.get_data_prepared())
        tree_mse = mean_squared_error(self.data_preparada.get_data_labels(), data_predictions)
        tree_rmse = np.sqrt(tree_mse)
        print('RMSE para DTR es: {}'.format(tree_rmse))
        return tree_rmse, tree_reg
    
    # Better Evaluation Using Cross-Validation
    # Evaluamos el más eficiente entre tree_reg y lin_reg
    def CrossValidation(self, tree_reg, lin_reg):
        scores = cross_val_score(tree_reg, self.data_preparada.get_data_prepared(), 
                                 self.data_preparada.get_data_labels(),scoring="neg_mean_squared_error", cv=10)
        tree_rmse_scores = np.sqrt(-scores)
        print('\n')
        self.display_scores(tree_rmse_scores)

        lin_scores = cross_val_score(lin_reg, self.data_preparada.get_data_prepared(), self.data_preparada.get_data_labels(),
                                     scoring="neg_mean_squared_error", cv=10)
        lin_rmse_scores = np.sqrt(-lin_scores)
        self.display_scores(lin_rmse_scores)
    
    #Random Forest Model
    def RandomForest(self):
        forest_reg = RandomForestRegressor()
        forest_reg.fit(self.data_preparada.get_data_prepared(), self.data_preparada.get_data_labels())
        #  Prediccion Random Forest
        some_data_rf = self.data_preparada.get_data().iloc[:5]
        some_labels_rf = self.data_preparada.get_data_labels().iloc[:5]
        some_data_prepared_rf = self.data_preparada.get_full_pipeline().transform(some_data_rf)
        print('\n***************** RANDOM FOREST ****************\n')
        print("Predictions:\t", forest_reg.predict(some_data_prepared_rf))
        print("Labels:\t\t", list(some_labels_rf))

        forest_scores = cross_val_score(forest_reg, self.data_preparada.get_data_prepared(), self.data_preparada.get_data_labels(),
                                        scoring="neg_mean_squared_error", cv=10)
        forest_rmse_scores = np.sqrt(-forest_scores)
        # medicion RMSE de la Prediccion
        data_predictions = forest_reg.predict(self.data_preparada.get_data_prepared())
        forest_mse = mean_squared_error(self.data_preparada.get_data_labels(), data_predictions)
        forest_rmse = np.sqrt(forest_mse)
        self.display_scores(forest_rmse_scores)
        return forest_rmse, forest_reg
    
    # Guardar el modelo de Scikit-Learn
    def GuardarModelo(self, modelo, nombre):
        filename = "_model.pkl"
        filename = nombre + filename    
        joblib.dump(modelo, filename)
        my_model_loaded = joblib.load(filename)
        print("Modelo: ", filename, "ha sido guardado con exito! :)")
        
    # Evaluacion con Grid Search para afinar el modelo
    # Se evalua el Random Forest usando hiperparametros, busca todas las combbinaciones posibles
    def GridSearch(self):
        param_grid = [
            {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}, 
            {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
        ]
        forest_reg = RandomForestRegressor()
        grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(self.data_preparada.get_data_prepared(), self.data_preparada.get_data_labels())
        print('Mejor combinacion de parametros es: {}'.format(grid_search.best_params_))
        # Mejor estimador
        print('Mejor estimador es: {}'.format(grid_search.best_estimator_))
        cvres = grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print('The RMSE score for the combination ({}) is: '.format(params,np.sqrt(-mean_score)))
    
    def RandomizedSearch(self, database):
        self.data_preparada = database
        param_grid = [
            {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}, 
            {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
        ]
        forest_reg = RandomForestRegressor()
        grid_search = RandomizedSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(self.data_preparada.get_data_prepared(), self.data_preparada.get_data_labels())
        # Mejor combinacion de parametros
        #grid_search.best_params_
        # Mejor estimador
        #grid_search.best_estimator_
        cvres = grid_search.cv_results_
        list_score_params = list()
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            #print(np.sqrt(-mean_score), params)
            list_score_params.append([np.sqrt(-mean_score), params])
        #Analizar el mejor modelo
        feature_importances = grid_search.best_estimator_.feature_importances_
        # Vamos a mostrar estas puntuaciones de importancia junto a los correspondientes nombres de los atributos
        extra_attribs = ["presion", "humedad"]
        encoder = self.data_preparada.get_encoder()
        cat_one_hot_attribs = list(encoder.classes_)
        num_attribs = list(self.data_preparada.get_data_num())
        attributes = num_attribs + extra_attribs + cat_one_hot_attribs
        puntuacion_atributo = sorted(zip(feature_importances, attributes), reverse=True)
        #se imprime la importancia de los atributos
        #for element in puntuacion_atributo:
        #    print(element)
        # Evaluar el modelo final en el conjunto de prueba. 
        # - Obtener los predictores y las etiquetas de su conjunto de prueba, 
        # - Ejecute el pipeline completo para transformar los datos (¡llame a transform (), no fit_transform ()!)
        final_model = grid_search.best_estimator_
        X_test = self.data_preparada.get_strat_test_set().drop("T", axis=1)
        y_test = self.data_preparada.get_strat_test_set()["T"].copy()
        X_test_prepared = self.data_preparada.get_full_pipeline().transform(X_test)
        final_predictions = final_model.predict(X_test_prepared)
        final_mse = mean_squared_error(y_test, final_predictions)
        final_rmse = np.sqrt(final_mse) # => evaluates to 48,209.6
        return final_rmse
    
    def SVM(self, database):
        self.data_preparada = database
        svm_reg = SVR(kernel='linear')
        svm_reg.fit(self.data_preparada.get_data_prepared(), self.data_preparada.get_data_labels())
        data_predictions = svm_reg.predict(self.data_preparada.get_data_prepared())
        svm_scores = cross_val_score(svm_reg, self.data_preparada.get_data_prepared(), self.data_preparada.get_data_labels(),
                                     scoring="neg_mean_squared_error", cv=10)
        svm_rmse_scores = np.sqrt(-svm_scores)
        svm_rmse_scores.mean()
        svm_rmse_scores.std()