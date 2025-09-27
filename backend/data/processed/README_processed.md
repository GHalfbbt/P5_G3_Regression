### 📊 Dataset

En esta carpeta se dejarán de manera provisional, los dataset construidos en el EDA (cse crean a través de la ejecución del notebook EDA_processed.ipynb).

El resto de notebooks de los algoritmos de regresión analizados, usaran estos datasets en función de si necesitan datos escalados o no.


Creamos dos versiones del dataset original:  
   - `featuress_no_scaling.csv`: para modelos robustos a la escala (árboles, Random Forest, XGBoost).  
   - `featuress_scaled`: aplicamos `StandardScaler`, necesario en modelos sensibles a magnitud (Regresión Lineal, KNN, SVM).
   - `target_y`: dataframe con la variable objetivo `Life expectancy`
