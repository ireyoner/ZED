
##Raport z zadania 'Uczenie Maszynowe'
####Autorzy:
Michał Bartecki 106518 

Marcin Błaszyk  106616
###Biblioteki i przygotowanie do pracy

W celu przygotowania do pracy należy umieścić dane wejściowe w katalogu data oraz zainstalować wymagane biblioteki.

W projekcie korzystaliśmy z python'a w wersji 3.5.x


```python
import os.path
import pandas as pd
import csv
import numpy as np
from sklearn.externals import joblib
#import joblib
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import time

RANDOM_STATE = 42

NA_FILL = 0# -999999999999999
```

##Wczytanie i oczyszczenie danych
Na początku przeprowadziliśmy wczytanie i oczyszczenie danych uczących.
Istotne jest by pamiętać o ustawieniu wektora na_values=["nan"] oraz zastosowaniu argumentu keep_default_na =False. W przeciwnym wypadku biblioteka pandas zamieni wszystkie wystąpienia klasy NA na wartości puste.


```python
def preprocess_data(df):
    lebels_to_delete =[ 'DA','DC','DT','DU','DG','DI','UNK','UNX','UNL','PR','PD','Y1','EU','N','15P','UQ','PX4','NAN']

    df_out = df[~df['res_name'].isin(lebels_to_delete)]

    #unikalne pary
    df_out = df_out.drop_duplicates(subset=['pdb_code','res_name'],keep='first')

    #minimum 5 wystapien
    df_out = df_out.groupby('res_name').filter(lambda x: len(x) >= 5)

    ##Usuwanie niepotrzebnych kolumn
    #local jakie chcemy zatrzymac
    keep = ['local_volume', 'local_electrons', 'local_mean', 'local_std', 'local_min', 'local_max', 'local_skewness', 'local_parts']
    columns_to_drop = list(df_out.columns[pd.Series(df_out.columns).str.startswith('local') | pd.Series(df_out.columns).str.startswith('dict')].difference(keep))
    df_out = df_out.drop(columns_to_drop,axis=1)
    columns_to_drop = ['title', 'pdb_code', 'res_id', 'chain_id','fo_col', 'fc_col', 'weight_col', 'grid_space', 'solvent_radius', 'solvent_opening_radius', 'part_step_FoFc_std_min', 'part_step_FoFc_std_max', 'part_step_FoFc_std_step']
    df_out = df_out.drop(columns_to_drop,axis=1)

    df_out = df_out.fillna(0)

    #Zbior df_out do pierwszej klasyfikacji - klasa res_name
    return df_out


def prepare_data(file, sep, use_cache = True):
    prepared_data_patch = 'cache/prepared_data_'+file
    if use_cache == True and os.path.isfile(prepared_data_patch):
        df = joblib.load(prepared_data_patch)
        return df
    else:
        df = pd.read_csv('data/'+file, sep=sep, na_values=["nan"], low_memory=False, keep_default_na =False)
        joblib.dump(df, prepared_data_patch)
        return df

def get_grouped_res_names(file):
    #pobranie zgrupowanych etykiet
    grouped_res_names = pd.read_csv(file,sep=",", na_values=["nan"], low_memory=False, keep_default_na =False)
    return grouped_res_names['res_name_group'].as_matrix()
```

##Próba użycia svd do redukcji wymiarów.
W celu poprawy jakości klasyfikacji próbowaliśmy zastosować transformację SVD dla danych wejściowych. Na podstawie dokonanej wcześniej analizy wiedzieliśmy, że niektóre atrybuty są mocno skorelowane i chcieliśmy w ten sposób ograniczyć ich liczbę. 
Aby dokonać transformację SVD należało najpierw przetransformować wszystkie atrybuty na wartości dodatnie. W tym celu dokonaliśmy normalizacji wszystkich atrybutów do przedziału wartości 0 - 1.
Dodatkowo należało usunąć z danych wszystkie wartości NA i NaN. Zrealizowaliśmy to poprzez ustawienie tych wartości na bardzo dużą liczbę.

Na tak przetworzonych danych zastosowaliśmy transformację SVD i wybraliśmy pierwsze noOfNewParams najistotnijszych parametrów.



```python
def use_svd(df, noOfNewParams = 100, NA_fallback_value = 10000):
    LA = np.linalg

    x = df.select_dtypes(include=['float64','int64'])

    dfNormalized = (x - x.mean()) / (x.max() - x.min())
    dfNormalized = dfNormalized - dfNormalized.min()
    dfNormalized = dfNormalized.fillna(NA_fallback_value)

    params = dfNormalized.select_dtypes(include=['float64','int64']).as_matrix()



    U, s, Vh = LA.svd(params, full_matrices=False)
    assert np.allclose(params, np.dot(U, np.dot(np.diag(s), Vh)))

    s[noOfNewParams:] = 0

    newParams = np.dot(U, np.diag(s))
    newParams = pd.DataFrame(newParams[:,:noOfNewParams])
    return newParams
```

Podczas testów okazało się, że taka transforamcja nie wpłyneła pozytywnie na jakość klasyfikacji. W ekserymencie uzyskaliśmy następujące wyniki:

| no of parmas | recall         | n_estimators |
|--------------|----------------|--------------|
|           40 | 0.376368613139 |          180 |
|           50 | 0.388686131387 |          140 |
|           60 | 0.405109489051 |          180 |
|           70 | 0.405565693431 |          180 |
|           80 | 0.405565693431 |          180 |
|           90 | 0.409671532847 |          180 |
|          100 |  0.41697080292 |          160 |
|          110 | 0.409215328467 |          180 |
|          120 | 0.410583941606 |          180 |
|          130 | 0.407390510949 |          160 |
|          140 | 0.409671532847 |          180 |
|          150 | 0.412408759124 |          180 |
|          160 | 0.400547445255 |          140 |
|          170 | 0.409671532847 |          180 |
|          180 | 0.407390510949 |          180 |
|--------------|----------------|--------------|
| bez SVD: 755 | 0.458941605839 |          160 |


Parametry testowe dla n_estimators wynosiły: np.arange(60, 300, 20)


```python
def merge_df(df, grouped_res_names, column_name):
    df[column_name] = grouped_res_names[column_name]
    return df
```

##Uczenie klasyfikatora
Do klasyfikacji wykorzystaliśmy algorytm RandomForest. Podczas uczenia poszukujemy najlepszego klasyfikatora przy pomocy parameter grid.
Najlepszy estymator zapisujemy zgodnie z zaleceniami przy pomocy bilbioteki joblib. Niestety po zserializowaniu i zapisaniu do plików każdy z klasyfikatorów zajmuje blisko 2gb pamięci. Wynika to ze sporej ilości drzew w utworzonym klasyfikatorze.


```python
def get_estimator(df,classes,cache_file,use_cache = True):

    X_train, X_test, y_train, y_test = train_test_split(df, classes, test_size=0.2, random_state=0, stratify=classes)

    score = 'recall_weighted'

    if use_cache == True and os.path.isfile(cache_file):
        best_estimator = joblib.load(cache_file)
    else:
        param_grid = {
            'n_estimators': np.arange(100, 200, 20),
            'max_features': ['sqrt'
                             , 'log2'
                            ]
        }

        print("starting classification: (",time.ctime(),')')
        rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True, warm_start=False, random_state=RANDOM_STATE)
        
        clf = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=2, scoring=score)
        clf.fit(X_train, y_train)
        #zapisanie klasyfikatora
        best_estimator = clf.best_estimator_
        joblib.dump(best_estimator, cache_file)

    y_true, y_pred = y_test, best_estimator.predict(X_test)
    print(cache_file)
    print(best_estimator.get_params())
    # print(classification_report(y_true, y_pred))
    print(score, recall_score(y_true, y_pred))
    return best_estimator

    return best_estimator
```

##Klasyfikacja
Wykorzystujemy wyżej zaimplementowane funkcje do przeprowadzenia klasyfikacji.
Po wczytaniu danych testowych ograniczamy je do zbioru kolumn identycznego ze zbiorem treningowym.
Wektory wynikowe dla obu klasyfikatorów zapisywane są następnie do plików w katalogu output.


```python
if __name__ == '__main__':
    for dir in ['cache', 'output']:
        if not os.path.exists(dir):
            os.mkdir(dir)

    train_data =  preprocess_data(prepare_data('all_summary.txt',';'))
    print("train_data.shape:", train_data.shape)

    orig_res_names = train_data['res_name'].as_matrix()
    grouped_res_names = get_grouped_res_names("data/grouped_res_name.txt")

    train_data = train_data.drop('res_name',axis=1)
    print("train_data.shape after drop('res_name'):", train_data.shape)

    test_data = prepare_data('test_data.txt',',')
    test_data = test_data.loc[:,train_data.columns]
    test_data = test_data.fillna(0)

    print("test_data.shape:", test_data.shape)

    # orig_res_names:

    orig_estimator = get_estimator(train_data,orig_res_names,'cache/clf_for_orig_res_names_all_params',use_cache = True)

    orig_test_predict = orig_estimator.predict(test_data)
    np.savetxt("output/orig_test_predict.csv", orig_test_predict, delimiter=",", fmt="%s")

    # for i in np.arange(40, 200, 20):
    #     cache_file = 'cache/clf_for_orig_res_names_svd_'+str(i)+'_params'
    #     df_svd = use_svd(df, noOfNewParams = i, NA_fallback_value = 10000)
    #     classify_data(df_svd,orig_res_names,i,cache_file,)

    # grouped_res_names:

    grouped_estimator = get_estimator(train_data,grouped_res_names,'cache/clf_for_grouped_res_names_all_params',use_cache = True)

    grouped_test_predict = grouped_estimator.predict(test_data)
    np.savetxt("output/grouped_test_predict", grouped_test_predict, delimiter=",", fmt="%s")

    # for i in np.arange(40, 200, 20):
    #     cache_file = 'cache/clf_for_grouped_res_names_svd_'+str(i)+'_params'
    #     df_svd = use_svd(df, noOfNewParams = i, NA_fallback_value = 10000)
    #     classify_data(df_svd,grouped_res_names,i,cache_file)
```

    train_data.shape: (11005, 756)
    train_data.shape after drop('res_name'): (11005, 755)
    test_data.shape: (18917, 755)
    cache/clf_for_orig_res_names_all_params
    {'n_jobs': -1, 'bootstrap': True, 'max_leaf_nodes': None, 'oob_score': True, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'n_estimators': 180, 'max_depth': None, 'warm_start': False, 'min_weight_fraction_leaf': 0.0, 'min_samples_split': 2, 'class_weight': None, 'criterion': 'gini', 'verbose': 0, 'random_state': 42}
    recall_weighted 0.405393053016
    cache/clf_for_grouped_res_names_all_params

    E:\Software\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1304: DeprecationWarning: The default `weighted` averaging is deprecated, and from version 0.18, use of precision, recall or F-score with multiclass or multilabel data or pos_label=None will result in an exception. Please set an explicit value for `average`, one of (None, 'micro', 'macro', 'weighted', 'samples'). In cross validation use, for instance, scoring="f1_weighted" instead of scoring="f1".
      sample_weight=sample_weight)
    E:\Software\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1304: DeprecationWarning: The default `weighted` averaging is deprecated, and from version 0.18, use of precision, recall or F-score with multiclass or multilabel data or pos_label=None will result in an exception. Please set an explicit value for `average`, one of (None, 'micro', 'macro', 'weighted', 'samples'). In cross validation use, for instance, scoring="f1_weighted" instead of scoring="f1".
      sample_weight=sample_weight)
    

    
    {'n_jobs': -1, 'bootstrap': True, 'max_leaf_nodes': None, 'oob_score': True, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'n_estimators': 180, 'max_depth': None, 'warm_start': False, 'min_weight_fraction_leaf': 0.0, 'min_samples_split': 2, 'class_weight': None, 'criterion': 'gini', 'verbose': 0, 'random_state': 42}
    recall_weighted 0.458941605839
    

##Podsumowanie
Musimy przyznać, że spodziewaliśmy się lepszych wyników przy zastosowaniu SVD. Uzyskana dokładność klasyfikacji wydaje się nam dosyć niska, jednak jest zrozumiała, ponieważ była to klasyfikacja wieloklasowa.

Algorytm Random Forest i tak spisał się lepiej niż wypróbowany przez nas wcześniej algorytm SVM, który dawał o ponad 20% gorszą jakość klasyfikacji.
