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


def merge_df(df, grouped_res_names, column_name):
    df[column_name] = grouped_res_names[column_name]
    return df


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

