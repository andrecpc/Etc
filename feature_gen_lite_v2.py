import os
import numpy as np
import pandas as pd
import swifter
from sklearn.model_selection import train_test_split
import utils
import catboost
from tqdm import tqdm
pd.options.display.max_columns = None
import time
start_time = time.time()

# DATA_PATH = "D:\IDAO-MuID\IDAO-2019-muon-id-master\data"
DATA_PATH = "../IDAO-MuID"
train_concat = pd.read_hdf('train_concat_31.01_rmse_foi.h5')

# train_concat = train_concat.loc[:10000, :]
print("--- %s seconds end loading ---" % (time.time() - start_time))
# full_train = utils.load_full_train_csv(DATA_PATH)
# closest_hits_features = full_train.swifter.apply(
    # utils.find_closest_hit_per_station, result_type="expand", axis=1)
# train_concat = pd.concat(
    # [full_train.loc[:, utils.SIMPLE_FEATURE_COLUMNS],
     # closest_hits_features], axis=1)
     
     

# def distance(x,y):
#     x1, y1, x2, y2 = [*x,*y]
#     return sqrt((x2-x1)**2+(y2-y1)**2)

# from sklearn.neighbors import KNeighborsClassifier

# from math import sin, cos, sqrt, atan2, radians

# knc = KNeighborsClassifier(metric=distance)
# dots = train_concat[['Lextra_X[0]','Lextra_Y[0]']].dropna()
# knc.fit(X=dots , y=np.ones(dots.shape[0]))
# distances, indexes = knc.kneighbors(X=dots,n_neighbors=6)

# for i in range(1,6):
#     dots['distance_%s'%i] = distances[:,i]
#     dots['indexes_%s'%i] = indexes[:,i]

# from sklearn.decomposition import PCA

# dots['mean'] = dots.iloc[:,dots.columns.str.contains('distance')].mean(axis=1)
# dots['median'] = dots.iloc[:,dots.columns.str.contains('distance')].median(axis=1)
# dots['std'] = dots.iloc[:,dots.columns.str.contains('distance')].std(axis=1)

# dots_pca = train_concat[['Lextra_X[0]','Lextra_Y[0]']].dropna()
# pca = PCA(n_components=1)

# dots['coords_pca'] = pca.fit_transform(dots_pca)

# train_concat_dots = pd.concat([train_concat,dots.drop(['Lextra_X[0]','Lextra_Y[0]'], 1)], axis=1)

# То же самое, что и выше, но только циклом с перебором всех пар в списке vals, который объявлен ниже
# train_concat_dots = train_concat.copy()
# 
# # Если правильно все сделал, то ближайшие похожие точки находятся на основе этой формулы
# def distance(x,y):
#     x1, y1, x2, y2 = [*x,*y]
#     return sqrt((x2-x1)**2+(y2-y1)**2)
# 
# from sklearn.neighbors import KNeighborsClassifier
# from math import sin, cos, sqrt, atan2, radians
# from sklearn.decomposition import PCA
# 
# vals = ['Lextra_X[0]','Lextra_Y[0]','Lextra_X[1]','Lextra_Y[1]','Lextra_X[2]','Lextra_Y[2]','Lextra_X[3]','Lextra_Y[3]']
# # Для каждой пары Lextra_ подбираются ближайшие похожие точки, помимо Lextra_ подобрать для всех таких координатных фич,
# # поэтому цикл сделал
# z = 0
# 
# while z < len(vals):
#     print (vals[z],vals[z+1])
#     knc = KNeighborsClassifier(metric=distance)
#     dots = train_concat[[vals[z],vals[z+1]]].dropna()
#     knc.fit(X=dots , y=np.ones(dots.shape[0]))
#     distances, indexes = knc.kneighbors(X=dots,n_neighbors=6)
# 
#     for i in range(1,6):
#         dots[str(z)+'_distance_%s'%i] = distances[:,i]
#         dots[str(z)+'_indexes_%s'%i] = indexes[:,i]
# 
#     # Попутно для них считаются эти три значения по дистанциям
#     dots[str(z)+'_mean'] = dots.iloc[:,dots.columns.str.contains(str(z)+'_distance')].mean(axis=1)
#     dots[str(z)+'_median'] = dots.iloc[:,dots.columns.str.contains(str(z)+'_distance')].median(axis=1)
#     dots[str(z)+'_std'] = dots.iloc[:,dots.columns.str.contains(str(z)+'_distance')].std(axis=1)
# 
#     # PCA, кажется, размерность сокращает и тож добавляется как фича
#     dots_pca = train_concat[[vals[z],vals[z+1]]].dropna()
#     pca = PCA(n_components=1)
# 
#     dots['coords_pca'] = pca.fit_transform(dots_pca)
# 
#     train_concat_dots = pd.concat([train_concat_dots,dots.drop([vals[z],vals[z+1]], 1)], axis=1)
# 
#     z = z + 2
#     print (z)
# 
# train_concat_dots.to_hdf('train_concat_dots.h5', 'data')
# print('train_concat_dots saved')
    
# Тут добавляем фичи кластеризации
# Снова на основе Lextra_, но можно и другие фичи выбрать
# train_concat_dots_cluster = train_concat_dots.copy()

train_concat_dots_cluster = train_concat.copy()

from sklearn import cluster, mixture   

def cluster_model(newdata, data, model_name, input_param):
    
    ds = data
    params = input_param
    
    if str.lower(model_name) == 'kmeans':                                
        cluster_obj = cluster.KMeans(n_clusters=params['n_clusters'])
    if str.lower(model_name) == str.lower('MiniBatchKMeans'):            
        cluster_obj = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    if str.lower(model_name) == str.lower('SpectralClustering'):         
        cluster_obj = cluster.SpectralClustering(n_clusters=params['n_clusters'])
    if str.lower(model_name) == str.lower('MeanShift'):                  
        cluster_obj = cluster.MeanShift(bandwidth=params['bandwidth'])
    if str.lower(model_name) == str.lower('DBSCAN'):                     
        cluster_obj = cluster.DBSCAN(eps=params['eps'])
    if str.lower(model_name) == str.lower('AffinityPropagation'):        
        cluster_obj = cluster.AffinityPropagation(damping=params['damping'], preference=params['preference'])
        cluster_obj.fit(ds)
    if str.lower(model_name) == str.lower('Birch'):                      
        cluster_obj = cluster.Birch(n_clusters=input_param['n_clusters'])
    if str.lower(model_name) == str.lower('GaussianMixture'):            
        cluster_obj = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')
        cluster_obj.fit(ds)
    if str.lower(model_name) in ['affinitypropagation', 'gaussianmixture']:
        model_result = cluster_obj.predict(ds)
    else:
        model_result = cluster_obj.fit_predict(ds)
    
    newdata[model_name] = pd.DataFrame(model_result)
    
    return(newdata)

cluster_list = ["KMeans", "MiniBatchKMeans", "DBSCAN", "Birch", "MeanShift"]

input_param = {'n_clusters':100, 'bandwidth':0.1, "damping":0.9, "eps":1, 'min_samples':3, "preference":-200}

# vals = ['Lextra_X[0]','Lextra_Y[0]','Lextra_X[1]','Lextra_Y[1]','Lextra_X[2]','Lextra_Y[2]','Lextra_X[3]','Lextra_Y[3]']
vals = ['PT']

pca_comp = train_concat_dots_cluster[vals].fillna(0)

for i in cluster_list:
    train_concat_dots_cluster = cluster_model(train_concat_dots_cluster, pca_comp, i, input_param)

train_concat_dots_cluster.to_hdf('train_concat_dots_cluster.h5', 'data')
print('train_concat_dots_cluster saved')
    
# Тут для исходных фич считаем 3 вида Scaler; отклонение, вроде

train_concat_dots_cluster_prep = train_concat_dots_cluster.copy()

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

ss = StandardScaler()
mms = MinMaxScaler()
rs = RobustScaler()

for elem in utils.SIMPLE_FEATURE_COLUMNS:
    train_concat_dots_cluster_prep[elem+'_ss'] = ss.fit_transform(train_concat_dots_cluster_prep[[elem]])
    train_concat_dots_cluster_prep[elem+'_mms'] = mms.fit_transform(train_concat_dots_cluster_prep[[elem]])
    train_concat_dots_cluster_prep[elem+'_rs'] = rs.fit_transform(train_concat_dots_cluster_prep[[elem]])


# Тут добавляем фичи логарифмов на основе базовых фич
train_concat_dots_cluster_prep_log = train_concat_dots_cluster_prep.copy()

for elem in utils.SIMPLE_FEATURE_COLUMNS:
    train_concat_dots_cluster_prep_log[elem+'_log'] = np.log(train_concat_dots_cluster_prep_log[elem])

train_concat_dots_cluster_prep.to_hdf('train_concat_dots_cluster_prep.h5', 'data')
print('train_concat_dots_cluster_prep saved')
    
# Тут добавляем mean, median, std на основе групп фич с одинаковыми префиксами, на основе базовых фич
train_concat_dots_cluster_prep_mms = train_concat_dots_cluster_prep.copy()

groups = [['ncl[0]', 'ncl[1]', 'ncl[2]', 'ncl[3]'], ['avg_cs[0]', 'avg_cs[1]', 'avg_cs[2]', 'avg_cs[3]'],
        ['MatchedHit_TYPE[0]', 'MatchedHit_TYPE[1]', 'MatchedHit_TYPE[2]', 'MatchedHit_TYPE[3]'],
        ['MatchedHit_X[0]', 'MatchedHit_X[1]', 'MatchedHit_X[2]', 'MatchedHit_X[3]'],
        ['MatchedHit_Y[0]', 'MatchedHit_Y[1]', 'MatchedHit_Y[2]', 'MatchedHit_Y[3]'],
        ['MatchedHit_Z[0]', 'MatchedHit_Z[1]', 'MatchedHit_Z[2]', 'MatchedHit_Z[3]'],
        ['MatchedHit_DX[0]', 'MatchedHit_DX[1]', 'MatchedHit_DX[2]', 'MatchedHit_DX[3]'],
        ['MatchedHit_DY[0]', 'MatchedHit_DY[1]', 'MatchedHit_DY[2]', 'MatchedHit_DY[3]'],
        ['MatchedHit_DZ[0]', 'MatchedHit_DZ[1]', 'MatchedHit_DZ[2]', 'MatchedHit_DZ[3]'],
        ['MatchedHit_T[0]', 'MatchedHit_T[1]', 'MatchedHit_T[2]', 'MatchedHit_T[3]'],
        ['MatchedHit_DT[0]', 'MatchedHit_DT[1]', 'MatchedHit_DT[2]', 'MatchedHit_DT[3]'],
        ['Lextra_X[0]', 'Lextra_X[1]', 'Lextra_X[2]', 'Lextra_X[3]'],
        ['Lextra_Y[0]', 'Lextra_Y[1]', 'Lextra_Y[2]', 'Lextra_Y[3]'],
        ['Mextra_DX2[0]', 'Mextra_DX2[1]', 'Mextra_DX2[2]', 'Mextra_DX2[3]'],
        ['Mextra_DY2[0]', 'Mextra_DY2[1]', 'Mextra_DY2[2]', 'Mextra_DY2[3]'], ['PT', 'P']]

grou = ['ncl', 'avg_cs',
        'MatchedHit_TYPE',
        'MatchedHit_X',
        'MatchedHit_Y',
        'MatchedHit_Z',
        'MatchedHit_DX',
        'MatchedHit_DY',
        'MatchedHit_DZ',
        'MatchedHit_T',
        'MatchedHit_DT',
        'Lextra_X',
        'Lextra_Y',
        'Mextra_DX2',
        'Mextra_DY2','P']
l = 0
for elem in groups:
    
    train_concat_dots_cluster_prep_mms[str(grou[l])+'_mean'] = train_concat_dots_cluster_prep_mms[elem].mean(axis=1)
    train_concat_dots_cluster_prep_mms[str(grou[l])+'_median'] = train_concat_dots_cluster_prep_mms[elem].median(axis=1)
    train_concat_dots_cluster_prep_mms[str(grou[l])+'_std'] = train_concat_dots_cluster_prep_mms[elem].std(axis=1)
    l = l + 1

train_concat_dots_cluster_prep_mms.to_hdf('train_concat_dots_cluster_prep_mms.h5', 'data')
print('train_concat_dots_cluster_prep_mms saved')
# На этом есть предварительный сет, в который фичи собирались по чуть-чуть разными методами
# train_concat_dots_cluster_prep_mms.tail()


# Тут как бы пойдет вторая часть, где получится другой сет, все генерится одним пакетом 
# conda install -c conda-forge featuretools

# Создаем новый черновой сет на основе базовых фич и добавляем столбец ID
# train_T = train_concat.loc[:, utils.SIMPLE_FEATURE_COLUMNS]
# train_T['ID'] = list(range(len(train_T.index)))
# 
# # Все парные фичи вырываются и транспонируются, создается подобие временного ряда, и из этого делаются отдельные сеты-сущности
# # Так надо, чтобы пакет работал
# def delaem_otdelnie_df(frame):
# 
#     df2 = pd.DataFrame([[-99, -99], [-99, -99]], columns=['ncl','ID'])
# 
#     for s in range(0,len(frame)):
# 
#         df4 = pd.DataFrame(frame[s:s+1].T.values, columns=['ncl'])
#         df4['ID'] = s
#         df2 = df2.append([df4], ignore_index=True)
# 
#     df2 = df2.drop(df2.index[[0,1]]) 
#     return df2
# 
# train_T_ncl = train_T[groups[0]]
# train_T_ncl = delaem_otdelnie_df(train_T_ncl)
# 
# train_T_avg_cs = train_T[groups[1]]
# train_T_avg_cs = delaem_otdelnie_df(train_T_avg_cs)
# 
# train_T_MatchedHit_X = train_T[groups[3]]
# train_T_MatchedHit_X = delaem_otdelnie_df(train_T_MatchedHit_X)
# 
# train_T_MatchedHit_Y = train_T[groups[4]]
# train_T_MatchedHit_Y = delaem_otdelnie_df(train_T_MatchedHit_Y)
# 
# train_T_MatchedHit_Z = train_T[groups[5]]
# train_T_MatchedHit_Z = delaem_otdelnie_df(train_T_MatchedHit_Z)
# 
# train_T_MatchedHit_DX = train_T[groups[6]]
# train_T_MatchedHit_DX = delaem_otdelnie_df(train_T_MatchedHit_DX)
# 
# train_T_MatchedHit_DY = train_T[groups[7]]
# train_T_MatchedHit_DY = delaem_otdelnie_df(train_T_MatchedHit_DY)
# 
# train_T_MatchedHit_DZ = train_T[groups[8]]
# train_T_MatchedHit_DZ = delaem_otdelnie_df(train_T_MatchedHit_DZ)
# 
# train_T_MatchedHit_T = train_T[groups[9]]
# train_T_MatchedHit_T = delaem_otdelnie_df(train_T_MatchedHit_T)
# 
# train_T_MatchedHit_DT = train_T[groups[10]]
# train_T_MatchedHit_DT = delaem_otdelnie_df(train_T_MatchedHit_DT)
# 
# train_T_Lextra_X = train_T[groups[11]]
# train_T_Lextra_X = delaem_otdelnie_df(train_T_Lextra_X)
# 
# train_T_Lextra_Y = train_T[groups[12]]
# train_T_Lextra_Y = delaem_otdelnie_df(train_T_Lextra_Y)
# 
# train_T_Mextra_DX2 = train_T[groups[13]]
# train_T_Mextra_DX2 = delaem_otdelnie_df(train_T_Mextra_DX2)
# 
# train_T_Mextra_DY2 = train_T[groups[14]]
# train_T_Mextra_DY2 = delaem_otdelnie_df(train_T_Mextra_DY2)
# 
# train_T_P = train_T[groups[15]]
# train_T_P = delaem_otdelnie_df(train_T_P)
# 
# # Новый черновой сет только с парными базовыми фичами кроме MatchedHit_TYPE
# # И ему тоже ID добавляем
# train_best_features = train_concat[['ncl[0]', 'ncl[1]', 'ncl[2]', 'ncl[3]', 'avg_cs[0]', 'avg_cs[1]', 'avg_cs[2]', 'avg_cs[3]',
#         'MatchedHit_X[0]', 'MatchedHit_X[1]', 'MatchedHit_X[2]', 'MatchedHit_X[3]',
#         'MatchedHit_Y[0]', 'MatchedHit_Y[1]', 'MatchedHit_Y[2]', 'MatchedHit_Y[3]',
#         'MatchedHit_Z[0]', 'MatchedHit_Z[1]', 'MatchedHit_Z[2]', 'MatchedHit_Z[3]',
#         'MatchedHit_DX[0]', 'MatchedHit_DX[1]', 'MatchedHit_DX[2]', 'MatchedHit_DX[3]',
#         'MatchedHit_DY[0]', 'MatchedHit_DY[1]', 'MatchedHit_DY[2]', 'MatchedHit_DY[3]',
#         'MatchedHit_DZ[0]', 'MatchedHit_DZ[1]', 'MatchedHit_DZ[2]', 'MatchedHit_DZ[3]',
#         'MatchedHit_T[0]', 'MatchedHit_T[1]', 'MatchedHit_T[2]', 'MatchedHit_T[3]',
#         'MatchedHit_DT[0]', 'MatchedHit_DT[1]', 'MatchedHit_DT[2]', 'MatchedHit_DT[3]',
#         'Lextra_X[0]', 'Lextra_X[1]', 'Lextra_X[2]', 'Lextra_X[3]',
#         'Lextra_Y[0]', 'Lextra_Y[1]', 'Lextra_Y[2]', 'Lextra_Y[3]',
#         'Mextra_DX2[0]', 'Mextra_DX2[1]', 'Mextra_DX2[2]', 'Mextra_DX2[3]',
#         'Mextra_DY2[0]', 'Mextra_DY2[1]', 'Mextra_DY2[2]', 'Mextra_DY2[3]', 'PT', 'P']]
# train_best_features['ID'] = list(range(len(train_best_features.index)))
# 
# 
# import featuretools as ft
# import warnings
# warnings.simplefilter('ignore')
# 
# # Создается набор сущностей
# es = ft.EntitySet('muons')
# 
# # Создается основная сущность
# es.entity_from_dataframe(
#     entity_id='main', # define entity id
#     dataframe=train_best_features, # select underlying data
#     index='ID', # define unique index column
#     # specify some datatypes manually (if needed)
# )
# 
# # Ко всем маленьким сетам, которые транспонировались добавляются ID
# 
# train_T_ncl['ID_'] = list(range(len(train_T_ncl.index)))
# train_T_avg_cs['ID_'] = list(range(len(train_T_avg_cs.index)))
# train_T_MatchedHit_X['ID_'] = list(range(len(train_T_MatchedHit_X.index)))
# train_T_MatchedHit_Y['ID_'] = list(range(len(train_T_MatchedHit_Y.index)))
# train_T_MatchedHit_Z['ID_'] = list(range(len(train_T_MatchedHit_Z.index)))
# train_T_MatchedHit_DX['ID_'] = list(range(len(train_T_MatchedHit_DX.index)))
# train_T_MatchedHit_DY['ID_'] = list(range(len(train_T_MatchedHit_DY.index)))
# train_T_MatchedHit_DZ['ID_'] = list(range(len(train_T_MatchedHit_DZ.index)))
# train_T_MatchedHit_T['ID_'] = list(range(len(train_T_MatchedHit_T.index)))
# train_T_MatchedHit_DT['ID_'] = list(range(len(train_T_MatchedHit_DT.index)))
# train_T_Lextra_X['ID_'] = list(range(len(train_T_Lextra_X.index)))
# train_T_Lextra_Y['ID_'] = list(range(len(train_T_Lextra_Y.index)))
# train_T_Mextra_DX2['ID_'] = list(range(len(train_T_Mextra_DX2.index)))
# train_T_Mextra_DY2['ID_'] = list(range(len(train_T_Mextra_DY2.index)))
# train_T_P['ID_'] = list(range(len(train_T_P.index)))
# 
# # Добавляем эти сеты к набору сущностей в виде самостоятельных сущностей
# es = es.entity_from_dataframe(
#     entity_id = 'ncl', 
#     dataframe = train_T_ncl,
#     index = 'ID_',)
# es = es.entity_from_dataframe(
#     entity_id = 'avg_cs', 
#     dataframe = train_T_avg_cs,
#     index = 'ID_',)
# es = es.entity_from_dataframe(
#     entity_id = 'MatchedHit_X', 
#     dataframe = train_T_MatchedHit_X,
#     index = 'ID_',)
# es = es.entity_from_dataframe(
#     entity_id = 'MatchedHit_Y', 
#     dataframe = train_T_MatchedHit_Y,
#     index = 'ID_',)
# es = es.entity_from_dataframe(
#     entity_id = 'MatchedHit_Z', 
#     dataframe = train_T_MatchedHit_Z,
#     index = 'ID_',)
# es = es.entity_from_dataframe(
#     entity_id = 'MatchedHit_DX', 
#     dataframe = train_T_MatchedHit_DX,
#     index = 'ID_',)
# es = es.entity_from_dataframe(
#     entity_id = 'MatchedHit_DY', 
#     dataframe = train_T_MatchedHit_DY,
#     index = 'ID_',)
# es = es.entity_from_dataframe(
#     entity_id = 'MatchedHit_DZ', 
#     dataframe = train_T_MatchedHit_DZ,
#     index = 'ID_',)
# es = es.entity_from_dataframe(
#     entity_id = 'MatchedHit_T', 
#     dataframe = train_T_MatchedHit_T,
#     index = 'ID_',)
# es = es.entity_from_dataframe(
#     entity_id = 'MatchedHit_DT', 
#     dataframe = train_T_MatchedHit_DT,
#     index = 'ID_',)
# es = es.entity_from_dataframe(
#     entity_id = 'Lextra_X', 
#     dataframe = train_T_Lextra_X,
#     index = 'ID_',)
# es = es.entity_from_dataframe(
#     entity_id = 'Lextra_Y', 
#     dataframe = train_T_Lextra_Y,
#     index = 'ID_',)
# es = es.entity_from_dataframe(
#     entity_id = 'Mextra_DX2', 
#     dataframe = train_T_Mextra_DX2,
#     index = 'ID_',)
# es = es.entity_from_dataframe(
#     entity_id = 'Mextra_DY2', 
#     dataframe = train_T_Mextra_DY2,
#     index = 'ID_',)
# es = es.entity_from_dataframe(
#     entity_id = 'T_P', 
#     dataframe = train_T_P,
#     index = 'ID_',)
# 
# # Указываем в наборе сущностей, какие сущности с какими связывать и по какому полю.
# es = es.add_relationship(ft.Relationship(
#     es['main']['ID'],
#     es['ncl']['ID']))
# es = es.add_relationship(ft.Relationship(
#     es['main']['ID'],
#     es['avg_cs']['ID']))
# es = es.add_relationship(ft.Relationship(
#     es['main']['ID'],
#     es['MatchedHit_X']['ID']))
# es = es.add_relationship(ft.Relationship(
#     es['main']['ID'],
#     es['MatchedHit_Y']['ID']))
# es = es.add_relationship(ft.Relationship(
#     es['main']['ID'],
#     es['MatchedHit_Z']['ID']))
# es = es.add_relationship(ft.Relationship(
#     es['main']['ID'],
#     es['MatchedHit_DX']['ID']))
# es = es.add_relationship(ft.Relationship(
#     es['main']['ID'],
#     es['MatchedHit_DY']['ID']))
# es = es.add_relationship(ft.Relationship(
#     es['main']['ID'],
#     es['MatchedHit_DZ']['ID']))
# es = es.add_relationship(ft.Relationship(
#     es['main']['ID'],
#     es['MatchedHit_T']['ID']))
# es = es.add_relationship(ft.Relationship(
#     es['main']['ID'],
#     es['MatchedHit_DT']['ID']))
# es = es.add_relationship(ft.Relationship(
#     es['main']['ID'],
#     es['Lextra_X']['ID']))
# es = es.add_relationship(ft.Relationship(
#     es['main']['ID'],
#     es['Lextra_Y']['ID']))
# es = es.add_relationship(ft.Relationship(
#     es['main']['ID'],
#     es['Mextra_DX2']['ID']))
# es = es.add_relationship(ft.Relationship(
#     es['main']['ID'],
#     es['Mextra_DY2']['ID']))
# es = es.add_relationship(ft.Relationship(
#     es['main']['ID'],
#     es['T_P']['ID']))
# 
# # Это два типа фич, которые генерит пакет
# # Полные списки
# # transform = ['percentile', 'cum_min', 'days_since', 'cum_sum', 'is_null', 'characters', 'time_since_previous', 'cum_count', 'mod', 'not', 'years', 'day', 'negate', 'time_since', 'haversine', 'year', 'second', 'latitude', 'hour', 'subtract', 'month', 'hours', 'diff', 'absolute', 'numwords', 'week', 'and', 'or', 'weekday', 'days', 'weeks', 'longitude', 'cum_mean', 'minutes', 'weekend', 'cum_max', 'minute', 'multiply', 'isin', 'months', 'seconds', 'add', 'divide']
# # aggregation = ['sum', 'median', 'trend', 'percent_true', 'mean', 'min', 'skew', 'avg_time_between', 'any', 'count', 'mode', 'std', 'max', 'num_unique', 'last', 'num_true', 'n_most_common', 'all', 'time_since_last']
# 
# # Порезаные списки, так как у нас нет фич дат
# transform = ['percentile',
# #              'mod',
#              'negate', 'latitude',
# #              'subtract',
#              'diff', 'absolute', 'longitude',
# #              'multiply',
# #              'add', 'divide'
#             ]
# aggregation = ['sum', 'median', 'trend', 'percent_true', 'mean', 'min', 'skew', 'count', 'mode', 'std', 'max', 'num_unique', 'num_true', 'n_most_common']
# 
# # А так можно вывести типы фич
# for el in ft.list_primitives().values:
#     print (el[0])
# ft.list_primitives().values
# 
# fm, feature_defs = ft.dfs(
#     entityset=es, 
#     target_entity="main", 
#     #features_only=True, was commented
# 
# 
# #     agg_primitives=[
# #         "avg_time_between",
# #         "time_since_last", 
# #         "num_unique", 
# #         "mean", 
# #         "sum", 
# #     ],
# #     trans_primitives=[
# #         "time_since_previous",
# #         #"add",
# #     ],
# 
#     agg_primitives=aggregation,
#     trans_primitives=transform,
# 
#     max_depth=1,
# #     cutoff_time=cutoff_times,
#     training_window=ft.Timedelta(60, "d"),
#     max_features=1000, #число генерируемых фич
#     chunk_size=4000,
#     verbose=True,
# )
# 
# fm = fm.drop_duplicates()
# print(fm.shape)
# fm[0:5]
# 
# fm.to_hdf('train_feature_gen.h5', 'data')

# fm - это итоговый датасет после работы этого пакета

print("--- %s seconds end script ---" % (time.time() - start_time))