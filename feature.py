import pandas as pd
import numpy as np
from tqdm import tqdm

#读取轨迹数据
i = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
user = pd.read_csv(r"G:\track data and travel prediction\dataset\DataTech_Travel_Train_User",
                   sep='|', names=['USER_ID', 'FLAG', 'TRAVEL_TYPE'])
#user = user.sample(100)
userid = list(user['USER_ID'])
dataset_sample = pd.DataFrame(columns=["USER_ID", "START_TIME", "LONGITUDE", "LATITUDE", "P_MONTH"])
for number in tqdm(i):
    filename = 'G:/track data and travel prediction/dataset/DataTech_Public_Trace/DataTech_Public_Trace_'
    filename += number
    dataset = pd.read_csv(filename, sep='|', names=["USER_ID", "START_TIME", "LONGITUDE", "LATITUDE", "P_MONTH"])
    dataset = dataset[~dataset['LONGITUDE'].isin([0])]
    dataset = dataset[~dataset['LATITUDE'].isin([0])]
    dataset = dataset[dataset['USER_ID'].isin(userid)]
    dataset_sample = dataset_sample.append(dataset, ignore_index=True)
dataset = dataset_sample.copy()
#预处理的操作，即把经纬度数据保留小数点后两位
#然后同一个小时内，如果有多条在地图网格的数据，只保留一条
dataset['longitude'] = (dataset['LONGITUDE'] * 100) // 1 / 100
dataset['latitude'] = (dataset['LATITUDE'] * 100) // 1 / 100
dataset['HOUR'] = dataset['START_TIME'] // 100 % 100
dataset.sort_values(['USER_ID', 'P_MONTH', 'HOUR'], inplace=True)
dataset.reset_index(drop=True, inplace=True)
dataset = dataset.drop_duplicates(['USER_ID', 'longitude', 'latitude', 'P_MONTH', 'HOUR'])
dataset = dataset.loc[:,('USER_ID', 'HOUR','longitude', 'latitude', 'P_MONTH')]

#筛选出每天都有轨迹记录的用户，即排除有缺失记录的用户
user_c = []
for userid in (dataset['USER_ID'].unique()):
    dt_user = dataset[dataset['USER_ID']==userid]
    if len(dt_user['P_MONTH'].unique()) == 30:
        user_c.append(userid)
    else:
        continue
dataset = dataset[dataset['USER_ID'].isin(user_c)]
      
#计算日均轨迹点数（日均记录数量）
num_of_points = dataset.copy()
num_of_points = num_of_points.groupby(['USER_ID', 'P_MONTH']).count()
num_of_points.reset_index(inplace=True)
num_of_points = num_of_points.loc[:,('USER_ID', 'longitude')]
num_of_points = num_of_points.groupby('USER_ID').mean()
num_of_points.reset_index(inplace=True)
num_of_points.columns = ['USER_ID', 'num_of_points']

#计算日均活动覆盖区域
covering = dataset.copy()
covering = covering.drop_duplicates(['USER_ID', 'longitude', 'latitude', 'P_MONTH'])
covering = covering.groupby(['USER_ID', 'P_MONTH']).count()
covering.reset_index(inplace=True)
covering = covering.loc[:,('USER_ID', 'longitude')]
covering = covering.groupby('USER_ID').mean()
covering.reset_index(inplace=True)
covering.columns = ['USER_ID', 'covering']

#计算回旋半径
turning = dataset.copy()
user_home = pd.DataFrame(columns=['USER_ID', 'LONGITUDE_HOME', 'LATITUDE_HOME'])
turning = turning.groupby(['USER_ID', 'longitude', 'latitude']).count()
turning.reset_index(inplace=True)
i = 0
for userid in turning['USER_ID'].unique():
    dt_user = turning[turning['USER_ID'] == userid]
    dt_user.sort_values(['P_MONTH'], inplace=True)
    dt_user.reset_index(drop=True, inplace=True)
    user_home.loc[i, 'USER_ID'] = userid
    user_home.loc[i, 'LONGITUDE_HOME'] = dt_user.loc[0, 'longitude']
    user_home.loc[i, 'LATITUDE_HOME'] = dt_user.loc[0, 'latitude']
    i += 1
turning = dataset.copy()

from haversine import haversine
turning_radius = pd.DataFrame(columns=['USER_ID', 'turning_radius'])
i = 0
for userid in tqdm(turning['USER_ID'].unique()):
    dt_user = turning[turning['USER_ID'] == userid]
    dt_user.reset_index(drop=True, inplace=True)
    dt_user = pd.merge(dt_user, user_home, on='USER_ID')
    j = 0
    l = len(dt_user)
    d_s = 0
    for j in range(l):
        d = haversine(dt_user.loc[j, 'LONGITUDE_HOME'], dt_user.loc[j, 'LATITUDE_HOME'],
                      dt_user.loc[j, 'longitude'], dt_user.loc[j, 'latitude'])
        d_s += np.abs(d)
        d_a = d_s / l
    turning_radius.loc[i, 'USER_ID'] = userid
    turning_radius.loc[i, 'turning_radius'] = d_a / 1000
    i += 1
    
#计算工作日和周末出行的差异
from datetime import date

ww = dataset.copy()
ww['DAY0618'] = ww['P_MONTH'].apply(lambda x:1 if(x%20180618==0) else 0)
ww['P_MONTH'] = ww['P_MONTH'].astype(np.str)
ww['P_MONTH'] = pd.to_datetime(ww['P_MONTH'], format='%Y-%m-%d')
ww['WEEKDAY'] = ww['P_MONTH'].apply(lambda x: date.isoweekday(x))
ww['ISWEEKENDS'] = ww['WEEKDAY'].apply(lambda x: 1 if ((x==6) | (x==7)) else 0)
ww['HOLIDAYS'] = ww['ISWEEKENDS'] + ww['DAY0618']
ww.drop(['WEEKDAY', 'ISWEEKENDS', 'DAY0618'], axis=1, inplace=True)
dataset_weekdays = ww[ww['HOLIDAYS']==0]
dataset_weekends = ww[ww['HOLIDAYS']==1]
weekdays = dataset_weekdays.groupby(['USER_ID', 'P_MONTH']).count()
weekends = dataset_weekends.groupby(['USER_ID', 'P_MONTH']).count()
weekdays.reset_index(inplace=True)
weekends.reset_index(inplace=True)
weekdays = weekdays.loc[:,('USER_ID', 'HOUR')]
weekends = weekends.loc[:,('USER_ID', 'HOUR')]
weekdays = weekdays.groupby('USER_ID').mean()
weekends = weekends.groupby('USER_ID').mean()
weekdays.reset_index(inplace=True)
weekends.reset_index(inplace=True)
weekdays.columns = ['USER_ID', 'WEEKDAYS_COUNT']
weekends.columns = ['USER_ID', 'WEEKENDS_COUNT']
ww = pd.merge(weekdays, weekends, on='USER_ID')
ww['weekdays_weekends'] = weekdays['WEEKDAYS_COUNT'] - weekends['WEEKENDS_COUNT']
ww = ww.loc[:,('USER_ID', 'weekdays_weekends')]

#计算生活熵
import math

life_en = dataset.copy()
life_en = life_en.drop_duplicates(['USER_ID', 'longitude', 'latitude', 'P_MONTH'])
life_en = life_en.groupby(['USER_ID', 'P_MONTH']).count()
life_en.reset_index(inplace=True)
life_en = life_en.loc[:,('USER_ID', 'longitude', 'P_MONTH')]  
entropy = pd.DataFrame(columns=['USER_ID', 'entropy'])   
i = 0
for userid in life_en['USER_ID'].unique():
    dt_user = turning[turning['USER_ID'] == userid]
    dt_user.reset_index(drop=True, inplace=True)
    s = dt_user['longitude'].sum()
    dt_user['p'] = dt_user['longitude'] / s
    l = len(dt_user)
    e = 0
    for j in range(l):
        p = dt_user.loc[j, 'p']
        e += (- (p * math.log2(p)))
    entropy.loc[i, 'USER_ID'] = userid
    entropy.loc[i, 'entropy'] = e
    i += 1


#计算Robust Coverage Similarity Metric
import scipy.spatial.distance as dist
import tensorflow as tf
rcsm = dataset.copy()
l = len(dataset['USER_ID'].unique())
holidays = ['20180602', '20180603', '20180609', '20180610', '20180616', '20180617', '20180618',
            '20180623', '20180624', '20180630']

def max_pooling(df):
    output = tf.nn.max_pool(value=df, ksize=[2, 2], strides=[1, 1])
    return(output)
def jac(vec1, vec2):
    vec1 = vec1.reshape(205312,)
    vec2 = vec2.reshape(205312,)
    matv = np.array([vec1, vec2])
    dis = dist.pdist(matv, 'jaccard')
    return(dis)
def shift_left(m):
    m = pd.DataFrame(m)
    m = m.shift(-1, axis=1)
    m.fillna(0, inplace=True)
    m = m.as_matrix()
    return(m)
def shift_right(m):
    m = pd.DataFrame(m)
    m = m.shift(1, axis=1)
    m.fillna(0, inplace=True)
    m = m.as_matrix()
    return(m)
def shift_upp(m):
    m = pd.DataFrame(m)
    m = m.shift(-1)
    m.fillna(0, inplace=True)
    m = m.as_matrix()
    return(m)
def shift_down(m):
    m = pd.DataFrame(m)
    m = m.shift(1)
    m.fillna(0, inplace=True)
    m = m.as_matrix()
    return(m)

#利用jaccard计算用户每天轨迹网格的相似度
similarity = pd.DataFrame(columns=['USER_ID', 'shift_similarity', 'maxpo_similarity'])
i = 0
for userid in tqdm(rcsm['USER_ID'].unique()):
    dataset_userid = rcsm[rcsm['USER_ID'] == userid]
    dataset_userid['P_MONTH1'] = dataset_userid['P_MONTH'].astype(np.str)
    dataset_userid = dataset_userid[~dataset_userid['P_MONTH1'].isin(holidays)]
    dataset_userid = dataset_userid.loc[:,('longitude', 'latitude', 'P_MONTH')]
    dataset_userid.reset_index(drop=True, inplace=True)
    metric = np.zeros((30, 512, 401), dtype=np.float16)
    con_1 = np.zeros((512, 401), dtype=np.float16)
    days_num = len(dataset_userid['P_MONTH'].unique())
    if days_num == 0:
        continue
    for j in range(len(dataset_userid)):
        row = round((float(dataset_userid.loc[j, 'longitude']) - 118.04) * 100)
        col = round((float(dataset_userid.loc[j, 'latitude']) - 27.17) * 100)
        day = dataset_userid.loc[j,'P_MONTH'] - 20180601
        metric[day, row, col] = 1
        con_1[row, col] = 1
    s_shift = 0
    s_maxpo = 0
    for k in range(30):
        m = metric[k]
        if np.max(m) == 0:
            continue
        else:
            ##shift函数操作
            con_l = shift_left(m)           
            s_shift += jac(metric[k], con_l)
            con_r = shift_right(m)
            s_shift += jac(metric[k], con_r)
            con_u = shift_upp(m)
            s_shift += jac(metric[k], con_u)
            con_d = shift_down(m)
            s_shift += jac(metric[k], con_d)
            '''
            ##max_pooling函数操作
            mp = pd.DataFrame(metric[k])
            mp = mp.as_matrix()
            mp = mp.reshape(1,512,401,1)
            mp = tf.nn.max_pool(mp, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
            sess=tf.Session()
            #sess.run(tf.initialize_all_variables())
            sess.run(tf.global_variables_initializer())
            mp = mp.eval(session=sess)
            mp = mp.reshape(512,401)
            con_l = shift_left(mp)           
            s_maxpo += jac(mp, con_l)
            con_r = shift_right(mp)
            s_maxpo += jac(mp, con_r)
            con_u = shift_upp(mp)
            s_maxpo += jac(mp, con_u)
            con_d = shift_down(mp)
            s_maxpo += jac(mp, con_d)   
            '''
    m_shift = s_shift[0] / (4*days_num)
    m_maxpo = s_maxpo[0] / (4*days_num)
    similarity.loc[i, 'USER_ID'] = userid
    similarity.loc[i, 'shift_similarity'] = m_shift                      
    similarity.loc[i, 'maxpo_similarity'] = m_maxpo
    i += 1

#特征整合
feature = pd.merge(num_of_points, covering, on='USER_ID')
feature = pd.merge(feature, turning_radius, on='USER_ID')
feature = pd.merge(feature, ww, on='USER_ID')
feature = pd.merge(feature, entropy, on='USER_ID')
feature.to_csv('feature', index=False)

reg = pd.merge(feature, user, on='USER_ID')

#随机森林分类
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X = reg.iloc[:,1:6]
y = reg.loc[:,'TRAVEL_TYPE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
clf = RandomForestClassifier(n_estimators=100, oob_score=True,n_jobs=-1,max_features='auto', min_samples_leaf=50, random_state=10)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
