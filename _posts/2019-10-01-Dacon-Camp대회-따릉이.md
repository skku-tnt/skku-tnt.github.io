---
layout: post
title: Dacon Camp 따릉이 대회
tags: [competition]
excerpt_separator: <!--more-->
---

Dacon Camp 따릉이 대회 입니다. 
<!--more-->

## 모듈 및 파일 로드


```python
from google.colab import drive
drive.mount('/gdrive', force_remount=True)
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&response_type=code&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly
    
    Enter your authorization code:
    ··········
    Mounted at /gdrive
    


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
```


```python
train_raw = pd.read_csv('/gdrive/My Drive/train.csv', encoding='utf-8') 
test_raw = pd.read_csv('/gdrive/My Drive/test.csv', encoding = 'utf-8')
```

##EDA 및 Data Preprocessing


```python
train_raw.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>hour</th>
      <th>hour_bef_temperature</th>
      <th>hour_bef_precipitation</th>
      <th>hour_bef_windspeed</th>
      <th>hour_bef_humidity</th>
      <th>hour_bef_visibility</th>
      <th>hour_bef_ozone</th>
      <th>hour_bef_pm10</th>
      <th>hour_bef_pm2.5</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>20</td>
      <td>16.3</td>
      <td>1.0</td>
      <td>1.5</td>
      <td>89.0</td>
      <td>576.0</td>
      <td>0.027</td>
      <td>76.0</td>
      <td>33.0</td>
      <td>49.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>13</td>
      <td>20.1</td>
      <td>0.0</td>
      <td>1.4</td>
      <td>48.0</td>
      <td>916.0</td>
      <td>0.042</td>
      <td>73.0</td>
      <td>40.0</td>
      <td>159.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>6</td>
      <td>13.9</td>
      <td>0.0</td>
      <td>0.7</td>
      <td>79.0</td>
      <td>1382.0</td>
      <td>0.033</td>
      <td>32.0</td>
      <td>19.0</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>23</td>
      <td>8.1</td>
      <td>0.0</td>
      <td>2.7</td>
      <td>54.0</td>
      <td>946.0</td>
      <td>0.040</td>
      <td>75.0</td>
      <td>64.0</td>
      <td>57.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>18</td>
      <td>29.5</td>
      <td>0.0</td>
      <td>4.8</td>
      <td>7.0</td>
      <td>2000.0</td>
      <td>0.057</td>
      <td>27.0</td>
      <td>11.0</td>
      <td>431.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_raw.isnull().sum()
```




    id                          0
    hour                        0
    hour_bef_temperature        2
    hour_bef_precipitation      2
    hour_bef_windspeed          9
    hour_bef_humidity           2
    hour_bef_visibility         2
    hour_bef_ozone             76
    hour_bef_pm10              90
    hour_bef_pm2.5            117
    count                       0
    dtype: int64




```python
# 결측치 분석
train_raw.loc[train_raw['hour_bef_temperature'].isnull()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>hour</th>
      <th>hour_bef_temperature</th>
      <th>hour_bef_precipitation</th>
      <th>hour_bef_windspeed</th>
      <th>hour_bef_humidity</th>
      <th>hour_bef_visibility</th>
      <th>hour_bef_ozone</th>
      <th>hour_bef_pm10</th>
      <th>hour_bef_pm2.5</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>934</th>
      <td>1420</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>39.0</td>
    </tr>
    <tr>
      <th>1035</th>
      <td>1553</td>
      <td>18</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 결측치 분석 결과 아무런 정보를 갖지 않는 data point인 것을 확인 후 제거
train = train_raw.loc[train_raw['hour_bef_temperature'].notnull()]
train.isnull().sum()
```




    id                          0
    hour                        0
    hour_bef_temperature        0
    hour_bef_precipitation      0
    hour_bef_windspeed          7
    hour_bef_humidity           0
    hour_bef_visibility         0
    hour_bef_ozone             74
    hour_bef_pm10              88
    hour_bef_pm2.5            115
    count                       0
    dtype: int64




```python
train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>hour</th>
      <th>hour_bef_temperature</th>
      <th>hour_bef_precipitation</th>
      <th>hour_bef_windspeed</th>
      <th>hour_bef_humidity</th>
      <th>hour_bef_visibility</th>
      <th>hour_bef_ozone</th>
      <th>hour_bef_pm10</th>
      <th>hour_bef_pm2.5</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1457.000000</td>
      <td>1457.000000</td>
      <td>1457.000000</td>
      <td>1457.000000</td>
      <td>1450.000000</td>
      <td>1457.000000</td>
      <td>1457.000000</td>
      <td>1383.000000</td>
      <td>1369.000000</td>
      <td>1342.000000</td>
      <td>1457.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1105.391901</td>
      <td>11.496911</td>
      <td>16.717433</td>
      <td>0.031572</td>
      <td>2.479034</td>
      <td>52.231297</td>
      <td>1405.216884</td>
      <td>0.039149</td>
      <td>57.168736</td>
      <td>30.327124</td>
      <td>108.684969</td>
    </tr>
    <tr>
      <th>std</th>
      <td>631.609634</td>
      <td>6.918890</td>
      <td>5.239150</td>
      <td>0.174917</td>
      <td>1.378265</td>
      <td>20.370387</td>
      <td>583.131708</td>
      <td>0.019509</td>
      <td>31.771019</td>
      <td>14.713252</td>
      <td>82.620202</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>3.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.000000</td>
      <td>78.000000</td>
      <td>0.003000</td>
      <td>9.000000</td>
      <td>8.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>555.000000</td>
      <td>6.000000</td>
      <td>12.800000</td>
      <td>0.000000</td>
      <td>1.400000</td>
      <td>36.000000</td>
      <td>879.000000</td>
      <td>0.025500</td>
      <td>36.000000</td>
      <td>20.000000</td>
      <td>37.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1113.000000</td>
      <td>11.000000</td>
      <td>16.600000</td>
      <td>0.000000</td>
      <td>2.300000</td>
      <td>51.000000</td>
      <td>1577.000000</td>
      <td>0.039000</td>
      <td>51.000000</td>
      <td>26.000000</td>
      <td>96.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1652.000000</td>
      <td>17.000000</td>
      <td>20.100000</td>
      <td>0.000000</td>
      <td>3.400000</td>
      <td>69.000000</td>
      <td>1994.000000</td>
      <td>0.052000</td>
      <td>69.000000</td>
      <td>37.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2179.000000</td>
      <td>23.000000</td>
      <td>30.000000</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>99.000000</td>
      <td>2000.000000</td>
      <td>0.125000</td>
      <td>269.000000</td>
      <td>90.000000</td>
      <td>431.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1457 entries, 0 to 1458
    Data columns (total 11 columns):
    id                        1457 non-null int64
    hour                      1457 non-null int64
    hour_bef_temperature      1457 non-null float64
    hour_bef_precipitation    1457 non-null float64
    hour_bef_windspeed        1450 non-null float64
    hour_bef_humidity         1457 non-null float64
    hour_bef_visibility       1457 non-null float64
    hour_bef_ozone            1383 non-null float64
    hour_bef_pm10             1369 non-null float64
    hour_bef_pm2.5            1342 non-null float64
    count                     1457 non-null float64
    dtypes: float64(9), int64(2)
    memory usage: 136.6 KB
    


```python
# 시간에 따른 이용량 시각화
train.groupby(['hour'], as_index = False)['count'].mean().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f0b6a378128>




<img src="/assets/img/output_11_1.png">


```python
# 1시간전 기온에 따른 이용량 분석
plt.scatter(train['hour_bef_temperature'], train['count'])
train[['hour_bef_temperature', 'count']].corr(method = 'pearson')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour_bef_temperature</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>hour_bef_temperature</th>
      <td>1.000000</td>
      <td>0.619404</td>
    </tr>
    <tr>
      <th>count</th>
      <td>0.619404</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




![png](assets/img/output_12_1.png)



```python
# 강수 유무에 따른 이용량 분석
train.groupby(['hour_bef_precipitation'])['count'].mean()
```




    hour_bef_precipitation
    0.0    111.130404
    1.0     33.673913
    Name: count, dtype: float64




```python
# 바람세기에 따른 이용량 분석
plt.scatter(train['hour_bef_windspeed'], train['count'])
train[['hour_bef_windspeed', 'count']].corr(method = 'pearson')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour_bef_windspeed</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>hour_bef_windspeed</th>
      <td>1.000000</td>
      <td>0.459906</td>
    </tr>
    <tr>
      <th>count</th>
      <td>0.459906</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




![png](assets/img/_14_1.png)



```python
# 습도에 따른 이용량 분석
plt.scatter(train['hour_bef_humidity'], train['count'])
train[['hour_bef_humidity', 'count']].corr(method = 'pearson')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour_bef_humidity</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>hour_bef_humidity</th>
      <td>1.000000</td>
      <td>-0.471142</td>
    </tr>
    <tr>
      <th>count</th>
      <td>-0.471142</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




![png](assets/img/output_15_1.png)



```python
# 가시성에 따른 이용량 분석
plt.scatter(train['hour_bef_visibility'], train['count'])
train[['hour_bef_visibility', 'count']].corr(method = 'pearson')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour_bef_visibility</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>hour_bef_visibility</th>
      <td>1.000000</td>
      <td>0.299094</td>
    </tr>
    <tr>
      <th>count</th>
      <td>0.299094</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




![png](assets/img/output_16_1.png)



```python
# 오존량에 따른 이용량 분석
plt.scatter(train['hour_bef_ozone'], train['count'])
train[['hour_bef_ozone', 'count']].corr(method = 'pearson')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour_bef_ozone</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>hour_bef_ozone</th>
      <td>1.000000</td>
      <td>0.477614</td>
    </tr>
    <tr>
      <th>count</th>
      <td>0.477614</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




![png](assets/img/output_17_1.png)



```python
# pm10에 따른 이용량 분석
plt.scatter(train['hour_bef_pm10'], train['count'])
train[['hour_bef_pm10', 'count']].corr(method = 'pearson')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour_bef_pm10</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>hour_bef_pm10</th>
      <td>1.000000</td>
      <td>-0.114288</td>
    </tr>
    <tr>
      <th>count</th>
      <td>-0.114288</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




![png](assets/img/output_18_1.png)



```python
# pm2.5에 따른 이용량 분석
plt.scatter(train['hour_bef_pm2.5'], train['count'])
train[['hour_bef_pm2.5', 'count']].corr(method = 'pearson')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour_bef_pm2.5</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>hour_bef_pm2.5</th>
      <td>1.000000</td>
      <td>-0.134293</td>
    </tr>
    <tr>
      <th>count</th>
      <td>-0.134293</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




![png](assets/img/output_19_1.png)



```python
# 오존농도를 기상청에서 알려주는 좋음/보통/나쁨/매우나쁨 을 기준으로 다른 class 부여
train['hour_bef_ozone'] = train['hour_bef_ozone'].apply(lambda x : 0 if x <= 0.03 else 
                                                                   1 if 0.03 < x and x <= 0.09 else 
                                                                   2 if 0.09 < x and x <= 0.151 else
                                                                   3 if 0.151 < x else x)
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.
    


```python
# 미세먼지를 기상청에서 알려주는 좋음/보통/나쁨/매우나쁨 을 기준으로 다른 class 부여
train['hour_bef_pm10'] = train['hour_bef_pm10'].apply(lambda x : 0 if x <= 30 else 
                                                                 1 if 30 < x and x <= 80 else 
                                                                 2 if 80 < x and x <= 150 else
                                                                 3 if 150 < x else x)
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.
    


```python
# 초미세먼지를 기상청에서 알려주는 좋음/보통/나쁨/매우나쁨 을 기준으로 다른 class 부여
train['hour_bef_pm2.5'] = train['hour_bef_pm2.5'].apply(lambda x : 0 if x <= 15 else 
                                                                   1 if 15 < x and x <= 35 else 
                                                                   2 if 35 < x and x <= 75 else
                                                                   3 if 75 < x else x)
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.
    


```python
train.median(axis = 0)
```




    id                        1113.0
    hour                        11.0
    hour_bef_temperature        16.6
    hour_bef_precipitation       0.0
    hour_bef_windspeed           2.3
    hour_bef_humidity           51.0
    hour_bef_visibility       1577.0
    hour_bef_ozone               1.0
    hour_bef_pm10                1.0
    hour_bef_pm2.5               1.0
    count                       96.0
    dtype: float64




```python
# 결측치를 median 값으로 처리
train.fillna(train.median(), inplace = True)
```

    /usr/local/lib/python3.6/dist-packages/pandas/core/generic.py:6287: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      self._update_inplace(new_data)
    


```python
# train data 확인
train.isnull().sum()
```




    id                        0
    hour                      0
    hour_bef_temperature      0
    hour_bef_precipitation    0
    hour_bef_windspeed        0
    hour_bef_humidity         0
    hour_bef_visibility       0
    hour_bef_ozone            0
    hour_bef_pm10             0
    hour_bef_pm2.5            0
    count                     0
    dtype: int64




```python
# train data 확인
train.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>hour</th>
      <th>hour_bef_temperature</th>
      <th>hour_bef_precipitation</th>
      <th>hour_bef_windspeed</th>
      <th>hour_bef_humidity</th>
      <th>hour_bef_visibility</th>
      <th>hour_bef_ozone</th>
      <th>hour_bef_pm10</th>
      <th>hour_bef_pm2.5</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>20</td>
      <td>16.3</td>
      <td>1.0</td>
      <td>1.5</td>
      <td>89.0</td>
      <td>576.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>49.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>13</td>
      <td>20.1</td>
      <td>0.0</td>
      <td>1.4</td>
      <td>48.0</td>
      <td>916.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>159.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>6</td>
      <td>13.9</td>
      <td>0.0</td>
      <td>0.7</td>
      <td>79.0</td>
      <td>1382.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>26.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# train data set의 id값 제거, count 값 분리, array형태 변환
train_x = np.array(train.iloc[:, 1:-1])
train_y = np.array(train.iloc[:, 0])
```

## Modeling


```python
# 모듈설치
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
```


```python
# 모델별 rmse를 확인하는 함수
def rmse(model):
  kf = KFold(10, shuffle = True, random_state = 42)
  rmse = np.sqrt(-cross_val_score(model, train_x, train_y, scoring = 'neg_mean_squared_error', cv = kf))
  print(np.mean(rmse))
```


```python
# 모델별 hyperparameter에 따른 rmse를 확인하는 함수
def girds(model, hyperparameters):
  kf = KFold(10, shuffle = True, random_state = 42)
  grid_search = GridSearchCV(model, param_grid = hyperparameters, scoring = 'neg_mean_squared_error', cv = kf)
  grid_search.fit(train_x, train_y)
  grid_search_result = grid_search.cv_results_
  for mean_score, params in zip(grid_search_result['mean_test_score'], grid_search_result['params']):
    print(np.sqrt(-mean_score), params)
```


```python
# 최종 제출본 만들기
def submission(prediction):
  sub = pd.DataFrame()
  sub['id'] = test['id']
  sub['pred'] = prediction
  sub.to_csv('submission.csv', index = False)
```


```python
lasso = make_pipeline(RobustScaler(), Lasso(random_state=3))
lasso_params = {'lasso__alpha' : [0.0001, 0.001, 0.01, 0.1]}
girds(lasso, lasso_params)
```

    52.75776537898072 {'lasso__alpha': 0.0001}
    52.75762990190659 {'lasso__alpha': 0.001}
    52.75632230762074 {'lasso__alpha': 0.01}
    52.74969915001244 {'lasso__alpha': 0.1}
    


```python
model_GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
rmse(model_GBoost)
```

    39.93306739247525
    


```python
model_rf = RandomForestRegressor()
rf_params = {'n_estimators': [3, 10, 30, 60, 90]}
girds(model_rf, rf_params)
```

    /usr/local/lib/python3.6/dist-packages/sklearn/model_selection/_search.py:814: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)
    

    43.87376948585613 {'n_estimators': 3}
    39.77442736314237 {'n_estimators': 10}
    39.145458200601915 {'n_estimators': 30}
    38.71933540674437 {'n_estimators': 60}
    38.703951845692636 {'n_estimators': 90}
    


```python
model_lgb = lgb.LGBMRegressor()
rmse(model_lgb)
```

    38.22539983547282
    


```python
model_xgb = xgb.XGBRegressor()
rmse(model_xgb)
```

    [07:38:56] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [07:38:56] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [07:38:56] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [07:38:56] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [07:38:57] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [07:38:57] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [07:38:57] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [07:38:57] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [07:38:57] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    [07:38:57] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    39.275307376492854
    


```python
# 가장 성능이 좋은 3개의 model을 선정해 학습
model_rf = RandomForestRegressor(n_estimators = 90)
model_rf.fit(train_x, train_y)
model_xgb = xgb.XGBRegressor()
model_xgb.fit(train_x, train_y)
model_lgb = lgb.LGBMRegressor()
model_lgb.fit(train_x, train_y)
```


```python
# test data set 처리
test_raw['hour_bef_ozone'] = test_raw['hour_bef_ozone'].apply(lambda x : 0 if x <= 0.03 else 
                                                                   1 if 0.03 < x and x <= 0.09 else 
                                                                   2 if 0.09 < x and x <= 0.151 else
                                                                   3 if 0.151 < x else x)
test_raw['hour_bef_pm10'] = test_raw['hour_bef_pm10'].apply(lambda x : 0 if x <= 30 else 
                                                                 1 if 30 < x and x <= 80 else 
                                                                 2 if 80 < x and x <= 150 else
                                                                 3 if 150 < x else x)
test_raw['hour_bef_pm2.5'] = test_raw['hour_bef_pm2.5'].apply(lambda x : 0 if x <= 15 else 
                                                                   1 if 15 < x and x <= 35 else 
                                                                   2 if 35 < x and x <= 75 else
                                                                   3 if 75 < x else x)
test = test_raw.fillna(train.median())
test_x = np.array(test.iloc[:, 1:])
print(test_x[0,0])
```


```python
# 3개의 모델을 적절한 비율로 ensemble하여 predict
ensemble = model_rf.predict(test_x) * 0.25 + model_lgb.predict(test_x) * 0.5 + model_xgb.predict(test_x) * 0.25
submission(ensemble)
```


```python

```
