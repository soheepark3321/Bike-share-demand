![header](https://capsule-render.vercel.app/api?type=waving&color=0:e1eec3,100:f05053&height=200&text=bike&nbsp;Share&nbsp;Demand&nbsp;Prediction&fontSize=55&fontAlignY=35&fontColor=FF4F8B&animation=fadeIn)

## Bike Shareing System
자전거 대여 사업은 정부/민간이 대여 체계를 구성하여 이용료를 지불하는 시민들에게 서비스를 제공하는 사업입니다.

자원을 공유하여 사회적 낭비를 감소시킬 수 있는 공유 경제 모델이며, 교통체증 및 환경오염이 감소되는 효과는 물론<br>
시민들은 이 서비스를 이용하여 적은 비용으로 편리한 이동이 가능합니다.

자전거 대여가 대중들 사이에서 활발히 이루어지기 위해서는 안정적인 자전거 공급을 위해 자전거 대여와 연관된 data들을 살펴볼 필요가 있으며, 직관적인 결과의 지표인 'count' 수의 흐름을 분석할 필요가 있습니다.

## Goal
1. [Kaggle] 2011 - 2012년 시간당 자전거 대여수 데이터를 이용하여 Test dataset(20일 ~ 말일)의 count 수를 예측
2. Capital Bike Share의 전반적인 자전거 대여 수 동향 예측 및 분석

## Data
<a href="https://www.kaggle.com/competitions/bike-sharing-demand/data"><img src="https://img.shields.io/badge/-white?style=flat-square&logo=kaggle&logoColor=20BEFF" width=100 /></a> <i>(data link)</i>

해당 dataset은 정형 및 시계열 데이터이며, 지도 학습 및 다중 회귀에 해당됩니다.<br>
따라서 해당 분석에 (교차검증)scikit-learn의 GridSearchCV과 회귀 모델을 사용하였습니다.

* **Train.csv** : 2011 - 2012년 월별 1 ~ 19일 (row)10,886 * (col)12 = 130,632개 data<br>
* **Test.csv**  : 2011 - 2012년 월별 20 ~ 말일 (row)6,493 * (col) 9 = 58,437개 data<br>
* Total     : 189,069건

|Coulmn|설명|
|:---:|:---|
|datetime|시간 (연/월/일/시/분/초)|
|season|계절 (봄 : 1 ~ 겨울 : 4)|
|holiday|주말 (휴일(1), 근무일(0))|
|workingday|근무일 (근무일(1), 휴일(0))|
|weather|날씨 (clean : 1, coludy : 2, snow&rain : 3, heavy snow&rain : 4)|
|temp|온도 (Celsius)|
|atemp|체감 온도 (Celsius)|
|humidity|습도|
|windspeed|풍속|
|casual|비회원 자전거 대여량 (test.csv (x))|
|registered|회원 자전거 대여량 (test.csv (x))|
|count|총 자전거 대여량 (test.csv (x))|

## Evaluation
1. Kaggle : RMSLE
2. Overall Data Analysis : MSE, RMSE, R2, Adjusted_R2

## Models
* Linear Regression
* Ridge
* Lasso
* Elasticnet
* Random Forest
* Gradient Boosting
* XGBoost (Extreme Gradient Boost)
* CatBoost
* LightGBM

## Conclusion

1. 사람들은 근무일보다 공휴일에 자전거를 더 많이 대여합니다.

2. 자전거가 많이 대여되는 시간은 근무일 기준, 출퇴근 때이고<br>
공휴일 기준, 늦은 아침부터 오후까지입니다.

3. 근무일 & 휴일 기준, 늦은 저녁 ~ 새벽까지가 대여율이 가장 낮습니다.

4. 자전거를 많이 빌리는 계절은 여름과 가을입니다.<br>
반대로 봄은 자전거 대여량이 가장 적습니다.

5. 날씨가 화창할 수록 자전거 대여량이 많습니다.<br>
눈과 비가 많이 오는 날에는 자전거 대여량이 거의 없습니다.

6. 극단적인 온도(너무 춥거나 더움)가 되면 자전거 대여율이 굉장히 낮습니다.<br>
자전거 대여율은 적당한 온도일 때 높습니다.

7. 전체적으로 앙상블 기법을 이용하는 모델의 성능이 우수하게 나타납니다.<br><br>
<가장 뛰어난 모델>
* Test lable 예측 : XGBoost
* 완전한 2011 - 2012 데이터의 lable 예측 : CatBoost
***

<div align=left>
📚 <b>Language<b> 📚  <br> </P>
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=yellow"/><br><br><br>

✏ <b>Library<b> ✏<br></P>
<img src="https://img.shields.io/badge/NumPy-blue?style=flat-square&logo=NumPy&logoColor=013243"/>
<img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit-learn-40AEF0?style=flat-square&logo=scikit-learn&logoColor=F7931E"/>
<img src="https://img.shields.io/badge/Matplotlib-004088?style=flat-square&logo=Matplotlib&logoColor=white"/>
<img src="https://img.shields.io/badge/Seaborn-26689A?style=flat-square&logo=Seaborn&logoColor=071D49"/>
<img src="https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=SciPy&logoColor=071D49"/> <br><br><br>


🛠 <b>Tools<b> 🛠<br></P>
<img src="https://img.shields.io/badge/Anaconda-44A833?style=flat-square&logo=Anaconda&logoColor=green"/>
<img src="https://img.shields.io/badge/Jupyter Notebook-F37626?style=flat-square&logo=Jupyter&logoColor=white"/></div>


![footer](https://capsule-render.vercel.app/api?type=waving&color=0:e1eec3,100:f05053&height=200&text=Personal&nbsp;Project&fontSize=35&fontAlignY=80&fontColor=FF4F8B&animation=fadeIn&section=footer)
