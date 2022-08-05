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
* **Train.csv** : 2011 - 2012년 월별 1 ~ 19일 (row)10,886 * (col)12 = 130,632개 data<br>
* **Test.csv**  : 2011 - 2012년 월별 20 ~ 말일 (row)6,493 * (col) 9 = 58,437개 data<br>
* Total     : 189,069건

|Coulmn|설명|
|------|---|
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
