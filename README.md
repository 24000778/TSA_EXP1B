# Ex.No: 1B                     CONVERSION OF NON STATIONARY TO STATIONARY DATA
# Date: 02/02/2026
# NAME: SINDHU PRIYA REDDY G
# 212224040319
# DATASET: Tesla stock price Dataset

### AIM:
To perform regular differncing,seasonal adjustment and log transformatio on international airline passenger data

### ALGORITHM:
1. Import the required packages like pandas and numpy
2. Read the data using the pandas
3. Perform the data preprocessing if needed and apply regular differncing,seasonal adjustment,log transformation.
4. Plot the data according to need, before and after regular differncing,seasonal adjustment,log transformation.
5. Display the overall results.
   
### PROGRAM:

```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

data=pd.read_csv('/content/Tesla Dataset.csv')
data.head()

data['date']=pd.to_datetime(data['date'])
data.set_index('date',inplace=True)
data['open_diff']=data['open']-data['open'].shift(1)

result = seasonal_decompose(data['open'], model='additive', period=12)
data['open_sea_diff']=result.resid

data['open_log'] = np.log(data['open'])
data['open_log_diff']=data['open_log']-data['open_log'].shift(1)

result = seasonal_decompose(data['open_log_diff'].dropna(), model='additive', period=12)
data['open_log_seasonal_diff']=result.resid

plt.figure(figsize=(16, 16))

plt.subplot(6, 1, 1)
plt.plot(data['open'], label='Original')
plt.legend(loc='best')
plt.title('Original Data')
plt.xlabel('date')
plt.ylabel('open')

plt.subplot(6, 1, 2)
plt.plot(data['open_diff'], label='Regular Difference')
plt.legend(loc='best')
plt.title('Regular Differencing')
plt.xlabel('date')
plt.ylabel('open')

plt.subplot(6, 1, 3)
plt.plot(data['open_sea_diff'], label='Seasonal Adjustment')
plt.legend(loc='best')
plt.title('Seasonal Adjustment')
plt.xlabel('date')
plt.ylabel('open')

plt.subplot(6, 1, 4)
plt.plot(data['open_log'], label='Log Transformation')
plt.legend(loc='best')
plt.title('Log Transformation')
plt.xlabel('Date')
plt.ylabel('Log(No of open)')

plt.subplot(6, 1, 5)
plt.plot(data['open_log_diff'], label='Log Transformation and Regular Differencing')
plt.legend(loc='best')
plt.title('Log Transformation and Regular Differencing')
plt.xlabel('Date')
plt.ylabel('RDiff(Log(No of open))')

plt.figure(figsize=(16, 16))
plt.subplot(6, 1, 6)
plt.plot(data['open_log_seasonal_diff'], label='Log Transformation and regular Differencing and Seasonal Differencing')
plt.legend(loc='best')
plt.title('Log Transformation and Regular Differencing and Seasonal Differencing')
plt.xlabel('Date')
plt.ylabel('SDiff(RDiff(Log(No of open)))')

plt.tight_layout()
plt.show()

data.plot(kind='line')

```

### OUTPUT:


ORIGINAL DATASET:

<img width="922" height="216" alt="image" src="https://github.com/user-attachments/assets/e6cd7967-bcd4-4751-acb4-1903a3ad04ec" />



REGULAR DIFFERENCING:

<img width="889" height="212" alt="image" src="https://github.com/user-attachments/assets/83953d89-7e4a-477d-affb-db7addf11ca4" />



SEASONAL ADJUSTMENT:

<img width="957" height="218" alt="image" src="https://github.com/user-attachments/assets/04561619-1e23-409f-acdf-d4066e77cfa2" />


LOG TRANSFORMATION:

<img width="849" height="223" alt="image" src="https://github.com/user-attachments/assets/30ea63da-ad8c-4549-ae42-98dd1d528754" />


LOG TRANSFORMATION and REGULAR DIFFERENCING and SEASONAL DIFFERENCING


<img width="1330" height="279" alt="image" src="https://github.com/user-attachments/assets/eb0882ba-4f37-49bf-b767-21493c54c59b" />



OVERALL VIEW

<img width="809" height="576" alt="image" src="https://github.com/user-attachments/assets/ede38bf3-c36a-4517-bec8-aa209fd9ece6" />






### RESULT:
Thus we have created the python code for the conversion of non stationary to stationary data on international airline passenger
data.
