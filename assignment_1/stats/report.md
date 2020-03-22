CSE472 (Machine Learning Sessional)
-----------------------------------

Assignment 1: Decision Tree and AdaBoost for Classification
-----------------------------------------------------------

Moaz Mahmud
1505064
-----------


Telco-customer-churn
--------------------
**Decision Tree**
|Performance measure                                | Training            | Test                |
|:--------------------------------------------------|:-------------------:|:-------------------:|
|Accuracy                                           | 97.95882144124955 % | 97.94180269694819 % |
|True positive rate (sensitivity, recall, hit rate) | 93.95134779750164 % | 93.96551724137932 % |
|True negative rate (specificity)                   | 99.44079747143203 % | 99.24599434495760 % |
|Positive predictive value (precision)              | 98.41597796143250 % | 97.61194029850746 % |
|False discovery rate                               | 1.584022038567493 % | 2.388059701492537 % |
|F1 score                                           | 96.13185334678775 % | 95.75402635431918 % |

**AdaBoost Accuracy**
|Number of boosting rounds| Training            | Test                |
|:------------------------|:-------------------:|:-------------------:|
|5                        | 78.34575789847355 % | 77.92760823278921 % |
|10                       | 79.33972310969116 % | 79.27608232789211 % |
|15                       | 79.60596379126731 % | 79.48899929027680 % |
|20                       | 79.97870074547390 % | 79.70191625266146 % |



Adult
-----
**Decision Tree**
|Performance measure                                | Training            | Test                |
|:--------------------------------------------------|:-------------------:|:-------------------:|
|Accuracy                                           | 96.11486486486487 % | 96.05404575464456 % |
|True positive rate (sensitivity, recall, hit rate) | 87.58262511803588 % | 93.96551724137932 % |
|True negative rate (specificity)                   | 98.86767543414237 % | 98.38838042180662 % |
|Positive predictive value (precision)              | 96.14720110573600 % | 94.18103448275862 % |
|False discovery rate                               | 3.852798894263994 % | 5.818965517241379 % |
|F1 score                                           | 91.66529402075440 % | 91.07328933657520 % |

**AdaBoost Accuracy**
|Number of boosting rounds| Training            | Test                |
|:------------------------|:-------------------:|:-------------------:|
|5                        | 83.26550982800983 % | 83.44848764010440 % |
|10                       | 83.65709459459460 % | 83.34101028711807 % |
|15                       | 84.29821867321867 % | 84.53861507753724 % |
|20                       | 84.69748157248156 % | 84.63073852295409 % |



Creditcard Fraud
----------------
**Decision Tree**
|Performance measure                                | Training            | Test                |
|:--------------------------------------------------|:-------------------:|:-------------------:|
|Accuracy                                           | 100.0 %             | 100.0 %             |
|True positive rate (sensitivity, recall, hit rate) | 100.0 %             | 100.0 %             |
|True negative rate (specificity)                   | 100.0 %             | 100.0 %             |
|Positive predictive value (precision)              | 100.0 %             | 100.0 %             |
|False discovery rate                               | 0.0 %               | 0.0 %               |
|F1 score                                           | 100.0 %             | 100.0 %             |


**AdaBoost Accuracy**
|Number of boosting rounds| Training            | Test                |
|:------------------------|:-------------------:|:-------------------:|
|5                        | 99.48148600012200 % | 98.68260551353988 % |
|10                       | 99.62788995302874 % | 98.56062454257136 % |
|15                       | 99.74989324711767 % | 98.56062454257136 % |
|20                       | 99.87799670591106 % | 98.78019029031470 % |