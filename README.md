# sporf-benchmarking
Benchmarking sporf in cuml against ydf and/or sklearn

```
conda create -n sporf_benchmarking python=3.13
conda activate sporf_benchmarking
pip install -U ydf
pip install -U pandas
cd src
python ./main.py
```

Outcome:
```
(sporf_benchmarking) scarlil1@degenerate:~/src/github/sporf-benchmarking/src$ python ./main.py 
   age     workclass  fnlwgt     education  education_num      marital_status  ...     sex capital_gain capital_loss hours_per_week      native_country  income
0   44       Private  228057       7th-8th              4  Married-civ-spouse  ...  Female            0            0             40  Dominican-Republic   <=50K
1   20       Private  299047  Some-college             10       Never-married  ...  Female            0            0             20       United-States   <=50K
2   40       Private  342164       HS-grad              9           Separated  ...  Female            0            0             37       United-States   <=50K
3   30       Private  361742  Some-college             10  Married-civ-spouse  ...    Male            0            0             50       United-States   <=50K
4   67  Self-emp-inc  171564       HS-grad              9  Married-civ-spouse  ...  Female        20051            0             30             England    >50K

[5 rows x 15 columns]
Train model on 22792 examples
Model trained in 0:00:01.014807
Type: "GRADIENT_BOOSTED_TREES"
Task: CLASSIFICATION
Label: "income"

Input Features (14):
	age
	workclass
	fnlwgt
	education
	education_num
	marital_status
	occupation
	relationship
	race
	sex
	capital_gain
	capital_loss
	hours_per_week
	native_country

No weights

Variable Importance: INV_MEAN_MIN_DEPTH:
    1.            "age"  0.226642 ################
    2.     "occupation"  0.219727 #############
    3.   "capital_gain"  0.214876 ############
    4.      "education"  0.213746 ###########
    5. "marital_status"  0.212739 ###########
    6.   "relationship"  0.206040 #########
    7.         "fnlwgt"  0.203843 ########
    8. "hours_per_week"  0.203735 ########
    9.   "capital_loss"  0.196549 ######
   10. "native_country"  0.190548 ####
   11.      "workclass"  0.187795 ###
   12.  "education_num"  0.184215 ##
   13.           "race"  0.180495 
   14.            "sex"  0.177647 

Variable Importance: NUM_AS_ROOT:
    1.            "age" 26.000000 ################
    2.   "capital_gain" 26.000000 ################
    3. "marital_status" 20.000000 ############
    4.   "relationship" 17.000000 ##########
    5.   "capital_loss" 14.000000 ########
    6. "hours_per_week" 14.000000 ########
    7.      "education" 12.000000 #######
    8.         "fnlwgt" 10.000000 #####
    9.           "race"  9.000000 #####
   10.  "education_num"  7.000000 ###
   11.            "sex"  4.000000 #
   12.     "occupation"  2.000000 
   13.      "workclass"  1.000000 
   14. "native_country"  1.000000 

Variable Importance: NUM_NODES:
    1.     "occupation" 724.000000 ################
    2.         "fnlwgt" 513.000000 ###########
    3.            "age" 483.000000 ##########
    4.      "education" 464.000000 ##########
    5. "hours_per_week" 339.000000 #######
    6.   "capital_gain" 326.000000 ######
    7. "native_country" 306.000000 ######
    8.   "capital_loss" 297.000000 ######
    9.   "relationship" 262.000000 #####
   10.      "workclass" 244.000000 #####
   11. "marital_status" 210.000000 ####
   12.  "education_num" 82.000000 #
   13.            "sex" 42.000000 
   14.           "race" 21.000000 

Variable Importance: SUM_SCORE:
    1.   "relationship" 3014.690076 ################
    2.   "capital_gain" 2065.521668 ##########
    3.      "education" 1144.490954 ######
    4. "marital_status" 1111.389695 #####
    5.     "occupation" 1094.619502 #####
    6.  "education_num" 796.666823 ####
    7.   "capital_loss" 584.055066 ###
    8.            "age" 582.288569 ###
    9. "hours_per_week" 366.856509 #
   10. "native_country" 263.872689 #
   11.         "fnlwgt" 216.537764 #
   12.      "workclass" 196.085503 #
   13.            "sex" 47.217730 
   14.           "race"  5.428727 



Loss: BINOMIAL_LOG_LIKELIHOOD
Validation loss value: 0.576162
Number of trees per iteration: 1
Node format: NOT_SET
Number of trees: 163
Total number of nodes: 8789

Number of nodes by tree:
Count: 163 Average: 53.9202 StdDev: 6.73147
Min: 31 Max: 63 Ignored: 0
----------------------------------------------
[ 31, 32)  1   0.61%   0.61%
[ 32, 34)  2   1.23%   1.84% #
[ 34, 35)  0   0.00%   1.84%
[ 35, 37)  1   0.61%   2.45%
[ 37, 39)  1   0.61%   3.07%
[ 39, 40)  4   2.45%   5.52% #
[ 40, 42)  2   1.23%   6.75% #
[ 42, 44)  3   1.84%   8.59% #
[ 44, 45)  0   0.00%   8.59%
[ 45, 47)  7   4.29%  12.88% ##
[ 47, 49)  4   2.45%  15.34% #
[ 49, 50) 15   9.20%  24.54% ####
[ 50, 52) 14   8.59%  33.13% ####
[ 52, 54) 11   6.75%  39.88% ###
[ 54, 55)  0   0.00%  39.88%
[ 55, 57) 23  14.11%  53.99% #######
[ 57, 59) 18  11.04%  65.03% #####
[ 59, 60) 34  20.86%  85.89% ##########
[ 60, 62) 14   8.59%  94.48% ####
[ 62, 63]  9   5.52% 100.00% ###

Depth by leafs:
Count: 4476 Average: 4.8702 StdDev: 0.41894
Min: 1 Max: 5 Ignored: 0
----------------------------------------------
[ 1, 2)    1   0.02%   0.02%
[ 2, 3)   14   0.31%   0.34%
[ 3, 4)   92   2.06%   2.39%
[ 4, 5)  351   7.84%  10.23% #
[ 5, 5] 4018  89.77% 100.00% ##########

Number of training obs by leaf:
Count: 4476 Average: 747.739 StdDev: 2462.57
Min: 5 Max: 20207 Ignored: 0
----------------------------------------------
[     5,  1015) 3898  87.09%  87.09% ##########
[  1015,  2025)  189   4.22%  91.31%
[  2025,  3035)   98   2.19%  93.50%
[  3035,  4045)   71   1.59%  95.08%
[  4045,  5055)   37   0.83%  95.91%
[  5055,  6065)   31   0.69%  96.60%
[  6065,  7076)   17   0.38%  96.98%
[  7076,  8086)   18   0.40%  97.39%
[  8086,  9096)   14   0.31%  97.70%
[  9096, 10106)   16   0.36%  98.06%
[ 10106, 11116)   11   0.25%  98.30%
[ 11116, 12126)   10   0.22%  98.53%
[ 12126, 13136)   10   0.22%  98.75%
[ 13136, 14147)    6   0.13%  98.88%
[ 14147, 15157)   10   0.22%  99.11%
[ 15157, 16167)    3   0.07%  99.17%
[ 16167, 17177)    0   0.00%  99.17%
[ 17177, 18187)    2   0.04%  99.22%
[ 18187, 19197)   17   0.38%  99.60%
[ 19197, 20207]   18   0.40% 100.00%

Attribute in nodes:
	724 : occupation [CATEGORICAL]
	513 : fnlwgt [NUMERICAL]
	483 : age [NUMERICAL]
	464 : education [CATEGORICAL]
	339 : hours_per_week [NUMERICAL]
	326 : capital_gain [NUMERICAL]
	306 : native_country [CATEGORICAL]
	297 : capital_loss [NUMERICAL]
	262 : relationship [CATEGORICAL]
	244 : workclass [CATEGORICAL]
	210 : marital_status [CATEGORICAL]
	82 : education_num [NUMERICAL]
	42 : sex [CATEGORICAL]
	21 : race [CATEGORICAL]

Attribute in nodes with depth <= 0:
	26 : capital_gain [NUMERICAL]
	26 : age [NUMERICAL]
	20 : marital_status [CATEGORICAL]
	17 : relationship [CATEGORICAL]
	14 : hours_per_week [NUMERICAL]
	14 : capital_loss [NUMERICAL]
	12 : education [CATEGORICAL]
	10 : fnlwgt [NUMERICAL]
	9 : race [CATEGORICAL]
	7 : education_num [NUMERICAL]
	4 : sex [CATEGORICAL]
	2 : occupation [CATEGORICAL]
	1 : native_country [CATEGORICAL]
	1 : workclass [CATEGORICAL]

Attribute in nodes with depth <= 1:
	73 : capital_gain [NUMERICAL]
	55 : capital_loss [NUMERICAL]
	53 : age [NUMERICAL]
	51 : marital_status [CATEGORICAL]
	49 : fnlwgt [NUMERICAL]
	41 : occupation [CATEGORICAL]
	34 : hours_per_week [NUMERICAL]
	34 : education [CATEGORICAL]
	29 : relationship [CATEGORICAL]
	17 : native_country [CATEGORICAL]
	17 : education_num [NUMERICAL]
	15 : workclass [CATEGORICAL]
	11 : race [CATEGORICAL]
	9 : sex [CATEGORICAL]

Attribute in nodes with depth <= 2:
	152 : capital_gain [NUMERICAL]
	125 : occupation [CATEGORICAL]
	122 : fnlwgt [NUMERICAL]
	120 : capital_loss [NUMERICAL]
	109 : age [NUMERICAL]
	100 : education [CATEGORICAL]
	86 : marital_status [CATEGORICAL]
	73 : hours_per_week [NUMERICAL]
	71 : relationship [CATEGORICAL]
	63 : native_country [CATEGORICAL]
	51 : workclass [CATEGORICAL]
	24 : education_num [NUMERICAL]
	15 : sex [CATEGORICAL]
	13 : race [CATEGORICAL]

Attribute in nodes with depth <= 3:
	354 : occupation [CATEGORICAL]
	268 : fnlwgt [NUMERICAL]
	251 : age [NUMERICAL]
	241 : capital_gain [NUMERICAL]
	224 : education [CATEGORICAL]
	206 : capital_loss [NUMERICAL]
	152 : hours_per_week [NUMERICAL]
	138 : native_country [CATEGORICAL]
	129 : workclass [CATEGORICAL]
	128 : relationship [CATEGORICAL]
	124 : marital_status [CATEGORICAL]
	50 : education_num [NUMERICAL]
	22 : sex [CATEGORICAL]
	17 : race [CATEGORICAL]

Attribute in nodes with depth <= 5:
	724 : occupation [CATEGORICAL]
	513 : fnlwgt [NUMERICAL]
	483 : age [NUMERICAL]
	464 : education [CATEGORICAL]
	339 : hours_per_week [NUMERICAL]
	326 : capital_gain [NUMERICAL]
	306 : native_country [CATEGORICAL]
	297 : capital_loss [NUMERICAL]
	262 : relationship [CATEGORICAL]
	244 : workclass [CATEGORICAL]
	210 : marital_status [CATEGORICAL]
	82 : education_num [NUMERICAL]
	42 : sex [CATEGORICAL]
	21 : race [CATEGORICAL]

Condition type in nodes:
	2271 : ContainsBitmapCondition
	2040 : HigherCondition
	2 : ContainsCondition
Condition type in nodes with depth <= 0:
	97 : HigherCondition
	66 : ContainsBitmapCondition
Condition type in nodes with depth <= 1:
	281 : HigherCondition
	207 : ContainsBitmapCondition
Condition type in nodes with depth <= 2:
	600 : HigherCondition
	524 : ContainsBitmapCondition
Condition type in nodes with depth <= 3:
	1168 : HigherCondition
	1134 : ContainsBitmapCondition
	2 : ContainsCondition
Condition type in nodes with depth <= 5:
	2271 : ContainsBitmapCondition
	2040 : HigherCondition
	2 : ContainsCondition

Training logs:
Number of iteration to final model: 163
	Iter:1 train-loss:1.015263 valid-loss:1.069998  train-accuracy:0.761895 valid-accuracy:0.736609
	Iter:2 train-loss:0.954158 valid-loss:1.005950  train-accuracy:0.761895 valid-accuracy:0.736609
	Iter:3 train-loss:0.905791 valid-loss:0.956294  train-accuracy:0.761895 valid-accuracy:0.736609
	Iter:4 train-loss:0.865922 valid-loss:0.915507  train-accuracy:0.802708 valid-accuracy:0.778663
	Iter:5 train-loss:0.832547 valid-loss:0.880311  train-accuracy:0.816491 valid-accuracy:0.792386
	Iter:6 train-loss:0.803634 valid-loss:0.850085  train-accuracy:0.816734 valid-accuracy:0.792386
	Iter:16 train-loss:0.651520 valid-loss:0.687656  train-accuracy:0.864121 valid-accuracy:0.854360
	Iter:26 train-loss:0.595837 valid-loss:0.635662  train-accuracy:0.870550 valid-accuracy:0.861000
	Iter:36 train-loss:0.567251 valid-loss:0.612381  train-accuracy:0.875664 valid-accuracy:0.860558
	Iter:46 train-loss:0.546733 valid-loss:0.599879  train-accuracy:0.881216 valid-accuracy:0.866313
	Iter:56 train-loss:0.531772 valid-loss:0.592656  train-accuracy:0.883894 valid-accuracy:0.871182
	Iter:66 train-loss:0.519639 valid-loss:0.588026  train-accuracy:0.886914 valid-accuracy:0.870297
	Iter:76 train-loss:0.508524 valid-loss:0.584333  train-accuracy:0.888959 valid-accuracy:0.871625
	Iter:86 train-loss:0.499619 valid-loss:0.581144  train-accuracy:0.891200 valid-accuracy:0.871625
	Iter:96 train-loss:0.492438 valid-loss:0.580198  train-accuracy:0.892222 valid-accuracy:0.868969
	Iter:106 train-loss:0.485893 valid-loss:0.580047  train-accuracy:0.894511 valid-accuracy:0.869854
	Iter:116 train-loss:0.479146 valid-loss:0.579222  train-accuracy:0.895729 valid-accuracy:0.869854
	Iter:126 train-loss:0.473333 valid-loss:0.578067  train-accuracy:0.897385 valid-accuracy:0.869854
	Iter:136 train-loss:0.467836 valid-loss:0.576896  train-accuracy:0.899089 valid-accuracy:0.868083
	Iter:146 train-loss:0.462477 valid-loss:0.577037  train-accuracy:0.900356 valid-accuracy:0.867198
	Iter:156 train-loss:0.457704 valid-loss:0.576897  train-accuracy:0.901427 valid-accuracy:0.867198
	Iter:166 train-loss:0.452869 valid-loss:0.576599  train-accuracy:0.902791 valid-accuracy:0.868526
	Iter:176 train-loss:0.446772 valid-loss:0.578243  train-accuracy:0.903911 valid-accuracy:0.868526
	Iter:186 train-loss:0.442635 valid-loss:0.577331  train-accuracy:0.904593 valid-accuracy:0.866755

DATASPEC:

Number of records: 22792
Number of columns: 15

Number of columns by type:
	CATEGORICAL: 9 (60%)
	NUMERICAL: 6 (40%)

Columns:

CATEGORICAL: 9 (60%)
	0: "income" CATEGORICAL has-dict vocab-size:3 zero-ood-items most-frequent:"<=50K" 17308 (75.9389%) dtype:DTYPE_BYTES
	2: "workclass" CATEGORICAL num-nas:1257 (5.51509%) has-dict vocab-size:8 num-oods:3 (0.0139308%) most-frequent:"Private" 15879 (73.7358%) dtype:DTYPE_BYTES
	4: "education" CATEGORICAL has-dict vocab-size:17 zero-ood-items most-frequent:"HS-grad" 7340 (32.2043%) dtype:DTYPE_BYTES
	6: "marital_status" CATEGORICAL has-dict vocab-size:8 zero-ood-items most-frequent:"Married-civ-spouse" 10431 (45.7661%) dtype:DTYPE_BYTES
	7: "occupation" CATEGORICAL num-nas:1260 (5.52826%) has-dict vocab-size:14 num-oods:4 (0.018577%) most-frequent:"Prof-specialty" 2870 (13.329%) dtype:DTYPE_BYTES
	8: "relationship" CATEGORICAL has-dict vocab-size:7 zero-ood-items most-frequent:"Husband" 9191 (40.3256%) dtype:DTYPE_BYTES
	9: "race" CATEGORICAL has-dict vocab-size:6 zero-ood-items most-frequent:"White" 19467 (85.4115%) dtype:DTYPE_BYTES
	10: "sex" CATEGORICAL has-dict vocab-size:3 zero-ood-items most-frequent:"Male" 15165 (66.5365%) dtype:DTYPE_BYTES
	14: "native_country" CATEGORICAL num-nas:407 (1.78571%) has-dict vocab-size:41 num-oods:1 (0.00446728%) most-frequent:"United-States" 20436 (91.2933%) dtype:DTYPE_BYTES

NUMERICAL: 6 (40%)
	1: "age" NUMERICAL mean:38.6153 min:17 max:90 sd:13.661 dtype:DTYPE_INT64
	3: "fnlwgt" NUMERICAL mean:189879 min:12285 max:1.4847e+06 sd:106423 dtype:DTYPE_INT64
	5: "education_num" NUMERICAL mean:10.0927 min:1 max:16 sd:2.56427 dtype:DTYPE_INT64
	11: "capital_gain" NUMERICAL mean:1081.9 min:0 max:99999 sd:7509.48 dtype:DTYPE_INT64
	12: "capital_loss" NUMERICAL mean:87.2806 min:0 max:4356 sd:403.01 dtype:DTYPE_INT64
	13: "hours_per_week" NUMERICAL mean:40.3955 min:1 max:99 sd:12.249 dtype:DTYPE_INT64

Terminology:
	nas: Number of non-available (i.e. missing) values.
	ood: Out of dictionary.
	manually-defined: Attribute whose type is manually defined by the user, i.e., the type was not automatically inferred.
	tokenized: The attribute value is obtained through tokenization.
	has-dict: The attribute is attached to a string dictionary e.g. a categorical attribute stored as a string.
	vocab-size: Number of unique values.

[0.01860435 0.36130956 0.83858865 0.04385567 0.02917649]
Test accuracy: 0.8738867847271983
Full evaluation report:
accuracy: 0.873887
confusion matrix:
    label (row) \ prediction (col)
    +-------+-------+-------+
    |       | <=50K |  >50K |
    +-------+-------+-------+
    | <=50K |  6962 |   450 |
    +-------+-------+-------+
    |  >50K |   782 |  1575 |
    +-------+-------+-------+
characteristics:
    name: '>50K' vs others
    ROC AUC: 0.929614
    PR AUC: 0.831149
    Num thresholds: 9606
loss: 0.276673
num examples: 9769
num examples (weighted): 9769

```
