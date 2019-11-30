programming language: python3

I finally select two models to get the final scores, the file structure and description are as below:

- linear
	- dataprocessing.py: data preprocessing
	- trainLinear.py: train a ridge model and make prediction
	- submission_ridge.csv: the 1st submission for final scores

- xgboost
	- dataprocessing.py: data preprocessing
	- trainXgboost.py: train a xgboost model and make prediction
	- submission_xgb.csv: the 2nd submission for final scores

- train.csv
- test.csv
- readme.txt


How to reproduce my results:
run trainLinear.py to get submission_ridge.csv
run trainXgboost.py to get submission_xgb.csv