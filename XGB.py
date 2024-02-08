import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
import scipy.stats as stats
import xgboost as xgb
import scipy.stats
from sklearn.model_selection import train_test_split
from shaphypetune import BoostRFE
from sklearn.metrics import roc_curve, auc
warnings.filterwarnings("ignore")

for fname in os.listdir("./Data/"):
	if os.path.isdir("./Data/"+str(fname)): continue

	base = str(fname).split(".")[0]
	base = "XGB/"+str(base)
	filename = str(fname)[:len(fname)-4]

	# Load data & drop unneeded variables
	df = pd.read_csv("./Data/"+str(fname))
	outcome = df[['Case-control status']]
	scaler = len(df[df['Case-control status'] == 0])/len(df[df['Case-control status'] == 1])

	df = df.drop(["Event","Patient ID","Case ID","Case-control status"], axis=1)

	sex = df['Sex_Male']
	df.drop("Sex_Male", axis=1, inplace=True)

	corr_matrix = df.corr().abs()
	upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
	to_drop = [column for column in upper.columns if any(upper[column] >= 0.75)]
	pd.DataFrame(to_drop, columns=['Features']).to_csv("./"+base+"/"+filename+".SurrogateMarkers.csv", index=False)
	df.drop(to_drop, axis=1, inplace=True)

	df['Sex_Male'] = sex

	X_train, X_test, ye_train, y_test = train_test_split(df, outcome, test_size=0.33, random_state=42)

	x_train, x_valid, y_train, y_valid = train_test_split(X_train, ye_train, test_size=0.33, random_state=42)
	clf_xgb = xgb.XGBClassifier(n_estimators=150, random_state=42, verbosity=0, eval_metric='auc', early_stopping_rounds=6)
	param_grid = {
		'booster':['gbtree'],
		'eta':[0.01, 0.1, 1, 10],
		'min_child_weight':[1, 3, 5],
		'max_depth':[1, 3, 5],
		'gamma':[0, 1, 3, 5],
		'subsample':[0.5, 0.75, 1],
		'lambda':[0, 1, 3, 5],
		'alpha':[0, 1, 3, 5],
		'scale_pos_weight':[scaler]
		}
	model = BoostRFE( clf_xgb, param_grid=param_grid, min_features_to_select=2, step=1, n_iter=8, sampling_seed=42, importance_type="feature_importances", train_importance=False, \
		greater_is_better=True, n_jobs=14)
	model.fit(x_train, y_train, eval_set=[(x_train,y_train),(x_valid, y_valid)])
	pickle.dump(model, open('./'+base+'/'+filename+'.Model.sav', 'wb'))

	# Can load model with:
	#model = pickle.load(open('./'+base+'/'+filename+'.Model.sav', 'rb'))
	# result = loaded_model.score(X_test, y_test)

	print(model.best_params_)
	print(df.columns[model.support_])

	X_train, X_test = X_train[X_train.columns[model.support_]], X_test[X_test.columns[model.support_]]
	model = xgb.XGBClassifier(**model.best_params_).fit(X_train, ye_train)

	train_pred = model.predict_proba(X_train)[:,1]
	fpr2, tpr2, _ = roc_curve(ye_train, train_pred)
	score2 = auc(fpr2, tpr2)

	ypred = model.predict_proba(X_test)[:,1]
	fpr, tpr, _ = roc_curve(y_test, ypred)
	score = auc(fpr, tpr)

	confidence = 0.95 # Adjusted to the designed confidence level
	z_level= scipy.stats.norm.ppf((1 + confidence)/2)
	ci_length = z_level * np.sqrt((score * (1 - score))/y_test.shape[0])
	ci_lower = score - ci_length
	ci_upper = score + ci_length

	plt.figure(figsize=(6,6), dpi=1200)
	plt.title("Receiver Operating Characteristic - Baseline All-Cause")
	plt.plot(fpr2, tpr2, label='Training (AUC=%.3f)'%score2, color='blue')
	plt.plot(fpr,tpr,label='Test (AUC=%0.3f [%.3f, %.3f])'%(score, ci_lower, ci_upper), color='darkorange')
	plt.plot([0,1],[0,1],'r--', label='Chance', linewidth=1)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title("Receiver Operating Characteristic")
	plt.legend(loc="lower right")
	plt.savefig("./"+base+"/"+filename+".AUC.png")
	plt.cla()

	plt.figure(figsize=(6,6), dpi=1200)
	xgb.plot_importance(model)
	plt.savefig("./"+base+"/"+filename+"FI.png")
	plt.cla()
