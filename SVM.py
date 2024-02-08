import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import scipy.stats as stats
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV, RFE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
warnings.filterwarnings("ignore")

for fname in os.listdir("./Data/"):
	if os.path.isdir("./Data/"+str(fname)): continue

	base = str(fname).split(".")[0]
	base = "SVM/"+str(base)
	filename = str(fname)[:len(fname)-4]

	# Load data & drop unneeded variables
	df = pd.read_csv("./Data/"+str(fname))
	outcome = df[['Case-control status']]
	df = df.drop(["Event","Patient ID","Case ID","Case-control status"], axis=1)

	# Remove categorical variable
	sex = df['Sex_Male']
	df.drop("Sex_Male", axis=1, inplace=True)

	# Identify and drop correlated features
	corr_matrix = df.corr().abs()
	upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
	to_drop = [column for column in upper.columns if any(upper[column] >= 0.65)]
	pd.DataFrame(to_drop, columns=['Feature']).to_csv("./"+base+"/"+filename+".SurrogateMarker.csv", index=False)
	df.drop(to_drop, axis=1, inplace=True)

	# Scale and add back in the cats
	scaled_df = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
	scaled_df['Sex_Male'] = sex

	X_train, X_test, y_train, y_test = train_test_split(scaled_df, outcome, test_size=0.33, random_state=42)

	svc = SVC(kernel='linear', probability=True)
	rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2), scoring='accuracy').fit(X_train, y_train)

	res = pd.DataFrame(list(scaled_df.columns[rfecv.support_]), columns=['Feature'])
	res['Coef'] = rfecv.estimator_.coef_[0]
	res = res.sort_values("Coef", ascending=False)
	res.to_csv("./"+base+"/"+filename+".FI.Coef.csv", index=False)

	plt.figure(figsize=(6,6), dpi=1200)
	plt.xlabel("Feature Coefficients")
	plt.ylabel("Features Selected")
	sns.barplot(x='Coef', y='Feature', data=res[ res['Coef'] != 0])
	plt.savefig("./"+base+"/"+filename+".FI.png")
	plt.cla()

	X_train, X_test = rfecv.transform(X_train), rfecv.transform(X_test)

	train_pred = rfecv.estimator_.predict_proba(X_train)[:,1]
	test_pred = rfecv.estimator_.predict_proba(X_test)[:,1]
	fpr1, tpr1, _ = roc_curve(y_train, train_pred)
	fpr2, tpr2, _ = roc_curve(y_test, test_pred)
	score1 = auc(fpr1, tpr1)
	score2 = auc(fpr2, tpr2)

	confidence = 0.95 # Adjusted to the designed confidence level
	z_level = stats.norm.ppf((1+confidence)/2)
	ci_length = z_level * np.sqrt((score2 * (1 - score2))/y_test.shape[0])
	ci_lower = score2 - ci_length
	ci_upper = np.min( [(score2 + ci_length), 1] )

	plt.figure(figsize=(6,6), dpi=1200)
	plt.plot(fpr1, tpr1, color='blue', label='Train (AUC=%.3f)'%score1)
	plt.plot(fpr2, tpr2, color='darkorange', label='Test (AUC=%.3f [%.3f, %.3f])'%(score2, ci_lower, ci_upper))
	plt.plot([0,1], [0,1], 'r--', label='Chance')
	plt.xlim([0,1])
	plt.ylim([0,1.05])
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.legend(loc="lower right")
	plt.savefig("./"+base+"/"+filename+".AUC.png")
