import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import scipy.stats as stats
import os
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
warnings.filterwarnings("ignore")

# Iterate through each timepoint/condition
for fname in os.listdir("./Data/"):
	if os.path.isdir("./Data/"+fname): continue
	if fname[-4:] != ".csv": continue
	base = str(fname).split(".")[0]
	base = "KNN/"+str(base)
	filename = str(fname)[:len(fname)-4]

	# Load data & drop unneeded variables
	df = pd.read_csv("./Data/"+fname)
	outcome = df[['Case-control status']]
	df = df.drop(["Event","Patient ID","Case ID","Case-control status"], axis=1)

	# Pull out cats
	sex = df['Sex_Male']
	df.drop("Sex_Male", axis=1, inplace=True)

	# Drop correlated features
	corr_matrix = df.corr().abs()
	upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
	to_drop = [column for column in upper.columns if any(upper[column] >= 0.65)]
	pd.DataFrame(to_drop, columns=['Feature']).to_csv("./"+base+"/"+filename+".SurrogateMarkers.csv", index=False)
	df.drop(to_drop, axis=1, inplace=True)

	scaled_df = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
	scaled_df['Sex_Male'] = sex

	X_train, X_test, y_train, y_test = train_test_split(scaled_df, outcome, test_size=0.33, random_state=42)

	knc = KNC(n_neighbors=2)
	kbest = SelectKBest(k=1)
	pipeline = Pipeline([('feature_selection',kbest),('KNN',knc)])
	params = dict( feature_selection__k=[3,4,5,6,7,8,9,10],
		KNN__n_neighbors=[2,3,4])
	best_model = GridSearchCV(pipeline,param_grid=params)
	best_model = best_model.fit(X_train, y_train)
	print(best_model.best_params_)
	print(best_model.best_estimator_.named_steps.feature_selection.get_feature_names_out())
	fi = pd.DataFrame({'Feature':best_model.best_estimator_.named_steps.feature_selection.get_feature_names_out()})
	fi.to_excel("./"+base+"/"+filename+".FI.xlsx", index=False)

	train_pred = best_model.predict_proba(X_train)[:,1]
	test_pred = best_model.predict_proba(X_test)[:,1]
	fpr1, tpr1, _ = roc_curve(y_train, train_pred)
	fpr2, tpr2, _ = roc_curve(y_test, test_pred)
	score1 = auc(fpr1, tpr1)
	score2 = auc(fpr2, tpr2)

	confidence = 0.95 # Adjusted to the designed confidence level
	z_level = stats.norm.ppf((1+confidence)/2)
	ci_length = z_level * np.sqrt((score2 * (1 - score2))/y_test.shape[0])
	ci_lower = score2 - ci_length
	ci_upper = np.min( [(score2 + ci_length), 100] )

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

