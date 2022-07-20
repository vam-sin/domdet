# libraries
import numpy as np 
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# load in the data
filename_X = 'sample_xgb_data/xgboost_input_res1039_100.npz'
X = np.load(filename_X, allow_pickle=True)['arr_0']

filename_y = 'sample_xgb_data/xgboost_labels_y_100.npz'
y = np.load(filename_y, allow_pickle=True)['arr_0']

print(X.shape, y.shape)

# split data into train and val sets
seed = 4
test_size = 0.2
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = test_size, random_state = seed)

# fit model no training data
print("Training")
model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=7, subsample=0.8, colsample_bytree=0.8, gamma=1, reg_alpha=1, objective="binary:logistic", scale_pos_weight=0.05, verbosity=2)
model.fit(X_train, y_train)

# val perf
# make predictions for test data
print("Validating")
y_pred = model.predict(X_val)
print(y_pred)

# evaluate predictions
prec = precision_score(y_val, np.round(y_pred))
rec = recall_score(y_val, np.round(y_pred))
acc = accuracy_score(y_val, np.round(y_pred))
aucroc = roc_auc_score(y_val, y_pred)

print("Precision: ", prec)
print("Recall: ", rec)
print("Accuracy: ", acc)
print("AUC-ROC: ", aucroc)

model.save_model("model_xgb_S1000.json")
