from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from Nets import Nets
from gdm_dataset import GDMDataset


data = Nets("microbiome", seed=None)
data.autoencoder.train()
embds = data.autoencoder.embs
X, y = embds, [net['type'] for index, net in data.nets.items()]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)


clf = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="sgd", learning_rate="adaptive", early_stopping=True).fit(X_train, y_train)
pred_prob_test = clf.predict_proba(X_test)
auc_score1 = roc_auc_score(y_test, pred_prob_test[:, 1])
print("acc", clf.score(X_test, y_test))
print("auc", auc_score1)
