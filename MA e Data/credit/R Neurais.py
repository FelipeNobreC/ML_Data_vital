from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.metrics import accuracy_score, classification_report


with open('/Users/felip/PycharmProjects/MA e Data/credit/credit.pkl', 'rb') as f:
  X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = pickle.load(f)

# 3 -> 100 -> 100 -> 1
# 3 -> 2 -> 2 -> 1
rede_neural_credit = MLPClassifier(max_iter=1500, verbose=True, tol=0.0000100,
                                   solver = 'adam', activation = 'relu',
                                   hidden_layer_sizes = (20,20))
rede_neural_credit.fit(X_credit_treinamento, y_credit_treinamento)

previsoes = rede_neural_credit.predict(X_credit_teste)
print(previsoes)

print(accuracy_score(y_credit_teste, previsoes))

print(classification_report(y_credit_teste, previsoes))