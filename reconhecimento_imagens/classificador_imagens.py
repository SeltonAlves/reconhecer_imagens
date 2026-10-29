# Importando as bibliotecas
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

print("--- 1. Carregando o Dataset de Imagens ---")
digitos = datasets.load_digits()

print(f"Total de imagens disponíveis: {len(digitos.images)}")
print(f"Formato de cada imagem: {digitos.images[0].shape} pixels")

X = digitos.data
y = digitos.target

print("\n--- 2. Dividindo os dados (Treino e Teste) ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"Imagens para treino: {len(X_train)}")
print(f"Imagens para teste: {len(X_test)}")

print("\n--- 3. Treinando a Rede Neural (Classificador) ---")
modelo = MLPClassifier(
    hidden_layer_sizes=(50,), max_iter=500, random_state=42, verbose=True)
modelo.fit(X_train, y_train)

print("\n--- 4. Avaliando o Modelo com os Dados de Teste ---")
previsoes = modelo.predict(X_test)

acuracia = accuracy_score(y_test, previsoes)
print(f"\nAcurácia do modelo: {acuracia * 100:.2f}%\n")

print("Relatório detalhado por dígito:")
print(classification_report(y_test, previsoes))

print(
    "\n--- 5. Testando visualmente com algumas previsões do conjunto de teste ---")
fig, axes = plt.subplots(2, 4, figsize=(8, 4))
for i, ax in enumerate(axes.ravel()):
  ax.imshow(X_test[i].reshape(8, 8), cmap=plt.cm.gray_r, interpolation="nearest")
  ax.set_title(f"Prev: {previsoes[i]} | Real: {y_test[i]}")
  ax.axis("off")

plt.tight_layout()
plt.show()