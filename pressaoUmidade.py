import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
dados = pd.read_csv('/Users/anakoressawa/TempoPython/tabela-variaveis-climaticas.csv', delimiter=',')

# Converter as colunas para valores numéricos
dados = dados.apply(pd.to_numeric, errors='coerce')

# Tratar valores nulos (substituindo por média, por exemplo)
dados = dados.fillna(dados.mean())

# Definir a variável dependente (y) e a variável independente (X)
X = dados[['Umidade']]  # Variável independente: Umidade
y = dados['Pressao']  # Variável dependente: Pressao

# Criar o modelo de regressão linear
modelo = LinearRegression()

# Ajustar o modelo aos dados
modelo.fit(X, y)

# Coeficientes da regressão
print('Coeficiente de regressão (Umidade):', modelo.coef_)
print('Intercepto:', modelo.intercept_)

# Fazer previsões com base nos dados
previsoes = modelo.predict(X)
print('Previsões da Pressão:', previsoes)

# Plotar gráfico de dispersão com a linha de regressão
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Dados reais')  # Gráfico de dispersão
plt.plot(X, previsoes, color='red', label='Linha de Regressão')  # Linha de regressão
plt.title('Regressão Linear: Pressão em Relação à Umidade')
plt.xlabel('Umidade')
plt.ylabel('Pressão')
plt.legend()
plt.grid(True)
plt.show()
