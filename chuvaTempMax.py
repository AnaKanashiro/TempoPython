import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar os dados
dados = pd.read_csv('/Users/anakoressawa/TempoPython/tabela-variaveis-climaticas.csv', delimiter=',')

# Converter as colunas para valores numéricos
dados = dados.apply(pd.to_numeric, errors='coerce')

# Tratar valores nulos (substituindo por média)
dados = dados.fillna(dados.mean())

# Definir as variáveis independentes (X) e dependente (y)
X = dados[['Umidade', 'Pressao', 'VelVento', 'TempMax', 'TempMin']]  # variáveis independentes
y = dados['Chuva']  # variável dependente

# Criar o modelo de regressão linear
modelo = LinearRegression()

# Ajustar o modelo aos dados
modelo.fit(X, y)

# Coeficientes da regressão
print('Coeficientes:', modelo.coef_)
print('Intercepto:', modelo.intercept_)

# Fazer previsões (exemplo)
previsoes = modelo.predict(X)
print('Previsões:', previsoes)

# Gerar o gráfico de regressão linear entre TempMax e Chuva
plt.figure(figsize=(10, 6))
sns.regplot(x='TempMax', y='Chuva', data=dados, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})

# Ajustando o título e os rótulos
plt.title('Regressão Linear entre TempMax e Chuva')
plt.xlabel('Temperatura Máxima (°C)')
plt.ylabel('Chuva (mm)')

# Exibir o gráfico
plt.show()
