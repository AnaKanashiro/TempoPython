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

# Definir a variável independente (X) e dependente (y)
X = dados[['Umidade']]  # variáveis independentes (umidade)
y = dados['Chuva']  # variável dependente (chuva)

# Criar o modelo de regressão linear
modelo = LinearRegression()

# Ajustar o modelo aos dados
modelo.fit(X, y)

# Coeficientes da regressão
print('Coeficiente angular (slope):', modelo.coef_[0])
print('Intercepto (intercept):', modelo.intercept_)

# Fazer previsões (exemplo)
previsoes = modelo.predict(X)
print('Previsões:', previsoes)

# Gerar gráfico de dispersão e a linha de regressão
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X['Umidade'], y=y, label='Dados reais')
plt.plot(X['Umidade'], previsoes, color='red', label='Linha de regressão', linewidth=2)

plt.title('Regressão Linear: Chuva em relação à Umidade')
plt.xlabel('Umidade (%)')
plt.ylabel('Chuva (mm)')
plt.legend()
plt.grid(True)
plt.show()
