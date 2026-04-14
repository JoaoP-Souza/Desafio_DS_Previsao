# Desafio_DS_Previsao
Esse repositório tem como objetivo documentar todo o processo desde análise de dados, até o deploy do modelo de predição baseado em machine learning. Possuindo um Readme.md que documenta todos os pontos requisitados pelo enunciado, um notebook jupyter com modelo desenvolvido e o diagrama.

--------------------------------------
## 1. Análise e Entendimento dos Dados
--------------------------------------

A primeira etapa do projeto consistiu em compreender a estrutura dos dados disponíveis e identificar quais informações poderiam contribuir para a previsão do preço dos imóveis. Para isso, foram utilizados dois conjuntos principais de dados: um contendo características físicas das propriedades e outro com informações demográficas agregadas por zipcode.

O arquivo `kc_house_data.csv` reúne dados individuais de imóveis com preço conhecido, sendo a coluna **price** a variável alvo do problema. Entre os atributos mais relevantes estão:

* **price**: preço de venda do imóvel
* **bedrooms** e **bathrooms**: quantidade de quartos e banheiros
* **sqft_living**: área útil interna do imóvel
* **sqft_lot**: tamanho total do terreno
* **floors**: número de andares
* **waterfront**: indica se o imóvel está à beira d’água
* **view**, **condition** e **grade**: indicadores de vista, conservação e padrão da construção
* **yr_built** e **yr_renovated**: ano de construção e que foi reformado
* **zipcode**, **lat** e **long**: localização geográfica
* **sqft_living15** e **sqft_lot15**: métricas médias de imóveis vizinhos

Essas variáveis representam fatores tradicionalmente importantes no mercado imobiliário, como tamanho, qualidade e localização.

O arquivo `zipcode_demographics.csv` traz informações socioeconômicas por região, permitindo complementar a análise com contexto demográfico. Entre as variáveis utilizadas, destacam-se:

* população total da região
* distribuição entre áreas urbanas, suburbanas e rurais
* renda média familiar
* renda média por pessoa
* valor médio dos imóveis da região
* indicadores de escolaridade da população

Esses dados ajudam a representar o perfil econômico de cada localidade, fator que costuma influenciar diretamente o valor dos imóveis.

### Integração dos dados

Foi realizada a junção entre os datasets utilizando a coluna **zipcode** como chave. Dessa forma, cada imóvel passou a contar também com informações demográficas da região em que está localizado. O **left join** foi usado para preservar todos os registros do dataset principal.

### Limpeza e preparação inicial

Após a integração, foi realizada uma etapa de tratamento para garantir consistência e qualidade da base de dados. Entre as principais ações executadas estiveram a verificação de valores ausentes, ajuste de tipos de dados, remoção de colunas não relevantes para a modelagem, tratamento de valores extremos claramente inconsistentes e preparação final das variáveis para o treinamento. A coluna de data também foi removida na etapa de modelagem, já que nesta primeira versão não agregou ganho relevante ao desempenho do modelo.

### Análise de outliers

Foram analisadas distribuições estatísticas e boxplots das variáveis numéricas para identificar possíveis outliers. Em bases imobiliárias, valores extremos são esperados, já que existem imóveis de luxo ou propriedades muito acima da média.

Por esse motivo, a estratégia adotada foi remover apenas casos claramente inconsistentes, como:

* número excessivo de quartos ou banheiros
* metragem incompatível com o restante da base
* valores fora de escala sem coerência aparente

Já imóveis premium e terrenos grandes foram mantidos, pois representam situações reais do mercado.

### Correlações e padrões encontrados

A análise de correlação indicou que algumas variáveis possuem relação mais forte com o preço dos imóveis. Entre as principais, destacam-se:

* **sqft_living (0.70)**
* **grade (0.67)**
* **sqft_above (0.61)**
* **sqft_living15 (0.58)**
* **hous_val_amt (0.58)**
* **medn_incm_per_prsn_amt (0.55)**
* **bathrooms (0.52)**
* **view (0.39)**
* **lat (0.31)**

Os resultados obtidos no notebook 'pre_processamento_dados.ipynb' mostram que imóveis maiores, de melhor padrão construtivo e localizados em regiões mais valorizadas tendem a apresentar preços superiores. Também ficou evidente a relevância do contexto socioeconômico, especialmente renda média regional e valor médio dos imóveis por zipcode.

----------------------------------------------------
### 2. Desenvolvimento do Modelo de Machine Learning
----------------------------------------------------

Após a etapa de análise e preparação dos dados, foi desenvolvido um modelo de Machine Learning com o objetivo de prever o preço dos imóveis a partir das características físicas e demográficas disponíveis.

Com base na análise de correlação e nos testes realizados, observou-se que algumas variáveis possuem maior influência na previsão de preço. Entre elas, destacam-se:

* **sqft_living**: área útil interna do imóvel
* **grade**: padrão construtivo e qualidade geral
* **sqft_above**: metragem acima do solo
* **sqft_living15**: padrão médio das casas vizinhas
* **bathrooms**: quantidade de banheiros
* **view**: qualidade da vista do imóvel
* **lat**: componente geográfico da localização
* **hous_val_amt**: valor médio dos imóveis da região
* **medn_incm_per_prsn_amt**: renda média por pessoa

Essas variáveis mostram que o preço depende tanto das características do imóvel quanto do contexto econômico e regional.
O modelo escolhido foi o **XGBoost Regressor**, por sua boa capacidade de lidar com relações não lineares entre variáveis e ouliers.
A Regressão Linear também foi testada, porém o XGBoost demonstrou melhor capacidade de generalização.


Para garantir que o modelo não aprendesse apenas os dados de treino, foram aplicadas técnicas de validação e teste.
O dataset final foi dividido em 80% treino e 20% teste. Além do train/test split, foi utilizada **K-Fold Cross Validation** com 5 divisões, permitindo uma estimativa mais robusta da performance do modelo.

#### Regularização e controle de complexidade

Também foram ajustados hiperparâmetros importantes do XGBoost, como:

* profundidade máxima das árvores
* learning rate
* número de estimadores
* subsample
* colsample_bytree
* parâmetros de regularização (`alpha`, `lambda`)

Esses controles ajudam a reduzir overfitting e melhorar a capacidade de generalização.

### Resultados obtidos

O modelo final apresentou os seguintes resultados:

#### Cross Validation

* **MAE médio:** 62.890
* **RMSE médio:** 119.338

#### Test Set

* **MAE:** 60.563
* **RMSE:** 119.180
* **R²:** 0.899

Os resultados mostram que o modelo consegue explicar aproximadamente **90% da variação dos preços**, indicando boa capacidade preditiva.


--------------------------
## 3. Estratégia de Deploy
--------------------------

Para disponibilizar o modelo em um ambiente de produção, uma alternativa viável é usar MLOps para torná-lo acessível via API. Isso tornaria possível à usuários ou sistemas externos, enviar os dados de um imóvel e receber automaticamente uma estimativa de preço baseada no modelo treinado.

O diagrama desenvolvido representa um fluxo de produção, contemplando a inferência do modelo, armazenamento histórico e preparação para aprendizado contínuo.

---

## Visão Geral da Arquitetura

```text id="vx2d5i"
Usuário → Frontend → API REST → Pré-processamento → Modelo XGBoost → Predição
                           ↓
                 Banco de Logs da API
                           ↓
               Histórico de Previsões
                           ↓
          Base nova para re-treinamento
```

---

### Usuário

Representa o cliente final ou sistema consumidor da solução (como uma aplicação web ou mobile), que informa os dados do imóvel que deseja avaliar.

### Frontend

É a interface utilizada para entrada das informações. Nela, o usuário preenche campos relacionados ao imóvel, como o número de quartos ou a metragem desejada. Essa camada pode ser implementada em tecnologias web baseadas em Javascript como React, Angular ou até mesmo em dashboards internos.

---

### API REST

A API representa o núcleo operacional da arquitetura, sendo responsável por receber as requisições enviadas pelo frontend e coordenar todo o fluxo de predição. Entre suas principais funções estão receber os dados informados pelo usuário, validar campos obrigatórios, verificar formatos inválidos, registrar logs da operação, acionar o pipeline de pré-processamento e retornar a resposta final ao frontend com o valor previsto. Para essa camada, tecnologias como FastAPI ou Flask seriam boas opções de implementação. O retorno da API é geralmente em formato JSON.

---

### Pré-processamento dos dados

Antes de enviar os dados ao modelo, é necessário aplicar o mesmo tratamento utilizado durante a fase de treinamento. Essa etapa inclui garantir a seleção e a ordem correta das colunas, tratar valores faltantes, converter tipos numéricos quando necessário, realizar o merge com a base demográfica por meio do zipcode, remover possíveis inconsistências e aplicar normalização caso seja exigida pelo pipeline. Manter essa consistência entre treino e produção é fundamental para evitar erros e garantir previsões confiáveis.

---

### Modelo XGBoost

Após o pré-processamento, os dados são enviados ao modelo treinado na etapa 2. Este fica salvo no ambiente de produção e é carregado pela API quando necessário.

---

### Predição

É o resultado final retornado ao usuário: o preço estimado do imóvel.

---

## Banco de Logs da API

Além de realizar previsões de preço, a aplicação também precisa registrar os eventos ocorridos em produção. Para isso, o banco de logs armazenaria informações como horário de cada requisição, dados enviados pelo usuário, tempo de resposta da API, status da chamada, possíveis erros ocorridos, versão do modelo utilizada e o preço previsto. Esses registros são importantes para auditoria, rastreabilidade, monitoramento e futuras melhorias da solução.


Exemplo dos dados da requisição:

```text id="qsv7sr"
request_id: 10452
timestamp: 14/04/2026 14:00
model_version: v1.0
predicted_price: 615000
latency_ms: 118
status: sucesso
```

---

### Histórico de Previsões

Parte dos logs gerados pela aplicação e das predições realizadas pode ser consolidada em uma base histórica. Essa base armazenaria as entradas utilizadas em cada previsão, o valor estimado pelo modelo, a data da consulta, a região analisada, a versão do modelo utilizada e, quando disponível posteriormente, o valor real de venda do imóvel. Essa camada é importante porque permite análises de negócio, acompanhamento do comportamento do mercado e avaliação contínua do desempenho do modelo ao longo do tempo.


---

## Base nova para re-treinamento

Com o passar do tempo, novas vendas reais e novos imóveis consultados podem ser incorporados ao histórico da aplicação. Após uma etapa de limpeza e validação, essas informações passam a compor uma nova base supervisionada, formada pelo histórico de previsões combinado com os novos preços reais observados no mercado. Essa base pode então ser utilizada no re-treinamento do modelo, permitindo sua atualização contínua e melhor adaptação às mudanças do cenário imobiliário.

---

## Monitoramento Operacional

Mesmo não aparecendo explicitamente no diagrama, essa camada é altamente recomendada em produção. Ela seria responsável por acompanhar indicadores como tempo médio de resposta da API, taxa de erro, disponibilidade do sistema, volume de consultas, possível degradação do modelo e mudanças no perfil dos dados recebidos. Para isso, poderiam ser utilizadas ferramentas como Grafana, Prometheus ou CloudWatch.

---

## Infraestrutura Recomendada

O deploy da solução poderia ser realizado em ambiente de nuvem, utilizando provedores como AWS, Google Cloud ou Azure. Também seria recomendado o uso de Docker para empacotamento da aplicação, além de pipelines de CI/CD para automatizar atualizações e facilitar futuras evoluções do sistema conforme o aumento da demanda.

---

## Benefícios da Arquitetura

Essa estratégia permite que o modelo passe a funcionar como um produto real de dados. Entre os principais ganhos estão a possibilidade de previsões automáticas em tempo real, integração com sistemas externos, histórico centralizado de uso, rastreabilidade das operações, melhoria contínua do modelo e escalabilidade futura da solução.


--------------------------
## 4. Aprendizado Contínuo
--------------------------
É importante ressaltar que o trabalho não termina quando um modelo como esse entra em produção. O mercado muda com o tempo e o  modelo deve permanecer atualizado sobre essas mudanças para continuar fornecendo  previsões consistentes. Para que isso possa ocorrer, é muito importante que haja a coleta constante de novos dados, como:

* características do imóvel consultado
* preço previsto pelo modelo
* preço real de venda
* data da transação
* região / zipcode
* tempo até venda

Após coletar e armazenar esses dados é importante realizar um novo treinamento do modelo, de forma periódica. o quão frequente esse deve ser realizado varia de acordo com os novos dados disponibilizados. É importante realizar uma análise e definir a recorrência com a qual o modelo será treinado. Em mercados mais dinâmicos, ciclos menores podem ser mais adequados. \Seguindo um fluxo onde os novos dados são coletados, pré-processados e validados, adicionados a base e então usados no treinamento. Análises devem então ser realizadas para as métricas relevantes (MAE, RMSE, R²...)  e se os resultados forem satisfatórios, um novo modelo pode ser gerado e integrado à solução.


Antes de substituir o modelo em produção, a nova versão deve ser comparada com a atual utilizando métricas como MAE, RMSE e R², além de critérios práticos como estabilidade entre regiões, desempenho em imóveis de alto valor, tempo de resposta e robustez a dados incompletos. A troca só deve acontecer caso a nova versão apresente melhora real ou desempenho equivalente com maior confiabilidade.

Para reduzir riscos, a substituição pode ser feita de forma gradual, liberando inicialmente a nova versão para apenas parte das requisições e comparando seu desempenho com o modelo atual. Caso os resultados sejam positivos, a nova versão passa a atender todo o tráfego.

Mesmo após a atualização, o modelo deve continuar sendo monitorado para identificar possíveis sinais de degradação, como aumento do erro médio, mudanças no perfil dos dados recebidos ou queda de desempenho em determinadas faixas de preço.

Além disso, cada versão publicada deve ser registrada com informações como data de treinamento, base utilizada, hiperparâmetros e métricas obtidas, permitindo rastreabilidade e rollback caso necessário.


-------------------------------
## 5. Comunicação com Stakeholders
-------------------------------

A apresentação dos resultados seria focada em linguagem simples, visual, demonstrando o valor da solução. demonstrando a eficácia do modelo, a precisão obtida e como isso poderia ajudar no negócio.

 #Exemplo:
"Foi desenvolvido um modelo capaz de estimar preços de imóveis com aproximadamente 90% de explicação da variação dos valores observados (R² = 0.89), utilizando dados físicos e demográficos. A solução pode apoiar precificação, análise de mercado e tomada de decisão comercial."


Ao invés de apenas manter uma linguagem técnica de ciência de dados, é importante traduzir isso para uma linguagem que os stakeholders entendam bem, e essa é o impacto gerado pela solução.
Na solução desenvolvida, a previsão fica a cerca de 60 mil dólares do preço real.
R² ≈ 0.89. O modelo consegue explicar cerca de 89% da variação dos preços observados no mercado (R² ≈ 0.89).
Explicar que o modelo generaliza bem para novos imóveis (como pode ser visto com as métricas obtidas entre treinamento e teste).

O Gráfico de dispersão mostrando o quanto as previsões se aproximam dos valores reais é uma boa representação visual a ser demonstrada para informar os resultados obtidos pelo modelo.

Também é importante demonstrar onde o modelo erra mais, identificando pontos de melhoria para versões futuras.


# Storytelling para áreas de negócio:

Demonstrar qual problema da área de negócios o modelo resolve. 

"Hoje a precificação
o depende muito de análise manual ou comparação limitada.
Com o modelo, passamos a ter uma estimativa automática baseada em milhares de imóveis históricos e dados regionais.
Isso permite decisões mais rápidas, padronizadas e escaláveis."

# Dar exemplos de aplicações práticas onde a solução poderia ser implementada:

apoio à definição de preço de venda.
triagem de imóveis subavaliados ou superavaliados.
recomendação para corretores.
inteligência de mercado por região.
estimativas rápidas em canais digitais.

Criar um dashboard contendo o histórico de métricas como o volume de previsões realizadas, a evolução do erro e da performance do modelo, também são formas de esclarecer e comunicar a evolução do trabalho realizado aos stakeholders.



