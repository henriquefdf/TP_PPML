# Detecção de Spam em SMS com Federated Learning

Este projeto implementa um sistema de detecção de spam em mensagens SMS utilizando técnicas de Aprendizado Federado, Regressão Logística e Redes Neurais (MLP e LSTM), com ênfase em privacidade dos dados.

## Sumário

- [Sobre o Projeto](#sobre-o-projeto)
- [Arquitetura e Fases](#arquitetura-e-fases)
- [Pré-requisitos](#pré-requisitos)
- [Execução](#execução)
- [Resultados Esperados](#resultados-esperados)
- [Referências](#referências)

---

## Sobre o Projeto

O objetivo é comparar abordagens centralizadas e federadas para detecção de spam em SMS, avaliando métricas como acurácia, precisão, recall, F1-score, AUC-ROC e matriz de confusão. O projeto utiliza o framework Flower para simular o aprendizado federado e TensorFlow para as redes neurais.

---

## Arquitetura e Fases

**Fase 1:**  
- Pré-processamento dos dados (limpeza, vetorização TF-IDF)
- Regressão Logística centralizada e federada

**Fase 2:**  
- Tokenização e padding de sequências
- Redes Neurais: MLP (com embeddings médios) e LSTM
- Avaliação centralizada e federada

**Métricas:**  
- Acurácia, Precisão, Recall, F1-Score, AUC-ROC, Matriz de Confusão

**Visualização:**  
- Comparação gráfica do desempenho dos modelos

---

## Pré-requisitos

- Python 3.8+
- Jupyter Notebook ou Google Colab

**Instale as dependências com:**
```bash
pip install flwr scikit-learn pandas numpy tensorflow matplotlib seaborn
```

**Dataset:**  
Baixe o arquivo `spam.csv` do [Kaggle SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) e coloque-o no mesmo diretório do notebook.

---

## Execução

1. **Clone este repositório ou baixe o notebook:**
   ```
   git clone https://github.com/henriquefdf/TP_PPML.git
   ```
2. **Abra o arquivo `federated_spam_sms.ipynb` no Jupyter Notebook ou Google Colab.**

3. **Execute as células sequencialmente.**

4. **Para rodar o aprendizado federado real:**  
   O Flower simula clientes federados. Para experimentos reais, execute múltiplos processos conforme a [documentação do Flower](https://flower.dev/docs/).

---

## Resultados Esperados

- **Tabela e gráficos** comparando o desempenho dos modelos centralizados e federados.
- **Métricas detalhadas** (acurácia, precisão, recall, F1, ROC-AUC) para cada abordagem.
- **Visualização da matriz de confusão** para análise de erros.
- **Código modular e comentado** para fácil adaptação e expansão.

---

## Referências

- [Flower - Framework de Federated Learning](https://flower.dev/)
- [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

**Dúvidas ou sugestões?**  
Abra uma issue ou envie um pull request!

---

> Projeto desenvolvido para a disciplina de Proteção da Privacidade no Aprendizado de Máquina — 2025.

---

