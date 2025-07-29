# ENGA74-TrabalhoFinal

Projeto final da disciplina de Mestrado ENGA74 - Inteligência Computacional da Universidade Federal da Bahia (UFBA).

## Autores

- Autor Principal: Ricardo Machado
- Autor Colaborador: Ricardo Gonçalves Pinto

## Descrição

Este repositório contém o projeto final desenvolvido para a disciplina Inteligência Computacional. 
O projeto consiste na previsão de irradiância solar da cidade de Natal com base em dados 
meteorológicos disponíveis pelo National Solar Radiation Database(NSRDB). O presente trabalho compara o 
desempenho de um Multilayer Perceptron, um Long Short Term Memory e um Gated Recurrent Unit 
para um conjunto de teste composto por dois anos de dados disponíıveis(2023 e 2024) e um conjunto
de treino composto pelos anos de 2020 a 2022.
## Arquivos principais

- `Artigo Final - ENGA74.pdf` — Artigo do trabalho.
- `main.py` — Código fonte do projeto.
- `prototype.ipynb` — Notebook com o protótipo do projeto.
- `optimization.ipynb` — Notebook utilizado para otimizar os hiperparâmetros.
- `README.md` — Este arquivo.

## Requisitos

- Python 3.13 ou superior
- Gerenciador de pacotes uv - [link](https://docs.astral.sh/uv/getting-started/installation/)
- Dependências listadas em pyproject.toml;


## Como Executar

1. Clone o repositório:
    ```bash
    git clone https://github.com/seu-usuario/ENGA74-TrabalhoFinal.git
    ```
2. Instalar as dependências necessárias:
    ```bash
    uv sync
    ```
3. Rodar o script main.py para executar as redes neurais já otimizadas:
    ```bash
    uv run main.py
    ```



## Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo LICENSE para mais informações.
