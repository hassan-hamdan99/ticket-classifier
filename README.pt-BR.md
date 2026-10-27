# 🎯 Sistema de Classificação de Chamados com Machine Learning

Sistema completo e modular para classificação automática de chamados/tickets usando Machine Learning. Classifica automaticamente chamados por departamento e categoria, com suporte a múltiplas origens de dados.

## 📋 Características Principais

- ✅ **Classificação Dupla**: Departamento + Categoria
- ✅ **Multi-origem**: Suporte para Email, Painel Web, API, etc.
- ✅ **Preprocessamento Inteligente**: Remove PIIs, normaliza texto
- ✅ **Alta Performance**: TF-IDF + Logistic Regression otimizado
- ✅ **Interface Flexível**: CLI, Python API e processamento em lote
- ✅ **Explicabilidade**: Visualize features importantes nas predições
- ✅ **Produção-Ready**: Logs, validação, tratamento de erros

## 🚀 Instalação Rápida

### 1. Clone o repositório
```bash
git clone <repo-url>
cd ticket-classifier
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. (Opcional) Instale modelo spaCy para NER
```bash
python -m spacy download pt_core_news_sm
```

## 📁 Estrutura do Projeto

```
ticket-classifier/
├── config.py              # Configurações centralizadas
├── utils.py               # Funções utilitárias
├── data_loader.py         # Carregamento de dados
├── model_trainer.py       # Treinamento de modelos
├── classifier.py          # Sistema principal (contém classificador_mensagem)
├── main.py                # Script de treinamento
├── predict.py             # Script de predição CLI
├── exemplo_uso.py         # Exemplos práticos
├── requirements.txt       # Dependências
├── data/                  # Dados de treinamento
│   └── exemplo_chamados.csv
├── models/                # Modelos treinados
├── tests/                 # Testes unitários
└── docs/                  # Documentação adicional
```

## 🎓 Guia de Uso

### Passo 1: Preparar os Dados

Crie um arquivo CSV ou Excel com as seguintes colunas:

| Coluna | Descrição | Obrigatória |
|--------|-----------|-------------|
| Origem | Canal de entrada (Email, Painel, etc.) | ✅ |
| Assunto | Título/assunto do chamado | ✅ |
| Mensagem | Corpo/descrição do chamado | ✅ |
| Departamento | Classificação de departamento | ✅ (treino) |
| Categoria | Classificação de categoria | ✅ (treino) |

Exemplo de dados (`data/exemplo_chamados.csv`):
```csv
Origem,Assunto,Mensagem,Departamento,Categoria
Email,Problema de login,Não consigo acessar o sistema,TI,Acesso
Painel,Solicitação de férias,Gostaria de solicitar férias em dezembro,RH,Benefícios
```

### Passo 2: Treinar os Modelos

```bash
# Treinar com dados padrão
python main.py

# Especificar diretório de dados
python main.py --data-dir /caminho/para/dados

# Listar arquivos disponíveis
python main.py --list-files

# Apenas analisar dados (sem treinar)
python main.py --no-train
```

### Passo 3: Fazer Predições

#### Opção 1: Uso Programático (Python)

```python
from classifier import classificador_mensagem

# Função principal do sistema
resultado = classificador_mensagem(
    origem="Email",
    assunto="Problema de acesso",
    mensagem="Não consigo fazer login no sistema"
)

print(f"Departamento: {resultado['departamento']['predicao']}")
print(f"Categoria: {resultado['categoria']['predicao']}")
print(f"Confiança: {resultado['departamento']['confianca']:.1%}")
```

#### Opção 2: CLI Interativo

```bash
# Modo interativo
python predict.py --interactive

# Predição direta
python predict.py --origem Email --assunto "Login" --mensagem "Erro ao acessar"

# Processar arquivo
python predict.py --file chamados.csv --output resultados.csv

# Listar modelos disponíveis
python predict.py --list-models
```

#### Opção 3: Processamento em Lote

```python
from classifier import ClassificadorChamados
import pandas as pd

# Carrega dados
df = pd.read_csv("chamados_novos.csv")

# Classifica
clf = ClassificadorChamados()
clf.carregar_todos_modelos()
df_resultado = clf.classificar_lote(df)

# Salva resultado
df_resultado.to_csv("chamados_classificados.csv", index=False)
```

## 🔧 Configuração Avançada

### Personalizando Parâmetros

Edite `config.py` para ajustar:

```python
# Parâmetros de ML
TFIDF_PARAMS = {
    "ngram_range": (1, 2),  # Uni e bigramas
    "max_features": 10000,   # Máximo de features
    "min_df": 2             # Mínimo de documentos
}

# Filtros de qualidade
MIN_SAMPLES_PER_CLASS_DEP = 5   # Mínimo para departamentos
MIN_SAMPLES_PER_CLASS_CAT = 30  # Mínimo para categorias

# Preprocessamento
REMOVE_PERSON_NAMES = True  # Remove nomes próprios
MASK_IDS_EMAILS = True      # Mascara emails e IDs
```

### Mapeamento de Colunas

O sistema reconhece automaticamente variações de nomes de colunas:

```python
COL_MAP_OPCOES = {
    "origem": ["Origem", "Canal", "Source", "Fonte"],
    "assunto": ["Assunto", "Título", "Subject"],
    "mensagem": ["Mensagem", "Descrição", "Body"],
    # ...
}
```

## 📊 Exemplos Práticos

### Exemplo 1: Classificação Simples

```python
from classifier import classificador_mensagem

resultado = classificador_mensagem(
    origem="Email",
    assunto="Computador lento",
    mensagem="Meu PC está travando constantemente"
)

# Resultado esperado:
# {
#   "departamento": {"predicao": "TI", "confianca": 0.92},
#   "categoria": {"predicao": "Hardware", "confianca": 0.88},
#   "sucesso": true
# }
```

### Exemplo 2: Integração com Sistema Existente

```python
class MeuSistema:
    def processar_chamado(self, dados):
        # Classifica automaticamente
        classificacao = classificador_mensagem(
            origem=dados['canal'],
            assunto=dados['titulo'],
            mensagem=dados['descricao']
        )

        # Usa a classificação
        dados['departamento'] = classificacao['departamento']['predicao']
        dados['prioridade'] = self.calcular_prioridade(classificacao)

        return self.criar_ticket(dados)
```

### Exemplo 3: Análise de Confiança

```python
resultado = classificador_mensagem(origem="Email", assunto="Dúvida", mensagem="...")

if resultado['departamento']['confianca'] < 0.7:
    print("Classificação incerta - revisar manualmente")
    print(f"Possibilidades: {resultado['departamento']['topk_labels']}")
```

## 🧪 Testes

Execute os testes unitários:

```bash
# Todos os testes
pytest tests/

# Com cobertura
pytest tests/ --cov=. --cov-report=html

# Teste específico
pytest tests/test_utils.py
```

## 📈 Métricas e Performance

O sistema reporta métricas detalhadas após o treinamento:

- **Accuracy**: Taxa de acertos geral
- **Precision**: Precisão por classe
- **Recall**: Cobertura por classe
- **F1-Score**: Média harmônica (principal métrica)
- **Cross-validation**: Validação cruzada com 5 folds

Exemplo de output:
```
Origem: Email
  ✅ departamento:
     - F1-Score: 0.923
     - Accuracy: 0.931
     - Classes: 5
  ✅ categoria:
     - F1-Score: 0.876
     - Accuracy: 0.889
     - Classes: 12
```

## 🔍 Explicabilidade

Visualize quais termos influenciaram a classificação:

```python
from classifier import ClassificadorChamados

clf = ClassificadorChamados()
resultado = clf.explicar_predicao(
    texto="VPN não conecta no notebook",
    origem="Email",
    tipo="departamento"
)

# Features importantes:
# 1. 'vpn' → contribuição: 0.823
# 2. 'conecta' → contribuição: 0.412
# 3. 'notebook' → contribuição: 0.387
```

## 🚨 Tratamento de Erros

O sistema possui tratamento robusto de erros:

```python
resultado = classificador_mensagem(origem="", assunto="", mensagem="")

if not resultado['sucesso']:
    for erro in resultado['erros']:
        print(f"Erro: {erro}")
# Output: "Erro: Origem não fornecida"
#         "Erro: Assunto e mensagem vazios"
```

## 🔐 Segurança e Privacidade

- **Remoção automática de PIIs**: CPF, CNPJ, emails, telefones
- **Mascaramento de dados sensíveis**: Substitui por placeholders
- **Logs sem dados pessoais**: Apenas métricas e metadados

## 🛠️ Troubleshooting

### Problema: "Nenhum modelo encontrado"
**Solução**: Execute `python main.py` para treinar os modelos

### Problema: "spaCy não disponível"
**Solução**: Instale com `pip install spacy` e baixe o modelo português

### Problema: "Poucas amostras para classe X"
**Solução**: Adicione mais exemplos ou ajuste `MIN_SAMPLES_PER_CLASS` em config.py

### Problema: "Classificação com baixa confiança"
**Solução**: Verifique qualidade dos dados de treino e considere re-treinar

## 📚 API Reference

### Função Principal

```python
def classificador_mensagem(
    origem: str,
    assunto: str,
    mensagem: str,
    classificar_departamento: bool = True,
    classificar_categoria: bool = True,
    top_k: int = 3,
    auto_carregar: bool = True
) -> Dict
```

**Parâmetros:**
- `origem`: Canal de origem (Email, Painel, API, etc.)
- `assunto`: Título/assunto do chamado
- `mensagem`: Corpo/descrição do chamado
- `classificar_departamento`: Se deve classificar departamento
- `classificar_categoria`: Se deve classificar categoria
- `top_k`: Número de predições alternativas
- `auto_carregar`: Carregar modelos automaticamente

**Retorno:**
```python
{
    "origem": str,
    "assunto": str,
    "mensagem": str,
    "departamento": {
        "predicao": str,
        "confianca": float,
        "topk_labels": List[str],
        "topk_probas": List[float]
    },
    "categoria": {...},  # Mesmo formato
    "sucesso": bool,
    "erros": List[str]
}
```

## 🤝 Contribuindo

1. Fork o repositório
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Add: nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Crie um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

## 👥 Suporte

Para dúvidas, sugestões ou problemas:
- Abra uma Issue neste repositório
- Consulte a [documentação completa](docs/)
- Entre em contato: suporte@exemplo.com

## 🎯 Roadmap

- [ ] Interface web para treinamento e predição
- [ ] Suporte a mais algoritmos (Random Forest, XGBoost)
- [ ] API REST para integração
- [ ] Dashboard de métricas em tempo real
- [ ] Suporte multilíngue
- [ ] Active learning para melhorar modelos

---

Desenvolvido com ❤️ para automatizar e melhorar o atendimento ao cliente