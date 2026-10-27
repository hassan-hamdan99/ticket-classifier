"""
Arquivo de configuração central do sistema de classificação de chamados.
Contém todas as configurações, parâmetros e constantes utilizadas no projeto.
"""

import os
from pathlib import Path

# ================================
# CONFIGURAÇÕES DE PATHS
# ================================

# Diretório base do projeto
BASE_DIR = Path(__file__).parent.absolute()

# Diretórios principais
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Criar diretórios se não existirem
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ================================
# CONFIGURAÇÕES DE MACHINE LEARNING
# ================================

# Parâmetros gerais
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Parâmetros do TF-IDF
TFIDF_PARAMS = {
    "ngram_range": (1, 2),  # Unigramas e bigramas
    "min_df": 2,  # Mínimo de documentos onde o termo deve aparecer
    "max_df": 0.95,  # Máximo de documentos (remove termos muito comuns)
    "max_features": 10000,  # Número máximo de features
    "strip_accents": "unicode",  # Remove acentos
    "lowercase": True,  # Converte para minúsculas
    "use_idf": True,  # Usa IDF (Inverse Document Frequency)
    "sublinear_tf": True,  # Aplica escala logarítmica ao TF
}

# Parâmetros do Logistic Regression
LOGISTIC_PARAMS = {
    "class_weight": "balanced",  # Balanceia classes desbalanceadas
    "max_iter": 4000,  # Número máximo de iterações
    "random_state": RANDOM_STATE,
    "solver": "lbfgs",  # Solver para otimização
    "multi_class": "multinomial",  # Estratégia multiclasse
    "n_jobs": -1,  # Usa todos os cores disponíveis
}

# ================================
# CONFIGURAÇÕES DE PREPROCESSAMENTO
# ================================

# Flags de preprocessamento
REMOVE_PERSON_NAMES = True  # Remove nomes próprios usando NER
MASK_IDS_EMAILS = True  # Substitui emails e IDs por placeholders
REMOVE_URLS = True  # Remove URLs
REMOVE_NUMBERS = False  # Mantém números (podem ser relevantes)
REMOVE_SPECIAL_CHARS = True  # Remove caracteres especiais excessivos
NORMALIZE_WHITESPACE = True  # Normaliza espaços em branco
MAX_TEXT_LENGTH = 5000  # Comprimento máximo de texto (caracteres)

# Placeholders para mascaramento
PLACEHOLDERS = {
    "email": "[EMAIL]",
    "url": "[URL]",
    "cpf": "[CPF]",
    "cnpj": "[CNPJ]",
    "telefone": "[TELEFONE]",
    "id": "[ID]",
    "nome": "[NOME]",
}

# ================================
# MAPEAMENTO DE COLUNAS
# ================================

# Possíveis nomes de colunas nos dados de entrada
COL_MAP_OPCOES = {
    "origem": ["Origem", "origem", "Canal", "canal", "Source", "source", "Fonte", "fonte"],
    "assunto": ["Assunto", "assunto", "Título", "titulo", "Subject", "subject", "Resumo", "resumo"],
    "mensagem": ["Mensagem", "mensagem", "Descrição", "descricao", "Description", "description",
                 "Texto", "texto", "Conteúdo", "conteudo", "Body", "body"],
    "departamento": ["Departamento", "departamento", "Área", "area", "Department", "department",
                     "Setor", "setor", "Team", "team"],
    "categoria": ["Categoria", "categoria", "Tipo", "tipo", "Category", "category",
                  "Classificação", "classificacao", "Type", "type", "Subtipo", "subtipo"],
}

# Colunas obrigatórias
REQUIRED_COLUMNS = ["origem", "assunto", "mensagem"]

# Colunas alvo (para classificação)
TARGET_COLUMNS = ["departamento", "categoria"]

# ================================
# FILTROS DE QUALIDADE
# ================================

# Número mínimo de amostras por classe para treinar
MIN_SAMPLES_PER_CLASS_DEP = 5  # Para departamentos
MIN_SAMPLES_PER_CLASS_CAT = 30  # Para categorias

# Tamanho mínimo de texto (caracteres)
MIN_TEXT_LENGTH = 10

# Confiança mínima para predições
MIN_CONFIDENCE_THRESHOLD = 0.3

# ================================
# CONFIGURAÇÕES DE LOGGING
# ================================

# Formato de log
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Nível de log padrão
LOG_LEVEL = "INFO"

# ================================
# CONFIGURAÇÕES DE MODELO
# ================================

# Formato de nome dos modelos salvos
MODEL_NAME_FORMAT = "{origem}_{tipo}_model.joblib"
VECTORIZER_NAME_FORMAT = "{origem}_{tipo}_vectorizer.joblib"
LABEL_ENCODER_NAME_FORMAT = "{origem}_{tipo}_label_encoder.joblib"
METADATA_NAME_FORMAT = "{origem}_{tipo}_metadata.json"

# Tipos de modelo
MODEL_TYPES = ["departamento", "categoria"]

# ================================
# CONFIGURAÇÕES DE SPACY (OPCIONAL)
# ================================

# Modelo do spaCy para NER (Entity Recognition)
SPACY_MODEL = "pt_core_news_sm"  # Modelo em português
SPACY_BATCH_SIZE = 1000  # Tamanho do batch para processamento
SPACY_DISABLE_COMPONENTS = ["tok2vec", "parser", "attribute_ruler", "lemmatizer"]  # Desabilita componentes não usados

# ================================
# CONFIGURAÇÕES DE PERFORMANCE
# ================================

# Tamanho do batch para processamento
BATCH_SIZE = 100

# Cache de predições (em segundos)
CACHE_TTL = 3600  # 1 hora

# Número de threads para processamento paralelo
N_JOBS = -1  # Usa todos os cores disponíveis

# ================================
# CONFIGURAÇÕES DE VALIDAÇÃO
# ================================

# Métricas para avaliação
METRICS = ["accuracy", "precision", "recall", "f1"]

# Estratégia de validação cruzada
CV_FOLDS = 5  # Número de folds para cross-validation

# ================================
# CONFIGURAÇÕES DE OUTPUT
# ================================

# Formato de saída padrão
DEFAULT_OUTPUT_FORMAT = "json"  # Opções: "json", "csv", "text"

# Número de predições top-K para retornar
DEFAULT_TOP_K = 3

# Verbosidade padrão
DEFAULT_VERBOSITY = 1  # 0: silencioso, 1: normal, 2: detalhado

# ================================
# MENSAGENS E TEXTOS
# ================================

# Mensagens de erro
ERROR_MESSAGES = {
    "no_data": "Nenhum arquivo de dados encontrado no diretório: {dir}",
    "invalid_columns": "Colunas obrigatórias não encontradas: {cols}",
    "no_model": "Modelo não encontrado: {model}",
    "prediction_error": "Erro ao fazer predição: {error}",
    "insufficient_samples": "Amostras insuficientes para a classe {cls}: {n} (mínimo: {min})",
}

# Mensagens de sucesso
SUCCESS_MESSAGES = {
    "model_trained": "Modelo treinado com sucesso: {model}",
    "model_saved": "Modelo salvo em: {path}",
    "data_loaded": "Dados carregados: {n} amostras",
    "prediction_complete": "Predição completa para {n} amostras",
}

# ================================
# CONFIGURAÇÕES DE API
# ================================

# Limites de taxa (rate limiting)
API_RATE_LIMIT = 100  # Requisições por minuto
API_TIMEOUT = 30  # Timeout em segundos

# ================================
# CONFIGURAÇÕES DE EXPORTAÇÃO
# ================================

# Formatos de exportação suportados
EXPORT_FORMATS = ["csv", "json", "excel", "parquet"]

# Encoding padrão
DEFAULT_ENCODING = "utf-8"

# ================================
# CONFIGURAÇÕES DE DESENVOLVIMENTO
# ================================

# Modo debug
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Ambiente
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # development, staging, production

# ================================
# VALIDAÇÃO DE CONFIGURAÇÕES
# ================================

def validar_configuracoes():
    """Valida as configurações do sistema."""
    erros = []

    # Valida diretórios
    if not DATA_DIR.exists():
        erros.append(f"Diretório de dados não existe: {DATA_DIR}")

    # Valida parâmetros numéricos
    if TEST_SIZE <= 0 or TEST_SIZE >= 1:
        erros.append(f"TEST_SIZE deve estar entre 0 e 1: {TEST_SIZE}")

    if MIN_SAMPLES_PER_CLASS_DEP < 2:
        erros.append(f"MIN_SAMPLES_PER_CLASS_DEP deve ser >= 2: {MIN_SAMPLES_PER_CLASS_DEP}")

    if len(erros) > 0:
        raise ValueError(f"Erros de configuração encontrados: {', '.join(erros)}")

    return True

# Executa validação ao importar
if __name__ != "__main__":
    try:
        validar_configuracoes()
    except ValueError as e:
        print(f"Aviso de configuração: {e}")