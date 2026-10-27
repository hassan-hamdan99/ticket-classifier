"""
Funções utilitárias para preprocessamento de texto e operações auxiliares.
Inclui limpeza de texto, remoção de PIIs, normalização e funções de suporte.
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
import warnings
import json
from pathlib import Path

# Importações opcionais
try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:  # ImportError or a broken spaCy install
    SPACY_AVAILABLE = False
    warnings.warn("spaCy não está instalado. Algumas funcionalidades de NER estarão desabilitadas.")

from config import (
    REMOVE_PERSON_NAMES, MASK_IDS_EMAILS, REMOVE_URLS,
    PLACEHOLDERS, SPACY_MODEL, MIN_TEXT_LENGTH,
    MAX_TEXT_LENGTH, NORMALIZE_WHITESPACE
)

# ================================
# PADRÕES REGEX
# ================================

# Padrões para identificação de PIIs
PATTERNS = {
    "email": re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        re.IGNORECASE
    ),
    "url": re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        re.IGNORECASE
    ),
    "cpf": re.compile(
        r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b'
    ),
    "cnpj": re.compile(
        r'\b\d{2}\.?\d{3}\.?\d{3}\/?\d{4}-?\d{2}\b'
    ),
    "telefone": re.compile(
        r'\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?(?:9\s?\d{4}|\d{4})[-\s]?\d{4}\b'
    ),
    "id_numerico": re.compile(
        r'\b(?:ID|id|Id|#)\s*:?\s*\d{5,}\b'
    ),
    "protocolo": re.compile(
        r'\b(?:protocolo|ticket|chamado)\s*:?\s*\d{5,}\b',
        re.IGNORECASE
    ),
}

# Carrega modelo spaCy se disponível
nlp = None
if SPACY_AVAILABLE and REMOVE_PERSON_NAMES:
    try:
        nlp = spacy.load(SPACY_MODEL)
    except Exception as e:
        warnings.warn(f"Não foi possível carregar o modelo spaCy '{SPACY_MODEL}': {e}")
        SPACY_AVAILABLE = False

# ================================
# FUNÇÕES DE NORMALIZAÇÃO
# ================================

def slugify(text: str, separator: str = "_") -> str:
    """
    Normaliza string removendo caracteres especiais e acentos.
    Útil para criar identificadores válidos.

    Args:
        text: Texto a ser normalizado
        separator: Separador para substituir espaços

    Returns:
        String normalizada

    Examples:
        >>> slugify("São Paulo - SP")
        'sao_paulo_sp'
    """
    if not text:
        return ""

    # Remove acentos
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ASCII', 'ignore').decode('ASCII')

    # Converte para minúsculas
    text = text.lower()

    # Remove caracteres especiais
    text = re.sub(r'[^a-z0-9\s-]', '', text)

    # Substitui espaços e hífens pelo separador
    text = re.sub(r'[\s-]+', separator, text)

    # Remove separadores nas extremidades
    text = text.strip(separator)

    return text

def limpar_texto(texto: str, remover_urls: bool = True) -> str:
    """
    Realiza limpeza básica do texto.

    Args:
        texto: Texto a ser limpo
        remover_urls: Se deve remover URLs

    Returns:
        Texto limpo
    """
    if not texto:
        return ""

    # Remove URLs se solicitado
    if remover_urls:
        texto = PATTERNS["url"].sub(" ", texto)

    # Remove múltiplos espaços
    texto = re.sub(r'\s+', ' ', texto)

    # Remove espaços nas extremidades
    texto = texto.strip()

    # Limita tamanho do texto
    if len(texto) > MAX_TEXT_LENGTH:
        texto = texto[:MAX_TEXT_LENGTH]

    return texto

# ================================
# FUNÇÕES DE MASCARAMENTO DE PIIs
# ================================

def mascarar_emails(texto: str) -> str:
    """Substitui emails por placeholder."""
    return PATTERNS["email"].sub(PLACEHOLDERS["email"], texto)

def mascarar_cpf(texto: str) -> str:
    """Substitui CPFs por placeholder."""
    return PATTERNS["cpf"].sub(PLACEHOLDERS["cpf"], texto)

def mascarar_cnpj(texto: str) -> str:
    """Substitui CNPJs por placeholder."""
    return PATTERNS["cnpj"].sub(PLACEHOLDERS["cnpj"], texto)

def mascarar_telefones(texto: str) -> str:
    """Substitui telefones por placeholder."""
    return PATTERNS["telefone"].sub(PLACEHOLDERS["telefone"], texto)

def mascarar_ids(texto: str) -> str:
    """Substitui IDs e protocolos por placeholder."""
    texto = PATTERNS["id_numerico"].sub(PLACEHOLDERS["id"], texto)
    texto = PATTERNS["protocolo"].sub(PLACEHOLDERS["id"], texto)
    return texto

# Aliases: preprocess_for_ml has boolean parameters with the same names as these functions
_mascarar_emails = mascarar_emails
_mascarar_ids = mascarar_ids


def remover_nomes_proprios(texto: str) -> str:
    """
    Remove ou mascara nomes próprios usando NER (Named Entity Recognition).
    Requer spaCy instalado e modelo carregado.

    Args:
        texto: Texto a ser processado

    Returns:
        Texto com nomes mascarados
    """
    if not SPACY_AVAILABLE or nlp is None:
        return texto

    try:
        doc = nlp(texto)

        # Substitui entidades de pessoa
        for ent in doc.ents:
            if ent.label_ == "PER":  # Pessoa
                texto = texto.replace(ent.text, PLACEHOLDERS["nome"])

        return texto
    except Exception as e:
        warnings.warn(f"Erro ao processar NER: {e}")
        return texto

# ================================
# FUNÇÃO PRINCIPAL DE PREPROCESSAMENTO
# ================================

def preprocess_for_ml(
    texto: str,
    remover_urls: bool = REMOVE_URLS,
    mascarar_emails: bool = MASK_IDS_EMAILS,
    mascarar_ids: bool = MASK_IDS_EMAILS,
    remover_nomes: bool = REMOVE_PERSON_NAMES,
    normalizar_espacos: bool = NORMALIZE_WHITESPACE
) -> str:
    """
    Pipeline completo de preprocessamento para ML.

    Args:
        texto: Texto a ser processado
        remover_urls: Se deve remover URLs
        mascarar_emails: Se deve mascarar emails
        mascarar_ids: Se deve mascarar IDs e documentos
        remover_nomes: Se deve remover nomes próprios
        normalizar_espacos: Se deve normalizar espaços

    Returns:
        Texto preprocessado

    Examples:
        >>> texto = "João enviou email para maria@example.com sobre o CPF 123.456.789-00"
        >>> preprocess_for_ml(texto)
        '[NOME] enviou email para [EMAIL] sobre o [CPF]'
    """
    if not texto:
        return ""

    # Converte para string se necessário
    texto = str(texto)

    # 1. Limpeza básica
    texto = limpar_texto(texto, remover_urls=remover_urls)

    # 2. Mascaramento de PIIs
    if mascarar_emails:
        texto = _mascarar_emails(texto)

    if mascarar_ids:
        texto = mascarar_cpf(texto)
        texto = mascarar_cnpj(texto)
        texto = mascarar_telefones(texto)
        texto = _mascarar_ids(texto)

    # 3. Remoção de nomes próprios (se disponível)
    if remover_nomes and SPACY_AVAILABLE:
        texto = remover_nomes_proprios(texto)

    # 4. Normalização final
    if normalizar_espacos:
        # Remove múltiplos espaços
        texto = re.sub(r'\s+', ' ', texto)
        # Remove espaços nas extremidades
        texto = texto.strip()

    # 5. Validação de tamanho mínimo
    if len(texto) < MIN_TEXT_LENGTH:
        return ""

    return texto

# ================================
# FUNÇÕES DE PREDIÇÃO
# ================================

def topk_predictions(
    probas: List[float],
    labels: List[str],
    k: int = 3
) -> Tuple[List[str], List[float]]:
    """
    Retorna as top-k predições com maior probabilidade.

    Args:
        probas: Lista de probabilidades
        labels: Lista de labels correspondentes
        k: Número de predições a retornar

    Returns:
        Tupla com (top_k_labels, top_k_probas)

    Examples:
        >>> probas = [0.1, 0.3, 0.6]
        >>> labels = ["A", "B", "C"]
        >>> topk_predictions(probas, labels, k=2)
        (["C", "B"], [0.6, 0.3])
    """
    if not probas or not labels:
        return [], []

    # Combina probas com labels e ordena
    combined = list(zip(probas, labels))
    combined.sort(reverse=True, key=lambda x: x[0])

    # Extrai top-k
    k = min(k, len(combined))
    top_k = combined[:k]

    # Separa labels e probas
    top_k_labels = [label for _, label in top_k]
    top_k_probas = [proba for proba, _ in top_k]

    return top_k_labels, top_k_probas

# ================================
# FUNÇÕES DE VALIDAÇÃO
# ================================

def validar_texto(texto: str) -> Tuple[bool, Optional[str]]:
    """
    Valida se o texto é adequado para processamento.

    Args:
        texto: Texto a ser validado

    Returns:
        Tupla (is_valid, error_message)
    """
    if not texto:
        return False, "Texto vazio"

    texto = str(texto).strip()

    if len(texto) < MIN_TEXT_LENGTH:
        return False, f"Texto muito curto (mínimo: {MIN_TEXT_LENGTH} caracteres)"

    if len(texto) > MAX_TEXT_LENGTH:
        return False, f"Texto muito longo (máximo: {MAX_TEXT_LENGTH} caracteres)"

    # Verifica se tem conteúdo além de espaços
    if not re.search(r'\w', texto):
        return False, "Texto não contém palavras válidas"

    return True, None

def combinar_textos(
    assunto: Optional[str],
    mensagem: Optional[str],
    separador: str = " [SEP] "
) -> str:
    """
    Combina assunto e mensagem em um único texto.

    Args:
        assunto: Assunto/título
        mensagem: Corpo da mensagem
        separador: Separador entre assunto e mensagem

    Returns:
        Texto combinado
    """
    partes = []

    if assunto and assunto.strip():
        partes.append(assunto.strip())

    if mensagem and mensagem.strip():
        partes.append(mensagem.strip())

    if not partes:
        return ""

    return separador.join(partes)

# ================================
# FUNÇÕES DE ESTATÍSTICA
# ================================

def calcular_estatisticas_texto(texto: str) -> Dict[str, Any]:
    """
    Calcula estatísticas básicas do texto.

    Args:
        texto: Texto para análise

    Returns:
        Dicionário com estatísticas
    """
    if not texto:
        return {
            "n_caracteres": 0,
            "n_palavras": 0,
            "n_frases": 0,
            "n_emails": 0,
            "n_urls": 0,
            "tem_pii": False
        }

    # Conta elementos
    n_caracteres = len(texto)
    n_palavras = len(texto.split())
    n_frases = len(re.findall(r'[.!?]+', texto))
    n_emails = len(PATTERNS["email"].findall(texto))
    n_urls = len(PATTERNS["url"].findall(texto))

    # Verifica PIIs
    tem_pii = any([
        n_emails > 0,
        n_urls > 0,
        bool(PATTERNS["cpf"].search(texto)),
        bool(PATTERNS["cnpj"].search(texto)),
        bool(PATTERNS["telefone"].search(texto))
    ])

    return {
        "n_caracteres": n_caracteres,
        "n_palavras": n_palavras,
        "n_frases": n_frases,
        "n_emails": n_emails,
        "n_urls": n_urls,
        "tem_pii": tem_pii
    }

# ================================
# FUNÇÕES DE ARQUIVO
# ================================

def salvar_json(data: Dict, filepath: Path) -> None:
    """
    Salva dicionário em arquivo JSON.

    Args:
        data: Dados a salvar
        filepath: Caminho do arquivo
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def carregar_json(filepath: Path) -> Dict:
    """
    Carrega dicionário de arquivo JSON.

    Args:
        filepath: Caminho do arquivo

    Returns:
        Dicionário carregado
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# ================================
# FUNÇÕES DE FORMATAÇÃO
# ================================

def formatar_resultado_predicao(
    predicao: str,
    confianca: float,
    top_k_labels: List[str] = None,
    top_k_probas: List[float] = None
) -> str:
    """
    Formata resultado de predição para exibição.

    Args:
        predicao: Label predito
        confianca: Confiança da predição
        top_k_labels: Top-k labels
        top_k_probas: Top-k probabilidades

    Returns:
        String formatada
    """
    resultado = f"Predição: {predicao} (Confiança: {confianca:.2%})"

    if top_k_labels and top_k_probas:
        resultado += "\nTop predições:"
        for label, proba in zip(top_k_labels, top_k_probas):
            resultado += f"\n  - {label}: {proba:.2%}"

    return resultado

def truncar_texto(texto: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Trunca texto mantendo legibilidade.

    Args:
        texto: Texto a truncar
        max_length: Comprimento máximo
        suffix: Sufixo para texto truncado

    Returns:
        Texto truncado
    """
    if not texto or len(texto) <= max_length:
        return texto

    return texto[:max_length - len(suffix)] + suffix

# ================================
# VALIDAÇÃO DE IMPORTAÇÕES
# ================================

def verificar_dependencias() -> Dict[str, bool]:
    """
    Verifica disponibilidade de dependências opcionais.

    Returns:
        Dicionário com status das dependências
    """
    deps = {
        "spacy": SPACY_AVAILABLE,
        "spacy_model": nlp is not None if SPACY_AVAILABLE else False,
    }

    return deps

# ================================
# EXPORTS
# ================================

__all__ = [
    # Normalização
    'slugify',
    'limpar_texto',

    # Preprocessamento
    'preprocess_for_ml',
    'mascarar_emails',
    'mascarar_cpf',
    'mascarar_cnpj',
    'mascarar_telefones',
    'mascarar_ids',
    'remover_nomes_proprios',

    # Predição
    'topk_predictions',

    # Validação
    'validar_texto',
    'combinar_textos',

    # Estatísticas
    'calcular_estatisticas_texto',

    # Arquivos
    'salvar_json',
    'carregar_json',

    # Formatação
    'formatar_resultado_predicao',
    'truncar_texto',

    # Verificação
    'verificar_dependencias',

    # Padrões
    'PATTERNS'
]