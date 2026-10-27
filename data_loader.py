"""
Módulo para carregamento e processamento de dados de chamados.
Suporta múltiplos formatos (Excel, CSV) e realiza validação e preprocessamento.
"""

import pandas as pd
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

from config import (
    DATA_DIR, COL_MAP_OPCOES, REQUIRED_COLUMNS,
    TARGET_COLUMNS, MIN_SAMPLES_PER_CLASS_DEP,
    MIN_SAMPLES_PER_CLASS_CAT, MIN_TEXT_LENGTH
)
from utils import preprocess_for_ml, combinar_textos, validar_texto

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# FUNÇÕES AUXILIARES
# ================================

def _primeira_col_disponivel(df: pd.DataFrame, opcoes: List[str]) -> Optional[str]:
    """
    Encontra a primeira coluna disponível no DataFrame de uma lista de opções.

    Args:
        df: DataFrame para verificar
        opcoes: Lista de possíveis nomes de colunas

    Returns:
        Nome da primeira coluna encontrada ou None
    """
    colunas_df = df.columns.tolist()

    for opcao in opcoes:
        if opcao in colunas_df:
            return opcao

    # Tenta busca case-insensitive
    colunas_lower = {col.lower(): col for col in colunas_df}
    for opcao in opcoes:
        if opcao.lower() in colunas_lower:
            return colunas_lower[opcao.lower()]

    return None

def _mapear_colunas(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Mapeia colunas do DataFrame para nomes padronizados.

    Args:
        df: DataFrame original

    Returns:
        Tupla (DataFrame com colunas renomeadas, dicionário de mapeamento)
    """
    mapeamento = {}
    df_mapped = df.copy()

    for col_padrao, opcoes in COL_MAP_OPCOES.items():
        col_encontrada = _primeira_col_disponivel(df, opcoes)

        if col_encontrada:
            mapeamento[col_encontrada] = col_padrao
            if col_encontrada != col_padrao:
                df_mapped = df_mapped.rename(columns={col_encontrada: col_padrao})
                logger.debug(f"Coluna '{col_encontrada}' mapeada para '{col_padrao}'")

    return df_mapped, mapeamento

def _validar_colunas(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Valida se o DataFrame tem as colunas obrigatórias.

    Args:
        df: DataFrame a validar

    Returns:
        Tupla (is_valid, lista de colunas faltando)
    """
    colunas_faltando = []

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            colunas_faltando.append(col)

    return len(colunas_faltando) == 0, colunas_faltando

def _ler_um_arquivo(filepath: Path) -> pd.DataFrame:
    """
    Lê um arquivo de dados (Excel ou CSV).

    Args:
        filepath: Caminho do arquivo

    Returns:
        DataFrame lido

    Raises:
        ValueError: Se o formato não for suportado
        FileNotFoundError: Se o arquivo não existir
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    # Determina formato pelo sufixo
    suffix = filepath.suffix.lower()

    try:
        if suffix in ['.xlsx', '.xlsm', '.xls']:
            # Lê Excel
            df = pd.read_excel(filepath, engine='openpyxl' if suffix == '.xlsx' else None)
            logger.info(f"Arquivo Excel lido: {filepath.name} ({len(df)} linhas)")

        elif suffix == '.csv':
            # Tenta diferentes encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            df = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    logger.info(f"Arquivo CSV lido com encoding {encoding}: {filepath.name} ({len(df)} linhas)")
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                raise ValueError(f"Não foi possível ler o arquivo CSV com nenhum encoding")

        else:
            raise ValueError(f"Formato de arquivo não suportado: {suffix}")

        return df

    except Exception as e:
        logger.error(f"Erro ao ler arquivo {filepath}: {e}")
        raise

# ================================
# FUNÇÃO PRINCIPAL DE CARREGAMENTO
# ================================

def carregar_chamados(
    data_dir: Optional[Path] = None,
    arquivo_especifico: Optional[str] = None,
    origem_filtro: Optional[str] = None,
    preprocessar: bool = True,
    validar: bool = True,
    min_samples: Optional[int] = None
) -> pd.DataFrame:
    """
    Carrega e processa dados de chamados.

    Args:
        data_dir: Diretório com os dados (padrão: DATA_DIR do config)
        arquivo_especifico: Nome de arquivo específico para carregar
        origem_filtro: Filtrar por origem específica
        preprocessar: Se deve aplicar preprocessamento
        validar: Se deve validar dados
        min_samples: Número mínimo de amostras por classe

    Returns:
        DataFrame processado com os chamados

    Raises:
        FileNotFoundError: Se nenhum arquivo for encontrado
        ValueError: Se os dados não passarem na validação

    Examples:
        >>> df = carregar_chamados()
        >>> df = carregar_chamados(origem_filtro="Email")
        >>> df = carregar_chamados(arquivo_especifico="dados_outubro.xlsx")
    """
    # Define diretório
    data_dir = Path(data_dir) if data_dir else DATA_DIR

    if not data_dir.exists():
        raise FileNotFoundError(f"Diretório de dados não encontrado: {data_dir}")

    # Lista arquivos disponíveis
    all_dfs = []

    if arquivo_especifico:
        # Carrega arquivo específico
        filepath = data_dir / arquivo_especifico
        if not filepath.exists():
            raise FileNotFoundError(f"Arquivo especificado não encontrado: {filepath}")

        df = _ler_um_arquivo(filepath)
        all_dfs.append(df)

    else:
        # Carrega todos os arquivos do diretório
        patterns = ['*.xlsx', '*.xlsm', '*.xls', '*.csv']
        files_found = []

        for pattern in patterns:
            files_found.extend(list(data_dir.glob(pattern)))

        if not files_found:
            raise FileNotFoundError(f"Nenhum arquivo de dados encontrado em: {data_dir}")

        logger.info(f"Encontrados {len(files_found)} arquivos de dados")

        for filepath in files_found:
            try:
                df = _ler_um_arquivo(filepath)
                all_dfs.append(df)
            except Exception as e:
                logger.warning(f"Erro ao ler {filepath.name}: {e}")
                continue

    if not all_dfs:
        raise ValueError("Nenhum arquivo pôde ser lido com sucesso")

    # Combina todos os DataFrames
    df_combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Total de {len(df_combined)} chamados carregados")

    # Mapeia colunas
    df_combined, mapeamento = _mapear_colunas(df_combined)

    # Valida colunas obrigatórias
    if validar:
        is_valid, cols_faltando = _validar_colunas(df_combined)
        if not is_valid:
            raise ValueError(f"Colunas obrigatórias não encontradas: {cols_faltando}")

    # Filtra por origem se especificado
    if origem_filtro and "origem" in df_combined.columns:
        df_combined = df_combined[df_combined["origem"] == origem_filtro].copy()
        logger.info(f"Filtrado para origem '{origem_filtro}': {len(df_combined)} chamados")

    # Remove linhas com valores nulos nas colunas obrigatórias
    for col in REQUIRED_COLUMNS:
        if col in df_combined.columns:
            n_before = len(df_combined)
            df_combined = df_combined.dropna(subset=[col])
            n_removed = n_before - len(df_combined)
            if n_removed > 0:
                logger.warning(f"Removidas {n_removed} linhas com valores nulos em '{col}'")

    # Preprocessamento
    if preprocessar:
        logger.info("Aplicando preprocessamento...")

        # Combina assunto e mensagem
        if "assunto" in df_combined.columns and "mensagem" in df_combined.columns:
            df_combined["texto_completo"] = df_combined.apply(
                lambda row: combinar_textos(row["assunto"], row["mensagem"]),
                axis=1
            )

            # Aplica preprocessamento
            df_combined["texto_processado"] = df_combined["texto_completo"].apply(preprocess_for_ml)

            # Remove textos vazios após processamento
            n_before = len(df_combined)
            df_combined = df_combined[df_combined["texto_processado"].str.len() >= MIN_TEXT_LENGTH]
            n_removed = n_before - len(df_combined)
            if n_removed > 0:
                logger.warning(f"Removidas {n_removed} linhas com texto insuficiente após processamento")

    # Validação de classes
    if validar and min_samples:
        for target_col in TARGET_COLUMNS:
            if target_col in df_combined.columns:
                value_counts = df_combined[target_col].value_counts()
                classes_insuficientes = value_counts[value_counts < min_samples]

                if not classes_insuficientes.empty:
                    logger.warning(f"Classes com amostras insuficientes em '{target_col}':")
                    for classe, count in classes_insuficientes.items():
                        logger.warning(f"  - {classe}: {count} amostras (mínimo: {min_samples})")

                    # Remove classes com poucas amostras
                    classes_validas = value_counts[value_counts >= min_samples].index
                    df_combined = df_combined[df_combined[target_col].isin(classes_validas)]

    logger.info(f"DataFrame final: {len(df_combined)} linhas, {len(df_combined.columns)} colunas")

    return df_combined

# ================================
# FUNÇÕES DE ANÁLISE
# ================================

def analisar_distribuicao_classes(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Analisa a distribuição de classes no DataFrame.

    Args:
        df: DataFrame com os dados

    Returns:
        Dicionário com distribuições por coluna alvo
    """
    distribuicoes = {}

    for target_col in TARGET_COLUMNS:
        if target_col in df.columns:
            distribuicoes[target_col] = df[target_col].value_counts()

    return distribuicoes

def listar_arquivos_dados(data_dir: Optional[Path] = None) -> List[Dict[str, any]]:
    """
    Lista arquivos de dados disponíveis com informações.

    Args:
        data_dir: Diretório dos dados

    Returns:
        Lista de dicionários com informações dos arquivos
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR

    if not data_dir.exists():
        return []

    arquivos_info = []
    patterns = ['*.xlsx', '*.xlsm', '*.xls', '*.csv']

    for pattern in patterns:
        for filepath in data_dir.glob(pattern):
            try:
                # Informações básicas do arquivo
                info = {
                    "nome": filepath.name,
                    "caminho": str(filepath),
                    "tamanho_mb": filepath.stat().st_size / (1024 * 1024),
                    "formato": filepath.suffix[1:].upper()
                }

                # Tenta ler para obter mais informações
                try:
                    df = _ler_um_arquivo(filepath)
                    df_mapped, _ = _mapear_colunas(df)

                    info.update({
                        "n_linhas": len(df),
                        "n_colunas": len(df.columns),
                        "colunas": df.columns.tolist(),
                        "tem_origem": "origem" in df_mapped.columns,
                        "tem_departamento": "departamento" in df_mapped.columns,
                        "tem_categoria": "categoria" in df_mapped.columns,
                    })

                    # Conta origens se disponível
                    if "origem" in df_mapped.columns:
                        info["origens"] = df_mapped["origem"].unique().tolist()

                except Exception as e:
                    info["erro_leitura"] = str(e)

                arquivos_info.append(info)

            except Exception as e:
                logger.error(f"Erro ao analisar {filepath}: {e}")

    return arquivos_info

def preparar_dados_treinamento(
    df: pd.DataFrame,
    target_col: str,
    text_col: str = "texto_processado",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Prepara dados para treinamento com split treino/teste.

    Args:
        df: DataFrame com os dados
        target_col: Nome da coluna alvo
        text_col: Nome da coluna de texto
        test_size: Proporção para teste
        random_state: Seed para reprodutibilidade

    Returns:
        Tupla (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split

    # Valida colunas
    if text_col not in df.columns:
        raise ValueError(f"Coluna de texto '{text_col}' não encontrada")

    if target_col not in df.columns:
        raise ValueError(f"Coluna alvo '{target_col}' não encontrada")

    # Remove valores nulos
    df_clean = df[[text_col, target_col]].dropna()

    if len(df_clean) == 0:
        raise ValueError("Nenhum dado válido após remoção de valores nulos")

    # Separa features e target
    X = df_clean[text_col]
    y = df_clean[target_col]

    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(f"Dados preparados - Treino: {len(X_train)}, Teste: {len(X_test)}")

    return X_train, X_test, y_train, y_test

# ================================
# FUNÇÕES DE VALIDAÇÃO DE DADOS
# ================================

def validar_dados_entrada(
    df: pd.DataFrame,
    origem_especifica: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """
    Valida completude e qualidade dos dados de entrada.

    Args:
        df: DataFrame a validar
        origem_especifica: Se especificado, valida apenas essa origem

    Returns:
        Tupla (is_valid, lista de avisos/erros)
    """
    avisos = []
    is_valid = True

    # Valida colunas obrigatórias
    cols_valid, cols_faltando = _validar_colunas(df)
    if not cols_valid:
        avisos.append(f"Colunas obrigatórias faltando: {cols_faltando}")
        is_valid = False

    # Verifica valores nulos
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            n_nulls = df[col].isnull().sum()
            if n_nulls > 0:
                pct = (n_nulls / len(df)) * 100
                avisos.append(f"Coluna '{col}' tem {n_nulls} ({pct:.1f}%) valores nulos")

    # Verifica tamanho mínimo de texto
    if "mensagem" in df.columns:
        textos_curtos = df[df["mensagem"].str.len() < MIN_TEXT_LENGTH]
        if len(textos_curtos) > 0:
            avisos.append(f"{len(textos_curtos)} mensagens muito curtas (< {MIN_TEXT_LENGTH} caracteres)")

    # Valida distribuição de classes
    for target_col in TARGET_COLUMNS:
        if target_col in df.columns:
            value_counts = df[target_col].value_counts()

            # Verifica classes com poucas amostras
            min_samples = (MIN_SAMPLES_PER_CLASS_DEP if target_col == "departamento"
                          else MIN_SAMPLES_PER_CLASS_CAT)

            classes_insuf = value_counts[value_counts < min_samples]
            if not classes_insuf.empty:
                avisos.append(f"{len(classes_insuf)} classes em '{target_col}' com < {min_samples} amostras")

            # Verifica desbalanceamento
            if len(value_counts) > 1:
                ratio = value_counts.min() / value_counts.max()
                if ratio < 0.1:
                    avisos.append(f"Forte desbalanceamento em '{target_col}' (ratio: {ratio:.2f})")

    # Valida origem específica se fornecida
    if origem_especifica and "origem" in df.columns:
        df_origem = df[df["origem"] == origem_especifica]
        if len(df_origem) == 0:
            avisos.append(f"Nenhum dado encontrado para origem '{origem_especifica}'")
            is_valid = False
        elif len(df_origem) < 100:
            avisos.append(f"Poucos dados para origem '{origem_especifica}': {len(df_origem)} amostras")

    return is_valid, avisos

# ================================
# EXPORTS
# ================================

__all__ = [
    'carregar_chamados',
    'analisar_distribuicao_classes',
    'listar_arquivos_dados',
    'preparar_dados_treinamento',
    'validar_dados_entrada'
]