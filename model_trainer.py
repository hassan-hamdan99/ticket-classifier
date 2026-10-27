"""
Módulo para treinamento e avaliação de modelos de classificação.
Implementa pipeline de ML com TF-IDF e Logistic Regression.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

from config import (
    MODELS_DIR, TFIDF_PARAMS, LOGISTIC_PARAMS,
    MODEL_NAME_FORMAT, VECTORIZER_NAME_FORMAT,
    LABEL_ENCODER_NAME_FORMAT, METADATA_NAME_FORMAT,
    CV_FOLDS, RANDOM_STATE, TEST_SIZE
)
from utils import salvar_json, slugify
from data_loader import preparar_dados_treinamento

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# CRIAÇÃO DE PIPELINE
# ================================

def criar_pipeline_tfidf_lr(
    tfidf_params: Optional[Dict] = None,
    lr_params: Optional[Dict] = None
) -> Pipeline:
    """
    Cria pipeline de TF-IDF + Logistic Regression.

    Args:
        tfidf_params: Parâmetros para TfidfVectorizer
        lr_params: Parâmetros para LogisticRegression

    Returns:
        Pipeline configurado
    """
    # Usa parâmetros padrão se não fornecidos
    tfidf_params = tfidf_params or TFIDF_PARAMS
    lr_params = lr_params or LOGISTIC_PARAMS

    # Cria componentes
    tfidf = TfidfVectorizer(**tfidf_params)
    lr = LogisticRegression(**lr_params)

    # Monta pipeline
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('classifier', lr)
    ])

    logger.debug(f"Pipeline criado com parâmetros TF-IDF: {tfidf_params}")
    logger.debug(f"Pipeline criado com parâmetros LR: {lr_params}")

    return pipeline

# ================================
# TREINAMENTO E AVALIAÇÃO
# ================================

def treinar_avaliar(
    X_train: pd.Series,
    X_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    pipeline: Optional[Pipeline] = None,
    cv_folds: int = CV_FOLDS,
    return_model: bool = True
) -> Dict[str, Any]:
    """
    Treina e avalia um modelo de classificação.

    Args:
        X_train: Textos de treino
        X_test: Textos de teste
        y_train: Labels de treino
        y_test: Labels de teste
        pipeline: Pipeline a usar (cria padrão se None)
        cv_folds: Número de folds para cross-validation
        return_model: Se deve retornar o modelo treinado

    Returns:
        Dicionário com métricas e opcionalmente o modelo
    """
    # Cria pipeline se não fornecido
    if pipeline is None:
        pipeline = criar_pipeline_tfidf_lr()

    # Codifica labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    logger.info(f"Classes encontradas: {list(label_encoder.classes_)}")
    logger.info(f"Treinando com {len(X_train)} amostras...")

    # Treinamento
    start_time = datetime.now()
    pipeline.fit(X_train, y_train_encoded)
    train_time = (datetime.now() - start_time).total_seconds()

    # Predições
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)

    # Métricas básicas
    accuracy = accuracy_score(y_test_encoded, y_pred)
    precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)

    # Cross-validation (opcional)
    cv_scores = None
    if cv_folds > 1:
        logger.info(f"Realizando cross-validation com {cv_folds} folds...")
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            pipeline, X_train, y_train_encoded,
            cv=skf, scoring='f1_weighted', n_jobs=-1
        )

    # Relatório de classificação
    class_report = classification_report(
        y_test_encoded, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    # Matriz de confusão
    cm = confusion_matrix(y_test_encoded, y_pred)

    # Monta resultado
    resultado = {
        "metricas": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        },
        "n_classes": len(label_encoder.classes_),
        "classes": list(label_encoder.classes_),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_time_seconds": train_time,
        "classification_report": class_report,
        "confusion_matrix": cm.tolist(),
    }

    if cv_scores is not None:
        resultado["cross_validation"] = {
            "scores": cv_scores.tolist(),
            "mean": float(cv_scores.mean()),
            "std": float(cv_scores.std()),
            "folds": cv_folds
        }

    if return_model:
        resultado["model"] = pipeline
        resultado["label_encoder"] = label_encoder

    logger.info(f"Treinamento concluído - F1-Score: {f1:.3f}, Accuracy: {accuracy:.3f}")

    return resultado

# ================================
# EXTRAÇÃO DE FEATURES IMPORTANTES
# ================================

def extrair_top_termos(
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    top_n: int = 10
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Extrai os termos mais importantes para cada classe.

    Args:
        pipeline: Pipeline treinado
        label_encoder: LabelEncoder usado no treinamento
        top_n: Número de termos top para cada classe

    Returns:
        Dicionário {classe: [(termo, peso), ...]}
    """
    # Extrai componentes
    tfidf = pipeline.named_steps['tfidf']
    classifier = pipeline.named_steps['classifier']

    # Vocabulário
    feature_names = tfidf.get_feature_names_out()

    # Coeficientes por classe
    coefs = classifier.coef_

    top_terms = {}

    for i, classe in enumerate(label_encoder.classes_):
        # Coeficientes desta classe
        class_coefs = coefs[i] if len(coefs.shape) > 1 else coefs[0]

        # Índices dos top termos
        top_indices = np.argsort(class_coefs)[-top_n:][::-1]

        # Extrai termos e pesos
        terms_weights = [
            (feature_names[idx], float(class_coefs[idx]))
            for idx in top_indices
        ]

        top_terms[classe] = terms_weights

    return top_terms

# ================================
# SALVAMENTO DE MODELOS
# ================================

def salvar_modelo(
    model_data: Dict[str, Any],
    origem: str,
    tipo: str,
    output_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Salva modelo, vectorizer, encoder e metadados.

    Args:
        model_data: Dicionário com model, label_encoder e métricas
        origem: Origem do modelo (ex: "Email")
        tipo: Tipo do modelo ("departamento" ou "categoria")
        output_dir: Diretório de saída (padrão: MODELS_DIR)

    Returns:
        Dicionário com caminhos dos arquivos salvos
    """
    output_dir = Path(output_dir) if output_dir else MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    origem_slug = slugify(origem)
    timestamp = datetime.now().isoformat()

    paths = {}

    # 1. Salva pipeline completo
    model_path = output_dir / MODEL_NAME_FORMAT.format(origem=origem_slug, tipo=tipo)
    joblib.dump(model_data["model"], model_path)
    paths["model"] = model_path
    logger.info(f"Modelo salvo em: {model_path}")

    # 2. Salva label encoder
    encoder_path = output_dir / LABEL_ENCODER_NAME_FORMAT.format(origem=origem_slug, tipo=tipo)
    joblib.dump(model_data["label_encoder"], encoder_path)
    paths["label_encoder"] = encoder_path

    # 3. Salva metadados
    metadata = {
        "origem": origem,
        "tipo": tipo,
        "timestamp_treinamento": timestamp,
        "metricas": model_data.get("metricas", {}),
        "n_classes": model_data.get("n_classes", 0),
        "classes": model_data.get("classes", []),
        "n_train": model_data.get("n_train", 0),
        "n_test": model_data.get("n_test", 0),
        "cross_validation": model_data.get("cross_validation", {}),
        "pipeline_params": {
            "tfidf": TFIDF_PARAMS,
            "logistic": LOGISTIC_PARAMS
        }
    }

    metadata_path = output_dir / METADATA_NAME_FORMAT.format(origem=origem_slug, tipo=tipo)
    salvar_json(metadata, metadata_path)
    paths["metadata"] = metadata_path

    # 4. Extrai e salva top termos (para interpretabilidade)
    if "model" in model_data and "label_encoder" in model_data:
        try:
            top_terms = extrair_top_termos(
                model_data["model"],
                model_data["label_encoder"],
                top_n=20
            )
            terms_path = output_dir / f"{origem_slug}_{tipo}_top_terms.json"
            salvar_json(top_terms, terms_path)
            paths["top_terms"] = terms_path
        except Exception as e:
            logger.warning(f"Não foi possível extrair top termos: {e}")

    logger.info(f"Modelo '{origem}_{tipo}' salvo com sucesso!")

    return paths

# ================================
# TREINAMENTO POR ORIGEM
# ================================

def treinar_por_origem(
    df: pd.DataFrame,
    origem: str,
    tipos: List[str] = ["departamento", "categoria"],
    text_col: str = "texto_processado",
    test_size: float = TEST_SIZE,
    save_models: bool = True,
    output_dir: Optional[Path] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Treina modelos para uma origem específica.

    Args:
        df: DataFrame com os dados
        origem: Origem para treinar
        tipos: Lista de tipos de modelo para treinar
        text_col: Coluna de texto
        test_size: Proporção para teste
        save_models: Se deve salvar os modelos
        output_dir: Diretório de saída

    Returns:
        Dicionário com resultados por tipo
    """
    # Filtra dados da origem
    df_origem = df[df["origem"] == origem].copy()

    if len(df_origem) == 0:
        raise ValueError(f"Nenhum dado encontrado para origem: {origem}")

    logger.info(f"Treinando modelos para origem '{origem}' ({len(df_origem)} amostras)")

    resultados = {}

    for tipo in tipos:
        if tipo not in df_origem.columns:
            logger.warning(f"Coluna '{tipo}' não encontrada para origem '{origem}'")
            continue

        # Remove valores nulos
        df_valid = df_origem.dropna(subset=[text_col, tipo])

        if len(df_valid) < 50:  # Mínimo absoluto para treinar
            logger.warning(f"Amostras insuficientes para '{origem}_{tipo}': {len(df_valid)}")
            continue

        logger.info(f"\nTreinando modelo '{origem}_{tipo}'...")

        try:
            # Prepara dados
            X_train, X_test, y_train, y_test = preparar_dados_treinamento(
                df_valid,
                target_col=tipo,
                text_col=text_col,
                test_size=test_size,
                random_state=RANDOM_STATE
            )

            # Treina e avalia
            resultado = treinar_avaliar(
                X_train, X_test, y_train, y_test,
                return_model=True
            )

            # Salva modelo se solicitado
            if save_models:
                paths = salvar_modelo(
                    resultado,
                    origem=origem,
                    tipo=tipo,
                    output_dir=output_dir
                )
                resultado["saved_paths"] = {k: str(v) for k, v in paths.items()}

            resultados[tipo] = resultado

        except Exception as e:
            logger.error(f"Erro ao treinar '{origem}_{tipo}': {e}")
            resultados[tipo] = {"erro": str(e)}

    return resultados

# ================================
# TREINAMENTO COMPLETO
# ================================

def treinar_todos_modelos(
    df: pd.DataFrame,
    origens: Optional[List[str]] = None,
    tipos: List[str] = ["departamento", "categoria"],
    text_col: str = "texto_processado",
    save_models: bool = True,
    output_dir: Optional[Path] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Treina modelos para todas as origens.

    Args:
        df: DataFrame com todos os dados
        origens: Lista de origens (None = todas)
        tipos: Tipos de modelo para treinar
        text_col: Coluna de texto
        save_models: Se deve salvar modelos
        output_dir: Diretório de saída

    Returns:
        Dicionário com resultados por origem e tipo
    """
    # Identifica origens disponíveis
    if origens is None:
        if "origem" not in df.columns:
            raise ValueError("Coluna 'origem' não encontrada no DataFrame")
        origens = df["origem"].unique().tolist()

    logger.info(f"Treinando modelos para {len(origens)} origens: {origens}")

    todos_resultados = {}

    for origem in origens:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processando origem: {origem}")
        logger.info(f"{'='*50}")

        try:
            resultados_origem = treinar_por_origem(
                df,
                origem=origem,
                tipos=tipos,
                text_col=text_col,
                save_models=save_models,
                output_dir=output_dir
            )

            todos_resultados[origem] = resultados_origem

        except Exception as e:
            logger.error(f"Erro ao processar origem '{origem}': {e}")
            todos_resultados[origem] = {"erro": str(e)}

    # Salva resumo geral
    if save_models and output_dir:
        resumo_path = Path(output_dir) / "resumo_treinamento.json"
        resumo = {
            "timestamp": datetime.now().isoformat(),
            "n_origens": len(origens),
            "origens": origens,
            "tipos": tipos,
            "resultados": {}
        }

        for origem, res_origem in todos_resultados.items():
            resumo["resultados"][origem] = {}
            for tipo, res_tipo in res_origem.items():
                if "erro" not in res_tipo:
                    resumo["resultados"][origem][tipo] = {
                        "f1_score": res_tipo.get("metricas", {}).get("f1_score"),
                        "accuracy": res_tipo.get("metricas", {}).get("accuracy"),
                        "n_classes": res_tipo.get("n_classes"),
                        "n_train": res_tipo.get("n_train")
                    }

        salvar_json(resumo, resumo_path)
        logger.info(f"\nResumo salvo em: {resumo_path}")

    return todos_resultados

# ================================
# FUNÇÕES AUXILIARES
# ================================

def avaliar_modelo_carregado(
    model_path: Path,
    encoder_path: Path,
    X_test: pd.Series,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Avalia um modelo já treinado e salvo.

    Args:
        model_path: Caminho do modelo
        encoder_path: Caminho do label encoder
        X_test: Textos de teste
        y_test: Labels de teste

    Returns:
        Dicionário com métricas
    """
    # Carrega modelo e encoder
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

    # Codifica labels
    y_test_encoded = label_encoder.transform(y_test)

    # Predições
    y_pred = model.predict(X_test)

    # Calcula métricas
    metrics = {
        "accuracy": accuracy_score(y_test_encoded, y_pred),
        "precision": precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
    }

    return metrics

# ================================
# EXPORTS
# ================================

__all__ = [
    'criar_pipeline_tfidf_lr',
    'treinar_avaliar',
    'extrair_top_termos',
    'salvar_modelo',
    'treinar_por_origem',
    'treinar_todos_modelos',
    'avaliar_modelo_carregado'
]