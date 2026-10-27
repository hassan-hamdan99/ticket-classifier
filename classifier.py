"""
Sistema principal de classificação de chamados.
Este é o arquivo central que implementa a classe ClassificadorChamados
e a função classificador_mensagem solicitada.
"""

import joblib
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from config import (
    MODELS_DIR, MODEL_NAME_FORMAT, LABEL_ENCODER_NAME_FORMAT,
    METADATA_NAME_FORMAT, DEFAULT_TOP_K, MIN_CONFIDENCE_THRESHOLD
)
from utils import (
    preprocess_for_ml, combinar_textos, validar_texto,
    topk_predictions, carregar_json, slugify
)

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# CLASSE PRINCIPAL
# ================================

class ClassificadorChamados:
    """
    Sistema completo de classificação de chamados/tickets.
    Suporta múltiplas origens e tipos de classificação.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        """
        Inicializa o classificador.

        Args:
            models_dir: Diretório dos modelos (padrão: MODELS_DIR)
        """
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.modelos_carregados = {}
        self.encoders_carregados = {}
        self.metadados_carregados = {}

        # Verifica se diretório existe
        if not self.models_dir.exists():
            logger.warning(f"Diretório de modelos não existe: {self.models_dir}")
            self.models_dir.mkdir(parents=True, exist_ok=True)

    def carregar_modelo(
        self,
        origem: str,
        tipo: str,
        force_reload: bool = False
    ) -> bool:
        """
        Carrega um modelo específico.

        Args:
            origem: Origem do modelo
            tipo: Tipo do modelo (departamento/categoria)
            force_reload: Força recarregamento mesmo se já carregado

        Returns:
            True se carregado com sucesso
        """
        origem_slug = slugify(origem)
        model_key = f"{origem_slug}_{tipo}"

        # Verifica se já está carregado
        if not force_reload and model_key in self.modelos_carregados:
            logger.debug(f"Modelo '{model_key}' já carregado")
            return True

        # Paths dos arquivos
        model_path = self.models_dir / MODEL_NAME_FORMAT.format(origem=origem_slug, tipo=tipo)
        encoder_path = self.models_dir / LABEL_ENCODER_NAME_FORMAT.format(origem=origem_slug, tipo=tipo)
        metadata_path = self.models_dir / METADATA_NAME_FORMAT.format(origem=origem_slug, tipo=tipo)

        # Verifica se arquivos existem
        if not model_path.exists():
            logger.warning(f"Modelo não encontrado: {model_path}")
            return False

        if not encoder_path.exists():
            logger.warning(f"Label encoder não encontrado: {encoder_path}")
            return False

        try:
            # Carrega modelo
            self.modelos_carregados[model_key] = joblib.load(model_path)
            logger.info(f"Modelo carregado: {model_key}")

            # Carrega encoder
            self.encoders_carregados[model_key] = joblib.load(encoder_path)

            # Carrega metadados se disponível
            if metadata_path.exists():
                self.metadados_carregados[model_key] = carregar_json(metadata_path)
            else:
                self.metadados_carregados[model_key] = {}

            return True

        except Exception as e:
            logger.error(f"Erro ao carregar modelo '{model_key}': {e}")
            # Remove parcialmente carregados em caso de erro
            self.modelos_carregados.pop(model_key, None)
            self.encoders_carregados.pop(model_key, None)
            self.metadados_carregados.pop(model_key, None)
            return False

    def carregar_todos_modelos(self) -> Dict[str, bool]:
        """
        Carrega todos os modelos disponíveis no diretório.

        Returns:
            Dicionário com status de carregamento por modelo
        """
        status = {}

        # Lista todos os arquivos de modelo
        model_files = list(self.models_dir.glob("*_model.joblib"))

        if not model_files:
            logger.warning(f"Nenhum modelo encontrado em: {self.models_dir}")
            return status

        for model_file in model_files:
            # Extrai origem e tipo do nome do arquivo
            parts = model_file.stem.replace("_model", "").rsplit("_", 1)
            if len(parts) == 2:
                origem, tipo = parts

                # Tenta carregar
                sucesso = self.carregar_modelo(origem, tipo)
                status[f"{origem}_{tipo}"] = sucesso

        logger.info(f"Carregados {sum(status.values())}/{len(status)} modelos")
        return status

    def prever_unico(
        self,
        texto: str,
        origem: str,
        tipo: str,
        top_k: int = DEFAULT_TOP_K,
        preprocessar: bool = True
    ) -> Dict[str, Any]:
        """
        Faz predição para um único texto.

        Args:
            texto: Texto para classificar
            origem: Origem do chamado
            tipo: Tipo de classificação
            top_k: Número de predições top
            preprocessar: Se deve preprocessar o texto

        Returns:
            Dicionário com predição e metadados
        """
        origem_slug = slugify(origem)
        model_key = f"{origem_slug}_{tipo}"

        # Verifica se modelo está carregado
        if model_key not in self.modelos_carregados:
            # Tenta carregar
            if not self.carregar_modelo(origem, tipo):
                return {
                    "erro": f"Modelo não disponível: {model_key}",
                    "sucesso": False
                }

        # Valida texto
        is_valid, erro = validar_texto(texto)
        if not is_valid:
            return {
                "erro": f"Texto inválido: {erro}",
                "sucesso": False
            }

        # Preprocessa se necessário
        if preprocessar:
            texto = preprocess_for_ml(texto)

        try:
            # Obtém modelo e encoder
            modelo = self.modelos_carregados[model_key]
            encoder = self.encoders_carregados[model_key]

            # Predição
            pred_encoded = modelo.predict([texto])[0]
            pred_proba = modelo.predict_proba([texto])[0]

            # Decodifica predição
            pred_label = encoder.inverse_transform([pred_encoded])[0]

            # Confiança da predição principal
            confianca = float(pred_proba[pred_encoded])

            # Top-k predições
            top_k_indices = np.argsort(pred_proba)[-top_k:][::-1]
            top_k_labels = encoder.inverse_transform(top_k_indices).tolist()
            top_k_probas = pred_proba[top_k_indices].tolist()

            return {
                "predicao": pred_label,
                "confianca": confianca,
                "topk_labels": top_k_labels,
                "topk_probas": top_k_probas,
                "modelo_usado": model_key,
                "sucesso": True
            }

        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return {
                "erro": str(e),
                "sucesso": False
            }

    def classificar_lote(
        self,
        df: pd.DataFrame,
        origem_col: str = "origem",
        text_col: str = "texto",
        tipos: List[str] = ["departamento", "categoria"],
        top_k: int = DEFAULT_TOP_K,
        preprocessar: bool = True
    ) -> pd.DataFrame:
        """
        Classifica um lote de chamados.

        Args:
            df: DataFrame com os chamados
            origem_col: Coluna com a origem
            text_col: Coluna com o texto
            tipos: Tipos de classificação a fazer
            top_k: Número de predições top
            preprocessar: Se deve preprocessar

        Returns:
            DataFrame com as classificações adicionadas
        """
        df_resultado = df.copy()

        # Adiciona colunas de resultado
        for tipo in tipos:
            df_resultado[f"{tipo}_pred"] = None
            df_resultado[f"{tipo}_conf"] = None

        # Processa cada linha
        for idx, row in df.iterrows():
            origem = row.get(origem_col, "default")
            texto = row.get(text_col, "")

            # Classifica para cada tipo
            for tipo in tipos:
                resultado = self.prever_unico(
                    texto=texto,
                    origem=origem,
                    tipo=tipo,
                    top_k=top_k,
                    preprocessar=preprocessar
                )

                if resultado.get("sucesso"):
                    df_resultado.at[idx, f"{tipo}_pred"] = resultado["predicao"]
                    df_resultado.at[idx, f"{tipo}_conf"] = resultado["confianca"]

        return df_resultado

    def listar_modelos_disponiveis(self) -> List[Dict[str, Any]]:
        """
        Lista todos os modelos disponíveis com suas informações.

        Returns:
            Lista de dicionários com informações dos modelos
        """
        modelos = []

        # Lista arquivos de modelo
        model_files = list(self.models_dir.glob("*_model.joblib"))

        for model_file in model_files:
            # Extrai origem e tipo
            parts = model_file.stem.replace("_model", "").rsplit("_", 1)
            if len(parts) != 2:
                continue

            origem, tipo = parts

            # Informações básicas
            info = {
                "origem": origem,
                "tipo": tipo,
                "arquivo": model_file.name,
                "tamanho_mb": model_file.stat().st_size / (1024 * 1024),
                "carregado": f"{origem}_{tipo}" in self.modelos_carregados
            }

            # Tenta ler metadados
            metadata_path = self.models_dir / METADATA_NAME_FORMAT.format(origem=origem, tipo=tipo)
            if metadata_path.exists():
                try:
                    metadata = carregar_json(metadata_path)
                    info.update({
                        "accuracy": metadata.get("metricas", {}).get("accuracy"),
                        "f1_score": metadata.get("metricas", {}).get("f1_score"),
                        "n_classes": metadata.get("n_classes"),
                        "classes": metadata.get("classes"),
                        "data_treinamento": metadata.get("timestamp_treinamento")
                    })
                except Exception as e:
                    logger.debug(f"Erro ao ler metadados de {origem}_{tipo}: {e}")

            modelos.append(info)

        return modelos

    def explicar_predicao(
        self,
        texto: str,
        origem: str,
        tipo: str,
        n_features: int = 10
    ) -> Dict[str, Any]:
        """
        Explica uma predição mostrando features importantes.

        Args:
            texto: Texto para classificar
            origem: Origem do chamado
            tipo: Tipo de classificação
            n_features: Número de features para mostrar

        Returns:
            Dicionário com predição e explicação
        """
        # Faz predição normal
        resultado = self.prever_unico(texto, origem, tipo)

        if not resultado.get("sucesso"):
            return resultado

        origem_slug = slugify(origem)
        model_key = f"{origem_slug}_{tipo}"

        try:
            # Obtém modelo
            modelo = self.modelos_carregados[model_key]
            encoder = self.encoders_carregados[model_key]

            # Extrai TF-IDF features
            tfidf = modelo.named_steps['tfidf']
            classifier = modelo.named_steps['classifier']

            # Transforma texto
            texto_tfidf = tfidf.transform([texto])

            # Identifica features não-zero
            feature_indices = texto_tfidf.nonzero()[1]
            feature_names = tfidf.get_feature_names_out()

            # Calcula importância das features
            pred_class = modelo.predict([texto])[0]
            coefs = classifier.coef_[pred_class] if len(classifier.coef_.shape) > 1 else classifier.coef_[0]

            # Features e suas contribuições
            features_contrib = []
            for idx in feature_indices:
                if idx < len(feature_names):
                    features_contrib.append({
                        "termo": feature_names[idx],
                        "tfidf": float(texto_tfidf[0, idx]),
                        "coef": float(coefs[idx]),
                        "contribuicao": float(texto_tfidf[0, idx] * coefs[idx])
                    })

            # Ordena por contribuição absoluta
            features_contrib.sort(key=lambda x: abs(x["contribuicao"]), reverse=True)

            resultado["explicacao"] = {
                "features_importantes": features_contrib[:n_features],
                "n_features_total": len(feature_indices)
            }

        except Exception as e:
            logger.warning(f"Não foi possível gerar explicação: {e}")

        return resultado

# ================================
# FUNÇÃO PRINCIPAL SOLICITADA
# ================================

def classificador_mensagem(
    origem: str,
    assunto: str,
    mensagem: str,
    classificar_departamento: bool = True,
    classificar_categoria: bool = True,
    top_k: int = 3,
    auto_carregar: bool = True
) -> Dict:
    """
    Classifica uma mensagem retornando departamento e/ou categoria.

    Esta é a função principal solicitada para classificação de chamados.

    Args:
        origem: Origem do chamado (ex: "Email", "Painel", "API")
        assunto: Assunto/título da mensagem
        mensagem: Conteúdo da mensagem
        classificar_departamento: Se deve classificar o departamento
        classificar_categoria: Se deve classificar a categoria
        top_k: Número de predições mais prováveis a retornar
        auto_carregar: Se deve carregar automaticamente os modelos

    Returns:
        Dicionário com estrutura:
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
            "categoria": {
                "predicao": str,
                "confianca": float,
                "topk_labels": List[str],
                "topk_probas": List[float]
            },
            "sucesso": bool,
            "erros": List[str]
        }

    Examples:
        >>> resultado = classificador_mensagem(
        ...     origem="Email",
        ...     assunto="Problema de login",
        ...     mensagem="Não consigo acessar o sistema"
        ... )
        >>> print(f"Departamento: {resultado['departamento']['predicao']}")
        TI
    """
    # Inicializa resultado
    resultado = {
        "origem": origem,
        "assunto": assunto,
        "mensagem": mensagem,
        "departamento": None,
        "categoria": None,
        "sucesso": True,
        "erros": []
    }

    # Valida entrada
    if not origem:
        resultado["erros"].append("Origem não fornecida")
        resultado["sucesso"] = False

    if not assunto and not mensagem:
        resultado["erros"].append("Assunto e mensagem vazios")
        resultado["sucesso"] = False

    if not resultado["sucesso"]:
        return resultado

    # Combina assunto e mensagem
    texto_completo = combinar_textos(assunto, mensagem)

    # Valida texto
    is_valid, erro = validar_texto(texto_completo)
    if not is_valid:
        resultado["erros"].append(f"Texto inválido: {erro}")
        resultado["sucesso"] = False
        return resultado

    # Cria instância do classificador
    classificador = ClassificadorChamados()

    # Auto-carrega modelos se necessário
    if auto_carregar:
        status = classificador.carregar_todos_modelos()
        if not any(status.values()):
            resultado["erros"].append("Nenhum modelo disponível")
            resultado["sucesso"] = False
            return resultado

    # Classifica departamento
    if classificar_departamento:
        try:
            pred_dept = classificador.prever_unico(
                texto=texto_completo,
                origem=origem,
                tipo="departamento",
                top_k=top_k,
                preprocessar=True
            )

            if pred_dept.get("sucesso"):
                resultado["departamento"] = {
                    "predicao": pred_dept["predicao"],
                    "confianca": pred_dept["confianca"],
                    "topk_labels": pred_dept["topk_labels"],
                    "topk_probas": pred_dept["topk_probas"]
                }
            else:
                # Tenta modelo genérico se específico não existe
                pred_dept_gen = classificador.prever_unico(
                    texto=texto_completo,
                    origem="geral",
                    tipo="departamento",
                    top_k=top_k,
                    preprocessar=True
                )

                if pred_dept_gen.get("sucesso"):
                    resultado["departamento"] = {
                        "predicao": pred_dept_gen["predicao"],
                        "confianca": pred_dept_gen["confianca"],
                        "topk_labels": pred_dept_gen["topk_labels"],
                        "topk_probas": pred_dept_gen["topk_probas"]
                    }
                else:
                    resultado["erros"].append(f"Modelo de departamento não disponível para origem '{origem}'")

        except Exception as e:
            resultado["erros"].append(f"Erro ao classificar departamento: {str(e)}")

    # Classifica categoria
    if classificar_categoria:
        try:
            pred_cat = classificador.prever_unico(
                texto=texto_completo,
                origem=origem,
                tipo="categoria",
                top_k=top_k,
                preprocessar=True
            )

            if pred_cat.get("sucesso"):
                resultado["categoria"] = {
                    "predicao": pred_cat["predicao"],
                    "confianca": pred_cat["confianca"],
                    "topk_labels": pred_cat["topk_labels"],
                    "topk_probas": pred_cat["topk_probas"]
                }
            else:
                # Tenta modelo genérico
                pred_cat_gen = classificador.prever_unico(
                    texto=texto_completo,
                    origem="geral",
                    tipo="categoria",
                    top_k=top_k,
                    preprocessar=True
                )

                if pred_cat_gen.get("sucesso"):
                    resultado["categoria"] = {
                        "predicao": pred_cat_gen["predicao"],
                        "confianca": pred_cat_gen["confianca"],
                        "topk_labels": pred_cat_gen["topk_labels"],
                        "topk_probas": pred_cat_gen["topk_probas"]
                    }
                else:
                    resultado["erros"].append(f"Modelo de categoria não disponível para origem '{origem}'")

        except Exception as e:
            resultado["erros"].append(f"Erro ao classificar categoria: {str(e)}")

    # Atualiza status de sucesso
    if resultado["erros"]:
        resultado["sucesso"] = False

    # Adiciona timestamp
    resultado["timestamp"] = datetime.now().isoformat()

    return resultado

# ================================
# FUNÇÕES AUXILIARES
# ================================

def classificar_arquivo(
    filepath: Path,
    origem_col: str = "origem",
    assunto_col: str = "assunto",
    mensagem_col: str = "mensagem",
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Classifica chamados de um arquivo CSV/Excel.

    Args:
        filepath: Caminho do arquivo
        origem_col: Nome da coluna de origem
        assunto_col: Nome da coluna de assunto
        mensagem_col: Nome da coluna de mensagem
        output_path: Caminho para salvar resultado

    Returns:
        DataFrame com classificações
    """
    # Lê arquivo
    if filepath.suffix.lower() == '.csv':
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    # Adiciona colunas de resultado
    df["departamento_pred"] = None
    df["departamento_conf"] = None
    df["categoria_pred"] = None
    df["categoria_conf"] = None

    # Processa cada linha
    for idx, row in df.iterrows():
        resultado = classificador_mensagem(
            origem=row.get(origem_col, ""),
            assunto=row.get(assunto_col, ""),
            mensagem=row.get(mensagem_col, ""),
            classificar_departamento=True,
            classificar_categoria=True
        )

        if resultado["sucesso"]:
            if resultado["departamento"]:
                df.at[idx, "departamento_pred"] = resultado["departamento"]["predicao"]
                df.at[idx, "departamento_conf"] = resultado["departamento"]["confianca"]

            if resultado["categoria"]:
                df.at[idx, "categoria_pred"] = resultado["categoria"]["predicao"]
                df.at[idx, "categoria_conf"] = resultado["categoria"]["confianca"]

    # Salva resultado se solicitado
    if output_path:
        if output_path.suffix.lower() == '.csv':
            df.to_csv(output_path, index=False)
        else:
            df.to_excel(output_path, index=False)

    return df

# ================================
# EXPORTS
# ================================

__all__ = [
    'ClassificadorChamados',
    'classificador_mensagem',
    'classificar_arquivo'
]