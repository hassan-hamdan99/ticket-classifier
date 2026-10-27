"""
Testes para o módulo classifier.py (arquivo principal).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path

from classifier import ClassificadorChamados, classificador_mensagem, classificar_arquivo


class TestClassificadorMensagem:
    """Testes para a função principal classificador_mensagem."""

    def test_classificacao_sucesso(self):
        """Testa classificação bem-sucedida."""
        with patch('classifier.ClassificadorChamados') as MockClf:
            # Mock do classificador
            mock_instance = MockClf.return_value
            mock_instance.carregar_todos_modelos.return_value = {'email_departamento': True}
            mock_instance.prever_unico.return_value = {
                'predicao': 'TI',
                'confianca': 0.95,
                'topk_labels': ['TI', 'Suporte', 'Admin'],
                'topk_probas': [0.95, 0.03, 0.02],
                'sucesso': True
            }

            resultado = classificador_mensagem(
                origem="Email",
                assunto="Problema de login",
                mensagem="Não consigo acessar o sistema"
            )

            assert resultado['sucesso']
            assert resultado['departamento'] is not None
            assert resultado['departamento']['predicao'] == 'TI'

    def test_classificacao_sem_origem(self):
        """Testa erro quando origem não é fornecida."""
        resultado = classificador_mensagem(
            origem="",
            assunto="Teste",
            mensagem="Teste"
        )

        assert not resultado['sucesso']
        assert "Origem não fornecida" in resultado['erros']

    def test_classificacao_sem_conteudo(self):
        """Testa erro quando assunto e mensagem estão vazios."""
        resultado = classificador_mensagem(
            origem="Email",
            assunto="",
            mensagem=""
        )

        assert not resultado['sucesso']
        assert "Assunto e mensagem vazios" in resultado['erros']

    def test_classificacao_apenas_departamento(self):
        """Testa classificação apenas de departamento."""
        with patch('classifier.ClassificadorChamados') as MockClf:
            mock_instance = MockClf.return_value
            mock_instance.carregar_todos_modelos.return_value = {'email_departamento': True}
            mock_instance.prever_unico.return_value = {
                'predicao': 'TI',
                'confianca': 0.95,
                'topk_labels': ['TI'],
                'topk_probas': [0.95],
                'sucesso': True
            }

            resultado = classificador_mensagem(
                origem="Email",
                assunto="Teste",
                mensagem="Teste",
                classificar_departamento=True,
                classificar_categoria=False
            )

            assert resultado['sucesso']
            assert resultado['departamento'] is not None
            assert resultado['categoria'] is None


class TestClassificadorChamados:
    """Testes para a classe ClassificadorChamados."""

    def test_inicializacao(self, tmp_path):
        """Testa inicialização do classificador."""
        clf = ClassificadorChamados(models_dir=tmp_path)
        assert clf.models_dir == tmp_path
        assert len(clf.modelos_carregados) == 0

    def test_carregar_modelo_sucesso(self, tmp_path):
        """Testa carregamento bem-sucedido de modelo."""
        # Cria arquivos mock
        model_file = tmp_path / "email_departamento_model.joblib"
        encoder_file = tmp_path / "email_departamento_label_encoder.joblib"

        # Mock dos arquivos
        with patch('joblib.load') as mock_load:
            mock_load.return_value = MagicMock()

            model_file.touch()
            encoder_file.touch()

            clf = ClassificadorChamados(models_dir=tmp_path)
            sucesso = clf.carregar_modelo("email", "departamento")

            assert sucesso
            assert "email_departamento" in clf.modelos_carregados

    def test_carregar_modelo_arquivo_nao_existe(self, tmp_path):
        """Testa falha quando arquivo de modelo não existe."""
        clf = ClassificadorChamados(models_dir=tmp_path)
        sucesso = clf.carregar_modelo("email", "departamento")

        assert not sucesso
        assert "email_departamento" not in clf.modelos_carregados

    def test_prever_unico_modelo_nao_carregado(self, tmp_path):
        """Testa predição quando modelo não está carregado."""
        clf = ClassificadorChamados(models_dir=tmp_path)
        resultado = clf.prever_unico("teste", "email", "departamento")

        assert not resultado['sucesso']
        assert 'erro' in resultado

    def test_prever_unico_texto_invalido(self, tmp_path):
        """Testa predição com texto inválido."""
        clf = ClassificadorChamados(models_dir=tmp_path)

        # Mock do modelo carregado
        clf.modelos_carregados['email_departamento'] = MagicMock()
        clf.encoders_carregados['email_departamento'] = MagicMock()

        resultado = clf.prever_unico("", "email", "departamento")

        assert not resultado['sucesso']
        assert 'Texto inválido' in resultado['erro']

    def test_listar_modelos_disponiveis(self, tmp_path):
        """Testa listagem de modelos disponíveis."""
        # Cria arquivos de modelo mock
        (tmp_path / "email_departamento_model.joblib").touch()
        (tmp_path / "painel_categoria_model.joblib").touch()

        clf = ClassificadorChamados(models_dir=tmp_path)
        modelos = clf.listar_modelos_disponiveis()

        assert len(modelos) == 2
        origens = [m['origem'] for m in modelos]
        assert 'email' in origens
        assert 'painel' in origens


class TestClassificarArquivo:
    """Testes para a função classificar_arquivo."""

    def test_classificar_arquivo_csv(self, temp_data_file):
        """Testa classificação de arquivo CSV."""
        with patch('classifier.classificador_mensagem') as mock_classificador:
            mock_classificador.return_value = {
                'sucesso': True,
                'departamento': {'predicao': 'TI', 'confianca': 0.95},
                'categoria': {'predicao': 'Acesso', 'confianca': 0.88}
            }

            df_resultado = classificar_arquivo(
                filepath=temp_data_file,
                origem_col='origem',
                assunto_col='assunto',
                mensagem_col='mensagem'
            )

            assert 'departamento_pred' in df_resultado.columns
            assert 'categoria_pred' in df_resultado.columns
            assert len(df_resultado) > 0


class TestIntegration:
    """Testes de integração."""

    def test_pipeline_completo(self, sample_data):
        """Testa pipeline completo de classificação."""
        with patch('classifier.ClassificadorChamados') as MockClf:
            # Setup do mock
            mock_instance = MockClf.return_value
            mock_instance.carregar_todos_modelos.return_value = {
                'email_departamento': True,
                'email_categoria': True
            }

            def mock_prever(texto, origem, tipo, **kwargs):
                if tipo == 'departamento':
                    return {
                        'predicao': 'TI',
                        'confianca': 0.9,
                        'topk_labels': ['TI', 'Suporte'],
                        'topk_probas': [0.9, 0.1],
                        'sucesso': True
                    }
                else:
                    return {
                        'predicao': 'Acesso',
                        'confianca': 0.85,
                        'topk_labels': ['Acesso', 'Sistema'],
                        'topk_probas': [0.85, 0.15],
                        'sucesso': True
                    }

            mock_instance.prever_unico.side_effect = mock_prever

            # Testa classificação
            resultado = classificador_mensagem(
                origem="Email",
                assunto="Login problem",
                mensagem="Cannot access the system"
            )

            assert resultado['sucesso']
            assert resultado['departamento']['predicao'] == 'TI'
            assert resultado['categoria']['predicao'] == 'Acesso'
            assert resultado['departamento']['confianca'] >= 0.8
            assert resultado['categoria']['confianca'] >= 0.8