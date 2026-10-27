"""
Testes para o módulo utils.py
"""

import pytest
from utils import (
    slugify, limpar_texto, preprocess_for_ml,
    mascarar_emails, mascarar_cpf, mascarar_telefones,
    validar_texto, combinar_textos, topk_predictions,
    calcular_estatisticas_texto, truncar_texto
)


class TestSlugify:
    """Testes para a função slugify."""

    def test_slugify_basico(self):
        assert slugify("São Paulo - SP") == "sao_paulo_sp"

    def test_slugify_com_acentos(self):
        assert slugify("Atenção: Urgência!") == "atencao_urgencia"

    def test_slugify_com_numeros(self):
        assert slugify("Teste 123 ABC") == "teste_123_abc"

    def test_slugify_vazio(self):
        assert slugify("") == ""

    def test_slugify_custom_separator(self):
        assert slugify("Hello World", separator="-") == "hello-world"


class TestLimparTexto:
    """Testes para a função limpar_texto."""

    def test_limpar_espacos_multiplos(self):
        texto = "Texto   com    muitos     espaços"
        assert limpar_texto(texto) == "Texto com muitos espaços"

    def test_remover_urls(self):
        texto = "Acesse https://www.site.com para mais info"
        assert limpar_texto(texto, remover_urls=True) == "Acesse para mais info"

    def test_manter_urls(self):
        texto = "Acesse https://www.site.com para mais info"
        assert "https://www.site.com" in limpar_texto(texto, remover_urls=False)

    def test_texto_vazio(self):
        assert limpar_texto("") == ""

    def test_trim_espacos(self):
        assert limpar_texto("  texto  ") == "texto"


class TestMascaramento:
    """Testes para funções de mascaramento de PIIs."""

    def test_mascarar_email(self):
        texto = "Entre em contato: joao@empresa.com"
        resultado = mascarar_emails(texto)
        assert "[EMAIL]" in resultado
        assert "joao@empresa.com" not in resultado

    def test_mascarar_cpf(self):
        texto = "CPF: 123.456.789-00"
        resultado = mascarar_cpf(texto)
        assert "[CPF]" in resultado
        assert "123.456.789-00" not in resultado

    def test_mascarar_telefone(self):
        texto = "Ligue: (11) 98765-4321"
        resultado = mascarar_telefones(texto)
        assert "[TELEFONE]" in resultado
        assert "98765-4321" not in resultado

    def test_mascarar_multiplos(self):
        texto = "João (11) 1234-5678 email: joao@test.com CPF: 111.222.333-44"
        resultado = preprocess_for_ml(texto, mascarar_emails=True, mascarar_ids=True)
        assert "[EMAIL]" in resultado
        assert "[TELEFONE]" in resultado
        assert "[CPF]" in resultado


class TestValidacao:
    """Testes para funções de validação."""

    def test_validar_texto_valido(self):
        is_valid, erro = validar_texto("Este é um texto válido para processamento")
        assert is_valid
        assert erro is None

    def test_validar_texto_vazio(self):
        is_valid, erro = validar_texto("")
        assert not is_valid
        assert "vazio" in erro.lower()

    def test_validar_texto_curto(self):
        is_valid, erro = validar_texto("ABC")
        assert not is_valid
        assert "curto" in erro.lower()

    def test_validar_texto_muito_longo(self):
        texto_longo = "a" * 10000
        is_valid, erro = validar_texto(texto_longo)
        assert not is_valid
        assert "longo" in erro.lower()


class TestCombinarTextos:
    """Testes para a função combinar_textos."""

    def test_combinar_assunto_mensagem(self):
        resultado = combinar_textos("Assunto", "Mensagem")
        assert "Assunto" in resultado
        assert "Mensagem" in resultado

    def test_combinar_apenas_assunto(self):
        resultado = combinar_textos("Assunto", "")
        assert resultado == "Assunto"

    def test_combinar_apenas_mensagem(self):
        resultado = combinar_textos("", "Mensagem")
        assert resultado == "Mensagem"

    def test_combinar_ambos_vazios(self):
        resultado = combinar_textos("", "")
        assert resultado == ""

    def test_combinar_com_separador_custom(self):
        resultado = combinar_textos("A", "B", separador=" | ")
        assert resultado == "A | B"


class TestTopKPredictions:
    """Testes para a função topk_predictions."""

    def test_topk_basico(self):
        probas = [0.1, 0.3, 0.6]
        labels = ["A", "B", "C"]
        top_labels, top_probas = topk_predictions(probas, labels, k=2)
        assert top_labels == ["C", "B"]
        assert top_probas == [0.6, 0.3]

    def test_topk_maior_que_lista(self):
        probas = [0.5, 0.5]
        labels = ["A", "B"]
        top_labels, top_probas = topk_predictions(probas, labels, k=5)
        assert len(top_labels) == 2

    def test_topk_lista_vazia(self):
        top_labels, top_probas = topk_predictions([], [], k=3)
        assert top_labels == []
        assert top_probas == []


class TestEstatisticas:
    """Testes para cálculo de estatísticas."""

    def test_calcular_estatisticas_basico(self):
        texto = "Este é um texto de teste."
        stats = calcular_estatisticas_texto(texto)
        assert stats["n_palavras"] == 6
        assert stats["n_caracteres"] == len(texto)

    def test_calcular_estatisticas_com_pii(self):
        texto = "Email: teste@example.com"
        stats = calcular_estatisticas_texto(texto)
        assert stats["tem_pii"]
        assert stats["n_emails"] == 1

    def test_calcular_estatisticas_vazio(self):
        stats = calcular_estatisticas_texto("")
        assert stats["n_caracteres"] == 0
        assert stats["n_palavras"] == 0
        assert not stats["tem_pii"]


class TestTruncarTexto:
    """Testes para truncar_texto."""

    def test_truncar_texto_longo(self):
        texto = "a" * 200
        resultado = truncar_texto(texto, max_length=100)
        assert len(resultado) == 100
        assert resultado.endswith("...")

    def test_nao_truncar_texto_curto(self):
        texto = "Texto curto"
        resultado = truncar_texto(texto, max_length=100)
        assert resultado == texto

    def test_truncar_com_suffix_custom(self):
        texto = "Texto longo para truncar"
        resultado = truncar_texto(texto, max_length=10, suffix="[...]")
        assert resultado.endswith("[...]")
        assert len(resultado) == 10