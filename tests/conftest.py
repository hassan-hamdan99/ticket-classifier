"""
Fixtures e configurações para os testes do sistema de classificação.
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
import sys
import os

# Adiciona o diretório pai ao path para importar módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def sample_data():
    """Cria DataFrame de exemplo para testes."""
    data = {
        'origem': ['Email'] * 10 + ['Painel'] * 10,
        'assunto': [
            'Login problem', 'Password reset', 'System error', 'Access denied', 'VPN issue',
            'Vacation request', 'Salary question', 'Benefits info', 'Training request', 'Leave application',
            'Computer slow', 'Software install', 'Printer error', 'Network issue', 'Hardware problem',
            'Invoice issue', 'Payment delay', 'Budget question', 'Expense report', 'Refund request'
        ],
        'mensagem': [
            'Cannot login to system', 'Need password reset', 'System showing errors', 'Access is denied', 'VPN not working',
            'Request vacation days', 'Question about salary', 'Need benefits information', 'Want training course', 'Applying for leave',
            'Computer running slow', 'Need software installation', 'Printer not working', 'Network connection issue', 'Hardware malfunction',
            'Invoice has errors', 'Payment is delayed', 'Budget clarification needed', 'Expense report submission', 'Requesting refund'
        ],
        'departamento': ['TI'] * 5 + ['RH'] * 5 + ['TI'] * 5 + ['Financeiro'] * 5,
        'categoria': ['Acesso'] * 5 + ['Benefícios'] * 5 + ['Hardware'] * 5 + ['Pagamentos'] * 5
    }

    return pd.DataFrame(data)


@pytest.fixture
def temp_data_file(sample_data):
    """Cria arquivo CSV temporário com dados de teste."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        sample_data.to_csv(f, index=False)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def sample_text():
    """Texto de exemplo para testes."""
    return "Não consigo fazer login no sistema desde ontem. Minha senha foi resetada mas continua sem funcionar."


@pytest.fixture
def sample_texts_with_pii():
    """Textos com informações pessoais para testar preprocessamento."""
    return [
        "João Silva enviou email para maria@empresa.com",
        "O CPF 123.456.789-00 está cadastrado incorretamente",
        "Ligue para (11) 98765-4321 para mais informações",
        "Acesse https://www.exemplo.com/reset para resetar"
    ]


@pytest.fixture
def mock_model_dir(tmp_path):
    """Cria diretório temporário para modelos."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return models_dir


@pytest.fixture
def classification_result():
    """Resultado de classificação esperado."""
    return {
        "origem": "Email",
        "assunto": "Problema de login",
        "mensagem": "Não consigo acessar o sistema",
        "departamento": {
            "predicao": "TI",
            "confianca": 0.95,
            "topk_labels": ["TI", "Suporte", "Administrativo"],
            "topk_probas": [0.95, 0.03, 0.02]
        },
        "categoria": {
            "predicao": "Acesso",
            "confianca": 0.88,
            "topk_labels": ["Acesso", "Sistema", "Senha"],
            "topk_probas": [0.88, 0.08, 0.04]
        },
        "sucesso": True,
        "erros": []
    }