#!/usr/bin/env python
"""
Script principal para treinamento de modelos de classificação.

Uso:
    python main.py                      # Treina com dados padrão
    python main.py --data-dir ./dados   # Especifica diretório de dados
    python main.py --list-files         # Lista arquivos disponíveis
    python main.py --no-train           # Apenas analisa dados sem treinar
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
import json

from config import DATA_DIR, MODELS_DIR
from data_loader import (
    carregar_chamados,
    listar_arquivos_dados,
    analisar_distribuicao_classes,
    validar_dados_entrada
)
from model_trainer import treinar_todos_modelos
from utils import verificar_dependencias

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Exibe banner do sistema."""
    print("\n" + "=" * 60)
    print(" SISTEMA DE CLASSIFICAÇÃO DE CHAMADOS - TREINAMENTO")
    print("=" * 60 + "\n")


def verificar_ambiente():
    """Verifica se o ambiente está configurado corretamente."""
    print("Verificando ambiente...")

    # Verifica dependências
    deps = verificar_dependencias()
    print(f"  - spaCy disponível: {deps['spacy']}")
    if deps['spacy']:
        print(f"  - Modelo spaCy carregado: {deps['spacy_model']}")

    # Verifica diretórios
    print(f"  - Diretório de dados: {DATA_DIR}")
    print(f"  - Diretório de modelos: {MODELS_DIR}")

    if not DATA_DIR.exists():
        print(f"    ⚠️  Criando diretório de dados...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not MODELS_DIR.exists():
        print(f"    ⚠️  Criando diretório de modelos...")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("")


def listar_dados(data_dir: Path):
    """Lista arquivos de dados disponíveis."""
    print("Arquivos de dados disponíveis:")
    print("-" * 40)

    arquivos = listar_arquivos_dados(data_dir)

    if not arquivos:
        print("  ❌ Nenhum arquivo de dados encontrado!")
        print(f"     Adicione arquivos Excel ou CSV em: {data_dir}")
        return False

    for arquivo in arquivos:
        print(f"\n  📄 {arquivo['nome']}")
        print(f"     - Formato: {arquivo['formato']}")
        print(f"     - Tamanho: {arquivo['tamanho_mb']:.2f} MB")

        if 'n_linhas' in arquivo:
            print(f"     - Linhas: {arquivo['n_linhas']:,}")
            print(f"     - Colunas: {arquivo['n_colunas']}")

            if arquivo.get('tem_origem'):
                print(f"     - ✅ Coluna 'origem' presente")
                if 'origens' in arquivo:
                    print(f"        Origens: {', '.join(arquivo['origens'][:5])}")

            if arquivo.get('tem_departamento'):
                print(f"     - ✅ Coluna 'departamento' presente")

            if arquivo.get('tem_categoria'):
                print(f"     - ✅ Coluna 'categoria' presente")

        if 'erro_leitura' in arquivo:
            print(f"     - ❌ Erro ao ler: {arquivo['erro_leitura']}")

    print("")
    return len(arquivos) > 0


def analisar_dados(df):
    """Analisa e exibe estatísticas dos dados."""
    print("\nAnálise dos dados carregados:")
    print("-" * 40)
    print(f"  Total de chamados: {len(df):,}")

    # Analisa distribuição
    distribuicoes = analisar_distribuicao_classes(df)

    for coluna, dist in distribuicoes.items():
        print(f"\n  Distribuição de '{coluna}':")
        print(f"    - Classes únicas: {len(dist)}")
        print(f"    - Top 5 classes:")
        for classe, count in dist.head(5).items():
            print(f"      • {classe}: {count:,} ({count/len(df)*100:.1f}%)")

    # Valida dados
    print("\nValidação dos dados:")
    is_valid, avisos = validar_dados_entrada(df)

    if is_valid:
        print("  ✅ Dados válidos para treinamento")
    else:
        print("  ❌ Problemas encontrados nos dados:")

    for aviso in avisos:
        print(f"    ⚠️  {aviso}")

    print("")
    return is_valid


def treinar_modelos(df):
    """Treina todos os modelos."""
    print("\nIniciando treinamento de modelos...")
    print("-" * 40)

    # Identifica origens disponíveis
    if 'origem' in df.columns:
        origens = df['origem'].value_counts()
        print(f"Origens encontradas:")
        for origem, count in origens.items():
            print(f"  • {origem}: {count:,} chamados")
    else:
        print("  ⚠️  Coluna 'origem' não encontrada - usando modo genérico")
        df['origem'] = 'geral'
        origens = {'geral': len(df)}

    print("")

    # Treina modelos
    start_time = datetime.now()

    try:
        resultados = treinar_todos_modelos(
            df,
            origens=list(origens.keys()),
            tipos=["departamento", "categoria"],
            text_col="texto_processado",
            save_models=True,
            output_dir=MODELS_DIR
        )

        tempo_total = (datetime.now() - start_time).total_seconds()

        # Exibe resumo
        print("\n" + "=" * 60)
        print(" RESUMO DO TREINAMENTO")
        print("=" * 60)

        modelos_sucesso = 0
        modelos_falha = 0

        for origem, res_origem in resultados.items():
            print(f"\n  Origem: {origem}")

            for tipo, res_tipo in res_origem.items():
                if "erro" in res_tipo:
                    print(f"    ❌ {tipo}: {res_tipo['erro']}")
                    modelos_falha += 1
                else:
                    metricas = res_tipo.get("metricas", {})
                    print(f"    ✅ {tipo}:")
                    print(f"       - F1-Score: {metricas.get('f1_score', 0):.3f}")
                    print(f"       - Accuracy: {metricas.get('accuracy', 0):.3f}")
                    print(f"       - Classes: {res_tipo.get('n_classes', 0)}")
                    modelos_sucesso += 1

        print(f"\n  Tempo total: {tempo_total:.1f} segundos")
        print(f"  Modelos treinados: {modelos_sucesso}")
        print(f"  Modelos com erro: {modelos_falha}")
        print(f"  Modelos salvos em: {MODELS_DIR}")

        return modelos_sucesso > 0

    except Exception as e:
        logger.error(f"Erro durante treinamento: {e}")
        return False


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Script de treinamento do sistema de classificação de chamados"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR),
        help="Diretório com os arquivos de dados"
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="Lista arquivos de dados disponíveis"
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Apenas analisa dados sem treinar"
    )
    parser.add_argument(
        "--arquivo",
        type=str,
        help="Arquivo específico para treinar"
    )
    parser.add_argument(
        "--origem",
        type=str,
        help="Filtrar por origem específica"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Modo verbose com mais detalhes"
    )

    args = parser.parse_args()

    # Configura logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Banner
    print_banner()

    # Verifica ambiente
    verificar_ambiente()

    # Define diretório de dados
    data_dir = Path(args.data_dir)

    # Lista arquivos se solicitado
    if args.list_files:
        if listar_dados(data_dir):
            print("Use --arquivo <nome> para treinar com arquivo específico")
        return 0

    # Carrega dados
    print("Carregando dados...")
    try:
        df = carregar_chamados(
            data_dir=data_dir,
            arquivo_especifico=args.arquivo,
            origem_filtro=args.origem,
            preprocessar=True,
            validar=True
        )
        print(f"  ✅ {len(df):,} chamados carregados\n")

    except FileNotFoundError as e:
        print(f"  ❌ Erro: {e}")
        print(f"\nAdicione arquivos de dados em: {data_dir}")
        print("Formatos suportados: .xlsx, .xls, .csv")
        return 1

    except ValueError as e:
        print(f"  ❌ Erro nos dados: {e}")
        return 1

    # Analisa dados
    dados_validos = analisar_dados(df)

    if not dados_validos and not args.no_train:
        print("⚠️  Dados com problemas - revise antes de treinar")
        resposta = input("\nDeseja continuar mesmo assim? (s/N): ")
        if resposta.lower() != 's':
            print("Treinamento cancelado.")
            return 0

    # Treina modelos (se não --no-train)
    if not args.no_train:
        sucesso = treinar_modelos(df)

        if sucesso:
            print("\n✅ Treinamento concluído com sucesso!")
            print("\nPróximos passos:")
            print("  1. Use 'python predict.py' para fazer predições")
            print("  2. Veja 'python exemplo_uso.py' para exemplos de uso")
            print("  3. Consulte README.md para documentação completa")
        else:
            print("\n❌ Treinamento falhou - verifique os logs")
            return 1
    else:
        print("\nModo análise - treinamento não executado")
        print("Remove --no-train para treinar os modelos")

    return 0


if __name__ == "__main__":
    sys.exit(main())