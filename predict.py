#!/usr/bin/env python
"""
Script CLI para fazer predições usando os modelos treinados.

Uso:
    # Modo interativo
    python predict.py --interactive

    # Predição direta
    python predict.py --origem Email --assunto "Problema" --mensagem "Não consigo acessar"

    # Processar arquivo
    python predict.py --file chamados.csv --output resultados.csv

    # Listar modelos disponíveis
    python predict.py --list-models
"""

import argparse
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

from classifier import ClassificadorChamados, classificador_mensagem, classificar_arquivo
from utils import truncar_texto, formatar_resultado_predicao
from config import MODELS_DIR


def print_banner():
    """Exibe banner do sistema."""
    print("\n" + "=" * 60)
    print(" SISTEMA DE CLASSIFICAÇÃO DE CHAMADOS - PREDIÇÃO")
    print("=" * 60 + "\n")


def listar_modelos():
    """Lista modelos disponíveis."""
    print("Modelos disponíveis:")
    print("-" * 40)

    clf = ClassificadorChamados()
    modelos = clf.listar_modelos_disponiveis()

    if not modelos:
        print("  ❌ Nenhum modelo encontrado!")
        print(f"     Execute 'python main.py' para treinar modelos")
        return False

    # Agrupa por origem
    por_origem = {}
    for modelo in modelos:
        origem = modelo['origem']
        if origem not in por_origem:
            por_origem[origem] = []
        por_origem[origem].append(modelo)

    for origem, mods in por_origem.items():
        print(f"\n  📦 Origem: {origem}")
        for mod in mods:
            status = "✅" if mod['carregado'] else "⭕"
            print(f"     {status} {mod['tipo']}:")
            print(f"        - Arquivo: {mod['arquivo']}")
            print(f"        - Tamanho: {mod['tamanho_mb']:.2f} MB")

            if mod.get('f1_score'):
                print(f"        - F1-Score: {mod['f1_score']:.3f}")
            if mod.get('n_classes'):
                print(f"        - Classes: {mod['n_classes']}")

    print("\n  ✅ = Carregado | ⭕ = Disponível")
    return len(modelos) > 0


def modo_interativo():
    """Executa modo interativo de predição."""
    print("Modo Interativo")
    print("-" * 40)
    print("Digite 'sair' para encerrar\n")

    clf = ClassificadorChamados()
    clf.carregar_todos_modelos()

    while True:
        try:
            # Coleta dados
            print("\n" + "-" * 30)
            origem = input("Origem (Email/Painel/API): ").strip()
            if origem.lower() == 'sair':
                break

            assunto = input("Assunto: ").strip()
            if assunto.lower() == 'sair':
                break

            mensagem = input("Mensagem: ").strip()
            if mensagem.lower() == 'sair':
                break

            # Faz predição
            print("\n🔍 Processando...")

            resultado = classificador_mensagem(
                origem=origem,
                assunto=assunto,
                mensagem=mensagem,
                classificar_departamento=True,
                classificar_categoria=True,
                top_k=3,
                auto_carregar=False  # Já carregamos acima
            )

            # Exibe resultado
            print("\n📊 Resultado:")
            print("-" * 30)

            if resultado['sucesso']:
                if resultado['departamento']:
                    dept = resultado['departamento']
                    print(f"\n🏢 Departamento: {dept['predicao']}")
                    print(f"   Confiança: {dept['confianca']:.1%}")

                    if len(dept['topk_labels']) > 1:
                        print("   Alternativas:")
                        for label, prob in zip(dept['topk_labels'][1:3], dept['topk_probas'][1:3]):
                            print(f"     • {label}: {prob:.1%}")

                if resultado['categoria']:
                    cat = resultado['categoria']
                    print(f"\n🏷️  Categoria: {cat['predicao']}")
                    print(f"   Confiança: {cat['confianca']:.1%}")

                    if len(cat['topk_labels']) > 1:
                        print("   Alternativas:")
                        for label, prob in zip(cat['topk_labels'][1:3], cat['topk_probas'][1:3]):
                            print(f"     • {label}: {prob:.1%}")
            else:
                print("❌ Erro na classificação:")
                for erro in resultado['erros']:
                    print(f"   • {erro}")

        except KeyboardInterrupt:
            print("\n\nInterrompido pelo usuário")
            break
        except Exception as e:
            print(f"\n❌ Erro: {e}")

    print("\nEncerrando modo interativo...")


def processar_arquivo(filepath: Path, output_path: Path = None):
    """Processa arquivo de chamados."""
    print(f"Processando arquivo: {filepath.name}")
    print("-" * 40)

    if not filepath.exists():
        print(f"❌ Arquivo não encontrado: {filepath}")
        return False

    try:
        # Lê arquivo
        if filepath.suffix.lower() == '.csv':
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        print(f"  ✅ {len(df):,} linhas carregadas")

        # Verifica colunas necessárias
        colunas_necessarias = ['origem', 'assunto', 'mensagem']
        colunas_faltando = [col for col in colunas_necessarias if col not in df.columns]

        if colunas_faltando:
            print(f"  ⚠️  Colunas faltando: {colunas_faltando}")
            print("     O arquivo deve conter: origem, assunto, mensagem")
            return False

        # Processa classificações
        print("\n🔍 Classificando chamados...")

        df_resultado = classificar_arquivo(
            filepath,
            origem_col='origem',
            assunto_col='assunto',
            mensagem_col='mensagem',
            output_path=output_path
        )

        # Estatísticas
        n_classificados_dept = df_resultado['departamento_pred'].notna().sum()
        n_classificados_cat = df_resultado['categoria_pred'].notna().sum()

        print(f"\n📊 Resultados:")
        print(f"  • Departamentos classificados: {n_classificados_dept}/{len(df)} ({n_classificados_dept/len(df)*100:.1f}%)")
        print(f"  • Categorias classificadas: {n_classificados_cat}/{len(df)} ({n_classificados_cat/len(df)*100:.1f}%)")

        # Confiança média
        if n_classificados_dept > 0:
            conf_media_dept = df_resultado['departamento_conf'].mean()
            print(f"  • Confiança média (depto): {conf_media_dept:.1%}")

        if n_classificados_cat > 0:
            conf_media_cat = df_resultado['categoria_conf'].mean()
            print(f"  • Confiança média (categoria): {conf_media_cat:.1%}")

        # Salva resultado
        if output_path:
            print(f"\n💾 Resultado salvo em: {output_path}")
        else:
            # Salva com nome padrão
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = filepath.parent / f"{filepath.stem}_classificado_{timestamp}.csv"
            df_resultado.to_csv(output_path, index=False)
            print(f"\n💾 Resultado salvo em: {output_path}")

        return True

    except Exception as e:
        print(f"❌ Erro ao processar arquivo: {e}")
        return False


def fazer_predicao_simples(origem: str, assunto: str, mensagem: str, formato: str = 'text'):
    """Faz uma predição simples e exibe resultado."""
    resultado = classificador_mensagem(
        origem=origem,
        assunto=assunto,
        mensagem=mensagem,
        classificar_departamento=True,
        classificar_categoria=True,
        top_k=3
    )

    if formato == 'json':
        # Saída JSON
        print(json.dumps(resultado, indent=2, ensure_ascii=False))

    else:
        # Saída texto formatado
        if resultado['sucesso']:
            print("✅ Classificação realizada:\n")

            # Exibe entrada
            print("📥 Entrada:")
            print(f"  Origem: {origem}")
            print(f"  Assunto: {truncar_texto(assunto, 50)}")
            print(f"  Mensagem: {truncar_texto(mensagem, 100)}")

            # Exibe resultados
            print("\n📊 Resultado:")

            if resultado['departamento']:
                dept = resultado['departamento']
                print(f"\n  Departamento: {dept['predicao']} ({dept['confianca']:.1%})")
                if len(dept['topk_labels']) > 1:
                    print(f"  Alternativas: {', '.join(dept['topk_labels'][1:3])}")

            if resultado['categoria']:
                cat = resultado['categoria']
                print(f"\n  Categoria: {cat['predicao']} ({cat['confianca']:.1%})")
                if len(cat['topk_labels']) > 1:
                    print(f"  Alternativas: {', '.join(cat['topk_labels'][1:3])}")
        else:
            print("❌ Erro na classificação:")
            for erro in resultado['erros']:
                print(f"  • {erro}")

    return resultado['sucesso']


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Script de predição para classificação de chamados"
    )

    # Modos de operação
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Modo interativo"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Lista modelos disponíveis"
    )

    # Predição direta
    parser.add_argument(
        "--origem",
        type=str,
        help="Origem do chamado (Email, Painel, API, etc.)"
    )
    parser.add_argument(
        "--assunto",
        type=str,
        help="Assunto do chamado"
    )
    parser.add_argument(
        "--mensagem",
        type=str,
        help="Mensagem/corpo do chamado"
    )

    # Processamento de arquivo
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Arquivo CSV/Excel para processar"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Arquivo de saída para resultados"
    )

    # Formato de saída
    parser.add_argument(
        "--format",
        choices=['text', 'json'],
        default='text',
        help="Formato de saída (text ou json)"
    )

    # Opções adicionais
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Número de predições alternativas"
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Explica a predição mostrando features importantes"
    )

    args = parser.parse_args()

    # Banner
    if args.format != 'json':
        print_banner()

    # Lista modelos
    if args.list_models:
        sucesso = listar_modelos()
        return 0 if sucesso else 1

    # Modo interativo
    if args.interactive:
        modo_interativo()
        return 0

    # Processa arquivo
    if args.file:
        filepath = Path(args.file)
        output_path = Path(args.output) if args.output else None
        sucesso = processar_arquivo(filepath, output_path)
        return 0 if sucesso else 1

    # Predição direta
    if args.origem and (args.assunto or args.mensagem):
        sucesso = fazer_predicao_simples(
            origem=args.origem,
            assunto=args.assunto or "",
            mensagem=args.mensagem or "",
            formato=args.format
        )
        return 0 if sucesso else 1

    # Se nenhum modo especificado, mostra ajuda
    if args.format != 'json':
        print("❌ Nenhum modo de operação especificado!\n")
        print("Opções disponíveis:")
        print("  --interactive        : Modo interativo")
        print("  --list-models        : Lista modelos disponíveis")
        print("  --file <arquivo>     : Processa arquivo de chamados")
        print("  --origem <> --mensagem <> : Predição direta")
        print("\nUse --help para mais informações")

    return 1


if __name__ == "__main__":
    sys.exit(main())