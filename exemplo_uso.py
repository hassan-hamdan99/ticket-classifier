#!/usr/bin/env python
"""
Exemplos práticos de uso do sistema de classificação de chamados.
Este arquivo demonstra diferentes formas de usar a biblioteca.
"""

import pandas as pd
from pathlib import Path
import json
from datetime import datetime

# Importa componentes do sistema
from classifier import ClassificadorChamados, classificador_mensagem
from utils import preprocess_for_ml, calcular_estatisticas_texto
from data_loader import carregar_chamados, listar_arquivos_dados
from model_trainer import treinar_por_origem


def print_section(title):
    """Imprime seção formatada."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60 + "\n")


# ==============================================================================
# EXEMPLO 1: USO BÁSICO DA FUNÇÃO PRINCIPAL
# ==============================================================================

def exemplo_uso_basico():
    """Demonstra o uso básico da função classificador_mensagem."""
    print_section("EXEMPLO 1: USO BÁSICO")

    # Caso simples de classificação
    resultado = classificador_mensagem(
        origem="Email",
        assunto="Problema de acesso ao sistema",
        mensagem="Não consigo fazer login no sistema desde ontem. Minha senha foi resetada mas continua sem funcionar."
    )

    print("📥 Entrada:")
    print(f"  Origem: {resultado['origem']}")
    print(f"  Assunto: {resultado['assunto']}")
    print(f"  Mensagem: {resultado['mensagem'][:100]}...")

    if resultado['sucesso']:
        print("\n✅ Classificação bem-sucedida!")

        if resultado['departamento']:
            dept = resultado['departamento']
            print(f"\n🏢 Departamento:")
            print(f"  Predição: {dept['predicao']}")
            print(f"  Confiança: {dept['confianca']:.1%}")
            print(f"  Top 3: {', '.join(dept['topk_labels'])}")

        if resultado['categoria']:
            cat = resultado['categoria']
            print(f"\n🏷️ Categoria:")
            print(f"  Predição: {cat['predicao']}")
            print(f"  Confiança: {cat['confianca']:.1%}")
            print(f"  Top 3: {', '.join(cat['topk_labels'])}")
    else:
        print(f"\n❌ Erros: {resultado['erros']}")

    return resultado


# ==============================================================================
# EXEMPLO 2: CLASSIFICAÇÃO EM LOTE
# ==============================================================================

def exemplo_classificacao_lote():
    """Demonstra classificação de múltiplos chamados."""
    print_section("EXEMPLO 2: CLASSIFICAÇÃO EM LOTE")

    # Dados de exemplo
    chamados = [
        {
            "origem": "Email",
            "assunto": "Férias",
            "mensagem": "Gostaria de solicitar minhas férias para dezembro"
        },
        {
            "origem": "Painel",
            "assunto": "Computador lento",
            "mensagem": "Meu computador está muito lento, preciso de ajuda"
        },
        {
            "origem": "API",
            "assunto": "Erro na integração",
            "mensagem": "A API está retornando erro 500 ao tentar autenticar"
        },
        {
            "origem": "Email",
            "assunto": "Reembolso",
            "mensagem": "Solicito reembolso das despesas da viagem a SP"
        }
    ]

    print(f"Processando {len(chamados)} chamados...\n")

    resultados = []
    for i, chamado in enumerate(chamados, 1):
        print(f"Chamado {i}:")
        print(f"  Assunto: {chamado['assunto']}")

        resultado = classificador_mensagem(**chamado)

        if resultado['sucesso']:
            dept = resultado['departamento']['predicao'] if resultado['departamento'] else 'N/A'
            cat = resultado['categoria']['predicao'] if resultado['categoria'] else 'N/A'

            print(f"  → Departamento: {dept}")
            print(f"  → Categoria: {cat}")
        else:
            print(f"  → Erro: {resultado['erros']}")

        resultados.append(resultado)
        print()

    return resultados


# ==============================================================================
# EXEMPLO 3: USO DA CLASSE ClassificadorChamados
# ==============================================================================

def exemplo_uso_classe():
    """Demonstra uso direto da classe ClassificadorChamados."""
    print_section("EXEMPLO 3: USO DA CLASSE ClassificadorChamados")

    # Cria instância do classificador
    clf = ClassificadorChamados()

    # Lista modelos disponíveis
    print("📦 Modelos disponíveis:")
    modelos = clf.listar_modelos_disponiveis()

    if not modelos:
        print("  ⚠️ Nenhum modelo encontrado. Execute 'python main.py' primeiro.")
        return

    for modelo in modelos[:3]:  # Mostra apenas 3 primeiros
        print(f"  • {modelo['origem']}_{modelo['tipo']}: "
              f"{'Carregado' if modelo['carregado'] else 'Disponível'}")

    # Carrega todos os modelos
    print("\n⏳ Carregando modelos...")
    status = clf.carregar_todos_modelos()
    print(f"  Carregados: {sum(status.values())}/{len(status)}")

    # Faz predição única
    texto = "Preciso resetar minha senha do email corporativo"
    print(f"\n🔍 Classificando: '{texto}'")

    resultado = clf.prever_unico(
        texto=texto,
        origem="Email",
        tipo="departamento",
        top_k=3
    )

    if resultado['sucesso']:
        print(f"  Predição: {resultado['predicao']} ({resultado['confianca']:.1%})")
        print(f"  Alternativas: {', '.join(resultado['topk_labels'][1:])}")
    else:
        print(f"  Erro: {resultado.get('erro')}")

    return clf


# ==============================================================================
# EXEMPLO 4: PREPROCESSAMENTO DE TEXTO
# ==============================================================================

def exemplo_preprocessamento():
    """Demonstra o preprocessamento de texto."""
    print_section("EXEMPLO 4: PREPROCESSAMENTO DE TEXTO")

    textos_exemplo = [
        "João Silva enviou email para maria@empresa.com sobre o CPF 123.456.789-00",
        "Acesse https://www.exemplo.com para mais informações. Tel: (11) 98765-4321",
        "O    texto   tem   muitos    espaços    extras    aqui",
        "TEXTO EM MAIÚSCULAS E com acentuação!"
    ]

    print("Demonstração de preprocessamento:\n")

    for i, texto_original in enumerate(textos_exemplo, 1):
        print(f"Exemplo {i}:")
        print(f"  Original: {texto_original}")

        # Aplica preprocessamento
        texto_processado = preprocess_for_ml(texto_original)
        print(f"  Processado: {texto_processado}")

        # Calcula estatísticas
        stats = calcular_estatisticas_texto(texto_original)
        print(f"  Estatísticas: {stats['n_palavras']} palavras, "
              f"PII: {'Sim' if stats['tem_pii'] else 'Não'}")
        print()

    return textos_exemplo


# ==============================================================================
# EXEMPLO 5: PROCESSAMENTO DE ARQUIVO CSV
# ==============================================================================

def exemplo_processar_csv():
    """Demonstra processamento de arquivo CSV."""
    print_section("EXEMPLO 5: PROCESSAMENTO DE ARQUIVO CSV")

    # Cria DataFrame de exemplo
    df_exemplo = pd.DataFrame({
        'origem': ['Email', 'Painel', 'Email', 'API', 'Painel'],
        'assunto': [
            'Problema de login',
            'Solicitar férias',
            'Computador travando',
            'Erro na API',
            'Reembolso viagem'
        ],
        'mensagem': [
            'Não consigo acessar o sistema',
            'Preciso tirar férias em dezembro',
            'PC está muito lento e travando',
            'API retorna erro 500',
            'Solicito reembolso da viagem'
        ]
    })

    print("📊 DataFrame de exemplo:")
    print(df_exemplo.to_string())

    # Salva temporariamente
    temp_file = Path("temp_chamados.csv")
    df_exemplo.to_csv(temp_file, index=False)
    print(f"\n💾 Arquivo temporário salvo: {temp_file}")

    # Processa cada linha
    print("\n🔍 Classificando chamados...")
    for idx, row in df_exemplo.iterrows():
        resultado = classificador_mensagem(
            origem=row['origem'],
            assunto=row['assunto'],
            mensagem=row['mensagem']
        )

        if resultado['sucesso'] and resultado['departamento']:
            df_exemplo.at[idx, 'departamento_pred'] = resultado['departamento']['predicao']
            df_exemplo.at[idx, 'departamento_conf'] = resultado['departamento']['confianca']

    print("\n✅ Resultado com classificações:")
    if 'departamento_pred' in df_exemplo.columns:
        print(df_exemplo[['assunto', 'departamento_pred', 'departamento_conf']].to_string())

    # Remove arquivo temporário
    if temp_file.exists():
        temp_file.unlink()
        print(f"\n🗑️ Arquivo temporário removido")

    return df_exemplo


# ==============================================================================
# EXEMPLO 6: EXPLICABILIDADE DO MODELO
# ==============================================================================

def exemplo_explicabilidade():
    """Demonstra como obter explicações das predições."""
    print_section("EXEMPLO 6: EXPLICABILIDADE DO MODELO")

    clf = ClassificadorChamados()
    clf.carregar_todos_modelos()

    texto = "Preciso de ajuda para configurar o VPN no meu notebook novo"

    print(f"📝 Texto: '{texto}'")
    print("\n🔍 Analisando features importantes...")

    resultado = clf.explicar_predicao(
        texto=texto,
        origem="Email",
        tipo="departamento",
        n_features=5
    )

    if resultado.get('sucesso'):
        print(f"\n✅ Predição: {resultado['predicao']} ({resultado['confianca']:.1%})")

        if 'explicacao' in resultado:
            print("\n📊 Features mais importantes:")
            features = resultado['explicacao']['features_importantes']

            for i, feat in enumerate(features[:5], 1):
                print(f"  {i}. '{feat['termo']}' → contribuição: {feat['contribuicao']:.3f}")
    else:
        print(f"❌ Erro: {resultado.get('erro')}")

    return resultado


# ==============================================================================
# EXEMPLO 7: VALIDAÇÃO E ESTATÍSTICAS
# ==============================================================================

def exemplo_validacao():
    """Demonstra validação e estatísticas de dados."""
    print_section("EXEMPLO 7: VALIDAÇÃO E ESTATÍSTICAS")

    # Lista arquivos de dados disponíveis
    print("📂 Arquivos de dados disponíveis:")
    arquivos = listar_arquivos_dados()

    if arquivos:
        for arq in arquivos:
            print(f"  • {arq['nome']} ({arq['tamanho_mb']:.2f} MB)")
    else:
        print("  ⚠️ Nenhum arquivo encontrado em data/")

    # Exemplo de validação de texto
    print("\n✔️ Validação de textos:")

    textos_teste = [
        "",  # Vazio
        "OK",  # Muito curto
        "Este é um texto válido para classificação",  # Válido
        "a" * 10000  # Muito longo
    ]

    from utils import validar_texto

    for texto in textos_teste:
        is_valid, erro = validar_texto(texto)
        preview = texto[:30] + "..." if len(texto) > 30 else texto
        status = "✅" if is_valid else "❌"
        print(f"  {status} '{preview}' → {erro if erro else 'Válido'}")

    return arquivos


# ==============================================================================
# EXEMPLO 8: INTEGRAÇÃO CUSTOMIZADA
# ==============================================================================

def exemplo_integracao_customizada():
    """Demonstra integração customizada com o sistema."""
    print_section("EXEMPLO 8: INTEGRAÇÃO CUSTOMIZADA")

    class MeuSistemaChamados:
        """Classe exemplo de integração."""

        def __init__(self):
            self.historico = []

        def processar_chamado(self, titulo, descricao, usuario):
            """Processa um chamado no sistema."""
            print(f"🎫 Novo chamado de {usuario}")

            # Classifica usando nosso sistema
            resultado = classificador_mensagem(
                origem="Sistema Interno",
                assunto=titulo,
                mensagem=descricao
            )

            if resultado['sucesso']:
                dept = resultado['departamento']['predicao'] if resultado['departamento'] else 'Geral'
                cat = resultado['categoria']['predicao'] if resultado['categoria'] else 'Outros'

                chamado = {
                    'id': len(self.historico) + 1,
                    'usuario': usuario,
                    'titulo': titulo,
                    'departamento': dept,
                    'categoria': cat,
                    'timestamp': datetime.now().isoformat()
                }

                self.historico.append(chamado)

                print(f"  ✅ Chamado #{chamado['id']} criado")
                print(f"     → Departamento: {dept}")
                print(f"     → Categoria: {cat}")

                return chamado
            else:
                print(f"  ❌ Erro ao classificar: {resultado['erros']}")
                return None

    # Usa o sistema customizado
    sistema = MeuSistemaChamados()

    chamados_exemplo = [
        ("Erro de login", "Não consigo acessar minha conta", "João Silva"),
        ("Férias dezembro", "Solicito férias de 15 a 30 de dezembro", "Maria Santos"),
        ("PC lento", "Computador está muito lento para trabalhar", "Pedro Costa")
    ]

    print("Processando chamados no sistema customizado:\n")
    for titulo, desc, usuario in chamados_exemplo:
        sistema.processar_chamado(titulo, desc, usuario)
        print()

    print(f"Total de chamados processados: {len(sistema.historico)}")

    return sistema


# ==============================================================================
# FUNÇÃO PRINCIPAL
# ==============================================================================

def main():
    """Executa todos os exemplos."""
    print("\n" + "=" * 60)
    print(" EXEMPLOS DE USO - SISTEMA DE CLASSIFICAÇÃO DE CHAMADOS")
    print("=" * 60)

    exemplos = [
        ("Uso Básico", exemplo_uso_basico),
        ("Classificação em Lote", exemplo_classificacao_lote),
        ("Uso da Classe ClassificadorChamados", exemplo_uso_classe),
        ("Preprocessamento de Texto", exemplo_preprocessamento),
        ("Processamento de CSV", exemplo_processar_csv),
        ("Explicabilidade do Modelo", exemplo_explicabilidade),
        ("Validação e Estatísticas", exemplo_validacao),
        ("Integração Customizada", exemplo_integracao_customizada)
    ]

    print("\nEscolha um exemplo para executar:")
    for i, (nome, _) in enumerate(exemplos, 1):
        print(f"  {i}. {nome}")
    print(f"  0. Executar todos")
    print(f"  Q. Sair")

    while True:
        escolha = input("\nEscolha (0-8 ou Q): ").strip().upper()

        if escolha == 'Q':
            print("Encerrando...")
            break

        try:
            num = int(escolha)
            if num == 0:
                # Executa todos
                for nome, func in exemplos:
                    try:
                        func()
                        input(f"\n[Pressione Enter para continuar...]")
                    except Exception as e:
                        print(f"❌ Erro no exemplo '{nome}': {e}")
            elif 1 <= num <= len(exemplos):
                # Executa exemplo específico
                nome, func = exemplos[num - 1]
                try:
                    func()
                except Exception as e:
                    print(f"❌ Erro: {e}")
                    print("\nDica: Certifique-se de ter executado 'python main.py' primeiro para treinar os modelos.")
            else:
                print("Opção inválida!")

        except ValueError:
            print("Opção inválida! Digite um número ou Q.")

        except KeyboardInterrupt:
            print("\n\nInterrompido pelo usuário.")
            break


if __name__ == "__main__":
    main()