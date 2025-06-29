# project_simulator.py

import json
import os
import time
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

# Carrega variáveis de ambiente
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# --- Constante para o limite de RPM ---
RPM_LIMIT = 14


def _sanitize_json_output(raw_output: str) -> str:
    """Limpa a saída de texto do LLM para extrair um bloco JSON."""
    json_start_index = raw_output.find('{')
    if json_start_index == -1:
        json_start_index = raw_output.find('[')
        if json_start_index == -1:
            return ""

    json_end_index = raw_output.rfind('}')
    if json_end_index == -1:
        json_end_index = raw_output.rfind(']')
        if json_end_index == -1:
            return ""

    return raw_output[json_start_index:json_end_index + 1]


def run_simulation(scenario_path: str):
    """Orquestra a simulação completa do projeto, sprint a sprint."""
    if not API_KEY:
        raise ValueError("API_KEY não encontrada. Defina OPENAI_API_KEY ou GOOGLE_API_KEY no seu arquivo .env")

    with open(scenario_path, 'r', encoding='utf-8') as f:
        cenario = json.load(f)
    print(f"Cenário '{scenario_path}' carregado.")

    llm = LLM(
        model='gemini/gemini-2.0-flash-lite',
        api_key=API_KEY,
        system_message="Você é um assistente especialista em gerenciamento de projetos ágeis.",
        temperature=0.1
    )

    sprint_planner_agent = Agent(
        role='Planejador de Sprints Ágil',
        goal="Analisar um backlog de tarefas e um número de sprints para criar um plano de trabalho balanceado.",
        backstory="Um Scrum Master experiente que sabe como balancear o trabalho ao longo das sprints.",
        llm=llm, verbose=False
    )

    task_allocator_agent = Agent(
        role='Gerente de Projetos IA para Alocação e Estimativa',
        goal=(
            "Alocar tarefas de uma sprint, reestimar o esforço em dias para cada tarefa com base no desenvolvedor "
            "alocado e calcular o custo (dias * custo diário do dev). Retornar a alocação em formato JSON."
        ),
        backstory=(
            "Um gerente de projetos eficiente que rapidamente realoca e reestima tarefas quando a equipe muda, "
            "focando em maximizar a produtividade e prever custos."
        ),
        llm=llm, verbose=False
    )

    summary_agent = Agent(
        role='Gerente de Relatórios de Projeto',
        goal=(
            "Consolidar os resultados de todas as sprints, analisando o impacto de eventos (saídas e contratações) "
            "nos custos e prazos, e gerar um relatório final conciso em formato JSON."
        ),
        backstory="Um PMO que transforma dados de execução de projeto em relatórios executivos claros e objetivos.",
        llm=llm, verbose=False
    )

    print("\n--- Fase 1: Planejando as Sprints com IA ---")
    backlog_completo = cenario['projeto']['backlog_completo']
    total_sprints = cenario['config']['total_sprints_planejadas']

    planning_task = Task(
        description=(
            f"Dado o backlog de tarefas: {json.dumps(backlog_completo, indent=2)}, "
            f"distribua essas tarefas em {total_sprints} sprints. O resultado deve ser um JSON onde as chaves "
            f"são os IDs das sprints (de '1' a '{total_sprints}') e os valores são listas de IDs de tarefas."
        ),
        agent=sprint_planner_agent,
        expected_output="Um único bloco de código JSON com o plano de tarefas para cada sprint."
    )
    planning_crew = Crew(agents=[sprint_planner_agent], tasks=[planning_task], process=Process.sequential, max_rpm=RPM_LIMIT)
    sprint_plan_raw = planning_crew.kickoff()
    sprint_plan = json.loads(_sanitize_json_output(str(sprint_plan_raw)))
    print("Plano de Sprints gerado.")
    print("Aguardando 60 segundos para evitar limite de RPM...")
    time.sleep(60)

    print("\n--- Fase 2: Iniciando a Simulação das Sprints ---")
    equipe_atual = cenario['equipe_inicial'].copy()
    resultados_simulacao = []

    for sprint_info in cenario['plano_sprints']:
        sprint_id = sprint_info['sprint_id']
        print(f"\n>>> Processando Sprint {sprint_id}... <<<")

        sprint_resultado = {'sprint_id': sprint_id, 'evento': None, 'alocacao': None, 'total_sprint_days': 0, 'total_sprint_cost': 0}

        evento = sprint_info.get('evento')
        if evento:
            sprint_resultado['evento'] = evento
            if evento['tipo'] == 'saida':
                dev_saindo = evento['dev']
                if dev_saindo in equipe_atual:
                    del equipe_atual[dev_saindo]
                    print(f"[EVENTO] Saída: {dev_saindo} deixou a equipe. Motivo: {evento.get('motivo', 'N/A')}")
            elif evento['tipo'] == 'contratacao':
                novo_dev = evento['dev']
                equipe_atual[novo_dev] = evento['dev_details']
                print(f"[EVENTO] Contratação: {novo_dev} se juntou à equipe.")

        tarefas_da_sprint_ids = sprint_plan.get(str(sprint_id), [])
        if not tarefas_da_sprint_ids:
            print("Nenhuma tarefa planejada para esta sprint.")
            resultados_simulacao.append(sprint_resultado)
            continue

        print(f"Alocando {len(tarefas_da_sprint_ids)} tarefas para a equipe atual: {list(equipe_atual.keys())}")
        tarefas_da_sprint_data = {task_id: backlog_completo[task_id] for task_id in tarefas_da_sprint_ids if task_id in backlog_completo}

        if tarefas_da_sprint_data:
            allocation_task = Task(
                description=(
                    f"Dada a equipe atual: {json.dumps(equipe_atual, indent=2)}, "
                    f"e as tarefas para esta sprint: {json.dumps(tarefas_da_sprint_data, indent=2)}. "
                    "Aloque cada tarefa a um desenvolvedor. Para cada alocação, reestime os 'dias' necessários "
                    "com base na adequação do desenvolvedor e calcule o 'custo' (dias * cost_per_day). "
                    "Retorne um JSON com uma lista de objetos, cada um contendo 'task_id', 'assignee', 'estimated_days', e 'estimated_cost'."
                ),
                agent=task_allocator_agent,
                expected_output="Um único bloco de código JSON com a lista de alocações e estimativas."
            )
            allocation_crew = Crew(agents=[task_allocator_agent], tasks=[allocation_task], process=Process.sequential, max_rpm=RPM_LIMIT)
            alocacao_raw = allocation_crew.kickoff()
            alocacao_clean = _sanitize_json_output(str(alocacao_raw))
            try:
                alocacao = json.loads(alocacao_clean)
                sprint_resultado['alocacao'] = alocacao
                sprint_resultado['total_sprint_days'] = sum(item.get('estimated_days', 0) for item in alocacao)
                sprint_resultado['total_sprint_cost'] = sum(item.get('estimated_cost', 0) for item in alocacao)
                print(f"Sprint {sprint_id} alocada. Custo estimado: ${sprint_resultado['total_sprint_cost']:.2f}, Duração estimada: {sprint_resultado['total_sprint_days']} dias.")
            except json.JSONDecodeError:
                print(f"Erro ao decodificar JSON da alocação para a sprint {sprint_id}. Saída bruta: {alocacao_clean}")
                sprint_resultado['alocacao'] = {"erro": "Falha ao decodificar JSON", "output_bruto": alocacao_clean}

        resultados_simulacao.append(sprint_resultado)
        print("Aguardando 60 segundos para evitar limite de RPM...")
        time.sleep(60)

    print("\n--- Fase 3: Gerando Relatório Final da Simulação ---")
    summary_task = Task(
        description=(
            "Com base nos resultados completos da simulação: "
            f"{json.dumps(resultados_simulacao, indent=2)}, "
            "gere uma análise final do projeto. A análise deve incluir: "
            "1. Um resumo executivo do desempenho do projeto. "
            "2. O impacto dos eventos (saídas e contratações) nos custos e prazos totais. "
            "3. Uma avaliação da resiliência da equipe e como as estimativas mudaram. "
            "4. Recomendações para projetos futuros."
        ),
        agent=summary_agent,
        expected_output="JSON com a síntese final do projeto."
    )
    summary_crew = Crew(agents=[summary_agent], tasks=[summary_task], process=Process.sequential, max_rpm=RPM_LIMIT)
    final_summary_raw = summary_crew.kickoff()
    final_summary = json.loads(_sanitize_json_output(str(final_summary_raw)))

    resultado_completo = {
        "cenario_usado": cenario,
        "plano_de_sprints_gerado": sprint_plan,
        "resultados_por_sprint": resultados_simulacao,
        "sintese_final": final_summary
    }

    output_filename = 'resultado_simulacao_completa.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(resultado_completo, f, ensure_ascii=False, indent=2)

    print(f"\nSimulação concluída! Relatório completo salvo em '{output_filename}'.")
    print("\n--- SÍNTESE FINAL ---")
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    if not os.path.exists('cenario_simulacao_llm.json'):
        print("Erro: Arquivo 'cenario_simulacao_llm.json' não encontrado.")
        print("Execute 'gerador_cenario_llm.py' primeiro para criar o cenário.")
    else:
        run_simulation('cenario_simulacao_llm.json')