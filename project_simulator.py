import json
import os
import time
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

# Carrega variáveis de ambiente
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Constante para o limite de RPM
RPM_LIMIT = 14
# Modelos LLM padrões para a simulação
LLM_MODEL_DEFAULT = 'gemini/gemini-2.0-flash-lite'
LLM_MODEL_SIMULATION = 'gemini/gemini-2.5-pro'


def _sanitize_json_output(raw_output: str) -> str:
    """
    Limpa a saída de texto do LLM para extrair um bloco JSON.
    Tenta encontrar o início de um objeto '{' ou de uma lista '['.
    Se encontrar múltiplos objetos JSON separados por vírgula, mas sem
    os colchetes da lista, os adiciona.
    """
    # Remove cercas de código e espaços em branco extras
    clean_output = raw_output.strip()
    if clean_output.startswith("```json"):
        clean_output = clean_output[7:]
    if clean_output.endswith("```"):
        clean_output = clean_output[:-3]
    clean_output = clean_output.strip()

    # Encontra o primeiro caractere JSON relevante
    json_start_index = -1
    first_brace = clean_output.find('{')
    first_bracket = clean_output.find('[')

    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        json_start_index = first_brace
    elif first_bracket != -1:
        json_start_index = first_bracket
    else:
        return "" # Nenhum JSON encontrado

    # Encontra o último caractere JSON relevante
    json_end_index = -1
    last_brace = clean_output.rfind('}')
    last_bracket = clean_output.rfind(']')

    if last_brace != -1 and (last_bracket == -1 or last_brace > last_bracket):
        json_end_index = last_brace
    elif last_bracket != -1:
        json_end_index = last_bracket
    else:
        return "" # Fim do JSON não encontrado

    # Extrai o conteúdo JSON
    json_content = clean_output[json_start_index:json_end_index + 1]

    # Verifica se é uma lista de objetos que não está entre colchetes
    if json_content.startswith('{') and json_content.endswith('}') and '},{' in json_content:
        return f'[{json_content}]'

    return json_content


def run_simulation(scenario_path: str):
    """Orquestra a simulação completa do projeto, sprint a sprint."""
    if not API_KEY:
        raise ValueError("API_KEY não encontrada. Defina OPENAI_API_KEY ou GOOGLE_API_KEY no seu arquivo .env")

    with open(scenario_path, 'r', encoding='utf-8') as f:
        cenario = json.load(f)
    print(f"Cenário '{scenario_path}' carregado.")

    # LLM padrão para a maioria dos agentes
    llm_default = LLM(
        model=LLM_MODEL_DEFAULT,
        api_key=API_KEY,
        system_message="Você é um assistente especialista em gerenciamento de projetos ágeis.",
        temperature=0
    )

    # LLM específico e mais poderoso para o agente de análise de impacto
    llm_impact = LLM(
        model=LLM_MODEL_SIMULATION,
        api_key=API_KEY,
        system_message="Você é um assistente especialista em análise quantitativa de impacto em projetos de software.",
        temperature=0
    )

    sprint_planner_agent = Agent(
        role='Planejador de Sprints Ágil',
        goal="Analisar um backlog de tarefas e um número de sprints para criar um plano de trabalho balanceado.",
        backstory="Um Scrum Master experiente que sabe como balancear o trabalho ao longo das sprints.",
        llm=llm_default, verbose=False
    )

    task_allocator_agent = Agent(
        role='Gerente de Projetos IA para Alocação e Estimativa',
        goal=(
            "Alocar tarefas de uma sprint, reestimar o esforço em dias para cada tarefa com base no desenvolvedor "
            "alocado e calcular o custo (dias * custo diário do dev). Retornar a alocação em formato JSON."
        ),
        backstory=(
            "Um gerente de projetos eficiente que rapidamente realoca e reestima tarefas quando a equipe muda, "
            "focando em maximizar a produtividade e prever custos, considerando a especialidade de cada dev."
        ),
        llm=llm_default, verbose=False
    )

    ramp_up_analyzer_agent = Agent(
        role='Analista de Onboarding de Talentos Técnicos',
        goal=(
            "Estimar o tempo de adaptação (ramp-up) em dias para um novo desenvolvedor, "
            "baseado em suas habilidades e no backlog de tarefas restante do projeto. "
            "Retornar um JSON com a estimativa."
        ),
        backstory=(
            "Um especialista em integração de equipes que entende a curva de aprendizado de novas tecnologias "
            "e a complexidade de projetos de software, prevendo quanto tempo um novo membro levará para se tornar produtivo."
        ),
        llm=llm_impact, verbose=False
    )

    event_impact_analyzer_agent = Agent(
        role='Analisador Quantitativo de Impacto de Eventos de Equipe',
        goal=(
            "Analisar o impacto da saída ou contratação de um desenvolvedor e quantificá-lo. "
            "Estimar o impacto financeiro (custo adicional) e no cronograma (dias de atraso) com base no backlog restante, "
            "truck factor e skills da equipe. O resultado deve ser um JSON."
        ),
        backstory=(
            "Um analista de projetos data-driven que traduz mudanças na equipe em métricas de custo e prazo, "
            "focando em fornecer dados para tomada de decisão."
        ),
        llm=llm_impact, verbose=False
    )

    summary_agent = Agent(
        role='Analista de Projetos Sênior Quantitativo',
        goal=(
            "Consolidar os resultados de todas as sprints e análises de impacto de eventos para gerar um relatório final "
            "quantitativo e qualitativo. O relatório deve calcular custos e prazos totais, analisar desvios, "
            "e fornecer um resumo executivo e recomendações baseadas em dados. O formato final deve ser JSON."
        ),
        backstory=(
            "Um especialista em análise de dados de projetos (PMO) que transforma logs de simulação em insights acionáveis, "
            "focando em métricas quantitativas como desvio de custo, impacto no cronograma e resiliência da equipe."
        ),
        llm=llm_impact, verbose=False
    )

    print("\n--- Fase 1: Planejando as Sprints com IA ---")
    projeto = cenario['projeto']
    backlog_completo = projeto['backlog_completo']
    total_sprints = cenario['config']['total_sprints_planejadas']
    componentes = projeto['componentes']
    truck_factors = projeto['truck_factors']

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
    completed_task_ids = set()

    for sprint_info in cenario['plano_sprints']:
        sprint_id = sprint_info['sprint_id']
        print(f"\n>>> Processando Sprint {sprint_id}... <<<")

        sprint_resultado = {'sprint_id': sprint_id, 'evento': None, 'analise_impacto_evento': None, 'alocacao': None, 'total_sprint_days': 0, 'total_sprint_cost': 0}

        evento = sprint_info.get('evento')
        if evento:
            sprint_resultado['evento'] = evento

            remaining_task_ids = set(backlog_completo.keys()) - completed_task_ids
            remaining_backlog = {task_id: backlog_completo[task_id] for task_id in remaining_task_ids}
            total_dias_restantes = sum(task['estimated_days'] for task in remaining_backlog.values())

            impact_analysis_desc = ""
            if evento['tipo'] == 'saida':
                dev_saindo = evento['dev']
                equipe_antes_saida = equipe_atual.copy()
                if dev_saindo in equipe_atual:
                    del equipe_atual[dev_saindo]
                    print(f"[EVENTO] Saída: {dev_saindo} deixou a equipe. Motivo: {evento.get('motivo', 'N/A')}")
                    impact_analysis_desc = (
                        f"O desenvolvedor '{dev_saindo}' saiu da equipe. A equipe anterior era {list(equipe_antes_saida.keys())} e agora é {list(equipe_atual.keys())}. "
                        f"O truck factor do projeto é: {json.dumps(truck_factors)}. "
                        f"O backlog restante tem {len(remaining_backlog)} tarefas, somando {total_dias_restantes} dias estimados. "
                        f"Analise o impacto quantitativo desta saída no projeto. Considere se o dev era um 'truck factor' para algum componente. "
                        f"Estime a variação de custo e de prazo (em dias de atraso) para o restante do projeto. "
                        f"Uma variação positiva indica um aumento (impacto negativo), enquanto uma variação negativa indica uma redução (impacto positivo)."
                    )


            elif evento['tipo'] == 'contratacao':
                novo_dev = evento['dev']
                equipe_atual[novo_dev] = evento['dev_details']
                print(f"[EVENTO] Contratação: {novo_dev} se juntou à equipe.")

                # Estimar o período de adaptação com LLM
                print("Estimando período de adaptação do novo desenvolvedor...")
                ramp_up_task_desc = (
                    f"O novo desenvolvedor é '{novo_dev}' com as seguintes competências: {json.dumps(evento['dev_details'])}. "
                    f"O backlog de tarefas restante do projeto é: {json.dumps(remaining_backlog, indent=2)}. "
                    f"Com base na complexidade das tarefas restantes e nas habilidades do desenvolvedor, "
                    f"estime quantos dias de adaptação (ramp-up) ele precisará para começar a contribuir efetivamente. "
                    "Considere que um dev sênior em uma tecnologia familiar pode levar de 2 a 5 dias, enquanto um dev "
                    "em um domínio desconhecido pode levar de 10 a 15 dias."
                )
                ramp_up_task = Task(
                    description=ramp_up_task_desc,
                    agent=ramp_up_analyzer_agent,
                    expected_output='Um único bloco de código JSON com a chave "dias_adaptacao_estimados". Ex: {"dias_adaptacao_estimados": 5}'
                )
                ramp_up_crew = Crew(agents=[ramp_up_analyzer_agent], tasks=[ramp_up_task], process=Process.sequential,
                                    max_rpm=RPM_LIMIT)
                ramp_up_result_raw = ramp_up_crew.kickoff()
                ramp_up_result_clean = _sanitize_json_output(str(ramp_up_result_raw))
                periodo_adaptacao_dias = 5  # Valor padrão
                try:
                    ramp_up_json = json.loads(ramp_up_result_clean)
                    periodo_adaptacao_dias = ramp_up_json.get('dias_adaptacao_estimados', 5)
                    print(f"Período de adaptação estimado pela IA: {periodo_adaptacao_dias} dias.")
                except json.JSONDecodeError:
                    print(
                        f"Erro ao decodificar JSON da estimativa de adaptação. Usando valor padrão de {periodo_adaptacao_dias} dias.")
                custo_adaptacao = periodo_adaptacao_dias * evento['dev_details']['cost_per_day']
                # Fim da estimativa de adaptação

                impact_analysis_desc = (
                    f"O desenvolvedor '{novo_dev}' ({json.dumps(evento['dev_details'])}) foi contratado. "
                    f"Foi estimado um período de adaptação de {periodo_adaptacao_dias} dias, com um custo de integração de {custo_adaptacao}. "
                    f"A nova equipe é {list(equipe_atual.keys())}. "
                    f"O truck factor do projeto é: {json.dumps(truck_factors)}. "
                    f"O backlog restante tem {len(remaining_backlog)} tarefas, somando {total_dias_restantes} dias estimados. "
                    f"Analise o impacto quantitativo desta contratação no projeto, considerando o período de adaptação. "
                    f"Estime a variação de custo e de prazo (em dias de atraso) para o restante do projeto. "
                    f"Uma variação negativa indica uma redução (impacto positivo), enquanto uma variação positiva indica um aumento (impacto negativo)."
                )

            if impact_analysis_desc:
                impact_task = Task(
                    description=impact_analysis_desc,
                    agent=event_impact_analyzer_agent,
                    expected_output="""
                    Um único bloco de código JSON com a seguinte estrutura:
                    {
                      "impacto_custo_estimado": <valor_numerico>,
                      "impacto_prazo_estimado_dias": <valor_numerico>,
                      "risco_truck_factor_afetado": <booleano>,
                      "analise_qualitativa": "<texto_explicativo>"
                    }
                    """
                )
                impact_crew = Crew(agents=[event_impact_analyzer_agent], tasks=[impact_task], process=Process.sequential, max_rpm=RPM_LIMIT)
                impact_analysis_raw = impact_crew.kickoff()
                impact_analysis_clean = _sanitize_json_output(str(impact_analysis_raw))
                try:
                    impact_analysis_json = json.loads(impact_analysis_clean)
                    sprint_resultado['analise_impacto_evento'] = impact_analysis_json
                    print(f"[ANÁLISE DE IMPACTO] {json.dumps(impact_analysis_json, ensure_ascii=False)}")
                except json.JSONDecodeError:
                    print(f"Erro ao decodificar JSON da análise de impacto. Saída bruta: {impact_analysis_clean}")
                    sprint_resultado['analise_impacto_evento'] = {"erro": "Falha ao decodificar JSON", "output_bruto": impact_analysis_clean}

                print("Aguardando 60 segundos para evitar limite de RPM...")
                time.sleep(60)


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
                    f"as tarefas para esta sprint: {json.dumps(tarefas_da_sprint_data, indent=2)}, "
                    f"os componentes do projeto: {json.dumps(componentes)}, "
                    f"e o truck factor (desenvolvedores-chave) de cada componente: {json.dumps(truck_factors, indent=2)}. "
                    "Aloque cada tarefa a um desenvolvedor. Considere que as tarefas podem estar relacionadas a um ou mais componentes. "
                    "Priorize alocar tarefas para os desenvolvedores listados no 'truck_factor' do componente correspondente, se eles estiverem na equipe atual. "
                    "Para cada alocação, reestime os 'dias' necessários com base na adequação do desenvolvedor e calcule o 'custo' (dias * cost_per_day). "
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
        completed_task_ids.update(tarefas_da_sprint_ids)

        if sprint_id < len(cenario['plano_sprints']):
            print("Aguardando 60 segundos para evitar limite de RPM...")
            time.sleep(60)

    print("\n--- Fase 3: Gerando Relatório Final da Simulação ---")
    summary_task = Task(
        description=(
            "Com base nos resultados completos da simulação: "
            f"{json.dumps(resultados_simulacao, indent=2)}, "
            "gere uma análise final do projeto. A análise deve ser um objeto JSON contendo uma única chave principal 'relatorio_final_projeto'. "
            "Dentro deste objeto, inclua: "
            "1. 'resumo_executivo': Um sumário do desempenho do projeto. "
            "2. 'analise_quantitativa_final': Métricas de custo e prazo (planejado vs. realizado) e seus desvios. "
            "3. 'analise_de_impacto_eventos': Uma lista detalhando o impacto de cada evento ocorrido. "
            "4. 'avaliacao_resiliencia_equipe': Uma análise sobre como a equipe lidou com as mudanças. "
            "5. 'recomendacoes': Recomendações para projetos futuros baseadas nos dados. "
            "Para a seção 'analise_de_impacto_eventos', use EXATAMENTE a seguinte estrutura para cada evento: "
            """
            {
              "sprint_id": <ID da Sprint do evento>,
              "evento": "<Nome do evento, ex: Saída da Desenvolvedora 'Carla'>",
              "descricao": "<Descrição do evento e seu contexto>",
              "impacto_quantitativo": {
                "custo_adicional_estimado": <valor numérico do custo extra gerado pelo evento>,
                "impacto_prazo_dias": <valor numérico do impacto em dias no cronograma>,
                "justificativa_custo": "<Explicação sobre a origem do custo adicional e como ele se relaciona com o impacto em dias no prazo. Por exemplo, 'O custo adicional é resultado do esforço extra necessário para cobrir as responsabilidades, o que gerou X dias de atraso.'>",
                "impacto_no_cronograma": "<Explicação sobre como o cronograma foi afetado>"
              },
              "impacto_qualitativo": "<Análise qualitativa do impacto do evento no projeto, equipe e riscos>"
            }
            """
            "Use os dados de 'analise_impacto_evento' de cada sprint para preencher esta seção. "
            "Seja explícito na 'justificativa_custo' sobre como o custo e o prazo estão conectados, sem usar o termo 'dias-homem'."
        ),
        agent=summary_agent,
        expected_output="Um único bloco de código JSON com a síntese final completa do projeto, aninhado sob a chave 'relatorio_final_projeto'."
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

    output_filename = 'resultado_simulacao.json'
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