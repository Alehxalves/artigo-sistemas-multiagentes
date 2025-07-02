import json
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

# Carrega variáveis de ambiente
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")

def gerar_cenario_com_llm(
    total_sprints: int = 5,
    tarefas_por_sprint: int = 5,
    llm_model: str = 'gemini/gemini-2.5-pro'
) -> None:
    """
    Usa um LLM para gerar um cenário de simulação de projeto e o salva em um arquivo JSON.
    """
    if not API_KEY:
        raise ValueError("API_KEY não encontrada. Defina OPENAI_API_KEY ou GOOGLE_API_KEY no seu arquivo .env")

    llm = LLM(
        model=llm_model,
        api_key=API_KEY,
        system_message="Você é um assistente especialista em planejamento de projetos de software.",
        temperature=0.7
    )

    scenario_generator = Agent(
        role='Gerador de Cenários de Projetos de Software',
        goal=(
            "Criar um cenário de projeto de software completo e realista em formato JSON. "
            "O JSON deve ser a ÚNICA coisa na sua resposta, sem comentários ou texto adicional."
        ),
        backstory=(
            "Um planejador de projetos experiente com um talento para criar simulações "
            "detalhadas e desafiadoras para testar a resiliência de equipes de desenvolvimento."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

    total_tarefas = total_sprints * tarefas_por_sprint
    generation_task = Task(
        description=f"""
        Gere um cenário de simulação de projeto de software em formato JSON.
        Siga EXATAMENTE a estrutura abaixo.

        O cenário deve conter:
        1.  'config': Parâmetros da simulação.
        2.  'projeto': 4 componentes, 'truck_factors', e um backlog com {total_tarefas} tarefas. Cada tarefa deve ter 'description' e 'estimated_days'.
        3.  'equipe_inicial': 4 desenvolvedores com 'skills', 'preferences', e 'cost_per_day'.
        4.  'plano_sprints': Uma lista de {total_sprints} sprints. O plano deve incluir pelo menos um evento de 'saida' e um evento de 'contratacao'.

        Estrutura do JSON de saída esperado:
        {{
          "config": {{
            "total_sprints_planejadas": {total_sprints},
            "tarefas_por_sprint": {tarefas_por_sprint}
          }},
          "projeto": {{
            "componentes": ["Componente A", "Componente B", "Componente C", "Componente D"],
            "truck_factors": {{
              "Componente A": {{"value": 2, "developers": ["NomeDev1", "NomeDev2"]}},
              "Componente B": {{"value": 1, "developers": ["NomeDev3"]}}
            }},
            "backlog_completo": {{
              "PROJ_TASK_001": {{"description": "Descrição detalhada da tarefa 1", "estimated_days": 3}},
              "PROJ_TASK_{total_tarefas:03d}": {{"description": "Descrição detalhada da tarefa {total_tarefas}", "estimated_days": 5}}
            }}
          }},
          "equipe_inicial": {{
            "NomeDev1": {{"skills": "...", "preferences": "...", "cost_per_day": 500}}
          }},
          "plano_sprints": [
            {{"sprint_id": 1, "evento": null}},
            {{"sprint_id": 2, "evento": {{"tipo": "saida", "dev": "NomeDev1", "motivo": "Motivo da saída"}}}},
            {{"sprint_id": 3, "evento": {{"tipo": "contratacao", "dev": "NovoDev", "dev_details": {{"skills": "...", "preferences": "...", "cost_per_day": 550}}}}}},
            {{"sprint_id": 4, "evento": null}}
          ]
        }}
        """,
        agent=scenario_generator,
        expected_output="Um único bloco de código JSON contendo o cenário completo."
    )

    crew = Crew(
        agents=[scenario_generator],
        tasks=[generation_task],
        process=Process.sequential,
        verbose=True
    )

    print("Gerando cenário com LLM... Isso pode levar alguns instantes.")
    result = crew.kickoff()

    raw_output = str(result).strip()
    if raw_output.startswith("```json"):
        raw_output = raw_output[7:]
    if raw_output.endswith("```"):
        raw_output = raw_output[:-3]

    try:
        scenario_data = json.loads(raw_output)
    except json.JSONDecodeError as e:
        print(f"Erro: O LLM não retornou um JSON válido. {e}")
        print("--- Saída Bruta Recebida ---")
        print(raw_output)
        return

    output_filename = 'cenario_simulacao_llm.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(scenario_data, f, ensure_ascii=False, indent=2)

    print(f"\nCenário gerado por LLM e salvo em '{output_filename}'")


if __name__ == '__main__':
    gerar_cenario_com_llm()