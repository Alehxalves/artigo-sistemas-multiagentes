import json
import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool
from dotenv import load_dotenv

from project_planner import create_planner_agent

# Carrega variáveis de ambiente (.env deve conter OPENAI_API_KEY ou GOOGLE_API_KEY)
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")

def create_crew(
    project_goal: str,
    developers: list,
    components: list,
    skills: dict,
    preferences: dict,
    truck_factors: dict
) -> dict:
    """
    Cria o Crew com base nos parâmetros, executa o pipeline,
    faz parse do JSON de saída (removendo fences ```json```) e salva em arquivo estruturado.
    """
    llm = LLM(
        model='gemini/gemini-2.0-flash-lite',
        api_key=API_KEY,
        system_message="Você é um assistente que responde sempre em português."
    )

    @tool
    def get_developer_skills(name: str) -> str:
        """Retorna habilidades de um desenvolvedor."""
        return skills.get(name, "Habilidades não encontradas.")

    @tool
    def get_developer_preferences(name: str) -> str:
        """Retorna preferências de um desenvolvedor."""
        return preferences.get(name, "Preferências não encontradas.")

    @tool
    def calculate_truck_factor(component: str) -> float:
        """Retorna o Truck Factor de um componente."""
        return truck_factors.get(component, 1.0)

    # A ferramenta get_task_details não é mais necessária, pois o planejador gera as tarefas.

    # Agentes
    planner_agent = create_planner_agent(llm)

    dev_data_analyst = Agent(
        role='Analista de Dados de Desenvolvedores',
        goal='Coletar habilidades e preferências dos desenvolvedores.',
        backstory='Especialista em RH e engenharia.',
        llm=llm,
        tools=[get_developer_skills, get_developer_preferences],
        verbose=False, allow_delegation=False
    )

    truck_factor_analyst = Agent(
        role='Especialista em Métricas de Resiliência',
        goal='Calcular o Truck Factor para cada componente.',
        backstory='Arquiteto de software.',
        llm=llm,
        tools=[calculate_truck_factor],
        verbose=False, allow_delegation=False
    )

    task_allocator = Agent(
        role='Gerente de Projetos IA',
        goal=(
            "Alocar tarefas de forma otimizada e retornar **APENAS JSON**, "
            "com justificativas em português, no formato:\n"
            "[\n"
            "  {\"taskId\":\"TASK_001\",\"assignee\":\"Alice\",\"justification\":\"...\"},\n"
            "  …\n"
            "]"
        ),
        backstory='Gerente de projetos usando IA.',
        llm=llm,
        tools=[],  # As tarefas agora vêm do contexto do planejador
        verbose=False, allow_delegation=False # Alterado para False para evitar erros
    )

    # Tarefas
    plan = Task(
        description=f"Analisar e detalhar o seguinte objetivo: '{project_goal}'",
        agent=planner_agent,
        expected_output="Um dicionário JSON de tarefas, onde as chaves são IDs e os valores são as descrições."
    )

    collect = Task(
        description=f"Para cada desenvolvedor em {developers}, colete habilidades e preferências.",
        agent=dev_data_analyst,
        expected_output="Resumo JSON das habilidades e preferências."
    )

    analyze = Task(
        description=f"Para cada componente em {components}, calcule o Truck Factor.",
        agent=truck_factor_analyst,
        expected_output="JSON com componentes e seus Truck Factors."
    )

    allocate = Task(
        description=(
            "Com base nos perfis, Truck Factors e na lista de tarefas gerada, "
            "alocar cada tarefa de forma otimizada."
        ),
        agent=task_allocator,
        context=[collect, analyze, plan],
        expected_output="JSON estruturado de alocações."
    )

    crew = Crew(
        agents=[planner_agent, dev_data_analyst, truck_factor_analyst, task_allocator],
        tasks=[plan, collect, analyze, allocate],
        process=Process.sequential,
        manager_llm=llm,
        verbose=True,
        max_rpm=15
    )

    result = crew.kickoff()

    # Salva entrada
    with open('allocation_input.json', 'w', encoding='utf-8') as f:
        json.dump({
            'project_goal': project_goal,
            'developers': developers,
            'components': components,
            'skills': skills,
            'preferences': preferences,
            'truck_factors': truck_factors
        }, f, ensure_ascii=False, indent=2)

    # --- Sanitiza fences Markdown ---
    raw = result.output if hasattr(result, 'output') else str(result)
    raw = raw.strip()
    if raw.startswith("```"):
        # Remove primeira linha e última fence
        parts = raw.split("\n")
        # Se fence indicar json, tira ela
        if parts[0].startswith("```"):
            parts = parts[1:]
        if parts[-1].strip().startswith("```"):
            parts = parts[:-1]
        raw = "\n".join(parts)

    # Tenta parsear JSON
    try:
        allocations = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM não retornou JSON válido: {e}\nSaída bruta:\n{raw}")

    # Salva resultado estruturado
    with open('allocation_result.json', 'w', encoding='utf-8') as f:
        json.dump(allocations, f, ensure_ascii=False, indent=2)

    print("Dados de entrada -> allocation_input.json")
    print("Alocações estruturadas -> allocation_result.json")

    return allocations


if __name__ == '__main__':
    # Defina o objetivo do projeto de alto nível
    project_goal = (
        "Desenvolver um novo módulo de e-commerce para a plataforma existente, "
        "incluindo um sistema de carrinho de compras, integração de pagamentos e "
        "uma interface de administração de produtos."
    )

    developers = ['Alice', 'Bob', 'Charlie', 'Diana']
    components = ['Backend API Gateway', 'Frontend Dashboard', 'Serviço de Autenticação', 'Módulo de Relatórios', 'Banco de Dados', 'Sistema de Pagamentos']
    skills = {
        'Alice': 'Python, Django, SQL, AWS, REST APIs',
        'Bob': 'JavaScript, React, Node.js, UX/UI',
        'Charlie': 'Java, Spring Boot, Kubernetes, DevOps, CI/CD',
        'Diana': 'Python, ML, TensorFlow, Análise de Dados'
    }
    preferences = {
        'Alice': 'Desafios de backend, arquitetura de sistemas, aprender Go',
        'Bob': 'Foco em experiência do usuário, performance web, componentes reutilizáveis',
        'Charlie': 'Infraestrutura como código, escalabilidade, segurança',
        'Diana': 'Processamento de linguagem natural, modelos de recomendação, visualização de dados'
    }
    truck_factors = {
        'Backend API Gateway': 1.5,
        'Frontend Dashboard': 3.0,
        'Serviço de Autenticação': 2.0,
        'Módulo de Relatórios': 4.0,
        'Banco de Dados': 2.5,
        'Sistema de Pagamentos': 1.0
    }

    create_crew(project_goal, developers, components, skills, preferences, truck_factors)
