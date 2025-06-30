from crewai import Agent


def create_planner_agent(llm):
    """
    Cria e retorna o Agente Planejador de Projeto.

    Este agente é responsável por decompor um objetivo de alto nível
    em uma lista de tarefas técnicas detalhadas.
    """
    project_planner_agent = Agent(
        role='Planejador de Projetos de Software',
        goal=(
            "Analisar um objetivo de projeto de alto nível e detalhá-lo em "
            "uma lista de tarefas de desenvolvimento específicas e acionáveis. "
            "Retorne a lista de tarefas em um formato JSON, como:\n"
            "{\n"
            '  "TASK_001": "Descrição da tarefa 1",\n'
            '  "TASK_002": "Descrição da tarefa 2",\n'
            "  ...\n"
            "}"
        ),
        backstory=(
            "Você é um gerente de projetos sênior e arquiteto de software com vasta "
            "experiência em decompor requisitos complexos em partes menores e gerenciáveis. "
            "Sua especialidade é criar planos de projeto claros que as equipes de "
            "desenvolvimento possam seguir para entregar software de alta qualidade."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
    return project_planner_agent
