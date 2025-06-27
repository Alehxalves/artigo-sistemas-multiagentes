# simulator.py

import json
from main import create_crew


def simulate_scenarios(scenarios: list) -> list:
    """
    Recebe uma lista de cenários, cada um contendo os mesmos parâmetros esperados por create_crew.
    Retorna uma lista de resultados e salva em JSON.
    """
    all_results = []
    for idx, scenario in enumerate(scenarios, start=1):
        print(f"--- Simulação {idx}/{len(scenarios)} ---")
        result = create_crew(
            developers=scenario['developers'],
            components=scenario['components'],
            skills=scenario['skills'],
            preferences=scenario['preferences'],
            truck_factors=scenario['truck_factors'],
            tasks_data=scenario['tasks_data']
        )
        all_results.append({
            'scenario_id': idx,
            'input': scenario,
            'result': result
        })

    # Salvar resultados finais
    with open('simulation_results.json', 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("Simulação concluída. Resultados salvos em simulation_results.json")
    return all_results


if __name__ == '__main__':
    # Exemplo de como carregar o cenário base
    with open('allocation_input.json', 'r') as f:
        base_scenario = json.load(f)

    # Criar múltiplos cenários (pode variar parâmetros como skills, truck_factors, etc.)
    scenarios = [
        base_scenario,
        # Você pode clonar e modificar base_scenario para testes A/B
    ]

    simulate_scenarios(scenarios)
