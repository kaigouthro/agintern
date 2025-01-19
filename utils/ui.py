import os
from colorama import Fore, Style
from terminaltables import AsciiTable
from itertools import cycle
import time
from typing import List, Any

def clear_console():
    """Clears the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_agent_info(manager: Any, stop_event: Any, outputs: List[str]):
    """
    Continuously displays comprehensive information about the agents.
    """
    animation = cycle(['🌑', '🌒', '🌓', '🌔', '🌕', '🌖', '🌗', '🌘'])
    while not stop_event.is_set():
        clear_console()

        header = [
            "👤 Agent",
            "🔁 Evolve Count",
            "💻 Code Executions",
            "👥 Active Agents",
            "📈 Usage Count",
            "🌟 Depth",
            "Working?",
            "📝 Last Input",
            "🚦 Status"
        ]

        agents_data = [header]
        agents = manager.get_agents()
        for agent in agents:
            active_agents = ", ".join(f"{k}->{v}" for k, v in agent.active_agents.items())
            agents_data.append([
                agent.name,
                agent.evolve_count,
                agent.number_of_code_executions,
                active_agents,
                agent.usage_count,
                agent.depth,
                "✅" if agent.working_agent else "❌",
                agent.last_input,
                agent.current_status
            ])

        table = AsciiTable(agents_data)
        print(Fore.CYAN + "🤖 [1m Agents Status:[0m " + Style.RESET_ALL)
        print(table.table)
        for output in outputs:
            print(output)
        print(f"Agents are running.. {next(animation)}", end='')  # '' returns the cursor to the start of the line

        time.sleep(1)

def print_final_output(outputs: List[str], manager: Any):
    """
    Print final outputs and agent statistics.
    """
    clear_console()
    for output in outputs:
        print(output)

    for agent in manager.get_agents():
        print_agent_statistics(agent)

def print_agent_statistics(agent: Any):
    """
    Print statistics for a given agent.
    """
    print(f"📊 Stats for {agent.name}:")
    stats = [
        f"🔁 Evolve Count: {agent.evolve_count}",
        f"💻 Code Executions: {agent.number_of_code_executions}",
        f"👥 Active Agents: {agent.active_agents}",
        f"📈 Usage Count: {agent.usage_count}",
        f"🏔️ Max Depth: {agent.max_depth}",
        f"🌟 Depth: {agent.depth}",
        f"🛠️ Working Agent: {agent.working_agent}",
        f"📝 Last Input: {agent.last_input}",
        f"🚦 Status: {agent.current_status}",
        f"{Fore.MAGENTA}Prompt for {agent.name}:{Style.RESET_ALL}",
        f"{Fore.LIGHTMAGENTA_EX}{agent.dynamic_prompt} {Style.RESET_ALL}"    ]    
    print(''.join(stats))

def format_text(question_number: int, user_input: str, response: str) -> str:
    """
    Formats the text with color and style.
    """
    formatted_text = f"{Fore.YELLOW}

🔍 Question {question_number}: {Style.RESET_ALL} {user_input}
{Fore.GREEN}💡 Response:{Style.RESET_ALL} {response}"
    return formatted_text