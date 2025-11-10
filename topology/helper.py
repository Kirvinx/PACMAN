# test_traps.py
from contest.agents.team_name_1.beliefline.topology import MapTopologyAnalyzer

class WallsGrid:
    def __init__(self, lines):
        self.width = len(lines[0])
        self.height = len(lines)
        self._grid = [
            [lines[self.height - 1 - y][x] == '%' for y in range(self.height)]
            for x in range(self.width)
        ]

    def __getitem__(self, x):
        return self._grid[x]

# ANSI color codes
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Walls
    WALL = '\033[48;5;240m\033[38;5;250m'  # Dark gray background, light gray text
    
    # Special nodes
    ARTICULATION = '\033[1m\033[38;5;196m' # Bright red, bold
    
    # Risk depth colors (for off pockets)
    RISK_COLORS = [
        '\033[38;5;255m',  # 0 (main region / no risk) – white / neutral
        '\033[38;5;214m',  # 1
        '\033[38;5;208m',  # 2
        '\033[38;5;202m',  # 3
        '\033[38;5;196m',  # 4
        '\033[38;5;160m',  # 5
        '\033[38;5;124m',  # 6
        '\033[38;5;88m',   # 7
        '\033[38;5;52m',   # 8
        '\033[38;5;53m',   # 9+
    ]

def print_legend():
    """Print a helpful legend explaining the visualization"""
    print("\n" + "="*60)
    print(f"{Colors.BOLD}LEGEND (Trap Risk):{Colors.RESET}")
    print(f"  {Colors.WALL}███{Colors.RESET} = Wall")
    print(f"  {Colors.ARTICULATION}⬤{Colors.RESET}   = Articulation door (choke tile)")
    print(f"  {Colors.RISK_COLORS[0]}·{Colors.RESET}   = Main region / no trap risk")
    print(f"  {Colors.RISK_COLORS[1]}1{Colors.RESET}   = Risk depth 1 (just past door)")
    print(f"  {Colors.RISK_COLORS[5]}5{Colors.RESET}   = Risk depth 5 (deep pocket)")
    print(f"  {Colors.RISK_COLORS[9]}9{Colors.RESET}   = Risk depth 9+ (very deep pocket)")
    print("="*60 + "\n")

def main():
    with open("fastCapture.lay") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    walls = WallsGrid(lines)

    analyzer = MapTopologyAnalyzer(walls)

    # Basic stats
    num_pockets = len(analyzer.pockets)
    num_off = len(analyzer.off_pockets)

    print(f"\n{Colors.BOLD}Trap Region Analysis:{Colors.RESET}")
    print(f"  Layout: {walls.width}x{walls.height}")
    print(f"  Articulation Points: {Colors.ARTICULATION}{len(analyzer.articulation_nodes)}{Colors.RESET}")
    print(f"  Pockets (components): {num_pockets}")
    print(f"  Main pocket index: {analyzer.main_pocket}")
    print(f"  Off pockets: {num_off} -> {analyzer.off_pockets}")

    print_legend()

    # Print with border
    print("  " + "┌" + "─" * walls.width + "┐")
    
    # Print risk depths visually with colors
    for y in range(walls.height - 1, -1, -1):
        row = "  │"
        for x in range(walls.width):
            pos = (x, y)
            if walls[x][y]:
                row += f"{Colors.WALL}█{Colors.RESET}"
            else:
                # Doors stay visible regardless of risk
                if pos in analyzer.articulation_nodes:
                    row += f"{Colors.ARTICULATION}⬤{Colors.RESET}"
                    continue

                risk = analyzer.trap_depth(pos)

                if risk > 0:
                    # Off pocket tile: show risk number
                    idx = min(risk, 9)
                    color = Colors.RISK_COLORS[idx]
                    char = str(risk) if risk <= 9 else '+'
                    row += f"{color}{char}{Colors.RESET}"
                else:
                    # Main pocket or non-trap tile
                    row += f"{Colors.RISK_COLORS[0]}·{Colors.RESET}"
        row += "│"
        print(row)
    
    print("  " + "└" + "─" * walls.width + "┘")

if __name__ == "__main__":
    main()
