from contest.agents.team_name_1.beliefline.topology3 import MapTopologyAnalyzer


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
    WALL = '\033[48;5;240m\033[38;5;250m'

    # Special nodes
    JUNCTION = '\033[1m\033[38;5;46m'
    ARTICULATION = '\033[1m\033[38;5;196m'

    # Tunnel depths (color gradient)
    DEPTH_COLORS = [
        '\033[38;5;255m',  # White (junction/open)
        '\033[38;5;51m',   # Cyan (depth 1)
        '\033[38;5;45m',   # Light blue (depth 2)
        '\033[38;5;39m',   # Blue (depth 3)
        '\033[38;5;33m',   # Medium blue (depth 4)
        '\033[38;5;27m',   # Dark blue (depth 5)
        '\033[38;5;21m',   # Darker blue (depth 6)
        '\033[38;5;129m',  # Purple (depth 7)
        '\033[38;5;93m',   # Dark purple (depth 8)
        '\033[38;5;57m',   # Darkest purple (depth 9+)
    ]


def print_legend():
    print("\n" + "=" * 60)
    print(f"{Colors.BOLD}LEGEND:{Colors.RESET}")
    print(f"  {Colors.WALL}███{Colors.RESET} = Wall")
    print(f"  {Colors.JUNCTION}◉{Colors.RESET}   = Junction (3+ connections)")
    print(f"  {Colors.ARTICULATION}→←↑↓↔↕┼↗↘↖↙✣{Colors.RESET}   = Articulation point showing blocked directions")
    print(f"  {Colors.DEPTH_COLORS[1]}1{Colors.RESET}   = Tunnel depth 1 (1 step from junction)")
    print(f"  {Colors.DEPTH_COLORS[5]}5{Colors.RESET}   = Tunnel depth 5")
    print(f"  {Colors.DEPTH_COLORS[9]}9{Colors.RESET}   = Tunnel depth 9+")
    print("=" * 60 + "\n")


def main():
    with open("alleyCapture.lay") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    walls = WallsGrid(lines)

    analyzer = MapTopologyAnalyzer(walls)

    print(f"\n{Colors.BOLD}Map Analysis:{Colors.RESET}")
    print(f"  Layout: {walls.width}x{walls.height}")
    print(f"  Junctions: {Colors.JUNCTION}{len(analyzer.junction_nodes)}{Colors.RESET}")
    print(f"  Articulation Points: {Colors.ARTICULATION}{len(analyzer.articulation_nodes)}{Colors.RESET}")

    print_legend()

    # Direction vectors and glyphs
    dir_to_symbol = {
        (1, 0): '→',
        (-1, 0): '←',
        (0, 1): '↑',
        (0, -1): '↓',
    }

    def combined_symbol(directions):
        """Return a single glyph summarizing multiple choke directions."""
        dirs = set(directions)
        if len(dirs) == 1:
            return dir_to_symbol[list(dirs)[0]]
        elif dirs == {(1, 0), (-1, 0)}:
            return '↔'
        elif dirs == {(0, 1), (0, -1)}:
            return '↕'
        elif dirs == {(1, 0), (0, 1)}:
            return '↗'
        elif dirs == {(1, 0), (0, -1)}:
            return '↘'
        elif dirs == {(-1, 0), (0, -1)}:
            return '↙'
        elif dirs == {(-1, 0), (0, 1)}:
            return '↖'
        elif len(dirs) >= 3:
            return '┼'  # multi-directional choke
        else:
            return '✣'  # fallback

    # Precompute articulation symbols
    articulation_symbols = {}
    for pos in analyzer.articulation_nodes:
        dirs = []
        for first_step, _length, _child in analyzer.choke_exits(pos):
            dx = first_step[0] - pos[0]
            dy = first_step[1] - pos[1]
            if (dx, dy) in dir_to_symbol:
                dirs.append((dx, dy))
        if dirs:
            articulation_symbols[pos] = Colors.ARTICULATION + combined_symbol(dirs) + Colors.RESET
        else:
            articulation_symbols[pos] = Colors.ARTICULATION + '⬤' + Colors.RESET

    # Print with border
    print("  " + "┌" + "─" * walls.width + "┐")

    for y in range(walls.height - 1, -1, -1):
        row = "  │"
        for x in range(walls.width):
            pos = (x, y)
            if walls[x][y]:
                row += f"{Colors.WALL}█{Colors.RESET}"
            elif pos in articulation_symbols:
                row += articulation_symbols[pos]
            elif pos in analyzer.junction_nodes:
                row += f"{Colors.JUNCTION}◉{Colors.RESET}"
            else:
                d = analyzer.depth_to_junction(pos)
                if d > 0:
                    depth_idx = min(d, 9)
                    color = Colors.DEPTH_COLORS[depth_idx]
                    row += f"{color}{d if d <= 9 else '+'}{Colors.RESET}"
                else:
                    row += f"{Colors.DEPTH_COLORS[0]}·{Colors.RESET}"
        row += "│"
        print(row)

    print("  " + "└" + "─" * walls.width + "┘")

    # Summary statistics
    print(f"\n{Colors.BOLD}Tunnel Depth Statistics:{Colors.RESET}")
    depth_counts = {}
    max_depth = 0
    for y in range(walls.height):
        for x in range(walls.width):
            if not walls[x][y]:
                d = analyzer.depth_to_junction((x, y))
                depth_counts[d] = depth_counts.get(d, 0) + 1
                max_depth = max(max_depth, d)

    for depth in sorted(depth_counts.keys()):
        count = depth_counts[depth]
        color = Colors.DEPTH_COLORS[min(depth, 9)] if depth > 0 else Colors.DEPTH_COLORS[0]
        bar = "█" * min(count // 2, 40)
        print(f"  Depth {depth:2d}: {color}{bar}{Colors.RESET} ({count} cells)")

    print(f"\n  Maximum tunnel depth: {Colors.DEPTH_COLORS[min(max_depth, 9)]}{max_depth}{Colors.RESET}")


if __name__ == "__main__":
    main()
