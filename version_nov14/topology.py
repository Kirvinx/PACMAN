from collections import deque


class MapTopologyAnalyzer:
    """
    Map Topology Analyzer
    ----------------------

    Performs structural analysis of a Pacman map grid.

    Stage 1: Graph topology
      • Builds the grid graph (legal positions + neighbors + degree)
      • Identifies junctions, corridors, and dead-ends
      • Builds a compressed graph (junctions, entry tiles, dead ends)
      • Finds articulation (choke) tiles using Tarjan's algorithm

    Stage 2: Trap-region analysis
      • Removes articulation tiles to find disconnected pockets
      • Identifies the main region vs smaller “off pockets”
      • Computes per-tile risk depth inside off pockets
        (distance to nearest articulation door)
      • Assigns risk = 1 to doors and dead-ends directly next to junctions
    """

    def __init__(self, walls):
        """
        :param walls: grid from game_state.get_walls()
                      walls[x][y] == True means wall, False means free.
        """
        self.walls = walls
        self.width, self.height = walls.width, walls.height

        # ---- Grid-level data ----
        self.legal_positions = []
        self.legal_positions_set = set()
        self.neighbors = {}
        self.degree = {}

        # ---- Node categories ----
        self.junction_nodes = set()    # degree >= 3
        self.corridor_nodes = set()    # degree == 2
        self.dead_end_nodes = set()    # degree == 1

        # ---- Compressed graph (junctions + entry + dead ends) ----
        self.compressed_adj = {}
        self.articulation_nodes = set()

        # ---- Pocket / trap analysis ----
        self.pocket_id = {}
        self.pockets = []
        self.pocket_exits = []
        self.main_pocket = None
        self.off_pockets = []
        self.pocket_risk_dist = {}

        # ---- Run full analysis ----
        self._build_grid_graph()
        self._build_compressed_graph()
        self._compute_articulation_points()
        self._compute_pocket_regions()
        self._compute_pocket_risks()

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def is_articulation_tile(self, pos):
        """True if 'pos' is a choke tile whose removal would disconnect the map."""
        return pos in self.articulation_nodes

    def is_junction(self, pos):
        """True if tile is a junction (degree >= 3)."""
        return pos in self.junction_nodes

    def is_in_trap_region(self, pos):
        """True if 'pos' lies in an off pocket (i.e., not in the main region)."""
        pocket = self.pocket_id.get(pos)
        return pocket is not None and pocket in self.off_pockets

    def trap_depth(self, pos):
        """
        Risk depth:
            • 0  → main region or not risky
            • 1  → articulation door or dead-end directly next to a junction
            • 2+ → progressively deeper inside a pocket
        """
        return self.pocket_risk_dist.get(pos, 0)

    # ----------------------------------------------------------------------
    # Stage 1: Grid + Compressed graph
    # ----------------------------------------------------------------------

    def _build_grid_graph(self):
        """Populate legal positions, neighbors, and classify nodes by degree."""
        for x in range(self.width):
            for y in range(self.height):
                if self.walls[x][y]:
                    continue
                pos = (x, y)
                self.legal_positions.append(pos)
                self.legal_positions_set.add(pos)

        for x, y in self.legal_positions:
            nbs = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < self.width
                    and 0 <= ny < self.height
                    and not self.walls[nx][ny]
                ):
                    nbs.append((nx, ny))
            self.neighbors[(x, y)] = nbs
            self.degree[(x, y)] = len(nbs)

        for pos, deg in self.degree.items():
            if deg == 1:
                self.dead_end_nodes.add(pos)
            elif deg == 2:
                self.corridor_nodes.add(pos)
            elif deg >= 3:
                self.junction_nodes.add(pos)

    def _build_compressed_graph(self):
        """
        Build a compressed graph connecting:
            • Junctions (deg >= 3)
            • Dead ends (deg == 1)
            • Entry corridor tiles (adjacent to junctions)
        """
        entry_tiles = set()
        for j in self.junction_nodes:
            for nb in self.neighbors[j]:
                if nb not in self.junction_nodes and nb not in self.dead_end_nodes:
                    entry_tiles.add(nb)

        self.compressed_nodes = (
            self.junction_nodes | self.dead_end_nodes | entry_tiles
        )
        self.compressed_adj = {node: [] for node in self.compressed_nodes}

        for node in self.compressed_nodes:
            for nb in self.neighbors[node]:
                if nb in self.compressed_nodes:
                    self.compressed_adj[node].append((nb, 1))
                else:
                    # Walk through corridor until another compressed node
                    prev, curr, length = node, nb, 1
                    while curr not in self.compressed_nodes:
                        nbs = [p for p in self.neighbors[curr] if p != prev]
                        if not nbs:
                            break
                        prev, curr = curr, nbs[0]
                        length += 1
                    if curr in self.compressed_nodes:
                        self.compressed_adj[node].append((curr, length))

        # Deduplicate (undirected)
        for node, edges in self.compressed_adj.items():
            uniq = {}
            for nb, dist in edges:
                if nb not in uniq or dist < uniq[nb]:
                    uniq[nb] = dist
            self.compressed_adj[node] = [(nb, d) for nb, d in uniq.items()]

    def _compute_articulation_points(self):
        """Run Tarjan’s algorithm on the compressed graph to find articulation points."""
        index, lowlink = {}, {}
        current = [0]
        self.articulation_nodes = set()

        def dfs(u, parent=None):
            index[u] = lowlink[u] = current[0]
            current[0] += 1
            child_count, is_art = 0, False

            for v, _ in self.compressed_adj.get(u, []):
                if v == parent:
                    continue
                if v not in index:
                    child_count += 1
                    dfs(v, u)
                    lowlink[u] = min(lowlink[u], lowlink[v])
                    if parent is not None and lowlink[v] >= index[u]:
                        is_art = True
                else:
                    lowlink[u] = min(lowlink[u], index[v])

            if parent is None and child_count > 1:
                is_art = True
            if is_art:
                self.articulation_nodes.add(u)

        for node in self.compressed_nodes:
            if node not in index:
                dfs(node)

        # Keep only articulation points that are not junctions
        self.articulation_nodes = {
            n for n in self.articulation_nodes if n not in self.junction_nodes
        }

        # Prune nested / deeper articulation points using compressed graph only
        self._prune_nested_articulations_on_compressed()

    def _prune_nested_articulations_on_compressed(self):
        """
        Keep only 'first layer' articulation nodes: those that touch the main
        non-articulation component in the compressed graph. Deeper ones are pruned.
        """
        if not self.articulation_nodes:
            return

        # 1) Connected components of NON-articulation compressed nodes
        comp_id = {}
        comp_sizes = []

        for node in self.compressed_nodes:
            if node in self.articulation_nodes or node in comp_id:
                continue

            cid = len(comp_sizes)
            stack = [node]
            comp_id[node] = cid
            size = 0

            while stack:
                u = stack.pop()
                size += 1
                for v, _ in self.compressed_adj.get(u, []):
                    if v in self.articulation_nodes or v in comp_id:
                        continue
                    comp_id[v] = cid
                    stack.append(v)

            comp_sizes.append(size)

        if not comp_sizes:
            return

        # 2) Choose main component (largest non-articulation component)
        main_comp = max(range(len(comp_sizes)), key=lambda i: comp_sizes[i])

        # 3) Keep only articulations that border the main component
        filtered = set()
        for a in self.articulation_nodes:
            neighbor_comps = set()
            for v, _ in self.compressed_adj.get(a, []):
                if v in self.articulation_nodes:
                    continue
                cid = comp_id.get(v)
                if cid is not None:
                    neighbor_comps.add(cid)

            # "First" articulation layer: has at least one neighbor in main_comp
            if main_comp in neighbor_comps:
                filtered.add(a)

        self.articulation_nodes = filtered


    # ----------------------------------------------------------------------
    # Stage 2: Pockets and risk propagation
    # ----------------------------------------------------------------------

    def _compute_pocket_regions(self):
        """Split map into pockets after removing articulation tiles."""
        visited = set()
        self.pocket_id, self.pockets, self.pocket_exits = {}, [], []

        for pos in self.legal_positions:
            if pos in visited or pos in self.articulation_nodes:
                continue

            stack = [pos]
            visited.add(pos)
            comp, exits = [], set()

            while stack:
                u = stack.pop()
                comp.append(u)
                for v in self.neighbors[u]:
                    if v in self.articulation_nodes:
                        exits.add(v)
                    elif v not in visited:
                        visited.add(v)
                        stack.append(v)

            idx = len(self.pockets)
            self.pockets.append(comp)
            self.pocket_exits.append(exits)
            for p in comp:
                self.pocket_id[p] = idx

        if self.pockets:
            self.main_pocket = max(range(len(self.pockets)), key=lambda i: len(self.pockets[i]))
            self.off_pockets = [i for i in range(len(self.pockets)) if i != self.main_pocket]
        else:
            self.main_pocket = None
            self.off_pockets = []

    def _compute_pocket_risks(self):
        """
        Compute per-tile risk distances inside off pockets:
          - Doors (articulations) = 1
          - Dead ends next to junctions = 1
          - Pocket interior starts at 2
        """
        self.pocket_risk_dist = {}

        # Doors = 1
        for art in self.articulation_nodes:
            self.pocket_risk_dist[art] = 1

        # Dead ends directly attached to junctions = 1
        for pos in self.dead_end_nodes:
            nbs = self.neighbors.get(pos, [])
            if nbs and nbs[0] in self.junction_nodes:
                self.pocket_risk_dist[pos] = 1

        # BFS propagation inside each off pocket
        for i in self.off_pockets:
            exits = self.pocket_exits[i]
            if not exits:
                continue

            q, visited = deque(), set()
            for exit_tile in exits:
                for nb in self.neighbors.get(exit_tile, []):
                    if self.pocket_id.get(nb) == i and nb not in visited:
                        visited.add(nb)
                        self.pocket_risk_dist[nb] = 2
                        q.append(nb)

            while q:
                u = q.popleft()
                du = self.pocket_risk_dist[u]
                for v in self.neighbors[u]:
                    if v in visited or self.pocket_id.get(v) != i:
                        continue
                    visited.add(v)
                    self.pocket_risk_dist[v] = du + 1
                    q.append(v)