from collections import deque


class MapTopologyAnalyzer:
    """
    Stage 1: static map analysis.

    - Works on the Pacman walls layout (grid).
    - Builds:
        * Grid graph (legal positions + neighbors + degree).
        * Compressed graph of junctions/dead-ends with corridor lengths.
        * Articulation points on compressed graph (classic "choke" nodes).
        * Tunnel depth: distance (in steps) from each corridor/dead tile to the nearest junction.

    Designed to be easily extended later with trap-region analysis.
    """

    def __init__(self, walls):
        """
        :param walls: grid from game_state.get_walls()
                      walls[x][y] == True means wall, False means free.
        """
        self.walls = walls
        self.width, self.height = walls.width, walls.height

        # ---- Grid-level data ----
        self.legal_positions = []          # list[(x, y)]
        self.legal_positions_set = set()   # set for fast lookup
        self.neighbors = {}                # pos -> [neighbor positions]
        self.degree = {}                   # pos -> number of neighbors

        # ---- Compressed graph (junctions + dead ends only) ----
        self.junction_nodes = set()        # positions with degree != 2
        self.corridor_nodes = set()        # degree == 2
        self.compressed_adj = {}           # pos -> list[(neighbor_pos, corridor_length)]

        # ---- Analysis results ----
        self.articulation_nodes = set()    # subset of junction_nodes
        self.tunnel_depth = {}             # pos -> int (steps to nearest junction)

        # Run analysis
        self._build_grid_graph()
        self._build_compressed_graph()
        self._compute_articulation_points()
        self._compute_tunnel_depths()

    # ----------------------------------------------------------------------
    # Public API (Stage 1)
    # ----------------------------------------------------------------------

    def depth_to_junction(self, pos):
        """
        Returns:
            Number of steps from 'pos' to the nearest junction (branching point)
            along 1-wide corridors / tunnels, or 0 if pos is a junction or not in a tunnel.
        """
        return self.tunnel_depth.get(pos, 0)

    def is_articulation_tile(self, pos):
        """
        Returns:
            True if 'pos' is a junction that is an articulation point in the
            compressed graph (removing it would disconnect regions). These are
            classic "bridge" choke points.
        """
        return pos in self.articulation_nodes

    def is_junction(self, pos):
        """Convenience: True if this tile is a junction/dead-end (degree != 2)."""
        return pos in self.junction_nodes

    def grid_degree(self, pos):
        """Degree in the raw grid graph (0–4)."""
        return self.degree.get(pos, 0)

    # ----------------------------------------------------------------------
    # Internal helpers: compressed graph + grid graoh
    # ----------------------------------------------------------------------

    def _build_grid_graph(self):
        """Populate legal_positions, neighbors, and degree on the raw grid."""
        self.dead_end_nodes = set()

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
                if 0 <= nx < self.width and 0 <= ny < self.height and not self.walls[nx][ny]:
                    nbs.append((nx, ny))
            self.neighbors[(x, y)] = nbs
            self.degree[(x, y)] = len(nbs)

        # classify nodes
        for pos in self.legal_positions:
            deg = self.degree[pos]
            if deg == 2:
                self.corridor_nodes.add(pos)
            elif deg == 1:
                self.dead_end_nodes.add(pos)      # true dead ends
            elif deg >= 3:
                self.junction_nodes.add(pos)      # real junctions
            # deg == 0 would be isolated; can ignore or handle separately

    def _build_compressed_graph(self):
        """
        Build a compressed graph that includes:
            - all junctions (deg >= 3)
            - all dead ends (deg == 1)
            - all corridor tiles adjacent to junctions ("entry tiles")

        Corridors between these nodes are represented as weighted edges
        with 'length' equal to the number of steps through corridor tiles.
        """

        # Step 1: determine which tiles should be treated as nodes
        self.dead_end_nodes = getattr(self, "dead_end_nodes", set())
        self.entry_tiles = set()

        # Corridor tiles immediately adjacent to junctions
        for j in self.junction_nodes:
            for nb in self.neighbors[j]:
                if nb not in self.junction_nodes and nb not in self.dead_end_nodes:
                    self.entry_tiles.add(nb)

        # The compressed graph will now include:
        # junctions, dead ends, and entry corridor tiles
        self.compressed_nodes = self.junction_nodes | self.dead_end_nodes | self.entry_tiles
        self.compressed_adj = {node: [] for node in self.compressed_nodes}

        # Step 2: traverse corridors to connect nodes
        for node in self.compressed_nodes:
            for nb in self.neighbors[node]:
                if nb in self.compressed_nodes:
                    # Direct neighbor is another compressed node
                    self.compressed_adj[node].append((nb, 1))
                else:
                    # Walk down the corridor until the next compressed node
                    prev = node
                    curr = nb
                    length = 1
                    while curr not in self.compressed_nodes:
                        nbs = self.neighbors[curr]
                        next_candidates = [p for p in nbs if p != prev]
                        if not next_candidates:
                            break
                        next_pos = next_candidates[0]
                        prev, curr = curr, next_pos
                        length += 1
                    if curr in self.compressed_nodes:
                        self.compressed_adj[node].append((curr, length))

        # Optional: remove duplicate edges (since it's an undirected graph)
        for node, edges in self.compressed_adj.items():
            unique = {}
            for nb, length in edges:
                if (nb not in unique) or (length < unique[nb]):
                    unique[nb] = length
            self.compressed_adj[node] = [(nb, length) for nb, length in unique.items()]



    # ----------------------------------------------------------------------
    # Internal helpers: articulation points (Tarjan)
    # ----------------------------------------------------------------------

    def _compute_articulation_points(self):
        """
        Run Tarjan's algorithm on the compressed graph to find articulation points.
        These are compressed nodes (junctions or dead ends) whose removal would
        disconnect the compressed graph. Typically only degree ≥3 junctions become
        articulation points, but we include all compressed nodes for completeness.
        """
        index = {}
        lowlink = {}
        self.articulation_nodes = set()
        current_index = [0]  # use list to make it mutable in nested func

        def dfs(u, parent=None):
            index[u] = lowlink[u] = current_index[0]
            current_index[0] += 1
            child_count = 0
            is_articulation = False

            for v, _length in self.compressed_adj.get(u, []):
                if v == parent:
                    continue
                if v not in index:
                    child_count += 1
                    dfs(v, u)
                    lowlink[u] = min(lowlink[u], lowlink[v])

                    # Non-root: articulation if no back-edge from v or its descendants
                    if parent is not None and lowlink[v] >= index[u]:
                        is_articulation = True
                else:
                    # Back edge
                    lowlink[u] = min(lowlink[u], index[v])

            # Root is articulation if it has more than one child
            if parent is None and child_count > 1:
                is_articulation = True

            if is_articulation:
                self.articulation_nodes.add(u)

        for node in self.compressed_nodes:
            if node not in index:
                dfs(node)
        
        # Keep only articulation points that are NOT junctions
        self.articulation_nodes = {
            n for n in self.articulation_nodes if n not in self.junction_nodes
        }


    # ----------------------------------------------------------------------
    # Internal helpers: tunnel depth
    # ----------------------------------------------------------------------

    def _compute_tunnel_depths(self):
        self.tunnel_depth = {pos: 0 for pos in self.legal_positions}

        # If there are no real junctions (weird map), fall back to dead ends
        seeds = set(self.junction_nodes)
        if not seeds:
            seeds = set(self.dead_end_nodes)
        if not seeds:
            return

        from collections import deque
        q = deque()
        visited = set()

        for j in seeds:
            q.append(j)     
            visited.add(j)
            self.tunnel_depth[j] = 0

        while q:
            pos = q.popleft()
            base_depth = self.tunnel_depth[pos]
            for nb in self.neighbors[pos]:
                if nb in visited:
                    continue
                # Only propagate through corridor-style tiles OR dead ends
                if self.degree[nb] <= 2:
                    visited.add(nb)
                    self.tunnel_depth[nb] = base_depth + 1
                    q.append(nb)

    def _compute_pocket_regions(self):
        """
        After self.articulation_nodes is known.
        Compute connected components of non-articulation tiles, and which
        articulation tiles border each component.
        """
        self.pocket_id = {}         # pos -> pocket index (or None for main region)
        self.pockets = []           # list[list[pos]]
        self.pocket_exits = []      # list[set[articulation_pos]]

        visited = set()
        for pos in self.legal_positions:
            if pos in visited or pos in self.articulation_nodes:
                continue

            # BFS/DFS to build one component
            stack = [pos]
            visited.add(pos)
            comp = []
            exits = set()

            while stack:
                u = stack.pop()
                comp.append(u)

                for v in self.neighbors[u]:
                    if v in self.articulation_nodes:
                        exits.add(v)         # this articulation borders the component
                    elif v not in visited:
                        visited.add(v)
                        stack.append(v)

            pocket_idx = len(self.pockets)
            self.pockets.append(comp)
            self.pocket_exits.append(exits)
            for u in comp:
                self.pocket_id[u] = pocket_idx
