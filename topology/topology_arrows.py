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
        self.junction_nodes = set()        # positions with degree >= 3
        self.dead_end_nodes = set()        # positions with degree == 1
        self.corridor_nodes = set()        # degree == 2

        self.compressed_nodes = set()      # junctions + dead-ends
        self.compressed_adj = {}           # pos -> list[(neighbor_pos, corridor_length)]
        self.corridor_meta = {}            # (u, v) -> (length, first_step_grid_pos)

        # ---- Analysis results ----
        self.articulation_nodes = set()    # subset of junction_nodes/dead-ends
        self.articulation_children = {}    # u -> set of child compressed nodes v
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
        return pos in self.junction_nodes or pos in self.dead_end_nodes

    def grid_degree(self, pos):
        """Degree in the raw grid graph (0–4)."""
        return self.degree.get(pos, 0)

    def choke_exits(self, pos):
        """
        For an articulation tile 'pos', return a list of:
            (first_step_grid_pos, corridor_length, compressed_target)

        for each direction that leads into a region separated by removing 'pos'.
        If 'pos' is not an articulation node, returns [].
        """
        if pos not in self.articulation_nodes:
            return []

        exits = []
        for child in self.articulation_children.get(pos, ()):
            meta = self.corridor_meta.get((pos, child))
            if meta is None:
                continue
            length, first_step = meta
            exits.append((first_step, length, child))
        return exits

    def choke_neighbor_tiles(self, pos):
        """
        Just the *grid neighbor tiles* from 'pos' that lead into chokehold regions.
        """
        return [first_step for (first_step, _len_, _child) in self.choke_exits(pos)]

    # ----------------------------------------------------------------------
    # Internal helpers: grid graph
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

    # ----------------------------------------------------------------------
    # Internal helpers: compressed graph
    # ----------------------------------------------------------------------

    def _build_compressed_graph(self):
        """
        Build compressed graph whose nodes are:
            - junctions (degree >= 3)
            - dead ends (degree == 1)

        Edges follow corridors (degree == 2 tiles) and store:
            - corridor length
            - first step from the source compressed node.
        """
        self.compressed_nodes = self.junction_nodes | self.dead_end_nodes
        self.compressed_adj = {node: [] for node in self.compressed_nodes}
        self.corridor_meta = {}

        for node in self.compressed_nodes:
            for nb in self.neighbors[node]:
                # immediate neighbor is always the "first step" from this node
                first_step = nb

                # Case 1: immediate neighbor is another compressed node
                if nb in self.compressed_nodes:
                    self.compressed_adj[node].append((nb, 1))
                    self.corridor_meta[(node, nb)] = (1, first_step)
                else:
                    # Case 2: walk through a corridor of degree-2 tiles
                    prev = node
                    curr = nb
                    length = 1

                    while curr not in self.compressed_nodes:
                        nbs = self.neighbors[curr]
                        next_candidates = [p for p in nbs if p != prev]
                        if not next_candidates:
                            # should not happen on normal Pacman maps,
                            # but we guard anyway.
                            break
                        next_pos = next_candidates[0]
                        prev, curr = curr, next_pos
                        length += 1

                    if curr in self.compressed_nodes:
                        self.compressed_adj[node].append((curr, length))
                        self.corridor_meta[(node, curr)] = (length, first_step)

    # ----------------------------------------------------------------------
    # Internal helpers: articulation points (Tarjan)
    # ----------------------------------------------------------------------

    def _compute_articulation_points(self):
        """
        Run Tarjan's algorithm on the compressed graph to find articulation points.
        These are compressed nodes (junctions or dead ends) whose removal would
        disconnect the compressed graph.

        Additionally, for each articulation node u, we record which adjacent
        compressed nodes v correspond to separated subtrees when u is removed.
        """
        index = {}
        lowlink = {}
        self.articulation_nodes = set()
        self.articulation_children = {node: set() for node in self.compressed_nodes}
        current_index = [0]  # use list to make it mutable in nested func

        def dfs(u, parent=None):
            index[u] = lowlink[u] = current_index[0]
            current_index[0] += 1
            child_count = 0
            is_articulation = False
            children = []

            for v, _length in self.compressed_adj.get(u, []):
                if v == parent:
                    continue
                if v not in index:
                    child_count += 1
                    children.append(v)
                    dfs(v, u)
                    lowlink[u] = min(lowlink[u], lowlink[v])

                    # Non-root: articulation if no back-edge from v or its descendants
                    if parent is not None and lowlink[v] >= index[u]:
                        is_articulation = True
                        self.articulation_children[u].add(v)
                else:
                    # Back edge
                    lowlink[u] = min(lowlink[u], index[v])

            # Root is articulation if it has more than one child
            if parent is None and child_count > 1:
                is_articulation = True
                # All children are separated components when root is removed
                for v in children:
                    self.articulation_children[u].add(v)

            if is_articulation:
                self.articulation_nodes.add(u)

        # Choose a deterministic and meaningful starting node
        main_root = max(self.compressed_nodes, key=lambda p: len(self.compressed_adj[p]))
        print(main_root)

        
        dfs(main_root)

    

    # ----------------------------------------------------------------------
    # Internal helpers: tunnel depth
    # ----------------------------------------------------------------------

    def _compute_tunnel_depths(self):
        """
        Compute distance from each tile to the nearest junction, propagating
        through corridor tiles (degree <= 2). If there are no junctions, use
        dead ends as seeds.
        """
        self.tunnel_depth = {pos: 0 for pos in self.legal_positions}

        # If there are no real junctions (weird map), fall back to dead ends
        seeds = set(self.junction_nodes)
        if not seeds:
            seeds = set(self.dead_end_nodes)
        if not seeds:
            return

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
