#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>

#include "G3dOGL/SimplicialComplex.h"

namespace hh {

#ifdef BUILD_LIBPSC

// helper class using a 3d delaunay triangulation to find cross-component candidates.
// manages all candidate edges (mesh and virtual) in a separate graph.
//
// high-level model:
// - _mesh stores the current geometric/topological state.
// - _dt stores one delaunay vertex per surviving mesh vertex id.
// - _edge_graph stores candidate edges, even when temporarily removed from heap.
// - _heap stores active candidates ordered by contraction cost.
// - disjoint-set maps maintain connected components for topological weighting.
//
// core consistency goals:
// - every edge in _heap must exist in _edge_graph.
// - every surviving vertex id in _mesh must have one disjoint-set entry.
// - if markov=true, candidate updates are local/incremental after each collapse.
// - if markov=false, candidate updates are driven by _edge_graph.unify().
class Contractor {
 private:
  using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;

  static_assert(std::is_same<Kernel::FT, double>::value, "Delaunay requires Kernel::FT to be double");

  using Vb = CGAL::Triangulation_vertex_base_with_info_3<int, Kernel>;
  using Tds = CGAL::Triangulation_data_structure_3<Vb>;
  using delaunay = CGAL::Delaunay_triangulation_3<Kernel, Tds>;
  using Vertex_handle = delaunay::Vertex_handle;
  using Point_3 = Kernel::Point_3;

  using PairInt = std::pair<int, int>;

  struct PairIntHash {
    std::size_t operator()(const PairInt& p) const noexcept {
      std::size_t h1 = std::hash<int>{}(p.first);
      std::size_t h2 = std::hash<int>{}(p.second);
      return h1 ^ (h2 + 0x9e3779b97f4a7c15ull + (h1 << 6) + (h1 >> 2));
    }
  };

  using PairIntSet = std::unordered_set<PairInt, PairIntHash>;
  using IntSet = std::unordered_set<int>;

 private:
  // selects the update policy:
  // - true: markov/incremental candidate maintenance using local neighborhood diffs.
  // - false: non-markov maintenance driven by direct edge-graph unification.
  bool markov;

  delaunay _dt;
  std::unordered_map<int, Vertex_handle> _vid2vh_map;

  SimplicialComplex& _mesh;
  MinHeap& _heap;

  // graph storing all candidate edges ever created.
  // edges may stay here even when erased from heap, which enables stable lookups.
  SimplicialComplex _edge_graph;

  // disjoint-set with explicit circular member lists.
  // _component_size is valid only for roots.
  std::unordered_map<int, int> _component_parent;
  std::unordered_map<int, int> _component_size;  // union by size for o(α(n)) amortized
  std::unordered_map<int, int> _component_next;  // circular linked list per component, for o(k) enumeration
  std::unordered_map<int, int> _component_prev;  // back links for o(1) removal

 private:
  // some utilities
  static inline PairInt normalize_pair(int a, int b) {
    if (a > b) std::swap(a, b);
    return {a, b};
  }

  void reserve_state_maps(std::size_t vertex_count) {
    _vid2vh_map.reserve(vertex_count);
    _component_parent.reserve(vertex_count);
    _component_size.reserve(vertex_count);
    _component_next.reserve(vertex_count);
    _component_prev.reserve(vertex_count);
  }

  bool needs_topology_component_sync() const { return _mesh._weighting_topo != 1.0; }

  // synchronize one mesh vertex's cached component id with current disjoint-set root.
  // this keeps simplicialcomplex::compute_contraction_cost_and_location() consistent
  // without requiring component-wide eager refreshes.
  void sync_component_id_for_vertex(int vid) {
    if (!needs_topology_component_sync()) return;
    Simplex v = _mesh.getSimplex(0, vid);
    if (!v) return;
    v->_component_id = ds_find(vid);
  }

  void sync_component_ids_for_pair_if_enabled(int vid0, int vid1, bool sync_enabled) {
    if (!sync_enabled) return;
    sync_component_id_for_vertex(vid0);
    sync_component_id_for_vertex(vid1);
  }

 private:
  // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // //

  // initialize component ids from current mesh connectivity.
  // after this function, each mesh vertex has _component_id == ds_find(vertex_id).
  void initialize_component_ids_from_mesh_edges() {
    // add vertices to the disjoint set
    for (auto v : _mesh.ordered_simplices_dim(0)) {
      ds_add(v->getId());
    }

    // merge connected vertices in disjoint set by mesh edges
    for (auto e : _mesh.ordered_simplices_dim(1)) {
      Simplex v0 = e->getChild(0);
      Simplex v1 = e->getChild(1);
      assertx(v0 && v1);
      ds_try_unite(v0->getId(), v1->getId());
    }

    // assign component ids from disjoint set
    for (auto v : _mesh.ordered_simplices_dim(0)) {
      int root = ds_find(v->getId());
      // assign to the "_mesh"
      v->_component_id = root;
    }
  }

  // insert a vertex into delaunay and register id -> handle mapping.
  // precondition: vid is not currently present in _vid2vh_map.
  void insert_delaunay_vertex(int vid, const Pointd& p) {
    Point_3 pt(p[0], p[1], p[2]);
    Vertex_handle vh = _dt.insert(pt);
    vh->info() = vid;
    _vid2vh_map[vid] = vh;
  }

  // collect delaunay edges whose endpoints are in different connected components.
  // returned pairs are normalized and unique.
  PairIntSet compute_cross_edges_from_delaunay() {
    PairIntSet pairs;
    std::unordered_map<int, int> root_cache;
    root_cache.reserve(_mesh.ordered_simplices_dim(0).size());
    for (auto v : _mesh.ordered_simplices_dim(0)) {
      int vid = v->getId();
      root_cache[vid] = ds_find(vid);
    }

    for (auto e = _dt.finite_edges_begin(); e != _dt.finite_edges_end(); ++e) {
      auto cell = e->first;
      int i = e->second;
      int j = e->third;

      Vertex_handle vh0 = cell->vertex(i);
      Vertex_handle vh1 = cell->vertex(j);
      int vid0 = vh0->info();
      int vid1 = vh1->info();
      assertx(vid0 != vid1);

      Simplex v0 = _mesh.getSimplex(0, vid0);
      Simplex v1 = _mesh.getSimplex(0, vid1);
      assertx(v0 && v1);

      if (root_cache[vid0] != root_cache[vid1]) {
        pairs.insert(normalize_pair(vid0, vid1));
      }
    }
    return pairs;
  }

  // collect all mesh edges; each edge is intra-component by construction.
  PairIntSet compute_mesh_edges() {
    PairIntSet pairs;
    pairs.reserve(_mesh.ordered_simplices_dim(1).size());
    for (auto e : _mesh.ordered_simplices_dim(1)) {
      Simplex v0 = e->getChild(0);
      Simplex v1 = e->getChild(1);
      assertx(v0 && v1);
      assertx(ds_find(v0->getId()) == ds_find(v1->getId()));

      pairs.insert(normalize_pair(v0->getId(), v1->getId()));
    }
    return pairs;
  }

  // try to add a candidate edge into _edge_graph and ensure heap has up-to-date cost.
  // force=true recomputes and reinserts even if edge already exists.
  void try_add_candi_edge_and_push_heap(int vid0, int vid1, bool force = false) {
    assertx(vid0 != vid1);
    auto [a, b] = normalize_pair(vid0, vid1);
    const bool sync_enabled = needs_topology_component_sync();

    // a helper function that:
    //   tries to add a candidate edge into "_edge_graph"
    //   the first boolean indicates whether the edge is newly added
    auto try_add_candi_edge = [&]() -> std::pair<bool, Simplex> {
      // check if already exists
      Simplex va = _edge_graph.getSimplex(0, a);
      Simplex vb = _edge_graph.getSimplex(0, b);
      Simplex e;

      // if both vertices exist
      if (va && vb) {
        // if the edge exists
        if (e = va->edgeTo(vb)) {
          // return the existing edge
          // "false" indicates not newly added
          return {false, e};
        }
      }

      // ensure vertices exist
      if (!va) {
        va = _edge_graph.createSimplex(0, a);
        // the position information might be required by "_heap"
        va->setPosition(_mesh.getSimplex(0, a)->getPosition());
      }
      if (!vb) {
        vb = _edge_graph.createSimplex(0, b);
        // the position information might be required by "_heap"
        vb->setPosition(_mesh.getSimplex(0, b)->getPosition());
      }

      // supposed the edge does not exist at first
      assertx(!va->edgeTo(vb));

      // create the edge
      e = _edge_graph.createSimplex(1);
      e->setChild(0, va);
      e->setChild(1, vb);
      va->addParent(e);
      vb->addParent(e);

      // successfully added
      return {true, e};
    };

    // try to add candidate edge to "_edge_graph"
    auto [added, e] = try_add_candi_edge();
    assertx(e);

    // just added this edge
    if (added) {
      sync_component_ids_for_pair_if_enabled(a, b, sync_enabled);
      std::tie(e->cost, e->w_p0) = _mesh.compute_contraction_cost_and_location(a, b);
      // must not exist before, and insert successfully
      assertx(_heap.insert(e));
    } else {
      // this edge exists before
      if (force) {
        // force to recompute; the edge may currently be out of heap
        // (logically removed but kept in edge graph for incremental bookkeeping)
        _heap.erase(e);
        sync_component_ids_for_pair_if_enabled(a, b, sync_enabled);
        std::tie(e->cost, e->w_p0) = _mesh.compute_contraction_cost_and_location(a, b);
        assertx(_heap.insert(e));
      } else {
        // exists before, and we do not require force update
        return;
      }
    }
  }

  // one-time initialization.
  // builds component ids, builds delaunay, discovers initial candidates,
  // then computes their contraction costs and fills heap.
  void initialize() {
    reserve_state_maps(_mesh.ordered_simplices_dim(0).size());

    // prepare "_component_id" of "_mesh"
    // useful for computing cross-component candidate edges
    initialize_component_ids_from_mesh_edges();

    // insert delaunay vertices in position-sorted order for determinism
    // (CGAL Delaunay insertion order can affect triangulation for degenerate configs)
    {
      std::vector<std::pair<Pointd, int>> pos_id_pairs;
      for (auto v : _mesh.ordered_simplices_dim(0)) {
        pos_id_pairs.push_back({v->getPosition(), v->getId()});
      }
      std::sort(pos_id_pairs.begin(), pos_id_pairs.end(),
                [](const auto& a, const auto& b) {
                  auto ta = std::make_tuple(a.first[0], a.first[1], a.first[2]);
                  auto tb = std::make_tuple(b.first[0], b.first[1], b.first[2]);
                  return ta < tb;
                });
      for (const auto& [pos, vid] : pos_id_pairs) {
        insert_delaunay_vertex(vid, pos);
      }
    }

    // derive pairs from:
    //   tetrahedra edges (bridging different components)
    //   mesh edges (bridging the same components)
    auto pairs_cross = compute_cross_edges_from_delaunay();
    auto pairs_self = compute_mesh_edges();

    // add pairs into the edge graph, compute costs, and push into heap
    for (const auto& [a, b] : pairs_cross) {
      try_add_candi_edge_and_push_heap(a, b);
    }
    for (const auto& [a, b] : pairs_self) {
      try_add_candi_edge_and_push_heap(a, b);
    }
  }

  // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // //

  // append one-ring neighbors in delaunay into adjacent_vertices.
  // neighbors are adjacent by at least one tetrahedral edge.
  void append_adjacent_vertices_of_delaunay_to_set(int vid, IntSet& adjacent_vertices) const {
    auto it = _vid2vh_map.find(vid);
    assertx(it != _vid2vh_map.end());

    Vertex_handle vh = it->second;
    std::vector<Vertex_handle> neighbors;
    neighbors.reserve(32);
    _dt.adjacent_vertices(vh, std::back_inserter(neighbors));

    for (Vertex_handle nh : neighbors) {
      if (_dt.is_infinite(nh)) continue;
      int nvid = nh->info();
      assertx(nvid != vid);
      adjacent_vertices.insert(nvid);
    }
  }

  // append one-ring neighbors in mesh via edge incidences.
  void append_adjacent_vertices_of_mesh_to_set(int vid, IntSet& adjacent_vertices) const {
    Simplex v = _mesh.getSimplex(0, vid);
    assertx(v);
    auto star = v->get_star();
    for (auto e : star) {
      if (e->getDim() != 1) continue;  // via an edge
      Simplex opp = e->opp_vertex(v);
      assertx(opp);
      adjacent_vertices.insert(opp->getId());
    }
  }

  // append cross-component delaunay edges incident to vid into cross_edges.
  void append_cross_edges_from_delaunay_for_vertex(int vid, PairIntSet& cross_edges,
                                                   std::unordered_map<int, int>* root_cache = nullptr) {
    auto it = _vid2vh_map.find(vid);
    assertx(it != _vid2vh_map.end());

    Simplex v = _mesh.getSimplex(0, vid);
    assertx(v);

    Vertex_handle vh = it->second;
    std::vector<Vertex_handle> neighbors;
    neighbors.reserve(32);
    _dt.adjacent_vertices(vh, std::back_inserter(neighbors));

    auto find_root_cached = [&](int qid) {
      if (!root_cache) return ds_find(qid);
      auto qit = root_cache->find(qid);
      if (qit != root_cache->end()) return qit->second;
      int root = ds_find(qid);
      root_cache->emplace(qid, root);
      return root;
    };

    const int root_vid = find_root_cached(vid);
    for (Vertex_handle neighbor_vh : neighbors) {
      if (_dt.is_infinite(neighbor_vh)) continue;
      int neighbor_vid = neighbor_vh->info();
      assertx(neighbor_vid != vid);

      Simplex neighbor_v = _mesh.getSimplex(0, neighbor_vid);
      assertx(neighbor_v);

      if (root_vid != find_root_cached(neighbor_vid)) {
        cross_edges.insert(normalize_pair(vid, neighbor_vid));
      }
    }
  }

  // union two components while preserving `vsid` as the representative.
  // this guarantees that removing `vtid` immediately afterwards is not root-removal,
  // avoiding component-wide parent/id refresh in the collapse path.
  void merge_components_keep_vsid_root(int vsid, int vtid) {
    assertx(vsid != vtid);
    Simplex vs = _mesh.getSimplex(0, vsid);
    Simplex vt = _mesh.getSimplex(0, vtid);
    assertx(vs && vt);

    ds_try_unite_keep_first_root(vsid, vtid);
    sync_component_id_for_vertex(vsid);
    sync_component_id_for_vertex(vtid);
  }

  // try to erase a vertex from the delaunay triangulation
  void try_erase_delaunay_vertex(int vid) {
    auto it = _vid2vh_map.find(vid);
    if (it == _vid2vh_map.end()) return;
    _dt.remove(it->second);
    _vid2vh_map.erase(it);
  }

  // collapse helper that updates component ids, optional delaunay, and mesh topology.
  // this function removes vtid and keeps vsid as survivor.
  void collapse_fn(int vsid, int vtid, bool delaunay = true) {
    // update the disjoint set, and "_component_id"
    merge_components_keep_vsid_root(vsid, vtid);

    if (delaunay) {
      // update the delaunay
      try_erase_delaunay_vertex(vtid);
      try_erase_delaunay_vertex(vsid);
    }

    // update "_mesh" here to obtain the collapsed location
    Simplex vs = _mesh.getSimplex(0, vsid);
    Simplex vt = _mesh.getSimplex(0, vtid);
    assertx(vs && vt);
    _mesh.unify(vs, vt);

    // remove "vtid" from the disjoint set
    ds_remove(vtid);

    if (delaunay) {
      // insert the merged vertex back to delaunay
      insert_delaunay_vertex(vsid, vs->getPosition());
    }
  }

  // collect mesh edges incident to any vertex in vids.
  void collect_adjacent_edges_from_mesh(const IntSet& vids, PairIntSet& adjacent_edges) const {
    adjacent_edges.clear();
    adjacent_edges.reserve(vids.size() * 8);
    for (int vid : vids) {
      Simplex v = _mesh.getSimplex(0, vid);
      assertx(v);
      for (auto e : v->get_star()) {
        if (e->getDim() != 1) continue;  // via an edge
        adjacent_edges.insert(normalize_pair(e->getChild(0)->getId(), e->getChild(1)->getId()));
      }
    }
  }

  // incrementally detect candidate-edge changes around one collapse.
  // tracked candidates include:
  // - cross-component delaunay edges
  // - all mesh edges (intra-component)
  // writes:
  // - removed_pairs: present before collapse but absent after collapse
  // - affected_edges_after: all candidates in the post-collapse local domain
  void collapse_and_detect_edge_changes(int vsid, int vtid, PairIntSet& removed_pairs,
                                        PairIntSet& affected_edges_after) {
    // collect affected vertices by one-hop closure in both delaunay and mesh
    auto collect_affected_vertices = [&](std::initializer_list<int> vids, IntSet& affected_verts) {
      affected_verts.clear();
      affected_verts.reserve(std::max<std::size_t>(affected_verts.bucket_count(), vids.size() * 16));
      for (int vid : vids) {
        affected_verts.insert(vid);
        append_adjacent_vertices_of_delaunay_to_set(vid, affected_verts);
        append_adjacent_vertices_of_mesh_to_set(vid, affected_verts);
      }
    };

    // collect affected candidate edges from a given affected-vertex set
    auto collect_affected_edges_from_vertices = [&](const IntSet& affected_verts, PairIntSet& affected_edges) {
      affected_edges.clear();
      affected_edges.reserve(std::max<std::size_t>(affected_edges.bucket_count(), affected_verts.size() * 8));
      std::unordered_map<int, int> root_cache;
      root_cache.reserve(affected_verts.size() * 2);
      for (int vid : affected_verts) {
        append_cross_edges_from_delaunay_for_vertex(vid, affected_edges, &root_cache);
      }
      PairIntSet mesh_edges;
      collect_adjacent_edges_from_mesh(affected_verts, mesh_edges);
      affected_edges.insert(mesh_edges.begin(), mesh_edges.end());
    };

    // collect affected closure before collapse
    IntSet affected_verts_before;
    collect_affected_vertices({vsid, vtid}, affected_verts_before);
    PairIntSet affected_edges_before;
    collect_affected_edges_from_vertices(affected_verts_before, affected_edges_before);

    collapse_fn(vsid, vtid);  // perform the collapse ("vt" will be removed)

    // keep a consistent comparison domain across pre/post states:
    // start from pre-collapse closure, remove deleted vertex, and merge in
    // post-collapse closure around the merged vertex.
    IntSet affected_verts_after = affected_verts_before;
    affected_verts_after.erase(vtid);
    IntSet affected_verts_post_vs;
    collect_affected_vertices({vsid}, affected_verts_post_vs);
    affected_verts_after.insert(affected_verts_post_vs.begin(), affected_verts_post_vs.end());

    collect_affected_edges_from_vertices(affected_verts_after, affected_edges_after);

    // compute edges that are removed
    removed_pairs.clear();
    removed_pairs.reserve(std::max<std::size_t>(removed_pairs.bucket_count(), affected_edges_before.size()));
    for (const auto& pair : affected_edges_before) {
      if (affected_edges_after.find(pair) == affected_edges_after.end()) {
        removed_pairs.insert(pair);
      }
    }
  }

  // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // //

  // update quadrics in closure(star(vs)) after collapse.
  // this captures local boundary-status changes and keeps costs consistent.
  void update_quadrics(int vsid) {
    Simplex vs = _mesh.getSimplex(0, vsid);
    assertx(vs);

    // affected region is closure(star(vs)): all simplices incident to vs,
    // plus child edges of incident faces. this captures neighbor-neighbor edges
    // whose boundary status can change when faces are merged/removed.
    std::set<Simplex> affected_vertices;
    std::set<Simplex> affected_edges;
    std::set<Simplex> affected_nonvertex_simplices;
    affected_vertices.insert(vs);

    for (auto s : vs->get_star()) {
      int dim = s->getDim();

      if (dim == 1) {
        affected_nonvertex_simplices.insert(s);
        affected_edges.insert(s);
        Simplex oppo = s->opp_vertex(vs);
        if (oppo) {
          affected_vertices.insert(oppo);
        }
      } else if (dim == 2) {
        affected_nonvertex_simplices.insert(s);
        for (auto e : s->children()) {
          assertx(e && e->getDim() == 1);
          affected_nonvertex_simplices.insert(e);
          affected_edges.insert(e);
          Simplex v0 = e->getChild(0);
          Simplex v1 = e->getChild(1);
          assertx(v0 && v1);
          affected_vertices.insert(v0);
          affected_vertices.insert(v1);
        }
      }
    }

    for (Simplex e : affected_edges) {
      _mesh.update_boundary_edge_weighting_(e);
    }

    for (Simplex s : affected_nonvertex_simplices) {
      s->compute_native_quadric_();
    }

    for (Simplex v : affected_vertices) {
      v->compute_native_quadric_();
      v->aggregate_();
    }
  }

  // erase a candidate edge from heap if present.
  // edge simplex remains in _edge_graph for stable incremental bookkeeping.
  void try_remove_candi_edge_from_graph_and_heap(int vid0, int vid1) {
    auto [a, b] = normalize_pair(vid0, vid1);
    Simplex va = _edge_graph.getSimplex(0, a);
    Simplex vb = _edge_graph.getSimplex(0, b);
    if (!va || !vb) return;

    Simplex e = va->edgeTo(vb);
    if (!e) return;

    _heap.erase(e);
  }

  // check whether a pair is still a valid candidate in current state.
  // valid iff:
  // - it is a current mesh edge, or
  // - it is a current delaunay edge across two different components.
  bool is_valid_candidate_pair_now(int vid0, int vid1) {
    if (vid0 == vid1) return false;
    auto [a, b] = normalize_pair(vid0, vid1);

    Simplex va = _mesh.getSimplex(0, a);
    Simplex vb = _mesh.getSimplex(0, b);
    if (!va || !vb) return false;

    // mesh-edge candidates are always valid
    if (va->edgeTo(vb)) {
      return true;
    }

    // virtual candidates must stay cross-component
    if (ds_find(a) == ds_find(b)) {
      return false;
    }

    // and remain adjacent in current delaunay triangulation
    auto it = _vid2vh_map.find(a);
    if (it == _vid2vh_map.end()) return false;

    Vertex_handle vh = it->second;
    std::vector<Vertex_handle> neighbors;
    neighbors.reserve(32);
    _dt.adjacent_vertices(vh, std::back_inserter(neighbors));
    for (Vertex_handle nh : neighbors) {
      if (_dt.is_infinite(nh)) continue;
      if (nh->info() == b) return true;
    }
    return false;
  }

  // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // //

  void ds_add(int vid) {
    assertx(_component_parent.find(vid) == _component_parent.end());
    _component_parent[vid] = vid;
    _component_size[vid] = 1;
    _component_next[vid] = vid;  // single-element circular list
    _component_prev[vid] = vid;
  }

  int ds_find(int vid) {
    auto it = _component_parent.find(vid);
    if (it == _component_parent.end()) {
      _component_parent[vid] = vid;
      _component_size[vid] = 1;
      _component_next[vid] = vid;
      _component_prev[vid] = vid;
      return vid;
    }

    // path-halving: every other node points to its grandparent.
    // this keeps amortized complexity near inverse-ackermann while staying simple.
    while (_component_parent[vid] != vid) {
      int gp = _component_parent[_component_parent[vid]];
      _component_parent[vid] = gp;
      vid = gp;
    }
    return vid;
  }

  // union by size.
  // additionally splices member circular lists in o(1).
  void ds_try_unite(int a, int b) {
    int pa = ds_find(a);
    int pb = ds_find(b);
    if (pa == pb) return;

    // splice the two circular linked lists in o(1) by swapping successors
    int a_next = _component_next[pa];
    int b_next = _component_next[pb];
    std::swap(_component_next[pa], _component_next[pb]);
    // update back links for the swapped successors
    _component_prev[a_next] = pb;
    _component_prev[b_next] = pa;

    // union by size: attach smaller tree under larger tree's root
    if (_component_size[pa] < _component_size[pb]) std::swap(pa, pb);
    // now pa is the larger (or equal) root
    _component_parent[pb] = pa;
    _component_size[pa] += _component_size[pb];
  }

  // make `vid` the explicit root of its current component in o(1).
  // this avoids component-wide parent rebuilds when we need to remove another
  // vertex immediately after a collapse.
  void ds_make_root(int vid) {
    int root = ds_find(vid);
    if (root == vid) return;

    auto it_size = _component_size.find(root);
    assertx(it_size != _component_size.end());
    int size = it_size->second;

    _component_parent[vid] = vid;
    _component_parent[root] = vid;
    _component_size[vid] = size;
    _component_size.erase(root);
  }

  // unite two components while forcing the root containing `a` to remain root.
  // used by collapse path so that removing `b` is not root-removal.
  void ds_try_unite_keep_first_root(int a, int b) {
    ds_make_root(a);
    int pa = a;
    int pb = ds_find(b);
    if (pa == pb) return;

    int a_next = _component_next[pa];
    int b_next = _component_next[pb];
    std::swap(_component_next[pa], _component_next[pb]);
    _component_prev[a_next] = pb;
    _component_prev[b_next] = pa;

    _component_parent[pb] = pa;
    _component_size[pa] += _component_size[pb];
    _component_size.erase(pb);
  }

  // remove vid from disjoint set while preserving component member list integrity.
  // if vid is root of a multi-vertex component, promotes next as new root.
  void ds_remove(int vid) {
    auto it = _component_parent.find(vid);
    if (it == _component_parent.end()) return;

    int root = ds_find(vid);
    auto size_it = _component_size.find(root);
    if (size_it == _component_size.end()) return;

    int size = size_it->second;
    if (size <= 1) {
      // single-element component; erase all state
      _component_parent.erase(vid);
      _component_size.erase(root);
      _component_next.erase(vid);
      _component_prev.erase(vid);
      return;
    }

    // unlink from the circular list in o(1) using back links
    int prev = _component_prev.at(vid);
    int next = _component_next.at(vid);
    _component_next[prev] = next;
    _component_prev[next] = prev;

    if (root == vid) {
      // removed node is the root; promote a new root and rebuild parents
      int new_root = next;
      _component_parent[new_root] = new_root;
      _component_size[new_root] = size - 1;
      _component_size.erase(root);
      int cur = new_root;
      do {
        _component_parent[cur] = new_root;
        cur = _component_next.at(cur);
      } while (cur != new_root);

      // keep a forwarding parent entry for the removed node so any stale
      // parent references still resolve correctly during path compression.
      _component_parent[vid] = new_root;
    } else {
      // removed node is not the root; only size changes
      _component_size[root] = size - 1;

      // keep a forwarding parent entry for removed non-root nodes.
      // this avoids dangling parent references from other nodes that may
      // still point to `vid` due to prior path compressions.
      _component_parent[vid] = root;
    }

    // clean up maps for the removed node
    _component_next.erase(vid);
    _component_prev.erase(vid);
    // _component_size is stored only for roots; never erase by "vid" unless it is root
  }

  // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // //

 public:
  Contractor(SimplicialComplex& mesh, MinHeap& heap, bool markov) : _mesh(mesh), _heap(heap), markov(markov) {
    initialize();
  }

  bool is_valid_candidate(Simplex e) {
    if (!e || e->getDim() != 1) return false;
    Simplex c0 = e->getChild(0);
    Simplex c1 = e->getChild(1);
    if (!c0 || !c1) return false;
    return is_valid_candidate_pair_now(c0->getId(), c1->getId());
  }

  // merge two vertices, and update things accordingly
  void merge(int vsid, int vtid) {
    Simplex vs = _mesh.getSimplex(0, vsid);
    Simplex vt = _mesh.getSimplex(0, vtid);
    assertx(vs && vt);

    // update merged tree stats eagerly.
    // this is equivalent to post-collapse update because collapse_fn
    // does not depend on subtree stats.
    vs->_subtree_depth = 1 + std::max(vs->_subtree_depth, vt->_subtree_depth);
    vs->_subtree_size += vt->_subtree_size;

    if (markov) {
      // incremental branch:
      // 1) detect local candidate deltas around collapse neighborhood.
      // 2) update local quadrics/tree stats.
      // 3) remove invalid candidates.
      // 4) add or refresh affected candidates.

      // update the disjoint set ("_component_id"), delaunay, and mesh
      PairIntSet removed_pairs_local;
      PairIntSet added_or_modified_pairs_local;
      collapse_and_detect_edge_changes(vsid, vtid, removed_pairs_local, added_or_modified_pairs_local);

      // update quadrics only in the local neighborhood affected by this collapse
      update_quadrics(vsid);

      // remove all invalidated candidates
      for (const auto& [a, b] : removed_pairs_local) {
        try_remove_candi_edge_from_graph_and_heap(a, b);
      }

      // add newly created candidates and refresh affected existing candidates
      for (const auto& [a, b] : added_or_modified_pairs_local) {
        try_add_candi_edge_and_push_heap(a, b, true);
      }

    } else {
      // non-markov branch:
      // keep delaunay untouched and update candidates directly from edge_graph unification.

      // in non-markov mode, vs aggregates vt's quadric explicitly.
      // do this before collapse while vt is still valid.
      ISimplex::add_quadric_(vs, vt);

      // edge collapse in "_mesh"
      _mesh.unify(vs, vt);

      // update "_edge_graph" solely based on the edge graph
      _edge_graph.unify(_edge_graph.getSimplex(0, vsid), _edge_graph.getSimplex(0, vtid), 0, &_heap);

      // recompute the cost
      const bool sync_enabled = needs_topology_component_sync();
      for (auto e : _edge_graph.getSimplex(0, vsid)->get_star()) {
        if (e->getDim() == 0) continue;
        assertx(e->getDim() == 1);
        _heap.erase(e);
        Simplex c0 = e->getChild(0);
        Simplex c1 = e->getChild(1);
        int id0 = c0->getId();
        int id1 = c1->getId();
        sync_component_ids_for_pair_if_enabled(id0, id1, sync_enabled);
        // recompute using the actual current edge endpoints.
        // using vt_copy here would be incorrect because vt may no longer be incident to this edge.
        std::tie(e->cost, e->w_p0) = _mesh.compute_contraction_cost_and_location(id0, id1);
        assertx(_heap.insert(e));
      }
    }
  }
};

#endif  // build_libpsc

}  // namespace hh