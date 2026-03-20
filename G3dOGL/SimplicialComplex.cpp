// -*- C++ -*-  Copyright (c) Microsoft Corporation; see license.txt
#include "G3dOGL/SimplicialComplex.h"

#include <cstdint>
#include <deque>
#include <iomanip>
#include <map>
#include <unordered_set>

#include "libHh/RangeOp.h"  // compare()
#include "libHh/Set.h"
#include "libHh/Stack.h"  // also vec_contains()
#include "libHh/StringOp.h"

#include "G3dOGL/Contractor.hpp"
#include "G3dOGL/SplitRecord.h"

namespace hh {

namespace {

constexpr double k_tolerance = 1e-12;                          // scalar attribute equality tolerance
constexpr double k_undefined = static_cast<double>(BIGFLOAT);  // undefined scalar attributes
constexpr double k_degenerate_face_eps = 1e-15;
HH_STAT(Sarea_dropped);
HH_STAT(Sarea_moved);

struct GridPoint {
  std::int64_t x;
  std::int64_t y;
  std::int64_t z;

  bool operator==(const GridPoint& other) const { return x == other.x && y == other.y && z == other.z; }
};

struct GridPointHash {
  std::size_t operator()(const GridPoint& p) const {
    std::size_t h1 = std::hash<std::int64_t>{}(p.x);
    std::size_t h2 = std::hash<std::int64_t>{}(p.y);
    std::size_t h3 = std::hash<std::int64_t>{}(p.z);
    std::size_t h = h1;
    h ^= h2 + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h ^= h3 + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
  }
};

struct GridPointLess {
  bool operator()(const GridPoint& a, const GridPoint& b) const {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    return a.z < b.z;
  }
};

// VertexVoxelInfo is no longer needed; vertices that collide on the grid
// are merged rather than displaced to nearby cells.

double squared_distance(const Pointd& a, const Pointd& b) {
  const Pointd diff = a - b;
  return mag2(diff);
}

Pointd zero_point() { return Pointd(0.0, 0.0, 0.0); }

std::int64_t llround_grid(double value) { return static_cast<std::int64_t>(std::llround(value)); }

GridPoint snap_point_to_grid(const Pointd& p, double voxel_size) {
  return {llround_grid(p[0] / voxel_size), llround_grid(p[1] / voxel_size), llround_grid(p[2] / voxel_size)};
}

Pointd grid_point_to_position(const GridPoint& p, double voxel_size) {
  return Pointd(static_cast<double>(p.x) * voxel_size, static_cast<double>(p.y) * voxel_size,
                static_cast<double>(p.z) * voxel_size);
}

bool is_on_voxel_grid(const Pointd& p, double voxel_size) {
  if (voxel_size <= 0.0) return true;
  for (int d : {0, 1, 2}) {
    double snapped = std::llround(p[d] / voxel_size) * voxel_size;
    if (std::abs(p[d] - snapped) > k_tolerance * std::max(1.0, std::abs(p[d]))) {
      return false;
    }
  }
  return true;
}

void copy_simplex_metadata(Simplex src, Simplex dst) {
  dst->setVAttribute(src->getVAttribute());
  dst->setArea(src->getArea());
  if (src->get_string()) dst->set_string(src->get_string());
  dst->_subtree_depth = src->_subtree_depth;
  dst->_subtree_size = src->_subtree_size;
  dst->_weighting_quadric = src->_weighting_quadric;
  dst->_component_id = src->_component_id;
  dst->_voxel_displaced = src->_voxel_displaced;
}

// Build a canonical copy of `src` where all vertices are snapped to the voxel
// grid.  When multiple source vertices snap to the same grid cell they are
// **merged** into a single canonical vertex (the one closest to the grid-cell
// center).  Edges and faces are remapped accordingly; self-loop edges (both
// endpoints merge to the same vertex) and degenerate faces (two or more
// vertices coincide after merging) are dropped.  Duplicate edges/faces that
// arise from the many-to-one vertex mapping are also deduplicated.
//
// This is idempotent: if the source mesh already has each vertex at a unique
// grid point, the output is structurally identical to the input (modulo
// canonical ID renumbering).
void build_voxel_canonical_copy(const SimplicialComplex& src, double voxel_size, SimplicialComplex& dst) {
  dst.clear();
  dst._weighting_topo = src._weighting_topo;
  dst._weighting_balanced = src._weighting_balanced;
  dst._alpha_balanced = src._alpha_balanced;
  dst._balanced_depth = src._balanced_depth;
  dst._ratio_degeneracy = src._ratio_degeneracy;
  dst._weighting_boundary = src._weighting_boundary;

  // -- step 1: snap every source vertex to the grid and bucket by grid point --

  struct BucketVertex {
    Simplex simplex;
    Pointd original_position;
  };

  std::map<GridPoint, std::vector<BucketVertex>, GridPointLess> buckets;
  for (auto v : src.ordered_simplices_dim(0)) {
    GridPoint gp = snap_point_to_grid(v->getPosition(), voxel_size);
    buckets[gp].push_back(BucketVertex{.simplex = v, .original_position = v->getPosition()});
  }

  // -- step 2: for each bucket choose one anchor; build old-id → new-id map --
  //    anchor = vertex closest to grid-cell center (deterministic tie-break)

  // Maps every source vertex id to the canonical new vertex id.
  std::unordered_map<int, int> vertex_remap;
  vertex_remap.reserve(src.ordered_simplices_dim(0).size() * 2);

  // We will create one new vertex per bucket, in GridPointLess order.
  for (auto& [gp, bv] : buckets) {
    const Pointd grid_center = grid_point_to_position(gp, voxel_size);

    // sort: closest to grid center first; then by original position; then by id
    std::sort(bv.begin(), bv.end(), [&](const BucketVertex& a, const BucketVertex& b) {
      const double da = squared_distance(a.original_position, grid_center);
      const double db = squared_distance(b.original_position, grid_center);
      if (da != db) return da < db;
      const auto pa = std::make_tuple(a.original_position[0], a.original_position[1], a.original_position[2]);
      const auto pb = std::make_tuple(b.original_position[0], b.original_position[1], b.original_position[2]);
      if (pa != pb) return pa < pb;
      return a.simplex->getId() < b.simplex->getId();
    });

    // Create one canonical vertex for this grid point.
    Simplex new_vertex = dst.createSimplex(0);
    new_vertex->setPosition(grid_center);
    copy_simplex_metadata(bv.front().simplex, new_vertex);
    new_vertex->_voxel_displaced = false;

    // If the bucket has multiple source vertices, aggregate subtree stats.
    if (bv.size() > 1) {
      int total_size = 0;
      int max_depth = 0;
      for (const auto& v : bv) {
        total_size += v.simplex->_subtree_size;
        max_depth = std::max(max_depth, v.simplex->_subtree_depth);
      }
      new_vertex->_subtree_size = total_size;
      new_vertex->_subtree_depth = max_depth;
    }

    const int new_id = new_vertex->getId();
    for (const auto& v : bv) {
      vertex_remap.emplace(v.simplex->getId(), new_id);
    }
  }

  // -- step 3: remap edges, dropping self-loops and deduplicating --

  struct CanonicalEdgeInfo {
    Simplex simplex;
    std::array<int, 2> canonical_verts;
  };
  std::vector<CanonicalEdgeInfo> canonical_edges;
  canonical_edges.reserve(src.ordered_simplices_dim(1).size());
  for (auto e : src.ordered_simplices_dim(1)) {
    int v0 = vertex_remap.at(e->getChild(0)->getId());
    int v1 = vertex_remap.at(e->getChild(1)->getId());
    if (v0 == v1) continue;  // self-loop from merged vertices
    std::array<int, 2> verts = {std::min(v0, v1), std::max(v0, v1)};
    canonical_edges.push_back(CanonicalEdgeInfo{.simplex = e, .canonical_verts = verts});
  }
  // Sort by canonical vertex pair; for duplicates keep the one with smallest original id
  std::sort(canonical_edges.begin(), canonical_edges.end(), [](const CanonicalEdgeInfo& a, const CanonicalEdgeInfo& b) {
    if (a.canonical_verts != b.canonical_verts) return a.canonical_verts < b.canonical_verts;
    return a.simplex->getId() < b.simplex->getId();
  });

  std::map<std::array<int, 2>, Simplex> edge_by_pair;
  for (const auto& info : canonical_edges) {
    if (edge_by_pair.count(info.canonical_verts)) continue;  // deduplicate
    Simplex edge = dst.createSimplex(1);
    Simplex v0 = dst.getSimplex(0, info.canonical_verts[0]);
    Simplex v1 = dst.getSimplex(0, info.canonical_verts[1]);
    edge->setChild(0, v0);
    edge->setChild(1, v1);
    v0->addParent(edge);
    v1->addParent(edge);
    copy_simplex_metadata(info.simplex, edge);
    edge_by_pair.emplace(info.canonical_verts, edge);
  }

  // -- step 4: remap faces, dropping degenerate and deduplicating --

  struct CanonicalFaceInfo {
    Simplex simplex;
    std::array<int, 3> canonical_verts;
  };
  std::vector<CanonicalFaceInfo> canonical_faces;
  canonical_faces.reserve(src.ordered_simplices_dim(2).size());
  for (auto f : src.ordered_simplices_dim(2)) {
    auto verts = compute_defining_vertex_ids(f);
    std::array<int, 3> mapped = {vertex_remap.at(verts[0]), vertex_remap.at(verts[1]), vertex_remap.at(verts[2])};
    std::sort(mapped.begin(), mapped.end());
    // skip degenerate faces (two or more vertices coincide after merge)
    if (mapped[0] == mapped[1] || mapped[1] == mapped[2]) continue;
    canonical_faces.push_back(CanonicalFaceInfo{.simplex = f, .canonical_verts = mapped});
  }
  std::sort(canonical_faces.begin(), canonical_faces.end(), [](const CanonicalFaceInfo& a, const CanonicalFaceInfo& b) {
    if (a.canonical_verts != b.canonical_verts) return a.canonical_verts < b.canonical_verts;
    return a.simplex->getId() < b.simplex->getId();
  });

  std::set<std::array<int, 3>> seen_faces;
  for (const auto& info : canonical_faces) {
    if (!seen_faces.insert(info.canonical_verts).second) continue;  // deduplicate
    // Ensure all required edges exist (merging can create faces whose edges
    // were self-loops and thus dropped; skip such faces).
    std::array<int, 2> e01 = {info.canonical_verts[0], info.canonical_verts[1]};
    std::array<int, 2> e02 = {info.canonical_verts[0], info.canonical_verts[2]};
    std::array<int, 2> e12 = {info.canonical_verts[1], info.canonical_verts[2]};
    if (!edge_by_pair.count(e01) || !edge_by_pair.count(e02) || !edge_by_pair.count(e12)) continue;
    Simplex face = dst.createSimplex(2);
    std::array<Simplex, 3> child_edges = {edge_by_pair.at(e01), edge_by_pair.at(e02), edge_by_pair.at(e12)};
    for (int i : {0, 1, 2}) {
      face->setChild(i, child_edges[i]);
      child_edges[i]->addParent(face);
    }
    copy_simplex_metadata(info.simplex, face);
  }
}

}  // namespace

// *** ISimplex

HH_ALLOCATE_POOL(ISimplex);

Array<Simplex> ISimplex::all_faces() const {
  Array<Simplex> faces;
  Queue<Simplex> queue;
  queue.enqueue(const_cast<Simplex>(this));
  while (!queue.empty()) {
    Simplex s = queue.dequeue();
    faces.push(s);
    if (s->getDim() != 0) {
      for (Simplex c : s->children())
        if (!queue.contains(c)) queue.enqueue(c);
    }
  }
  return faces;
}

PArray<Simplex, 20> ISimplex::get_star() const {
  // Buggy: only valid for simplicial complex with DIM <= 3.
  PArray<Simplex, 20> simplices;
  Simplex s = const_cast<Simplex>(this);
  simplices.push(s);
  for (Simplex ss : s->getParents()) simplices.push(ss);
  if (s->getDim() == 0) {
    int index = simplices.num();
    // for each edge
    for_intL(i, 1, index) {
      // add in faces
      for (Simplex f : simplices[i]->getParents()) {
        bool found = false;
        // only, if not already there
        for_intL(j, index, simplices.num()) {
          if (f == simplices[j]) {
            found = true;
            break;
          }
        }
        if (!found) simplices.push(f);
      }
    }
  }
  // ensure returned simplex list has a fixed order, independent of memory addresses.
  std::sort(simplices.begin(), simplices.end(), SimplexIdCompare());
  return simplices;
}

PArray<Simplex, 20> ISimplex::faces_of_vertex() const {
  PArray<Simplex, 20> simplices;
  Simplex s = const_cast<Simplex>(this);
  assertx(s->getDim() == 0);
  for (Simplex e : s->getParents())
    for (Simplex f : e->getParents())
      if (!simplices.contains(f)) simplices.push(f);
  // ensure returned simplex list has a fixed order, independent of memory addresses.
  std::sort(simplices.begin(), simplices.end(), SimplexIdCompare());
  return simplices;
}

void ISimplex::polygon(Polygon& poly) const {
  assertx(_dim == 2);
  Simplex s0[2];
  Simplex s1;
  poly.init(0);
  s0[0] = getChild(0)->getChild(0);
  poly.push(s0[0]->getPosition().cast<float>());
  s0[1] = getChild(0)->getChild(1);
  poly.push(s0[1]->getPosition().cast<float>());
  s1 = getChild(1);
  const int child_index = s1->getChild(0) != s0[0] && s1->getChild(0) != s0[1] ? 0 : 1;
  poly.push(s1->getChild(child_index)->getPosition().cast<float>());
  return;
}

// *** SimplicialComplex

void SimplicialComplex::clear() {
  for_int(i, MAX_DIM + 1) {
    for (Simplex s : this->simplices_dim(i)) delete s;
    _simplices[i].clear();
    _free_sid[i] = 1;
  }
}

// Make *this SC a copy of the orig.
void SimplicialComplex::copy(const SimplicialComplex& orig) {
  clear();

  for_int(i, MAX_DIM + 1) {
    for (Simplex s : orig.simplices_dim(i)) {
      Simplex news = createSimplex(s->getDim(), s->getId());
      for (auto [ci, c] : enumerate<int>(s->children())) {
        Simplex this_child = getSimplex(c->getDim(), c->getId());
        news->setChild(ci, this_child);
        this_child->addParent(news);
      }
      if (s->getDim() == 0) news->setPosition(s->getPosition());

      news->_flags = s->_flags;
      news->_area = s->_area;
      news->setVAttribute(s->getVAttribute());
    }
  }

  _material_strings = orig._material_strings;

  for_int(i, MAX_DIM + 1) assertx(num(i) == orig.num(i));
}

// Check as much as possible if SimplicialComplex is ok.
void SimplicialComplex::ok() const {
  HH_ATIMER("__ok");
  for_int(i, MAX_DIM + 1) {
    for (Simplex si : this->simplices_dim(i)) {
      assertx(si->getDim() <= MAX_DIM && si->getDim() >= 0);
      for (Simplex c : si->children()) {
        if (!valid(c)) std::cerr << "Simplex " << si->getDim() << " " << si->getId() << "has invalid child.\n";

        if (!vec_contains(c->_parent, si))
          std::cerr << "Simplex " << c->getDim() << " " << c->getId() << "does not know for a parent (" << si->getDim()
                    << " " << si->getId() << ") of which it is a child.\n";
      }

      for (Simplex p : si->getParents()) {
        if (!valid(p)) std::cerr << "Simplex " << si->getDim() << " " << si->getId() << "has invalid parent.\n";

        bool found = false;
        for (Simplex pc : p->children())
          if (pc == si) found = true;

        if (!found) {
          std::cerr << "Simplex " << p->getDim() << " " << p->getId() << " does not know for a child (" << si->getDim()
                    << " " << si->getId() << ") of which it is a parent.\n";
        }
      }
      // check for duplicates
      int num_identical = 0;
      for (Simplex p1 : si->getParents()) {
        for (Simplex p2 : si->getParents()) {
          if (p1 == p2) {
            num_identical++;
            continue;
          }
          if (equal(p1, p2))
            std::cerr << "Simplex " << si->getDim() << " " << si->getId() << " has duplicate parents.\n";
        }
      }
      if (num_identical != narrow_cast<int>(si->_parent.size()))
        std::cerr << "Simplex " << si->getDim() << " " << si->getId() << " has duplicate parents.\n";
    }
  }
}

// Return simplicial complex representing starbar of simplex s.
// id's of simplices in resulting SC are also meaningful in *this.
void SimplicialComplex::starbar(Simplex s, SimplicialComplex& res) const {
  assertx(s->getDim() == 0);

  res.clear();
  // commented primarily for unify record reasons
  // res._material_strings = _material_strings;

  // note: it cycles through lower dimension parents first
  for (Simplex curr : s->get_star()) {
    Simplex news;

    // create a copy of current simplex in resulting simplicial complex
    news = assertx(res.createSimplex(curr->getDim(), curr->getId()));
    if (news->getDim() == 0) news->setPosition(s->getPosition());
    news->setVAttribute(curr->getVAttribute());
    news->_flags = curr->_flags;
    news->_area = curr->_area;

    for (auto [ci, c] : enumerate<int>(curr->children())) {
      Simplex res_child = res.getSimplex(c->getDim(), c->getId());
      // note some children might not be ancestors of s
      if (!res_child) {
        // if so, first create them
        res_child = res.createSimplex(c->getDim(), c->getId());
        if (c->getDim() == 0) res_child->setPosition(c->getPosition());
        res_child->setVAttribute(c->getVAttribute());
        res_child->_flags = c->_flags;
        res_child->_area = c->_area;

        // update child pointers (all must exist)
        for (auto [cci, cc] : enumerate<int>(c->children())) {
          Simplex res_childchild = res.getSimplex(cc->getDim(), cc->getId());
          assertx(res_childchild);  // all must exist
          res_child->setChild(cci, res_childchild);
          res_childchild->addParent(res_child);
        }
      }
      news->setChild(ci, res_child);
      res_child->addParent(news);
    }
  }
}

// Perform union of two simplicial complex where id's are meaningful within this SC.
void SimplicialComplex::scUnion(const SimplicialComplex& s1, const SimplicialComplex& s2,
                                SimplicialComplex& res) const {
  res.copy(s1);

  for_int(i, MAX_DIM + 1) {
    for (Simplex s2_s : s2.simplices_dim(i)) {
      Simplex res_news = res.getSimplex(s2_s->getDim(), s2_s->getId());

      // create it if it doesn't exist in res
      if (!res_news) {
        res_news = res.createSimplex(s2_s->getDim(), s2_s->getId());
        if (res_news->getDim() == 0) res_news->setPosition(s2_s->getPosition());
        res_news->setVAttribute(s2_s->getVAttribute());
        res_news->_flags = s2_s->_flags;
        res_news->_area = s2_s->_area;
        // update its links
        for (auto [s2_ci, s2_c] : enumerate<int>(s2_s->children())) {
          Simplex res_child = res.getSimplex(s2_c->getDim(), s2_c->getId());
          assertx(res_child);  // all children must exist

          res_news->setChild(s2_ci, res_child);

          // update p
          res_child->addParent(res_news);
        }
      }
    }
  }
}

// Given a simplex s, return with res containing all simplices adjacent to s.
void SimplicialComplex::star(Simplex s, Array<Simplex>& res) const {
  res.init(0);
  // res.push(s);
  for (Simplex curr : s->get_star()) res.push(curr);
}

// Removes simplex and all of its ancestors from SC.
void SimplicialComplex::destroySimplex(Simplex s, int area_test) {
  assertx(valid(s));
  if (area_test) assertx(s->getArea() == 0.f);

  // remove all references from its children
  for (Simplex c : s->children())
    if (c) vec_remove_ordered(c->_parent, s);

  Stack<Simplex> todel;
  // find all parents to be removed
  for (Simplex p : s->getParents())
    if (p) todel.push(p);

  // destroy parents!
  while (!todel.empty()) {
    Simplex del = todel.pop();
    destroySimplex(del);
  }

  _free_sid[s->getDim()] -= 1;
  _simplices[s->getDim()].remove(s->getId());
  delete s;
}

// Lisp-like equal comparison: two simplices equal if their children are equal.
// Note eq => equal but NOT equal => eq   (more expensive than eq).
bool SimplicialComplex::equal(Simplex s1, Simplex s2) const {
  if (s1->getDim() != s2->getDim()) return false;

  if (s1 == s2 || s1->getId() == s2->getId()) return true;

  int num_equal = 0;
  for (Simplex c1 : s1->children())
    for (Simplex c2 : s2->children())
      if (equal(c1, c2)) num_equal++;

  assertx(num_equal <= s1->getDim() + 1);

  if (num_equal == s1->getDim() + 1) return true;

  return false;
}

// Lisp like eq comparison: two simplices eq if they are identical or their children are identical eq.
// Note: children must be distinct for this to work i.e. no degenerate simplices allowed.
// Note: no duplicate simplices allowed.
bool SimplicialComplex::eq1simp(Simplex s1, Simplex s2) const {
  assertx(s1->getDim() == 1);
  assertx(s2->getDim() == 1);
  Simplex s1v1 = s1->getChild(0);
  Simplex s1v2 = s1->getChild(1);
  Simplex s2v1 = s2->getChild(0);
  Simplex s2v2 = s2->getChild(1);

  return (s1v1 == s2v1 && s1v2 == s2v2) || (s1v1 == s2v2 && s1v2 == s2v1);
}

bool SimplicialComplex::eq2simp(Simplex s1, Simplex s2) const {
  assertx(s1->getDim() == 2);
  assertx(s2->getDim() == 2);

  Simplex s1verts[3];
  Simplex s2verts[3];
  s1->vertices(s1verts);
  s2->vertices(s2verts);

  return ((s1verts[0] == s2verts[0] && ((s1verts[1] == s2verts[1] && s1verts[2] == s2verts[2]) ||
                                        (s1verts[1] == s2verts[2] && s1verts[2] == s2verts[1]))) ||
          (s1verts[0] == s2verts[1] && ((s1verts[1] == s2verts[2] && s1verts[2] == s2verts[0]) ||
                                        (s1verts[1] == s2verts[0] && s1verts[2] == s2verts[2]))) ||
          (s1verts[0] == s2verts[2] && ((s1verts[1] == s2verts[0] && s1verts[2] == s2verts[1]) ||
                                        (s1verts[1] == s2verts[1] && s1verts[2] == s2verts[0]))));
}

void SimplicialComplex::unify(Simplex vs, Simplex vt, int propagate_area, MinHeap* heap) {
  assertx(vs->getDim() == 0 && vt->getDim() == 0);

  std::map<Simplex, int> modified_simplices;
  if (heap) {
    for (auto v : {vs, vt}) {
      for (auto s : v->get_star()) {
        if (s->getDim() == 0) {
          continue;
        }
        assertx(s->getDim() == 1);
        modified_simplices[s] = -1;
      }
    }
  }

  Simplex both = vs->edgeTo(vt);
  Stack<Simplex> check_principal;

  // propagate material
  if (both) {
    // new principal edges
    for (Simplex s : both->get_star()) {
      if (s == both) continue;

      assertx(s->getDim() == 2);
      for (Simplex c : s->children())
        if (c->getParents().size() == 1) c->setVAttribute(s->getVAttribute());
    }

    // new principal verts
    if (vs->getParents().size() == 1) vs->setVAttribute(both->getVAttribute());
  }

  double cmp_area = 0.0;

  if (propagate_area) {
    for (Simplex s : vs->get_star()) {
      if (s->isPrincipal()) {
        assertx(s->getArea() > 0.f);
      } else {
        assertx(s->getArea() == 0.f);
      }
    }

    for (Simplex s : vt->get_star()) {
      if (s->isPrincipal()) {
        assertx(s->getArea() > 0.f);
      } else {
        assertx(s->getArea() == 0.f);
      }
    }

    if (both) {
      // distribute area
      for (Simplex spx : both->get_star()) {
        // consider only principal simplices
        if (!spx->isPrincipal()) continue;

        bool area_given = false;
        bool drop_area = false;

        // give it's area to manifold adjacent component
        for (Simplex c : spx->children()) {
          if (c == both) continue;

          if (c->isManifold()) {
            Simplex spx_adj = nullptr;
            // find adjacent component
            for (Simplex p : c->getParents()) {
              if (p != spx) {
                spx_adj = p;
                break;
              }
            }
            assertx(spx_adj && spx_adj->isPrincipal());

            Sarea_moved.enter(spx->getArea());
            Sarea_dropped.enter(spx->getArea());
            spx_adj->setArea(spx_adj->getArea() + spx->getArea());
            spx->setArea(0.f);
            area_given = true;
            break;
          }

          // if all children are boundary the ancestor will be a
          // a principal simplex and we can give its area away
          // to the ancestor
          if (!c->is_boundary()) {
            // otherwise drop the area
            drop_area = true;
            Sarea_dropped.enter(spx->getArea());
            spx->setArea(0.f);
          }
        }

        // if area should not be dropped and is not given away
        if (!drop_area && !area_given) {
          assertx(spx->getDim() == 1 || spx->getDim() == 2);

          Simplex spx_adj = nullptr;
          if (spx->getDim() == 2) {
            // facet
            spx_adj = spx->opp_edge(vt);
          }

          if (spx->getDim() == 1) {
            // vert
            spx_adj = vs;
          }

          assertx(spx_adj);
          check_principal.push(spx_adj);

          Sarea_dropped.enter(spx->getArea());
          Sarea_moved.enter(spx->getArea());
          spx_adj->setArea(spx_adj->getArea() + spx->getArea());
          spx->setArea(0.f);
        }
      }
    }

    if (vt->isPrincipal()) {
      cmp_area = vt->getArea();
      Sarea_dropped.enter(cmp_area);
      vt->setArea(0.f);
    } else if (vs->isPrincipal()) {
      cmp_area = vs->getArea();
      Sarea_dropped.enter(cmp_area);
      vs->setArea(0.f);
    }
  }

  if (both) destroySimplex(both, propagate_area);
  // both = nullptr;  // now undefined

  // remap all references of vt to vs in simplices adjacent to vt.
  std::vector<Simplex> worklist[MAX_DIM + 1];
  for (Simplex s : vs->get_star())
    if (s != vs) worklist[s->getDim()].push_back(s);

  Stack<Simplex> affected_spx[MAX_DIM + 2];
  replace(vt, vs, affected_spx[1]);

  // remove vt
  vt->_parent.clear();
  destroySimplex(vt, propagate_area);
  // vt = nullptr;  // now undefined

  assertx(Sarea_moved.sum() <= Sarea_dropped.sum());

  // remove duplicate simplices
  int dim = 1;
  while (!affected_spx[dim].empty()) {
    Simplex vs_new_ancestor = affected_spx[dim].pop();
    for (Simplex vs_ancestor : worklist[dim]) {  // was ForStack which went in reverse order
      assertx(vs_ancestor->getDim() == vs_new_ancestor->getDim());
      assertx(vs_new_ancestor != vs_ancestor);

      if (eq1simp(vs_new_ancestor, vs_ancestor)) {
        // remove duplicate
        if (propagate_area) {
          if (vs_new_ancestor->isPrincipal() && vs_ancestor->isPrincipal()) {
            Sarea_dropped.enter(vs_new_ancestor->getArea());
            Sarea_moved.enter(vs_new_ancestor->getArea());
            vs_ancestor->setArea(vs_ancestor->getArea() + vs_new_ancestor->getArea());
            vs_new_ancestor->setArea(0.f);
          } else {
            if (vs_new_ancestor->getArea() > 0.f) {
              assertx(vs_new_ancestor->isPrincipal());
              Sarea_dropped.enter(vs_new_ancestor->getArea());
              vs_new_ancestor->setArea(0.f);
            }
            if (vs_ancestor->getArea() > 0.f) {
              assertx(vs_ancestor->isPrincipal());
              Sarea_dropped.enter(vs_new_ancestor->getArea());
              vs_ancestor->setArea(0.f);
            }
          }
        }

        replace(vs_new_ancestor, vs_ancestor, affected_spx[dim + 1]);
        if (dim + 1 == 3) assertx(affected_spx[3].empty());
        vs_new_ancestor->_parent.clear();
        destroySimplex(vs_new_ancestor, propagate_area);
        break;
      }
    }
  }

  if (heap) {
    for (auto s : vs->get_star()) {
      if (s->getDim() == 0) {
        continue;
      }
      assertx(s->getDim() == 1);
      modified_simplices[s] += 1;
    }

    // update the heap
    for (auto [e, cnt] : modified_simplices) {
      if (cnt == 0) {
        assertx(e->getDim() == 1);
        continue;
      } else if (cnt == 1) {
        assertnever("should never add a new contraction pair");
      } else {
        assertx(cnt == -1);
        // "e" have been destroyed, cannot access
        heap->erase(e);
      }
    }

    return;
  }

  dim = 2;
  while (!affected_spx[dim].empty()) {
    Simplex vs_new_ancestor = affected_spx[dim].pop();
    for (Simplex vs_ancestor : worklist[dim]) {  // was ForStack which went in reverse order
      assertx(vs_ancestor->getDim() == vs_new_ancestor->getDim());
      assertx(vs_new_ancestor != vs_ancestor);

      if (eq2simp(vs_new_ancestor, vs_ancestor)) {
        replace(vs_new_ancestor, vs_ancestor, affected_spx[dim + 1]);
        if (dim + 1 == 3) assertx(affected_spx[3].empty());

        // remove duplicate
        if (propagate_area) {
          Sarea_dropped.enter(vs_new_ancestor->getArea());
          Sarea_moved.enter(vs_new_ancestor->getArea());
          vs_ancestor->setArea(vs_ancestor->getArea() + vs_new_ancestor->getArea());
          vs_new_ancestor->setArea(0.f);
        }

        vs_new_ancestor->_parent.clear();
        destroySimplex(vs_new_ancestor, propagate_area);
        break;
      }
    }
  }

  // distribute vt area if any
  if (propagate_area && cmp_area != 0.f) {
    for (Simplex s : vs->get_star())
      if (s->isPrincipal()) {
        Sarea_moved.enter(cmp_area);
        s->setArea(s->getArea() + cmp_area);
        break;
      }
  }

  if (propagate_area) {
    assertx(Sarea_moved.sum() <= Sarea_dropped.sum());
    while (!check_principal.empty()) {
      Simplex s = check_principal.pop();
      assertx(s->isPrincipal());
    }
  }
}

void SimplicialComplex::replace(Simplex src, Simplex tgt, Stack<Simplex>& affected_parents) {
  // remove references from children
  for (auto [ci, c] : enumerate<int>(src->children())) {
    if (!c) continue;
    src->_child[ci] = nullptr;
    vec_remove_ordered(c->_parent, src);
  }

  // replace references from parents
  // and add reference to parent from tgt
  for (Simplex p : src->getParents()) {
    if (!p) continue;
    for (auto [ci, c] : enumerate<int>(p->children()))
      if (c == src) p->setChild(ci, tgt);

    if (!affected_parents.contains(p)) affected_parents.push(p);
    tgt->addParent(p);
  }
}

void SimplicialComplex::write(std::ostream& os) const {
  const auto old_precision = os.precision();
  os << std::setprecision(17);

  // dump materials

  if (_material_strings.num()) {
    os << "[Attributes]\n";
    for_int(attrid, _material_strings.num()) os << _material_strings[attrid] << "\n";
    os << "[EndAttributes]\n";
  }

  // dump simplicial complex
  for_int(dim, MAX_DIM + 1) {
    for (Simplex s : this->ordered_simplices_dim(dim)) {
      os << "Simplex " << dim << " " << s->getId() << "  ";
      if (dim == 0) {
        Pointd pos = s->getPosition();
        os << " " << pos[0] << " " << pos[1] << " " << pos[2];
      } else {  // dim != 0
        // iterate over children
        for (Simplex c : s->children()) os << " " << c->getId();
      }

      // print vattributes
      string out;
      if (s->get_string()) {
        if (!out.empty()) out += " ";
        out += s->get_string();
      }

      if (s->isPrincipal()) {
        assertx(s->getVAttribute() != -1);
        if (!out.empty()) out += " ";
        out += sform("attrid=%d", s->getVAttribute());
      }

      if (s->isPrincipal()) {
        if (!out.empty()) out += " ";
        out += sform("area=%.17g", s->getArea());
      }

      if (!out.empty()) os << "  {" << out << "}";
      assertx(os << "\n");
    }
  }

  os << std::setprecision(old_precision);
}

void SimplicialComplex::read(std::istream& is) {
  string line;
  auto parse_line = &SimplicialComplex::readLine;
  for (;;) {
    if (!my_getline(is, line)) break;
    if (line == "") continue;
    if (line == "#") break;        // done parsing simplex, before vsplit records
    if (line[0] == '#') continue;  // skip comment
    // if attribute change state and read next line
    if (starts_with(line, "[Attributes]")) {
      parse_line = &SimplicialComplex::attrReadLine;
      continue;
    }

    if (starts_with(line, "[EndAttributes]")) {
      parse_line = &SimplicialComplex::readLine;
      continue;
    }
    (this->*parse_line)(line.c_str());
  }
}

void SimplicialComplex::readLine(const char* str) {
  char* sline = const_cast<char*>(str);
  if (sline[0] == '#') return;
  char* va_field = strchr(sline, '{');
  if (va_field) {
    *va_field++ = 0;
    char* s = strchr(va_field, '}');
    if (!s) {
      if (Warning("No matching '}'")) SHOW(sline, va_field);
      va_field = nullptr;
    } else {
      *s = 0;
    }
  }
  if (const char* s = after_prefix(sline, "Simplex ")) {
    const int dim = int_from_chars(s), sid = int_from_chars(s);
    Simplex sd = assertx(createSimplex(dim, sid));
    // read and update children pointers
    if (dim == 0) {
      // read position
      Pointd pos;
      for_int(i, 3) pos[i] = double_from_chars(s);
      sd->setPosition(pos);
    } else {  // dim != 0
      // read connectivity
      for_int(i, dim + 1) {
        const int child = int_from_chars(s);
        Simplex spxChild = assertx(getSimplex(dim - 1, child));
        sd->setChild(i, spxChild);
      }
      // Update children's parent pointers
      for (Simplex c : sd->children()) c->addParent(sd);
    }
    while (std::isspace(*s)) s++;
    assert_no_more_chars(s);

    // read in vattributes
    sd->set_string(va_field);

    string str2;
    const char* attrid = GMesh::string_key(str2, va_field, "attrid");
    if (attrid) sd->setVAttribute(to_int(attrid));

    const char* area = GMesh::string_key(str2, va_field, "area");
    if (area) sd->setArea(to_double(area));
    return;
  }
  if (const char* s = after_prefix(sline, "Unify ")) {
    const int vi1 = int_from_chars(s), vi2 = int_from_chars(s);
    assert_no_more_chars(s);
    unify(getSimplex(0, vi1), getSimplex(0, vi2));
    return;
  }
  assertnever("Unrecognized line '" + string(sline) + "'");
}

void SimplicialComplex::attrReadLine(const char* str) {
  if (str[0] == '#') return;

  _material_strings.push(str);
}

// Construct skeleteon from this SC containing only simplices of dimension <= dim.
void SimplicialComplex::skeleton(int dim) {
  if (dim + 1 > MAX_DIM) return;
  for (Simplex s : Array<Simplex>{this->simplices_dim(dim + 1)}) destroySimplex(s);
}

#if 0
void SimplicialComplex::attrReadLine(char* sline) {
  if (sline[0] == '#') return;
  if (const char* s = after_prefix(sline, "Simplex ")) {
    const int dim = int_from_chars(s), sid = int_from_chars(s);
    // update position for 0-simplices
    if (dim == 0) {
      Point pos;
      for_int(i, 3) pos[i] = float_from_chars(s);
      getSimplex(dim, sid)->setPosition(pos);
    }
    // read and update color for 0,1,2-simp
    if (*s) {
      Point rgb;
      for_int(i, 3) rgb[i] = float_from_chars(s);
      getSimplex(dim, sid)->setColor(rgb);
    }
    assert_no_more_chars();
  }
}
#endif

int SimplicialComplex::compare_normal(const GMesh& mesh, Corner c1, Corner c2) {
  // if nothing to compare
  assertx(c1 && c2);
  Vector n1, n2;
  assertx(parse_key_vec(mesh.get_string(c1), "normal", n1));
  assertx(parse_key_vec(mesh.get_string(c2), "normal", n2));
  return compare(n1, n2, k_tolerance);
}

// Allocates new simplex of dimension dim and inserts into simplicial complex.
// Note: children and parents if any must be specified later.
Simplex SimplicialComplex::createSimplex(int dim) {
  int sid = _free_sid[dim]++;
  Simplex s = new ISimplex(dim, sid);
  // _child = nullptr's; _parent = empty

  _simplices[dim].enter(sid, s);

  return s;
}

// Same as createSimplex(dim) but use the given id instead of next available one.
Simplex SimplicialComplex::createSimplex(int dim, int id) {
  int sid = id;
  Simplex s = new ISimplex(dim, id);
  // _child = nullptr's; _parent = empty

  _simplices[dim].enter(sid, s);

  _free_sid[dim] = id >= _free_sid[dim] ? id + 1 : _free_sid[dim];

  return s;
}

std::tuple<std::array<double, 3>, std::vector<py::dict>, std::vector<double>>
SimplicialComplex::perform_simplification(bool markov, double voxel_size) {
  if (voxel_size > 0.0) {
    SimplicialComplex canonical_mesh;
    build_voxel_canonical_copy(*this, voxel_size, canonical_mesh);
    canonical_mesh._active_voxel_size = voxel_size;
    return canonical_mesh.perform_simplification_impl(markov);
  }

  _active_voxel_size = 0.0;
  return perform_simplification_impl(markov);
}

/* perform simplicial complex simplification, until a single vertex */
std::tuple<std::array<double, 3>, std::vector<py::dict>, std::vector<double>>
SimplicialComplex::perform_simplification_impl(bool markov) {
  // update "_total_initial_vertices"
  _total_initial_vertices = 0;
  for (auto v : ordered_simplices_dim(0)) {
    _total_initial_vertices += v->_subtree_size;
  }

  // in markov mode, update boundary edge weightings first
  if (markov) {
    // iterate through all edges
    for (auto e : ordered_simplices_dim(1)) {
      update_boundary_edge_weighting_(e);
    }
  }

  // compute quadric for each simplex
  for (int d : {0, 1, 2}) {
    for (auto s : ordered_simplices_dim(d)) {
      s->compute_native_quadric_();
    }
  }

  // aggregate into the vertices
  for (auto v : ordered_simplices_dim(0)) {
    v->aggregate_();
  }

  // the min heap
  MinHeap heap;

  // useful for edge collapse
  Contractor contractor(*this, heap, markov);

  // all the edge collapse recordings
  std::vector<EcolRecord> ecol_record_lst;

  // see if this is a valid vertex unification
  auto is_valid_ecol = [&](int vsid, int vtid, Pointd p_new) -> bool {
    Simplex vs = getSimplex(0, vsid);
    Simplex vt = getSimplex(0, vtid);
    auto face_s = vs->faces_of_vertex();
    auto face_t = vt->faces_of_vertex();
    std::set<Simplex> face_set_s(face_s.begin(), face_s.end());
    std::set<Simplex> face_set_t(face_t.begin(), face_t.end());

    // symmetric difference
    std::set<Simplex> affected_faces = [&] {
      std::set<Simplex> result;
      for (auto f : face_set_s)
        if (!face_set_t.count(f)) result.insert(f);
      for (auto f : face_set_t)
        if (!face_set_s.count(f)) result.insert(f);
      return result;
    }();

    // get vertex position (possibly replaced)
    auto get_v_pos = [&](Simplex v, bool replace = true) -> Pointd {
      assertx(v->getDim() == 0);
      if (replace) {
        int vid = v->getId();
        if (vid == vsid || vid == vtid) {
          return p_new;
        }
      }
      return v->getPosition();
    };

    // get face normals
    auto get_f_nor = [&](Simplex f, bool replace = true) -> Pointd {
      assertx(f->getDim() == 2);
      Simplex v012[3];
      f->vertices(v012);
      Pointd v0 = get_v_pos(v012[0], replace);
      Pointd v1 = get_v_pos(v012[1], replace);
      Pointd v2 = get_v_pos(v012[2], replace);
      if (is_degenerate_triangle_(v0, v1, v2)) return zero_point();
      Pointd normal = cross(v1 - v0, v2 - v0);
      normal.normalize();
      return normal;
    };

    // compute original face normals
    for (Simplex f : affected_faces) {
      Pointd nor_before = get_f_nor(f, false);
      Pointd nor_after = get_f_nor(f, true);
      if (mag2(nor_before) == 0.0 || mag2(nor_after) == 0.0) continue;
      // detect flipped faces
      //   the threshold is from: https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification/blob/65df07dc54766e3ee480482f1c881a62767831cc/src.gl/Simplify.h#L219
      if (dot(nor_before, nor_after) < 0.2) {
        return false;
      }
    }

    // TODO
    //   detect self-intersection
    //   but we may never implement it, because the original input mesh may already contain self-intersection

    return true;
  };

  // the list of contraction costs
  std::vector<double> cost_lst;

  // for loop, until only one vertex left
  while (!heap.empty()) {
    // get the candidate with the lowest cost
    Simplex pair = heap.min();
    assertx(pair->getDim() == 1);

    // lazy invalidation is only enabled in markov mode.
    // non-markov branch relies on a stronger invariant that every edge in
    // _edge_graph star(vs) is present in _heap before local recomputation.
    if (markov && !contractor.is_valid_candidate(pair)) {
      assertx(heap.erase(pair));
      continue;
    }

    // get the values
    double cost = pair->cost;
    int vsid = pair->getChild(0)->getId();
    int vtid = pair->getChild(1)->getId();
    double w_p0 = pair->w_p0;
    if (w_p0 == 0.0) {
      std::swap(vsid, vtid);
    }

    Simplex vs = getSimplex(0, vsid);
    Simplex vt = getSimplex(0, vtid);

    // compute new location
    Pointd delta_p = vt->getPosition() - vs->getPosition();
    if (w_p0 == 0.5) {
      delta_p *= 0.5;
    }
    Pointd p_new = vt->getPosition() - delta_p;

    // In voxel mode, re-snap delta_p and p_new to exact grid multiples to
    // eliminate floating-point drift that would break S1==S2 idempotency.
    if (_active_voxel_size > 0.0) {
      for (int d = 0; d < 3; ++d) {
        delta_p[d] = std::llround(delta_p[d] / _active_voxel_size) * _active_voxel_size;
        p_new[d] = std::llround(p_new[d] / _active_voxel_size) * _active_voxel_size;
      }
      assertx(is_on_voxel_grid(vs->getPosition(), _active_voxel_size));
      assertx(is_on_voxel_grid(vt->getPosition(), _active_voxel_size));
    }

    // see if it will cause flipped faces
    constexpr double FP64_INF = std::numeric_limits<double>::max();
    if (cost < FP64_INF && !is_valid_ecol(vsid, vtid, p_new)) {
      assertx(heap.erase(pair));
      pair->cost = FP64_INF;  // penalize such a collapse
      assertx(heap.insert(pair));
      continue;
    }

    // record the collapse step
    auto ecol_record = compute_ecol_record(vsid, vtid, w_p0);
    ecol_record_lst.push_back(ecol_record);

    // update source position, because we will later unify and discard "vt"
    vs->setPosition(p_new);
    if (_active_voxel_size > 0.0) {
      vs->_voxel_displaced = vs->_voxel_displaced || vt->_voxel_displaced || w_p0 == 0.5;
    }

    // perform edge collapse
    contractor.merge(vsid, vtid);

    // append the old cost
    cost_lst.push_back(cost);
  }

  // check the simplices' number
  //   vertices have not been fully removed (because we need to access "getId()" before)
  //   but edges/faces are completely removed
  assertx(num(0) == 1 && num(1) == 0 && num(2) == 0);

  // reverse the collapse operations
  std::reverse(ecol_record_lst.begin(), ecol_record_lst.end());

  // reverse the cost list
  std::reverse(cost_lst.begin(), cost_lst.end());

  // create the vertex id remapping table
  std::unordered_map<int, int> remap;  // old vert id --> increasing vert id
  int vert_idx_inc = 1;                // >= 1
  for (const auto& record : ecol_record_lst) {
    // if insertion is new, then vertex id increases
    vert_idx_inc += int(std::get<1>(remap.emplace(record.vsid, vert_idx_inc)));
    vert_idx_inc += int(std::get<1>(remap.emplace(record.vtid, vert_idx_inc)));
  }

  // the vertex id of the starting point
  int start_v_idx = ecol_record_lst[0].vsid;

  // remap the vertex index
  for (auto& [vsid, vtid, position_bit, delta_p, topo_record] : ecol_record_lst) {
    // apply "remap" and update vertex ids
    vsid = remap.at(vsid);
    vtid = remap.at(vtid);

    // apply "remap" and update "topo_record"
    constexpr int INT_MAX_VAL = std::numeric_limits<int>::max();
    for (auto& [defining_vertex_ids, label, dim, simplex_id] : topo_record) {
      for (auto& v : defining_vertex_ids) {
        if (v != INT_MAX_VAL) {
          v = remap.at(v);
        }
      }
      std::sort(defining_vertex_ids.begin(), defining_vertex_ids.end());
    }
  }

  // the starting location
  Pointd starting_point_ = getSimplex(0, start_v_idx)->getPosition();
  if (_active_voxel_size > 0.0) {
    // Re-snap to exact grid representation.
    for (int d = 0; d < 3; ++d)
      starting_point_[d] = std::llround(starting_point_[d] / _active_voxel_size) * _active_voxel_size;
  }
  std::array<double, 3> starting_point{starting_point_[0], starting_point_[1], starting_point_[2]};

  // compute string code by reconstructing the simplicial complex
  SimplicialComplex K_recon;
  std::stringstream ss;
  ss << std::setprecision(17);
  ss << "Simplex 0 1 " << starting_point[0] << " " << starting_point[1] << " " << starting_point[2] << "\n";
  K_recon.read(ss);

  // the output list for vertex splitting operations
  std::vector<py::dict> vsplit_lst;

  // compute the string code during reconstruction of the simplicial complex
  //   this is because :
  //     we have reordered the vertices, making them starting from 1 and increasing during vertex splitting
  //     the simplex order of the string code really matters
  //     we need to reconstruct to obtain the order
  //     and use the order to compute string code
  for (const auto& ecol_record : ecol_record_lst) {
    const auto& [vsid, vtid, position_bit, delta_p, topo_record_lst] = ecol_record;

    Simplex vs = assertx(K_recon.getSimplex(0, vsid));

    // compute the ordered queue of adjacent simplices
    Pqueue<Simplex> pq[3];
    for (auto s : vs->get_star()) {
      pq[s->getDim()].enter_unsorted(s, static_cast<double>(s->getId()));
    }

    // to find the topological label for the query simplex
    using TopoKey = std::pair<int, DefiningVertIds>;
    std::map<TopoKey, std::deque<int>> labels_by_key;
    std::vector<const TopoRecord*> topo_records_sorted;
    topo_records_sorted.reserve(topo_record_lst.size());
    for (const auto& record : topo_record_lst) {
      topo_records_sorted.push_back(&record);
    }
    std::sort(topo_records_sorted.begin(), topo_records_sorted.end(), [](const TopoRecord* a, const TopoRecord* b) {
      if (a->dim != b->dim) return a->dim < b->dim;
      return a->simplex_id < b->simplex_id;
    });
    for (const TopoRecord* record : topo_records_sorted) {
      labels_by_key[{record->dim, record->defining_vertex_ids}].push_back(record->label);
    }

    auto find_label = [&](Simplex s) {
      const TopoKey key{s->getDim(), compute_defining_vertex_ids(s)};
      auto it = labels_by_key.find(key);
      assertx(it != labels_by_key.end() && !it->second.empty());
      const int label = it->second.front();
      it->second.pop_front();
      return label;
    };

    // compute the string code
    std::string string_code = "";
    for (int dim : {0, 1, 2}) {
      pq[dim].sort();
      while (!pq[dim].empty()) {
        Simplex s = pq[dim].remove_min();
        if (dim == 0) {
          assertx(s == vs);
          string_code += std::string("9") + std::to_string(find_label(s));
        } else if (dim == 1) {
          string_code += std::string("8") + std::to_string(find_label(s));
        } else {
          assertx(dim == 2);
          string_code += std::string("7") + std::to_string(find_label(s));
        }
      }
    }

    // prepare the python dict
    vsplit_lst.push_back(ecol_record.to_dict(string_code));  // convert to python dictionary

    // build the "SplitRecord" object
    std::stringstream ss;
    ss << vsid << " " << vtid << "\n";
    ss << string_code << "\n";
    ss << position_bit << " " << delta_p[0] << " " << delta_p[1] << " " << delta_p[2] << "\n";
    ss << "-1\n-1\n";
    SplitRecord split_record;
    split_record.read(ss);

    // apply vertex split
    split_record.applySplit(K_recon);
  }

  // output the starting location as well as the operation sequence
  return std::make_tuple(starting_point, vsplit_lst, cost_lst);
}

}  // namespace hh
