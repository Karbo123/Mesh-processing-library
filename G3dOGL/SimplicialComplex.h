// -*- C++ -*-  Copyright (c) Microsoft Corporation; see license.txt
#ifndef MESH_PROCESSING_G3DOGL_SIMPLICIALCOMPLEX_H_
#define MESH_PROCESSING_G3DOGL_SIMPLICIALCOMPLEX_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

#ifdef BUILD_LIBPSC
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <limits>
#include <sstream>
namespace py = pybind11;
#endif

#include "libHh/A3dStream.h"
#include "libHh/Flags.h"
#include "libHh/GMesh.h"
#include "libHh/Map.h"
#include "libHh/MeshOp.h"
#include "libHh/PArray.h"
#include "libHh/Polygon.h"
#include "libHh/Pqueue.h"
#include "libHh/Queue.h"
#include "libHh/Stack.h"
#include "libHh/Timer.h"

namespace hh {

class ISimplex;
using Simplex = ISimplex*;

// Double precision types for high-accuracy quadric and cost calculations
using Pointd = Vec3<double>;
using Matrix3d = std::array<Pointd, 3>;
using DefiningVertIds = std::array<int, 3>;

/* make two vectors perpendicular (double precision) */
inline void orthogonalize_(Pointd& v0, Pointd& v1) {
  v0.normalize();
  v1 = v1 - dot(v0, v1) * v0;
  v1.normalize();
}

// compute bounding edges of the (non-degenerate) convex hull of four input points
// handles cases:
// - 0D : all points coincident     => no edges
// - 1D : all points collinear      => one edge between the two most distant points
// - 2D : all points coplanar       => convex hull in 2D projection (using monotone chain algorithm)
// - 3D : points form a tetrahedron => all edges of the tetrahedron
// Note: uses double precision (Pointd) for accurate geometric tests
inline std::vector<std::array<int, 2>> computeConvexHullEdge(const Pointd& v0, const Pointd& v1, const Pointd& v2,
                                                             const Pointd& v3, double kEpsCoord) {
  constexpr double kEpsFp64 = 1e-12;

  // small epsilon relative to the bounding scale of the points
  // by setting kEpsCoord==0, we assume the tetrahedron is never degenerate
  if (kEpsCoord == 0.0) {
    return {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
  }
  double kEpsCoordSq = kEpsCoord * kEpsCoord;
  double kEpsCoordCu = kEpsCoordSq * kEpsCoord;

  // utilities
  using Vec3d = Vec3<double>;
  auto mk_edge = [](int i, int j) -> std::array<int, 2> {
    auto [a, b] = std::minmax(i, j);
    return {a, b};
  };
  struct P2 {
    double x, y;
    int id;
  };
  using PointsP2 = std::array<P2, 4>;

  // already in double precision
  std::array<Vec3d, 4> pts = {v0, v1, v2, v3};

  // normalize point coordinates within a unit sphere
  Vec3d center = (pts[0] + pts[1] + pts[2] + pts[3]) / 4.0;
  double scale = 0.0;
  for (int i = 0; i < 4; ++i) {
    scale = std::max(scale, mag(pts[i] - center));
  }
  if (scale < kEpsFp64) return {};  // all points coincident (0D)
  for (int i : {0, 1, 2, 3}) pts[i] = (pts[i] - center) / scale;

  // check the volume (3D)
  Vec3d e1 = pts[1] - pts[0], e2 = pts[2] - pts[0], e3 = pts[3] - pts[0];
  double triple_prod = std::abs(dot(e1, cross(e2, e3)));
  if (triple_prod > kEpsCoordCu) {  // not degenerate, return all possible edges
    return {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
  }

  // find the triangle with the largest area (best 2D projection plane)
  constexpr int tris[4][3] = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
  double best_area = 0.0;
  Vec3d normal = {0., 0., 1.};
  int ref0 = 0, ref1 = 1;
  for (const auto& t : tris) {
    Vec3d n = cross(pts[t[1]] - pts[t[0]], pts[t[2]] - pts[t[0]]);
    double area = mag(n);
    if (area > best_area) {
      best_area = area;
      normal = n;
      ref0 = t[0];
      ref1 = t[1];
    }
  }

  // check if all points collinear (1D)
  if (best_area < kEpsCoordSq) {
    int bi = 0, bj = 1;
    double max_d = 0.0;
    for (int i = 0; i < 4; ++i)
      for (int j = i + 1; j < 4; ++j) {
        double d = mag(pts[i] - pts[j]);
        if (d > max_d) {
          max_d = d;
          bi = i;
          bj = j;
        }
      }
    return max_d > kEpsCoord ? std::vector<std::array<int, 2>>{mk_edge(bi, bj)} : std::vector<std::array<int, 2>>{};
  }

  // find convex hull via monotone chain (2D)
  auto compute_monotone_chain = [&](PointsP2 points) {
    auto cross2 = [](const P2& O, const P2& A, const P2& B) {
      return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
    };

    std::sort(points.begin(), points.end(),
              [](const P2& a, const P2& b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });

    std::array<P2, 8> hull;
    int k = 0;

    // lower hull
    for (int i = 0; i < 4; ++i) {
      while (k >= 2 && cross2(hull[k - 2], hull[k - 1], points[i]) <= 0) k--;
      hull[k++] = points[i];
    }

    // upper hull
    for (int i = 2, t = k + 1; i >= 0; i--) {
      while (k >= t && cross2(hull[k - 2], hull[k - 1], points[i]) <= 0) k--;
      hull[k++] = points[i];
    }
    int hull_size = std::max(0, k - 1);

    std::vector<std::array<int, 2>> edges;
    if (hull_size >= 2) {
      edges.reserve(hull_size);
      for (int i = 0; i < hull_size; ++i) {
        edges.push_back(mk_edge(hull[i].id, hull[(i + 1) % hull_size].id));
      }
    }
    return edges;
  };

  normal.normalize();
  Vec3d X = pts[ref1] - pts[ref0];
  X.normalize();
  Vec3d Y = cross(normal, X);

  PointsP2 p2s;
  for (int i = 0; i < 4; ++i) {
    Vec3d v = pts[i] - pts[ref0];
    p2s[i] = {dot(v, X), dot(v, Y), i};
  }

  return compute_monotone_chain(p2s);
}

class ISimplex : noncopyable {
 public:
  friend class SimplicialComplex;
  static constexpr int MAX_DIM = 2;

  // 0-dim simplex has _child[0] which is ignored
  ISimplex(int dim, int id) : _dim(dim), _id(id) {
    for_int(i, _dim + 1) _child[i] = nullptr;
#ifdef BUILD_LIBPSC
    if (dim == 2) {
      // default:
      //   for verts, edges ==> 0.0
      //   for faces        ==> 1.0
      _weighting_quadric = 1.0;
    }
#endif
  }

  void setChild(int num, Simplex child);
  void addParent(Simplex parent);
  void removeParent(Simplex parent);

  Simplex getChild(int num) const { return _child[num]; }
  CArrayView<Simplex> children() const { return _child.head(getDim() + 1); }
  const std::vector<Simplex>& getParents() const { return _parent; }
  // All descendents (>= 1 dim); iterates by generations children first, grandchildren next, etc.
  Array<Simplex> all_faces() const;
  // All ancestors (>= 1 dim); iterates by generations parents first, grandparents next, etc.
  PArray<Simplex, 20> get_star() const;
  PArray<Simplex, 20> faces_of_vertex() const;  // Faces adjacent to simplex (which must be a vertex).
  int getDim() const { return _dim; }
  int getId() const { return _id; }

  double length2() const;
  double length() const { return sqrt(length2()); }
  void polygon(Polygon& p) const;

  // 2-simplices
  void vertices(Simplex va[3]);
  Simplex opp_edge(Simplex v);
  // Simplex opp_vertex(Simplex e);

  // 1-simplices
  Simplex opp_vertex(Simplex v);

  // 0-simplices
  Simplex edgeTo(Simplex opp_v);

  // attribute mod and access function
  void setPosition(const Pointd& pos) { assertx(_dim == 0), _position = pos; }
  void setVAttribute(int va) { _vattribute = va; }
  void setArea(double area) { _area = area; }
  Flags& flags() {
    assertnever("no longer supported");
    return _flags;
  }

  const Pointd& getPosition() const { return _position; }
  int getVAttribute() const { return _vattribute; }
  double getArea() const { return _area; }
  const Flags& flags() const {
    assertnever("no longer supported");
    return _flags;
  }

  // predicates
  bool hasColor() const { return _vattribute >= 0; }
  bool isPrincipal() const { return _parent.empty(); }
  bool is_boundary() const { return _parent.size() == 1; }
  bool isManifold() const { return _parent.size() == 2; }

  // for gemorph
  const char* get_string() const { return _string.get(); }
  void set_string(const char* s) { _string = make_unique_c_string(s); }
  void update_string(const char* key, const char* val) { GMesh::update_string_ptr(_string, key, val); }

  HH_POOL_ALLOCATION(ISimplex);

 private:
  int _dim;                          // dimension of the simplex
  int _id;                           // simplex id
  Vec<Simplex, MAX_DIM + 1> _child;  // simplices it contains
  std::vector<Simplex> _parent;      // simplices it belongs to
  // Attributes
  Flags _flags;
  Pointd _position;     // simplices of dimension 0 only (double precision)
  int _vattribute{-1};  // visual attributes
  double _area{0.0};
  // for geomorph
  unique_ptr<char[]> _string;

#ifdef BUILD_LIBPSC
 public:
  // for contraction edges only
  //   we store the information in edges of a separate simplicial complex
  double cost;
  double w_p0;  // "p0" is "getChild(0)"

  // for tracking the subtree depth in the binary tree of edge collapses
  //   used for balanced tree construction when _weighting_balanced > 0
  int _subtree_depth = 1;
  int _subtree_size = 1;

  // quadric information (double precision for numerical stability)
  Matrix3d _A{};   // initialized to all zeros
  Pointd _b{};     // initialized to all zeros
  double _c{0.0};  // initialized to 0.0

  // the id of connected component, ideally >=0
  int _component_id = -1;

  // the volume weighting for quadric, setting to negative to disable
  double _weighting_quadric = 0.0;

  /* add two quadrics: q = a + b (double precision) */
  static void add_quadric_(Simplex q, const Simplex a, double w = 1.0) {
    for_int(i, 3) q->_A[i] += a->_A[i] * w;
    q->_b += a->_b * w;
    q->_c += a->_c * w;
  }

  /* quadric multiplied by a factor: q *= w (double precision) */
  static void weight_quadric_(Simplex q, double w) {
    for_int(i, 3) q->_A[i] *= w;
    q->_b *= w;
    q->_c *= w;
  }

  /* compute fundamental quadric for the simplex, optionally weighting with a scaled volume */
  void compute_native_quadric_() {
    int dim = getDim();

    if (dim == 0) {
      const Pointd& p = getPosition();

      // A = Identity
      for_int(i, 3) for_int(j, 3) _A[i][j] = (i == j ? 1.0 : 0.0);

      // b = -p
      _b = -p;

      // c = p^T p
      _c = mag2(p);

      // setting this will penalize the movements of vertices
      // allowing good distribution of vertices
      if (_weighting_quadric >= 0) {
        ISimplex::weight_quadric_(this, _weighting_quadric);
      }

    } else if (dim == 1) {
      // get defining points
      const Pointd& p0 = getChild(0)->getPosition();
      const Pointd& p1 = getChild(1)->getPosition();

      Pointd p = 0.5 * (p0 + p1);

      Pointd dir = p1 - p0;
      dir.normalize();

      // A = I - dir * dir^T
      for_int(i, 3) for_int(j, 3) _A[i][j] = (i == j ? 1.0 : 0.0) - dir[i] * dir[j];

      // b = -A * p
      for_int(i, 3) {
        _b[i] = 0;
        for_int(j, 3) _b[i] -= _A[i][j] * p[j];
      }

      // c = p^T A p
      _c = 0;
      for_int(i, 3) for_int(j, 3) _c += p[i] * _A[i][j] * p[j];

      // weighted by length
      if (_weighting_quadric >= 0) {
        ISimplex::weight_quadric_(this, mag(p1 - p0) * _weighting_quadric);
      }

    } else if (dim == 2) {
      // get defining points
      Simplex v012[3];
      vertices(v012);
      Pointd p0 = v012[0]->getPosition();
      Pointd p1 = v012[1]->getPosition();
      Pointd p2 = v012[2]->getPosition();

      // Sort vertices by position to ensure deterministic Gram-Schmidt ordering,
      // making the quadric computation independent of vertex ID assignment.
      {
        auto pos_key = [](const Pointd& q) { return std::make_tuple(q[0], q[1], q[2]); };
        if (pos_key(p1) < pos_key(p0)) std::swap(p0, p1);
        if (pos_key(p2) < pos_key(p0)) std::swap(p0, p2);
        if (pos_key(p2) < pos_key(p1)) std::swap(p1, p2);
      }

      Pointd p = (p0 + p1 + p2) / 3.0;

      Pointd u, v;
      u = p1 - p0;
      v = p2 - p0;
      orthogonalize_(u, v);

      // A = I - sum(tangent_i * tangent_i^T)
      for_int(i, 3) for_int(j, 3) _A[i][j] = (i == j ? 1.0 : 0.0) - u[i] * u[j] - v[i] * v[j];

      // b = -A * p
      for_int(i, 3) {
        _b[i] = 0;
        for_int(j, 3) _b[i] -= _A[i][j] * p[j];
      }

      // c = p^T A p
      _c = 0;
      for_int(i, 3) for_int(j, 3) _c += p[i] * _A[i][j] * p[j];

      // weighted by area
      if (_weighting_quadric >= 0) {
        // originally intended for area weighting, so take the square root for length weighting
        double factor = std::sqrt(_weighting_quadric);
        Pointd ud = (p1 - p0) * factor;
        Pointd vd = (p2 - p0) * factor;
        double scaled_area = 0.5 * mag(cross(ud, vd));
        ISimplex::weight_quadric_(this, scaled_area);
      }
    }
  }

  /* compute the vertex quadrics
     by aggregating neighboring simplices' quadrics to it
   */
  void aggregate_() {
    assertx(getDim() == 0);

    // Sort star simplices by defining vertex positions for deterministic aggregation.
    // Floating-point addition is not associative, so iteration order matters.
    auto star = get_star();
    std::sort(star.begin(), star.end(), [](Simplex a, Simplex b) {
      if (a->getDim() != b->getDim()) return a->getDim() < b->getDim();
      // Sort by defining vertex positions lexicographically
      auto get_pos_key = [](Simplex s) {
        if (s->getDim() == 0) {
          const Pointd& p = s->getPosition();
          return std::array<double, 9>{p[0], p[1], p[2], 0, 0, 0, 0, 0, 0};
        } else if (s->getDim() == 1) {
          Pointd p0 = s->getChild(0)->getPosition();
          Pointd p1 = s->getChild(1)->getPosition();
          auto t0 = std::make_tuple(p0[0], p0[1], p0[2]);
          auto t1 = std::make_tuple(p1[0], p1[1], p1[2]);
          if (t1 < t0) std::swap(p0, p1);
          return std::array<double, 9>{p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], 0, 0, 0};
        } else {
          Simplex v012[3];
          s->vertices(v012);
          Pointd p0 = v012[0]->getPosition();
          Pointd p1 = v012[1]->getPosition();
          Pointd p2 = v012[2]->getPosition();
          auto key = [](const Pointd& q) { return std::make_tuple(q[0], q[1], q[2]); };
          if (key(p1) < key(p0)) std::swap(p0, p1);
          if (key(p2) < key(p0)) std::swap(p0, p2);
          if (key(p2) < key(p1)) std::swap(p1, p2);
          return std::array<double, 9>{p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]};
        }
      };
      return get_pos_key(a) < get_pos_key(b);
    });

    // Aggregate in deterministic order
    for (auto s : star) {
      int dim = s->getDim();
      if (dim == 0) {
        assertx(s == this);
      } else {
        ISimplex::add_quadric_(this, s, 1.0 / (dim + 1));
      }
    }
  }
#endif
};

#ifdef BUILD_LIBPSC

struct TopoRecord {
  DefiningVertIds defining_vertex_ids;
  int label;  // topological label
};

/* see if these three vertex can constitute a face simplex */
inline Simplex is_face(Simplex v0, Simplex v1, Simplex v2) {
  // 1. see if the vertices exist
  if (!v0 || !v1 || !v2) return nullptr;

  // 2. see if edges exist
  Simplex edge0 = v0->edgeTo(v1);
  Simplex edge1 = v0->edgeTo(v2);
  Simplex edge2 = v1->edgeTo(v2);
  if (!edge0 || !edge1 || !edge2) {
    // edges must exist
    return nullptr;
  }

  // 3. see if the face exist
  const auto& parents_e0 = edge0->getParents();
  const auto& parents_e1 = edge1->getParents();
  const auto& parents_e2 = edge2->getParents();
  for (auto parent : parents_e0) {
    if (parent->getDim() == 2 && std::find(parents_e1.begin(), parents_e1.end(), parent) != parents_e1.end() &&
        std::find(parents_e2.begin(), parents_e2.end(), parent) != parents_e2.end()) {
      // find the face
      return parent;
    }
  }

  // not exist
  return nullptr;
}

/* compute the vertex ids that define the simplex */
inline DefiningVertIds compute_defining_vertex_ids(Simplex s) {
  constexpr int INT_MAX_VAL = std::numeric_limits<int>::max();
  int dim = s->getDim();
  if (dim == 0) {
    return {s->getId(), INT_MAX_VAL, INT_MAX_VAL};
  } else if (dim == 1) {
    DefiningVertIds out = {s->getChild(0)->getId(), s->getChild(1)->getId(), INT_MAX_VAL};
    std::sort(out.begin(), out.end());
    return out;
  } else {
    assertx(dim == 2);
    Simplex v012[3];
    s->vertices(v012);
    DefiningVertIds out = {v012[0]->getId(), v012[1]->getId(), v012[2]->getId()};
    std::sort(out.begin(), out.end());
    return out;
  }
}

#endif

// this is useful for getting the edge pair with minimal cost, or querying the existence
class MinHeap {
 public:
  // add new element
  bool insert(Simplex s) {
    // must be en edge
    assertx(s->getDim() == 1);

    // if exist, do nothing
    if (iter_lookup.count(s)) {
      return false;
    }

    // insert new
    auto iter = cost_sorted.insert(s).first;
    iter_lookup[s] = iter;
    return true;
  }

  // remove element
  bool erase(Simplex s) {
    // "e" might already been destroyed

    // may not exist, fail to delete
    auto it = iter_lookup.find(s);
    if (it == iter_lookup.end()) {
      return false;
    }

    // erase
    cost_sorted.erase(it->second);
    iter_lookup.erase(it);
    return true;
  }

  Simplex min() const {
    // obtain the element with minimum cost
    if (cost_sorted.empty()) return nullptr;
    return *cost_sorted.begin();
  }

  bool empty() const { return cost_sorted.empty(); }

  void clear() {
    cost_sorted.clear();
    iter_lookup.clear();
  }

 private:
  struct CompareByCost {
    // helper to create a canonical tuple from edge endpoint positions for comparison (double precision)
    static std::tuple<double, double, double, double, double, double> make_canonical_edge_key_by_coords(Simplex e) {
      assertx(e && e->getDim() == 1);
      Simplex v0 = e->getChild(0);
      Simplex v1 = e->getChild(1);
      assertx(v0 && v1);
      const Pointd& p0 = v0->getPosition();
      const Pointd& p1 = v1->getPosition();

      // compare positions lexicographically to determine order
      auto as_tuple = [](const Pointd& p) { return std::make_tuple(p[0], p[1], p[2]); };
      auto t0 = as_tuple(p0);
      auto t1 = as_tuple(p1);
      if (t0 <= t1) {
        return std::make_tuple(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]);
      } else {
        return std::make_tuple(p1[0], p1[1], p1[2], p0[0], p0[1], p0[2]);
      }
    }

    bool operator()(Simplex a, Simplex b) const {
#ifdef BUILD_LIBPSC
      if (a == b) {
        return false;
      }

      // primary comparison: by cost
      if (a->cost != b->cost) {
        return a->cost < b->cost;
      }

      // tiebreaker: compare by vertex positions (geometry) to ensure determinism
      // independent of vertex ID assignment.
      assertx(a->getDim() == 1 && b->getDim() == 1);

      // create canonical edge keys and compare
      auto a_key_coord = make_canonical_edge_key_by_coords(a);
      auto b_key_coord = make_canonical_edge_key_by_coords(b);
      if (a_key_coord != b_key_coord) {
        return a_key_coord < b_key_coord;
      }

      assertnever(
          "Degenerate candidate edges detected: two distinct edges share identical endpoint coordinates. "
          "This usually indicates duplicated/overlapping vertex positions. "
          "Please preprocess input mesh to remove coincident vertices.");
      return false;
#else
      // dummy
      return true;
#endif
    }
  };

  std::set<Simplex, CompareByCost> cost_sorted;
  std::unordered_map<Simplex, std::set<Simplex, CompareByCost>::iterator> iter_lookup;
};

struct SimplexIdCompare {
  // use getid() for deterministic sorting instead of relying on pointer addresses,
  // to avoid results being affected by random memory allocation.
  bool operator()(const Simplex& s1, const Simplex& s2) const {
    if (s1->getDim() != s2->getDim()) return s1->getDim() < s2->getDim();
    return s1->getId() < s2->getId();
  }
};

class SimplicialComplex : noncopyable {
  struct OrderedSimplices_range;

 public:
  static constexpr int MAX_DIM = ISimplex::MAX_DIM;

  SimplicialComplex() { for_int(i, MAX_DIM + 1) _free_sid[i] = 1; }
  ~SimplicialComplex() { clear(); }

  void clear();

  // I/O
  void readGMesh(std::istream& is);
  void read(std::istream& is);
  void write(std::ostream& os) const;

  // modification functions
  Simplex createSimplex(int dim);
  Simplex createSimplex(int dim, int id);
  void destroySimplex(Simplex s, int area_test = 0);
  void unify(Simplex vs, Simplex vt, int propagate_area = 0, MinHeap* heap = nullptr);
  void copy(const SimplicialComplex& orig);
  void skeleton(int dim);

  // access (const) functions
  int num(int dim) const { return _simplices[dim].num(); }
  int getMaxId(int dim) const { return _free_sid[dim]; }
  bool valid(Simplex s) const;
  Simplex getSimplex(Simplex s) const { return getSimplex(s->getDim(), s->getId()); }  // Convenience.
  Simplex getSimplex(int dim, int id) const;
  int materialNum() const { return _material_strings.num(); }
  const char* getMaterial(int matid) const { return _material_strings[matid].c_str(); }
  const Map<int, Simplex>::cvalues_range simplices_dim(int dim) const { return _simplices[dim].values(); }
  OrderedSimplices_range ordered_simplices_dim(int dim) const { return OrderedSimplices_range(*this, dim); }
  void starbar(Simplex s, SimplicialComplex& result) const;
  void star(Simplex s, Array<Simplex>& ares) const;
  void ok() const;
  void scUnion(const SimplicialComplex& s1, const SimplicialComplex& s2, SimplicialComplex& result) const;

  // static constexpr FlagMask ALL = ~0u, SHARP = 1;  // flags

 private:                              // functions
  void readLine(const char* str);      // connectivity
  void attrReadLine(const char* str);  // attributes
  bool equal(Simplex s1, Simplex s2) const;
  bool eq1simp(Simplex s1, Simplex s2) const;
  bool eq2simp(Simplex s1, Simplex s2) const;
  void replace(Simplex src, Simplex tgt, Stack<Simplex>& affected_parents);
  int compare_normal(const GMesh& mesh, Corner c1, Corner c2);

  struct OrderedSimplices_range {
    using Container = Array<Simplex>;
    OrderedSimplices_range(const SimplicialComplex& sc, int dim) : _simplices(sc.simplices_dim(dim)) {
      const auto by_increasing_id = [&](Simplex s1, Simplex s2) { return s1->getId() < s2->getId(); };
      sort(_simplices, by_increasing_id);
    }
    Container::iterator begin() const { return const_cast<Container&>(_simplices).begin(); }
    Container::iterator end() const { return const_cast<Container&>(_simplices).end(); }
    int size() const { return _simplices.num(); }

   private:
    Container _simplices;
  };

  // one array per dimension
  Vec<Map<int, Simplex>, MAX_DIM + 1> _simplices;
  Array<string> _material_strings;
  Vec<int, MAX_DIM + 1> _free_sid;

#ifdef BUILD_LIBPSC
 public:
  //  the multiplicative factor for the cost if the vertices are from different components
  double _weighting_topo = 1.0;

  // weighting factor for encouraging balanced binary tree construction
  //   0.0 means no balancing consideration
  //   positive values penalize unbalanced merges (larger difference in subtree depths)
  double _weighting_balanced = 0.0;
  double _alpha_balanced = 1.0;
  bool _balanced_depth = true;

  // total number of initial vertices (for normalizing subtree size to 0~1 range)
  int _total_initial_vertices = 0;

  // the aspect ratio threshold to consider a simplex degenerate
  double _ratio_degeneracy = 0.0;  // for computing candidate pairs

  // the weighting for boundary edges (only used in markov mode)
  // this is set from weighting_e[0] when edges is empty and weighting_e has size 1
  double _weighting_boundary = 0.0;

  /* update boundary edge weightings (for markov mode)
     boundary edge: an edge that is adjacent to exactly one face (i.e., has exactly one face parent)
     only updates _weighting_quadric, native quadric should be computed separately
   */
  void update_boundary_edge_weighting_(Simplex e) {
    if (_weighting_boundary <= 0.0) return;
    assertx(e->getDim() == 1);

    // count how many face parents this edge has
    int face_parent_count = 0;
    for (auto p : e->getParents()) {
      if (p->getDim() == 2) {
        face_parent_count++;
      }
    }

    // boundary edge: adjacent to exactly one face, or isolated edge (no face)
    bool is_boundary = (face_parent_count <= 1);

    if (is_boundary) {
      // set the weighting for boundary edge
      e->_weighting_quadric = _weighting_boundary;
    } else {
      // non-boundary edge: zero out its contribution
      e->_weighting_quadric = 0.0;
    }
  }

  /* the data type for recording the operations */
  struct EcolRecord {
    int vsid;
    int vtid;
    int position_bit;
    Pointd delta_p;
    std::vector<TopoRecord> topo_record_lst;  // you need to convert this into a code string
    /* convert to python dictionary */
    py::dict to_dict(const std::string& code) const {
      py::dict d;
      d["vsid"] = vsid - 1;  // >= 0
      d["vtid"] = vtid - 1;  // >= 0
      d["code"] = code;
      d["position_bit"] = position_bit;
      d["delta_p"] = py::cast(std::array<double, 3>{delta_p[0], delta_p[1], delta_p[2]});  // to a list (double)
      return d;
    }
  };

  /* compute the edge collapse record
      note : 
        the topology can be changed, but vertex simplex should not be destroyed
        because we need to access vertex information
   */
  EcolRecord compute_ecol_record(int vsid, int vtid, double w_p0) {
    Simplex vs = getSimplex(0, vsid);
    Simplex vt = getSimplex(0, vtid);
    assertx(vs && vt);

    int position_bit = 1;
    Pointd delta_p = vt->getPosition() - vs->getPosition();
    if (w_p0 == 0.5) {
      position_bit = 0;
      delta_p *= 0.5;
    }

    // source simplex (in the form of vertex ids) ---> topological label
    std::vector<TopoRecord> topo_record_lst;

    // get the neighborhood of both "vs" and "vt"
    auto vs_star = vs->get_star();
    auto vt_star = vt->get_star();
    std::set<Simplex, SimplexIdCompare> vst_star;  // some are overlapped, so we use "std::set"
    vst_star.insert(vs_star.begin(), vs_star.end());
    vst_star.insert(vt_star.begin(), vt_star.end());

    // a helper function
    auto add = [&topo_record_lst, vsid, vtid](Simplex s, int label) {
      auto v_ids = compute_defining_vertex_ids(s);
      for (auto& vid : v_ids) {
        // the target vertex has not been created during vertex split, so replace with source vertex
        if (vid == vtid) {
          vid = vsid;
        }
      }
      topo_record_lst.push_back({v_ids, label});
    };

    // handle point
    add(vs, int(vs->edgeTo(vt) != nullptr));

    // handle edge/face
    for (auto s : vst_star) {
      int dim = s->getDim();

      if (dim == 0) {
        continue;  // we have processed before
      } else if (dim == 1) {
        auto contains = [s](Simplex v) { return s->getChild(0) == v || s->getChild(1) == v; };
        bool contains_vs = contains(vs);
        bool contains_vt = contains(vt);
        Simplex v_opp;
        if (contains_vs) {
          if (contains_vt) {
            // contains_vs, contains_vt
            continue;  // this is just an edge: (vs, vt)
          }
          // contains_vs, !contains_vt
          v_opp = s->opp_vertex(vs);
          if (v_opp->edgeTo(vt) == nullptr) {
            add(s, 0);
          } else {
            if (!is_face(vs, vt, v_opp)) {
              add(s, 2);
            } else {
              add(s, 3);
            }
          }
        } else {  // !contains_vs, contains_vt
          assertx(contains_vt);
          v_opp = s->opp_vertex(vt);
          if (v_opp->edgeTo(vs) == nullptr) {
            add(s, 1);
          } else {
            /* ignore this case 
               because the "s" is not from the source
             */
          }
        }

      } else if (dim == 2) {
        auto contains = [s](Simplex v) {
          Simplex v012[3];
          s->vertices(v012);
          return v012[0] == v || v012[1] == v || v012[2] == v;
        };
        bool contains_vs = contains(vs);
        bool contains_vt = contains(vt);
        Simplex e_opp;
        if (contains_vs) {
          if (contains_vt) {
            // contains_vs, contains_vt
            continue;  // this is not a source face: (vs, vt, ?)
          }
          // contains_vs, !contains_vt
          e_opp = s->opp_edge(vs);
          if (!is_face(e_opp->getChild(0), e_opp->getChild(1), vt)) {
            add(s, 0);
          } else {
            add(s, 2);
          }
        } else {
          // !contains_vs, contains_vt
          assertx(contains_vt);
          e_opp = s->opp_edge(vt);
          if (!is_face(e_opp->getChild(0), e_opp->getChild(1), vs)) {
            add(s, 1);
          } else {
            /* ignore this case 
               because the "s" is not from the source
             */
          }
        }
      }
    }

    return {.vsid = vsid,
            .vtid = vtid,
            .position_bit = position_bit,
            .delta_p = delta_p,
            .topo_record_lst = topo_record_lst};
  }

  /* evaluate the cost of edge contraction between two vertices (double precision)
   */
  std::pair<double, double> compute_contraction_cost_and_location(Simplex v0, Simplex v1) {
    assert(v0 && v1);
    const Pointd& p0 = v0->getPosition();
    const Pointd& p1 = v1->getPosition();
    Pointd mid = 0.5 * (p0 + p1);

    // fused quadric (double precision)
    Matrix3d A = {v0->_A[0] + v1->_A[0], v0->_A[1] + v1->_A[1], v0->_A[2] + v1->_A[2]};
    Pointd b = v0->_b + v1->_b;
    double c = v0->_c + v1->_c;

    // to evaluate the cost (double precision)
    auto evaluate_cost = [&A, &b, &c](const Pointd& x) {
      double val = c;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) val += x[i] * A[i][j] * x[j];
      val += 2.0 * dot(b, x);
      // ideally >= 0, but might due to numerical error
      return std::max(val, 0.0);
    };

    // evaluate cost at three points, take the one with minimum cost
    std::array<Pointd, 3> candidates = {p0, p1, mid};
    int i_best = 2;
    double cost_min = std::numeric_limits<double>::max();
    for (int i : {2, 1, 0}) {
      double cost = evaluate_cost(candidates[i]);
      if (cost < cost_min) {
        i_best = i;
        cost_min = cost;
      }
    }

    // When both endpoints have equal cost (and are better than midpoint),
    // use lexicographic position comparison for geometry-deterministic selection.
    // This ensures the result is independent of vertex ID assignment.
    if (i_best == 0 || i_best == 1) {
      double cost_other = evaluate_cost(candidates[1 - i_best]);
      if (cost_other == cost_min) {
        auto pos_tuple = [](const Pointd& q) { return std::make_tuple(q[0], q[1], q[2]); };
        i_best = pos_tuple(candidates[0]) <= pos_tuple(candidates[1]) ? 0 : 1;
      }
    }

    // fused location is at: p0 * w_p0 + p1 * (1 - w_p0)
    double w_p0;
    if (i_best == 0) {
      w_p0 = 1.0;
    } else if (i_best == 1) {
      w_p0 = 0.0;
    } else {
      w_p0 = 0.5;
    }

    // if the two vertices are in different components, we may need to penalize the cost
    if (v0->_component_id != v1->_component_id) {
      cost_min *= _weighting_topo;
    }

    // add penalty for unbalanced merges when _weighting_balanced > 0.
    if (_weighting_balanced > 0.0) {
      // determine the value for balancing
      double value;
      if (_balanced_depth) {
        double depth_0 = static_cast<double>(v0->_subtree_depth);
        double depth_1 = static_cast<double>(v1->_subtree_depth);
        value = 1.0 + std::max(depth_0, depth_1);
      } else {
        // use normalized size (0~1 range): size / total_initial_vertices
        // this ensures that the sum of all current vertices' normalized sizes equals 1
        double size_0 = static_cast<double>(v0->_subtree_size);
        double size_1 = static_cast<double>(v1->_subtree_size);
        assertx(_total_initial_vertices > 0);
        value = (size_0 + size_1) / static_cast<double>(_total_initial_vertices);  // in range (0, 1]
      }
      // piecewise function
      double ratio;
      if (_alpha_balanced > 0.0) {
        ratio = std::pow(_alpha_balanced, value);  // exponential growth
      } else if (_alpha_balanced == 0.0) {
        ratio = value;  // linear growth
      } else {
        ratio = std::pow(value, -_alpha_balanced);  // polynomial growth
      }
      // the multiplicative factor
      double balance_factor = 1.0 + _weighting_balanced * ratio;
      cost_min *= balance_factor;
    }

    // return the cost and the location weighting
    return {cost_min, w_p0};
  }

  std::pair<double, double> compute_contraction_cost_and_location(int v0_id, int v1_id) {
    Simplex v0 = getSimplex(0, v0_id);
    Simplex v1 = getSimplex(0, v1_id);
    return compute_contraction_cost_and_location(v0, v1);
  }

  /* perform simplicial complex simplification, until a single vertex
       note : because it depends on "SplitRecord", so it is best to move the implementation to the cpp file
  */
  std::tuple<std::array<double, 3>, std::vector<py::dict>, std::vector<double>> perform_simplification(
      bool markov = false, double voxel_size = 0.0);
#endif
};

inline Simplex SimplicialComplex::getSimplex(int dim, int id) const {
  if (_simplices[dim].contains(id))
    return _simplices[dim].get(id);
  else
    return nullptr;
}

inline Simplex ISimplex::opp_vertex(Simplex v1) {
  assertx(getDim() == 1);
  if (_child[0] == v1) return _child[1];
  if (_child[1] == v1) return _child[0];
  // no opposite to v1 on this edge
  return nullptr;
}

inline Simplex ISimplex::opp_edge(Simplex v1) {
  assertx(getDim() == 2);
  for (Simplex edge : children())
    if (edge->_child[0] != v1 && edge->_child[1] != v1) return edge;
  // no opposite to v1 on this face
  return nullptr;
}

inline void ISimplex::setChild(int num, Simplex child) {
  assertx(child->_dim == _dim - 1);
  _child[num] = child;
}

inline void ISimplex::addParent(Simplex parent) {
  assertx(parent->_dim == _dim + 1);
  _parent.push_back(parent);
}

inline void ISimplex::removeParent(Simplex old_parent) {
  assertx(old_parent->_dim == _dim + 1);
  assertx(vec_remove_ordered(_parent, old_parent));
}

// inline const Point& ISimplex::getColor() const {
//     string str;
//     const char* s = assertx(GMesh::string_key(str, getVAttribute(), "rgb"));
//     Point co; for_int(c, 3) co[c] = float_from_chars(s);
//     assert_no_more_chars(s);
//     return co;
// }

inline double ISimplex::length2() const {
  assertx(getDim() == 1);
  const Pointd& p0 = getChild(0)->getPosition();
  const Pointd& p1 = getChild(1)->getPosition();
  return dist2(p0, p1);
}

inline Simplex ISimplex::edgeTo(Simplex opp_v) {
  assertx(_dim == 0);
  for (Simplex e : getParents())
    if (e->opp_vertex(this) == opp_v) return e;
  return nullptr;
}

inline void ISimplex::vertices(Simplex va[3]) {
  assertx(_dim == 2);
  Simplex va0 = getChild(0)->getChild(0);
  Simplex va1 = getChild(0)->getChild(1);
  Simplex va2 = getChild(1)->getChild(0);
  if (va2 == va0 || va2 == va1) {
    va2 = getChild(1)->getChild(1);
    assertx(va2 != va0 && va2 != va1);
  }
  va[0] = va0;
  va[1] = va1;
  va[2] = va2;
}

HH_INITIALIZE_POOL(ISimplex);

inline bool SimplicialComplex::valid(Simplex s) const {
  if (!s) return false;
  if (s->getDim() > MAX_DIM || s->getDim() < 0) return false;
  return _simplices[s->getDim()].contains(s->getId());
}

}  // namespace hh

#endif  // MESH_PROCESSING_G3DOGL_SIMPLICIALCOMPLEX_H_
