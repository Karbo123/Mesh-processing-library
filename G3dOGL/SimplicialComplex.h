// -*- C++ -*-  Copyright (c) Microsoft Corporation; see license.txt
#ifndef MESH_PROCESSING_G3DOGL_SIMPLICIALCOMPLEX_H_
#define MESH_PROCESSING_G3DOGL_SIMPLICIALCOMPLEX_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <unordered_map>
#include <vector>

#ifdef BUILD_LIBPSC
#include <libqhullcpp/Qhull.h>
#include <libqhullcpp/QhullFacet.h>
#include <libqhullcpp/QhullFacetList.h>
#include <libqhullcpp/QhullVertexSet.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <limits>
#include <sstream>
#include <tuple>
namespace py = pybind11;
using namespace orgQhull;
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

#ifdef BUILD_LIBPSC

using Matrix3 = std::array<Point, 3>;
using DefiningVertIds = std::array<int, 3>;

/* make two vectors perpendicular */
inline void orthogonalize_(Point& v0, Point& v1) {
  v0.normalize();
  v1 = v1 - float(dot(v0, v1)) * v0;
  v1.normalize();
}

/* useful for computing connected components */
class DisjointSet {
 public:
  Simplex find(Simplex x) {  // find the root with path compression
    if (parent[x] != x) parent[x] = find(parent[x]);
    return parent[x];
  }

  void unite(Simplex x, Simplex y) {  // unite two sets with union by rank
    auto px = find(x);
    auto py = find(y);
    if (px != py) {
      // Union by rank
      if (rank[px] < rank[py]) {
        parent[px] = py;
      } else if (rank[px] > rank[py]) {
        parent[py] = px;
      } else {
        parent[py] = px;
        rank[px]++;  // increase rank if both have the same rank
      }
    }
  }

  void add(Simplex x) {  // add a new element
    if (!parent.count(x)) {
      parent[x] = x;
      rank[x] = 0;  // initialize rank to 0
    }
  }

 public:
  std::unordered_map<Simplex, Simplex> parent;  // node --> root node
  std::unordered_map<Simplex, int> rank;        // node --> rank (tree height)
};

/* use QHull to compute Delaunay triangulation */
inline std::vector<std::array<int, 4>> compute_delaunay_3d(const std::vector<Point>& verts) {
  std::vector<std::array<int, 4>> tets;

  // too few points, no tetrahedra exist
  if (verts.size() < 4) {
    return tets;
  }

  // flatten point data
  std::vector<coordT> points_flat;
  for (const auto& pt : verts) {
    points_flat.insert(points_flat.end(), pt.begin(), pt.end());
  }

  // use Qhull C++ API to compute 3D Delaunay triangulation
  Qhull qh;
  try {
    qh.runQhull("qhull", 3, static_cast<int>(verts.size()), points_flat.data(), "d Qz");

    const QhullFacetList& facets = qh.facetList();
    for (auto f = facets.begin(); f != facets.end(); ++f) {
      if (!f->isUpperDelaunay()) {  // only lower hull facets are part of Delaunay triangulation
        const QhullVertexSet& vs = f->vertices();
        if (vs.size() == 4) {
          std::array<int, 4> tet;
          int i = 0;
          for (const auto& v : vs) {
            tet[i++] = v.point().id();
          }
          tets.push_back(tet);
        }
      }
    }
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("Qhull error: ") + e.what());
  }

  return tets;
}

#endif

// compute bounding edges of the (non-degenerate) convex hull of four input points
// handles cases:
// - 0D : all points coincident     => no edges
// - 1D : all points collinear      => one edge between the two most distant points
// - 2D : all points coplanar       => convex hull in 2D projection (using monotone chain algorithm)
// - 3D : points form a tetrahedron => all edges of the tetrahedron
inline std::vector<std::array<int, 2>> computeConvexHullEdge(const Point& v0, const Point& v1, const Point& v2,
                                                             const Point& v3, double kEpsCoord) {
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

  // convert to double precision (for more accurate geometric tests)
  std::array<Vec3d, 4> pts = {v0.cast<double>(), v1.cast<double>(), v2.cast<double>(), v3.cast<double>()};

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

  float length2() const;
  float length() const { return sqrt(length2()); }
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
  void setPosition(const Point& pos) { assertx(_dim == 0), _position = pos; }
  void setVAttribute(int va) { _vattribute = va; }
  void setArea(float area) { _area = area; }
  Flags& flags() {
    assertnever("no longer supported");
    return _flags;
  }

  const Point& getPosition() const { return _position; }
  int getVAttribute() const { return _vattribute; }
  float getArea() const { return _area; }
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
  Point _position;      // simplices of dimension 0 only
  int _vattribute{-1};  // visual attributes
  float _area{0.f};
  // for geomorph
  unique_ptr<char[]> _string;

#ifdef BUILD_LIBPSC
 public:
  // for contraction edges only
  //   we store the information in edges of a separate simplicial complex
  float cost;
  float w_p0;  // "p0" is "getChild(0)"

  // for tracking the subtree depth in the binary tree of edge collapses
  //   used for balanced tree construction when _weighting_balanced > 0
  int _subtree_depth = 1;
  int _subtree_size = 1;

  // quadric information
  Matrix3 _A{};  // initialized to all zeros
  Point _b{};    // initialized to all zeros
  float _c{0};   // initialized to 0.0

  // the id of connected component, ideally >=0
  int _component_id = -1;

  // the volume weighting for quadric, setting to negative to disable
  float _weighting_quadric = 0.0;

  /* add two quadrics: q = a + b */
  static void add_quadric_(Simplex q, const Simplex a, float w = 1.0) {
    for_int(i, 3) q->_A[i] += a->_A[i] * w;
    q->_b += a->_b * w;
    q->_c += a->_c * w;
  }

  /* quadric multiplied by a factor: q *= w */
  static void weight_quadric_(Simplex q, float w) {
    for_int(i, 3) q->_A[i] *= w;
    q->_b *= w;
    q->_c *= w;
  }

  /* compute fundamental quadric for the simplex, optionally weighting with a scaled volume */
  void compute_native_quadric_() {
    int dim = getDim();

    if (dim == 0) {
      const Point& p = getPosition();

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
      const Point& p0 = getChild(0)->getPosition();
      const Point& p1 = getChild(1)->getPosition();

      Point p = 0.5f * (p0 + p1);

      Point dir = p1 - p0;
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
      Point p0 = v012[0]->getPosition();
      Point p1 = v012[1]->getPosition();
      Point p2 = v012[2]->getPosition();

      Point p = (p0 + p1 + p2) / 3.0f;

      Point u, v;
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
        float factor = std::sqrt(_weighting_quadric);
        u = (p1 - p0) * factor;
        v = (p2 - p0) * factor;
        float scaled_area = 0.5f * mag(cross(u, v));
        ISimplex::weight_quadric_(this, scaled_area);
      }
    }
  }

  /* compute the vertex quadrics
     by aggregating neighboring simplices' quadrics to it
   */
  void aggregate_() {
    assertx(getDim() == 0);

    // get its neighborhood, including the vertex itself
    for (auto s : get_star()) {
      int dim = s->getDim();
      if (dim == 0) {
        // it is just itself, don't need to add itself
        assertx(s == this);
      } else {
        // divide by the number of corners
        ISimplex::add_quadric_(this, s, 1.0f / (dim + 1));
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

 private:
  struct CompareByCost {
    // helper to create a canonical tuple from two positions for comparison
    static std::tuple<float, float, float, float, float, float> make_canonical_edge_key(const Point& p0,
                                                                                        const Point& p1) {
      // compare positions lexicographically to determine order
      auto as_tuple = [](const Point& p) { return std::make_tuple(p[0], p[1], p[2]); };
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
      // primary comparison: by cost
      if (a->cost != b->cost) {
        return a->cost < b->cost;
      }

      // tiebreaker: compare by vertex positions (geometry) to ensure determinism
      // independent of vertex ID assignment.
      assertx(a->getDim() == 1 && b->getDim() == 1);
      Point a_p0 = a->getChild(0)->getPosition();
      Point a_p1 = a->getChild(1)->getPosition();
      Point b_p0 = b->getChild(0)->getPosition();
      Point b_p1 = b->getChild(1)->getPosition();

      // create canonical edge keys and compare
      auto a_key = make_canonical_edge_key(a_p0, a_p1);
      auto b_key = make_canonical_edge_key(b_p0, b_p1);
      return a_key < b_key;
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
  void readQHull(std::istream& is);

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
  float _weighting_topo = 1.0f;

  // weighting factor for encouraging balanced binary tree construction
  //   0.0 means no balancing consideration
  //   positive values penalize unbalanced merges (larger difference in subtree depths)
  float _weighting_balanced = 0.0f;
  float _alpha_balanced = 1.0f;
  bool _balanced_depth = true;

  // the aspect ratio threshold to consider a simplex degenerate
  float _ratio_degeneracy = 0.0f;  // for computing candidate pairs

  // whether to entirely recompute connected component ids after each edge collapse
  bool _recompute_component_ids = false;

  /* the data type for recording the operations */
  struct EcolRecord {
    int vsid;
    int vtid;
    int position_bit;
    Point delta_p;
    std::vector<TopoRecord> topo_record_lst;  // you need to convert this into a code string
    /* convert to python dictionary */
    py::dict to_dict(const std::string& code) const {
      py::dict d;
      d["vsid"] = vsid - 1;  // >= 0
      d["vtid"] = vtid - 1;  // >= 0
      d["code"] = code;
      d["position_bit"] = position_bit;
      d["delta_p"] = py::cast(std::array<float, 3>{delta_p[0], delta_p[1], delta_p[2]});  // to a list
      return d;
    }
  };

  /* compute connected components for vertices (update "_component_id") */
  int compute_connected_components_() {
    // gather all edges
    std::vector<Simplex> edges;
    for (auto e : ordered_simplices_dim(1)) {
      edges.push_back(e);
    }

    // Union-Find
    DisjointSet ds;

    // initialize the disjoint set
    for (auto e : edges) {
      Simplex v0 = e->getChild(0);
      Simplex v1 = e->getChild(1);
      ds.add(v0);
      ds.add(v1);
      ds.unite(v0, v1);
    }

    // assign a component id to each vertex
    std::unordered_map<Simplex, int> rep_to_id;  // root --> id
    int id_counter = 0;
    std::vector<Simplex> sorted_vertices;
    for (auto& [vtx, _] : ds.parent) sorted_vertices.push_back(vtx);
    // sort vertices by id to ensure deterministic assignment of connected component ids.
    std::sort(sorted_vertices.begin(), sorted_vertices.end(), SimplexIdCompare());
    for (Simplex vtx : sorted_vertices) {
      Simplex root = ds.find(vtx);
      if (!rep_to_id.count(root)) {
        rep_to_id[root] = id_counter++;
      }
      vtx->_component_id = rep_to_id[root];  // assign
    }

    // total number of connected components
    return id_counter;
  }

  /* compute candidate vertex pairs, as described by the "Progressive Simplicial Complexes" paper */
  auto compute_candidate_pairs() {
    // gather all the vertices into an organized array
    std::vector<Point> verts;
    std::vector<Simplex> verts_ptr;

    // gather all vertices
    for (auto s : ordered_simplices_dim(0)) {
      verts.push_back(s->getPosition());
      verts_ptr.push_back(s);
    }

    // compute Delaunay triangulation
    const auto& tets = compute_delaunay_3d(verts);

    // compute connected component id for each vertex
    compute_connected_components_();

    // candidate vertex pairs
    std::set<std::tuple<int, int>> candidate_pairs;

    // a helper to add a pair of vertices
    auto add = [&candidate_pairs](Simplex v0, Simplex v1) {
      assertx(v0->getDim() == 0 && v1->getDim() == 0);
      if (v0->getId() > v1->getId()) {
        std::swap(v0, v1);
      }
      // add in sorted order
      candidate_pairs.insert({v0->getId(), v1->getId()});
    };

    // add all the edges presented in the simplicial complex
    for (auto e : ordered_simplices_dim(1)) {
      // this is unique
      Simplex v0 = e->getChild(0);
      Simplex v1 = e->getChild(1);
      assertx(v0->_component_id == v1->_component_id);
      add(v0, v1);
    }

    // add edges from the Delaunay complex that connect with different components
    for (const auto& tet : tets) {
      auto edges =
          computeConvexHullEdge(verts[tet[0]], verts[tet[1]], verts[tet[2]], verts[tet[3]], _ratio_degeneracy);
      // for each edge pair from the tetrahedron
      for (auto [i, j] : edges) {
        Simplex v0 = verts_ptr[tet[i]];
        Simplex v1 = verts_ptr[tet[j]];
        assertx(v0->_component_id >= 0);
        assertx(v1->_component_id >= 0);
        // only add the edge if they are in different components
        if (v0->_component_id != v1->_component_id) {
          add(v0, v1);
        }
      }
    }

    return candidate_pairs;
  }

  /* compute the edge collapse record
      note : 
        the topology can be changed, but vertex simplex should not be destroyed
        because we need to access vertex information
   */
  EcolRecord compute_ecol_record(int vsid, int vtid, float w_p0) {
    Simplex vs = getSimplex(0, vsid);
    Simplex vt = getSimplex(0, vtid);
    assertx(vs && vt);

    int position_bit = 1;
    Point delta_p = vt->getPosition() - vs->getPosition();
    if (w_p0 == 0.5f) {
      position_bit = 0;
      delta_p *= 0.5f;
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

  /* evaluate the cost of edge contraction between two vertices
   */
  std::pair<float, float> compute_contraction_cost_and_location(int v0_id, int v1_id) {
    Simplex v0 = getSimplex(0, v0_id);
    Simplex v1 = getSimplex(0, v1_id);
    assert(v0 && v1);
    const Point& p0 = v0->getPosition();
    const Point& p1 = v1->getPosition();
    Point mid = 0.5f * (p0 + p1);

    // fused quadric
    Matrix3 A = {v0->_A[0] + v1->_A[0], v0->_A[1] + v1->_A[1], v0->_A[2] + v1->_A[2]};
    Point b = v0->_b + v1->_b;
    float c = v0->_c + v1->_c;

    // to evaluate the cost
    auto evaluate_cost = [&A, &b, &c](const Point& x) {
      float val = c;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) val += x[i] * A[i][j] * x[j];
      val += 2.0f * dot(b, x);
      // ideally >= 0, but might due to numerical error
      return std::max(val, 0.0f);
    };

    // evaluate cost at three points, take the one with minimum cost
    std::array<Point, 3> candidates = {p0, p1, mid};
    int i_best = 2;
    float cost_min = std::numeric_limits<float>::max();
    for (int i : {2, 1, 0}) {
      float cost = evaluate_cost(candidates[i]);
      if (cost < cost_min) {
        i_best = i;
        cost_min = cost;
      }
    }

    // fused location is at: p0 * w_p0 + p1 * (1 - w_p0)
    float w_p0;
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
    if (_weighting_balanced > 0.0f) {
      // determine the value for balancing
      float value;
      if (_balanced_depth) {
        float depth_0 = static_cast<float>(v0->_subtree_depth);
        float depth_1 = static_cast<float>(v1->_subtree_depth);
        value = 1.0f + std::max(depth_0, depth_1);
      } else {
        float size_0 = static_cast<float>(v0->_subtree_size);
        float size_1 = static_cast<float>(v1->_subtree_size);
        value = size_0 + size_1;
      }
      // piecewise function
      float ratio;
      if (_alpha_balanced > 0.0f) {
        ratio = std::pow(_alpha_balanced, value);  // exponential growth
      } else if (_alpha_balanced == 0.0f) {
        ratio = value;  // linear growth
      } else {
        ratio = std::pow(value, -_alpha_balanced);  // polynomial growth
      }
      // the multiplicative factor
      float balance_factor = 1.0f + _weighting_balanced * ratio;
      cost_min *= balance_factor;
    }

    // return the cost and the location weighting
    return {cost_min, w_p0};
  }

  /* perform simplicial complex simplification, until a single vertex
       note : because it depends on "SplitRecord", so it is best to move the implementation to the cpp file
  */
  std::tuple<std::array<float, 3>, std::vector<py::dict>, std::vector<float>> perform_simplification(
      bool markov = false);
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

inline float ISimplex::length2() const {
  assertx(getDim() == 1);
  return dist2(getChild(0)->getPosition(), getChild(1)->getPosition());
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
