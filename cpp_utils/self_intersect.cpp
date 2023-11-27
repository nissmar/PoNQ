#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Polygon_mesh_processing/polygon_mesh_to_polygon_soup.h>
#include <fstream>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Surface_mesh<K::Point_3> Mesh;
typedef boost::graph_traits<Mesh>::face_descriptor face_descriptor;

namespace PMP = CGAL::Polygon_mesh_processing;
// COMPUTES THE NUMBER OF SELF INTERSECTIONS
int main(int argc, char *argv[])
{
  const char *filename = argv[1];
  std::ifstream input(filename);
  Mesh mesh;
  PMP::IO::read_polygon_mesh(filename, mesh);
  if (!CGAL::is_triangle_mesh(mesh))
  {
    std::cerr << "Not a valid input file." << std::endl;
    return 1;
  }

  std::vector<K::Point_3> soup_points;
  std::vector< std::vector<std::size_t> > soup_polygons;
  PMP::polygon_mesh_to_polygon_soup(mesh, soup_points, soup_polygons); 
  PMP::merge_duplicate_points_in_polygon_soup(soup_points, soup_polygons);
  std::cout << PMP::does_triangle_soup_self_intersect(soup_points, soup_polygons)<<std::endl;
}
