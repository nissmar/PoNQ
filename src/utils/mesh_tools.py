"""Various mesh utilities"""
import numpy as np
import trimesh
import igl
from sklearn.neighbors import KDTree
from skimage import measure
from scipy.spatial import Delaunay

SIGNS = np.array(
    [
        [(-1) ** i, (-1) ** j, (-1) ** k]
        for i in range(2)
        for j in range(2)
        for k in range(2)
    ]
)  # +1 or -1 for all coordinates


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Mesh operations
def NDCnormalize(vertices, return_scale=False):
    """normalization in half unit ball"""
    vM = vertices.max(0)
    vm = vertices.min(0)
    scale = np.sqrt(((vM - vm) ** 2).sum(-1))
    mean = (vM + vm) / 2.0
    nverts = (vertices - mean) / scale
    if return_scale:
        return nverts, mean, scale
    return nverts


def normalize(v: np.ndarray, retscale=False):
    """center and normalize vertices in [-1,1]^3"""
    center = (v.max(0) + v.min(0)) / 2
    v -= center
    scale = np.abs(v).max()
    v /= scale
    if retscale:
        return v, center, scale
    return v


def matched_normalize(v: np.ndarray, center: np.ndarray, scale: float):
    """center and normalize vertices in [-1,1]^3"""
    return (v - center) / scale


def subdivide_and_smooth(mesh: trimesh.Trimesh):
    """subdivides trimesh object"""
    mesh = mesh.subdivide()
    trimesh.smoothing.filter_humphrey(mesh)
    return mesh


def sample_surface(ref_mesh: trimesh.Trimesh, surface_samples=int(1e4)):
    """samples surface of a trimesh object"""
    sampled_v, sampled_faces = trimesh.sample.sample_surface_even(
        ref_mesh, surface_samples
    )
    sampled_normals = ref_mesh.face_normals[sampled_faces]
    return sampled_v, sampled_normals


def extract_largest_connected_component(new_mesh: trimesh.Trimesh):
    """extract largest component of a trimesh"""
    connected_c = trimesh.graph.connected_components(new_mesh.edges)
    max_c = np.array([e.shape for e in connected_c]).argmax()
    keep_vertices = connected_c[max_c]
    mask = np.zeros(len(new_mesh.vertices), dtype=bool)
    mask[keep_vertices] = True
    new_mesh.update_vertices(mask)
    return new_mesh


def mesh_grid(grid_size: int, normalize=False):
    """create mesh grid with default indexing"""
    xx, yy, zz = np.mgrid[:grid_size, :grid_size, :grid_size]
    grid_3d = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
    if normalize:
        return 2 * (grid_3d / (grid_size - 1)) - 1
    return grid_3d


def mesh_from_voxels(vox: np.ndarray, iso=0.0, ret=False):
    """marching cube from NxNxN array"""
    im_res = vox.shape[0]
    vox_v, vox_f, _, _ = measure.marching_cubes(
        vox, iso, spacing=[1.0 for i in range(3)]
    )
    vox_v = 2 * (vox_v / (im_res - 1)) - 1
    nf = vox_f.copy()
    if ret:
        vox_f[:, 0], vox_f[:, 1] = nf[:, 1], nf[:, 0]
    return vox_v.astype(np.float64), vox_f


# Metrics
def mesh_hausdorff(v1: np.ndarray, f1: np.ndarray, v2: np.ndarray, f2: np.ndarray):
    """signed distance: vertices -> mesh"""
    directed_hausdorff1 = igl.signed_distance(v1, v2, f2)[0]
    directed_hausdorff2 = igl.signed_distance(v2, v1, f1)[0]
    return max(directed_hausdorff1.max(), directed_hausdorff2.max())


def mesh_chamfer(v1: np.ndarray, f1: np.ndarray, v2: np.ndarray, f2: np.ndarray):
    """signed distance: vertices -> mesh"""
    d1 = igl.signed_distance(v1, v2, f2)[0]
    d2 = igl.signed_distance(v2, v1, f1)[0]
    return (d1**2).mean() + (d2**2).mean()


def points_distance(
    points1: np.ndarray, points2: np.ndarray, norm="L2", print_result=False
):
    """norm choice: L1, L2, Directed Hausdorff (1->2), Hausdorff"""
    tree = KDTree(points1, leaf_size=32)
    d1, _ = tree.query(points2)
    tree = KDTree(points2, leaf_size=32)
    d2, _ = tree.query(points1)

    if norm == "L1":
        dist = d2.mean() + d1.mean()
    elif norm == "L2":
        dist = (d2**2).mean() + (d1**2).mean()
    elif norm == "Directed Hausdorff":
        dist = d2.max()
    elif norm == "Hausdorff":
        dist = max(d1.max(), d2.max())

    if print_result:
        print("{0}: {1:.4e}".format(norm, dist))
    else:
        return dist


def mask_relevant_voxels(grid_n: int, samples: np.ndarray):
    """subselects voxels which collide with pointcloud"""
    samples_low = np.floor((samples + 1) / 2 * (grid_n - 1)).astype(np.int64)
    mask = np.zeros((grid_n, grid_n, grid_n))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                mask[
                    samples_low[:, 0] + i, samples_low[:, 1] +
                    j, samples_low[:, 2] + k
                ] += 1
    return mask.reshape((grid_n**3)) > 0

# IO


def load_shape(path, normalize='NDC', sample_n=None):
    '''returns points+normals'''
    # mesh
    v, f = igl.read_triangle_mesh(path)
    if normalize == 'NDC':
        v = 2*NDCnormalize(v)
    ref_mesh = trimesh.Trimesh(v, f)
    samples, face_index = trimesh.sample.sample_surface_even(
        ref_mesh, sample_n)
    input_points = np.array(samples)
    input_normals = np.array(ref_mesh.face_normals[face_index])
    return input_points, input_normals

def load_and_sample_shape(model_name: str, src_dir: str, sample_n=1e5, rescale_f=None):
    """load and sample shape with normalization options"""
    v, f = igl.read_triangle_mesh(src_dir + model_name)
    if rescale_f == "NDC":
        v = 2 * NDCnormalize(v)
    elif not (rescale_f is None):
        v = 2 * NDCnormalize(v)
    if sample_n == 0:
        return v, f
    ref_mesh = trimesh.Trimesh(v, f)
    samples, _ = trimesh.sample.sample_surface_even(ref_mesh, int(sample_n))
    samples = np.array(samples)
    return v, f, samples


def sample_mesh_with_normals(ref_mesh, n_samples):
    samples, face_index = trimesh.sample.sample_surface_even(
        ref_mesh, n_samples)
    samples = np.array(samples)
    normals = np.array(ref_mesh.face_normals[face_index])
    return samples, normals


def export_ply(ref_mesh: trimesh.Trimesh, model_name: str, target_dir="./"):
    result = trimesh.exchange.ply.export_ply(ref_mesh, encoding="ascii")
    output_file = open((target_dir + model_name + ".ply"), "wb+")
    output_file.write(result)
    output_file.close()


def export_obj(nv: np.ndarray, nf: np.ndarray, name: str, export_lines=False):
    if name[:-4] != ".obj":
        name += ".obj"
    try:
        file = open(name, "x")
    except:
        file = open(name, "w")
    for e in nv:
        file.write("v {} {} {}\n".format(*e))
    file.write("\n")
    for face in nf:
        header = "l " if export_lines else "f "
        file.write(header + " ".join([str(fi + 1) for fi in face]) + "\n")
    file.write("\n")


def export_off(nv: np.ndarray, nf: np.ndarray, name: str):
    name += ".off"
    try:
        file = open(name, "x")
    except:
        file = open(name, "w")

    file.write("OFF \n")
    file.write("{} {} 0 \n".format(len(nv), len(nf)))
    for e in nv:
        file.write("{} {} {}\n".format(*e))
    for face in nf:
        file.write("{} ".format(len(face)) +
                   " ".join([str(fi) for fi in face]) + "\n")
    file.write("\n")


def export_vmesh(points, values, name, noise=10**-7):
    p1 = points[values >= 0]
    p2 = points[values < 0]

    name += ".vmesh"
    try:
        file = open(name, "x")
    except:
        file = open(name, "w")

    file.write("{} {}\n".format(len(points), len(p1)))
    p1 = p1.astype(np.float64) + np.random.randn(*p1.shape) * noise
    p2 = p2.astype(np.float64) + np.random.randn(*p2.shape) * noise

    for e in p1:
        file.write("{} {} {}\n".format(*e))
    for e in p2:
        file.write("{} {} {}\n".format(*e))
    file.write("\n")


# plot
def meshplot_add_points(mp, points, size=0.04, c=None):
    mp.add_points(points, c, shading={"point_size": size})


# scipy.Spatial
def delaunay_triangle_faces(scipy_delaunay: Delaunay):
    opp_face = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
    tet = scipy_delaunay.simplices
    delaunay_faces = []
    for (i, j, k) in opp_face:
        delaunay_faces.append(np.column_stack(
            (tet[:, i], tet[:, j], tet[:, k])))
    delaunay_faces = np.concatenate(delaunay_faces)
    delaunay_faces = np.unique(delaunay_faces, axis=0)
    return delaunay_faces


def tet_circumcenter(verts):
    # ba = b - a
    ba_x = verts[:, 1, 0] - verts[:, 0, 0]
    ba_y = verts[:, 1, 1] - verts[:, 0, 1]
    ba_z = verts[:, 1, 2] - verts[:, 0, 2]
    # ca = c - a
    ca_x = verts[:, 2, 0] - verts[:, 0, 0]
    ca_y = verts[:, 2, 1] - verts[:, 0, 1]
    ca_z = verts[:, 2, 2] - verts[:, 0, 2]
    # da = d - a
    da_x = verts[:, 3, 0] - verts[:, 0, 0]
    da_y = verts[:, 3, 1] - verts[:, 0, 1]
    da_z = verts[:, 3, 2] - verts[:, 0, 2]
    # Squares of lengths of the edges incident to `a'.
    len_ba = ba_x * ba_x + ba_y * ba_y + ba_z * ba_z
    len_ca = ca_x * ca_x + ca_y * ca_y + ca_z * ca_z
    len_da = da_x * da_x + da_y * da_y + da_z * da_z
    # Cross products of these edges.
    # c cross d
    cross_cd_x = ca_y * da_z - da_y * ca_z
    cross_cd_y = ca_z * da_x - da_z * ca_x
    cross_cd_z = ca_x * da_y - da_x * ca_y
    # d cross b
    cross_db_x = da_y * ba_z - ba_y * da_z
    cross_db_y = da_z * ba_x - ba_z * da_x
    cross_db_z = da_x * ba_y - ba_x * da_y
    # b cross c
    cross_bc_x = ba_y * ca_z - ca_y * ba_z
    cross_bc_y = ba_z * ca_x - ca_z * ba_x
    cross_bc_z = ba_x * ca_y - ca_x * ba_y
    # Calculate the denominator of the formula.
    div_den = (ba_x * cross_cd_x + ba_y *
               cross_cd_y + ba_z * cross_cd_z)
    # coplanar vertices
    mask_div_den = np.abs(div_den) == 0
    div_den[mask_div_den] = 1
    denominator = 0.5 / div_den
    # Calculate offset (from `a') of circumcenter.
    circ_x = (len_ba * cross_cd_x + len_ca * cross_db_x +
              len_da * cross_bc_x) * denominator
    circ_y = (len_ba * cross_cd_y + len_ca * cross_db_y +
              len_da * cross_bc_y) * denominator
    circ_z = (len_ba * cross_cd_z + len_ca * cross_db_z +
              len_da * cross_bc_z) * denominator

    out = np.column_stack((circ_x, circ_y, circ_z))+verts[:, 0]
    out[mask_div_den] = verts[mask_div_den].mean(1)
    return out
