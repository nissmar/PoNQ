import numpy as np
from scipy.spatial import Delaunay
import torch
from mesh_tools import tet_circumcenter, SIGNS
import networkx as nx


def quadrics_score(quadrics, points):
    '''quadrics: Nx4x4 points: Nx3'''
    new_points = torch.cat((points, torch.ones_like(points[..., :1])), -1)
    return (new_points[..., None, :]@quadrics@new_points[..., None]).squeeze(-1).squeeze(-1)


def face_orientation(p1, p2, p3, vp1):
    return (np.cross(p2 - p1, p3 - p1) * (vp1 - (p1+p2+p3)/3.)).sum() > 0


class SurfaceFromQuadrics(Delaunay):
    '''Mesh from PoNQ'''
    def __init__(self, vstars: torch.tensor, eigs: torch.tensor, quadrics: torch.tensor, normals: torch.tensor, add_corners=True, compute_mincut=True, rvd_tresh=-1, grid_scale=32, correct_tet_color=True, **kwargs) -> None:
        self.add_corners = add_corners
        if self.add_corners:
            vstars = torch.cat(
                (vstars, torch.tensor(SIGNS, device=vstars.device)))
            quadrics = torch.cat((quadrics, torch.eye(
                4, device=vstars.device).repeat(8, 1, 1)))
            normals = torch.cat((normals, -torch.tensor(SIGNS/np.sqrt(
                (SIGNS**2).sum(-1, keepdims=True)), device=vstars.device)))
            eigs = torch.cat((eigs, torch.ones(8, 3, device=vstars.device)))
        self.vstars = vstars
        self.quadrics = quadrics/eigs[:, 0, None, None]
        self.normals = normals
        self.eigs = eigs
        self.circum_tresh = 3*(2/grid_scale)**2
        super().__init__(vstars.cpu().detach().numpy(), **kwargs)
        self.scaling = 1-self.eigs[:, 1]/self.eigs[:, 0]
        self.circum_centers = tet_circumcenter(self.points[self.simplices])
        self.triangle_faces, self.triangle_faces_neighbors = self.get_triangle_faces()
        self.in_mask = self.order_neighbors()
        self.triangle_faces_scores = self.get_faces_score(self.triangle_faces)
        self.triangle_normals = self.normals[self.triangle_faces].mean(
            1).cpu().detach().numpy()
        self.triangle_areas = np.sqrt((np.cross(
            self.points[self.triangle_faces[:, 1]] - self.points[self.triangle_faces[:, 0]], self.points[self.triangle_faces[:, 2]] - self.points[self.triangle_faces[:, 0]])**2).sum(-1))
        self.tet_colors = self.get_init_tet_color()
        if correct_tet_color:
            self.correct_tet_color()
        self.add_void_vertices()
        self.final_scores = self.get_faces_score(
            self.triangle_faces)*grid_scale**2 + (self.get_face_normal_align()/1.5)**2
        if compute_mincut:
            self.min_cut()

    def get_triangle_faces(self):
        opp_face = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
        ii = np.arange(len(self.neighbors))
        triangle_faces = -np.ones((len(self.neighbors)*4, 3), dtype=int)

        for j in range(4):
            triangle_faces[4*ii + j] = self.simplices[:, opp_face[j]]
        triangle_faces_neighbors = np.column_stack((np.arange(
            len(self.neighbors)).repeat(4), self.neighbors.reshape(len(self.neighbors)*4)))

        return triangle_faces, triangle_faces_neighbors

    def outside_tet_mask(self):
        return (self.simplices >= len(self.points)-8).any(-1)

    def face_orientation(self, p1, p2, p3, vp1):
        return (np.cross(p2 - p1, p3 - p1) * (vp1 - (p1+p2+p3)/3.)).sum(-1) > 0

    def order_neighbors(self):
        opp_vert = self.simplices.reshape(
            len(self.simplices[self.triangle_faces_neighbors]))
        in_mask = self.face_orientation(
            *np.transpose(self.points[self.triangle_faces], (1, 0, 2)), self.points[opp_vert])

        in_triangle = self.triangle_faces
        flipped_triangles = np.fliplr(in_triangle)
        self.triangle_faces = in_triangle * \
            in_mask[:, None] + (1-in_mask[:, None])*flipped_triangles
        return in_mask

    def get_faces_score(self, faces):
        tri_score = []
        for i, j, k in [[0, 1, 2], [1, 2, 0], [2, 1, 0]]:
            # scaling = self.scaling[faces[:, i]]
            quadric_score = quadrics_score(self.quadrics[faces[:, i]], self.vstars[faces[:, j]])+quadrics_score(
                self.quadrics[faces[:, i]], self.vstars[faces[:, k]])
            tri_score.append(quadric_score)
        return torch.row_stack(tri_score).sum(0).cpu().detach().numpy()

    def get_faces_score_inv_scaling(self, faces, tresh=.8):
        glob_scaling = (self.scaling[faces[:, 0]] < 1-tresh)*(
            self.scaling[faces[:, 1]] < 1-tresh)*(self.scaling[faces[:, 2]] < 1-tresh)
        return glob_scaling == 0

    def get_face_normal_align(self):
        predicted_vertices_normals = self.normals[self.triangle_faces].cpu(
        ).detach().numpy()
        tf_vertices_length = np.sqrt((predicted_vertices_normals ** 2).sum(-1))
        predicted_vertices_normals /= tf_vertices_length[..., None]
        tf_normals = np.cross(self.points[self.triangle_faces[:, 1]]-self.points[self.triangle_faces[:, 0]],
                              self.points[self.triangle_faces[:, 2]]-self.points[self.triangle_faces[:, 0]])
        tf_normals /= np.sqrt((tf_normals**2).sum(-1, keepdims=True))

        angle = np.arccos(
            (tf_normals[:, None, :]*predicted_vertices_normals).sum(-1))
        return (angle).sum(-1)

    def get_init_tet_color(self):
        vects = (self.points[self.simplices] - self.circum_centers[:, None, :])
        tet_color = ((vects*self.normals.cpu().detach().numpy()
                      [self.simplices]).sum(-1) > 0).sum(-1)
        tet_color[(self.simplices >= len(self.points)-8).any(-1)] = 0
        return tet_color

    def correct_tet_color(self):
        bary_centers = self.points[self.simplices].mean(1)
        vects_from_bary = bary_centers[:, None, :]-self.points[self.simplices]
        dist_to_plane = (
            vects_from_bary*self.normals[self.simplices].cpu().detach().numpy()).sum(-1)
        is_prob_in = dist_to_plane.mean(-1) < 0
        is_prob_out = dist_to_plane.mean(-1) > 0
        self.tet_colors[(self.tet_colors == 0)*is_prob_in] = 1
        self.tet_colors[(self.tet_colors == 4)*is_prob_out] = 1

    def add_void_vertices(self):
        # identify points with no tets
        is_represented = np.zeros((len(self.points), 2), dtype=bool)
        is_represented[self.simplices[self.tet_colors == 0].flatten(),
                       0] = True
        is_represented[self.simplices[self.tet_colors == 4].flatten(),
                       1] = True

        lack_outside = np.logical_not(is_represented[:, 0])
        lack_inside = np.logical_not(is_represented[:, 1])
        lack_outside[len(self.points)-8:] = False
        lack_inside[len(self.points)-8:] = False

        void_points = np.arange(len(self.points))[lack_inside+lack_outside]

        circum_centers = tet_circumcenter(self.points[self.simplices])
        vects = (circum_centers[:, None, :]-self.points[self.simplices])
        voro_dist = (vects*self.normals.cpu().detach().numpy()
                     [self.simplices]).sum(-1)
        vects = (self.points[self.simplices].mean(1)[
                 :, None, :]-self.points[self.simplices])
        voro_dist = (vects**2).sum(-1).mean(-1)[:, None].repeat(4, 1)
        circum_idx = np.arange(len(self.simplices))[:, None].repeat(4, 1)
        for i in void_points:
            if (lack_outside[i] and lack_outside[i]):
                pass
            else:
                if (voro_dist[self.simplices == i].min()) < self.circum_tresh:
                    candidate_tet = circum_idx[self.simplices ==
                                               i][voro_dist[self.simplices == i].argmin()]
                    if lack_outside[i]:
                        self.tet_colors[candidate_tet] = 0
                    else:
                        self.tet_colors[candidate_tet] = 4

    def min_cut(self):
        scores = self.final_scores
        self.tet_colors = self.tet_colors.astype(int)
        tet_edges = self.triangle_faces_neighbors.copy()
        non_inf = (tet_edges != -1).all(-1)
        min_in = np.arange(*self.tet_colors.shape)[self.tet_colors == 4].min()
        min_out = np.arange(*self.tet_colors.shape)[self.tet_colors == 0].min()
        tet_edges[self.tet_colors[tet_edges] == 4] = min_in
        tet_edges[self.tet_colors[tet_edges] == 0] = min_out

        relevant_edges = (tet_edges[:, 0] != tet_edges[:, 1])*non_inf
        edges = np.fliplr(tet_edges[relevant_edges])
        r_scores = scores[relevant_edges]

        network = nx.DiGraph()
        network.add_nodes_from(np.unique((edges)))
        for e, s in zip(edges, r_scores):
            if network.has_edge(e[0], e[1]):
                network[e[0]][e[1]]['capacity'] += s
            else:
                network.add_edge(e[0], e[1], capacity=s)
        cut, partition = nx.minimum_cut(network, min_in, min_out)
        self.tet_colors[np.array(list(partition[1]))] = 0 if (
            min_out in partition[1]) else 4
        self.tet_colors[np.array(list(partition[0]))] = 4 if (
            min_out in partition[1]) else 0

    def get_surface(self, return_scores=False, color_func=None, treshold=0, open_treshold=None, return_indices=False):
        neigh_color = self.tet_colors[self.triangle_faces_neighbors]
        neigh_color[self.triangle_faces_neighbors == -1] = 0
        neigh_color = neigh_color > treshold
        # two triangles per face: select only one
        in_t = (neigh_color[:, 0] == 0)*(neigh_color[:, 1] == 1)
        in_triangle = self.triangle_faces[in_t]
        in_scores = self.triangle_faces_scores[in_t]

        # select only the relevant vertices
        un = np.unique(in_triangle)
        inv = np.arange(in_triangle.max() + 1)
        inv[un] = np.arange(len(un))
        nvertices = self.points[un]
        in_triangle = inv[in_triangle]
        to_return = [nvertices]
        if not open_treshold is None:
            in_scores = self.get_faces_score_inv_scaling(
                self.triangle_faces, open_treshold).cpu().detach().numpy()
            to_return.append(in_triangle[in_scores[in_t]])
            in_scores = in_scores[in_t][in_scores[in_t]]
        else:
            to_return.append(in_triangle)
        if return_scores:
            to_return.append(in_scores)
        elif not color_func is None:
            to_return.append(color_func[in_t])
        if return_indices:
            to_return.append(un)
        return to_return
