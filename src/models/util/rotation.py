import torch
import math
import numpy as np

def t_get_relative_transform(cam_i, cam_j, t_lambda=0.5):
    """
    Construct the transform matrix which maps from the
    coordinate system of canonical (cam2) to the system
    in viewpoint (cam1).

    The coordinates of both cameras are with respect to
    world coordinates. If we wanted to map from the
    canonical to the viewpoint camera, we need to
    derive the relative matrix that does that [2].

    If P_{wv} and P_{wc} denote the transform matrices
    that map from world->viewpoint and world->canonical
    respectively, then we need to make the following
    matrix P_{vc} mapping from view to canonical:

             -----------------------------
    P_{vc} = | Rc.T * Rv    Rc.T*(tv-tc) |
             |     0             1       |
             -----------------------------

    `cam_batch` is in the (Blender) format [1]:
    (t_x, t_y, t_z, theta_x, theta_y, theta_z),
    where in Blender the z axis points up and y
    is depth. But in PyTorch it's the other way
    around, so when we construct R and T we need
    to swap z and y around.

    References:

    [1] https://bit.ly/2Y7yhTj
    [2] https://bit.ly/3hEeJhh

    NOTE:
    - Zeroing of z
    - Division of offsets

    """

    # t_get_theta expects angles in
    # (z,y,x) format. But y and z
    # needs to be switched around
    # due to the difference in coord
    # system btwn blender and pytorch
    cam_i_x = cam_i[:, 3:4]
    cam_i_y = cam_i[:, 4:5]
    cam_i_z = cam_i[:, 5:6]
    cam_j_x = cam_j[:, 3:4]
    cam_j_y = cam_j[:, 4:5]
    cam_j_z = cam_j[:, 5:6]
    # Switch them here
    R_i = t_get_theta(torch.cat((
        cam_i_y,
        cam_i_z,
        cam_i_x), dim=1))
    R_j = t_get_theta(torch.cat((
        cam_j_y,
        cam_j_z,
        cam_j_x), dim=1))
    # Translations are in format (x,y,z),
    # so here switch y and z around like
    # before. Objects are on a flat plane
    # so zero out the z axis, and also
    # divide tx,ty by 2 so it lies in the range
    # [-4, +4] (roughly).
    cam_i_tx = cam_i[:, 0].view(-1, 1, 1) * t_lambda
    cam_i_ty = cam_i[:, 1].view(-1, 1, 1) * t_lambda
    cam_i_tz = cam_i[:, 2].view(-1, 1, 1) * 0.
    T_i = torch.cat((cam_i_tx,
                     cam_i_tz,
                     cam_i_ty), dim=1)

    cam_j_tx = cam_j[:, 0].view(-1, 1, 1) * t_lambda
    cam_j_ty = cam_j[:, 1].view(-1, 1, 1) * t_lambda
    cam_j_tz = cam_j[:, 2].view(-1, 1, 1) * 0.
    T_j = torch.cat((cam_j_tx,
                     cam_j_tz,
                     cam_j_ty), dim=1)

    """
    def compute_relative(R_i, R_j, T_i, T_j):
    # relative pose from camera i to camera j
    R_rel = np.dot(R_j, R_i.transpose(1,0))
    T_rel = np.dot(-R_rel, T_i) + T_j
    return R_rel, T_rel
    """

    R_rel = torch.bmm(R_j, R_i.transpose(2,1))
    T_rel = torch.bmm(-R_rel, T_i) + T_j

    return torch.cat((R_rel, T_rel), dim=2).detach()

def t_rot_matrix_x(theta):
    """
    theta: measured in radians
    """
    bs = theta.size(0)
    mat = torch.zeros((bs, 3, 3)).float()
    if theta.is_cuda:
        mat = mat.cuda()
    mat[:, 0, 0] = 1.
    mat[:, 1, 1] = torch.cos(theta).view(-1)
    mat[:, 1, 2] = -torch.sin(theta).view(-1)
    mat[:, 2, 1] = torch.sin(theta).view(-1)
    mat[:, 2, 2] = torch.cos(theta).view(-1)
    return mat

def t_rot_matrix_y(theta):
    """
    theta: measured in radians
    """
    bs = theta.size(0)
    mat = torch.zeros((bs, 3, 3)).float()
    if theta.is_cuda:
        mat = mat.cuda()
    mat[:, 0, 0] = torch.cos(theta).view(-1)
    mat[:, 0, 2] = torch.sin(theta).view(-1)
    mat[:, 1, 1] = 1.
    mat[:, 2, 0] = -torch.sin(theta).view(-1)
    mat[:, 2, 2] = torch.cos(theta).view(-1)
    return mat

def t_rot_matrix_z(theta):
    """
    theta: measured in radians
    """
    bs = theta.size(0)
    mat = torch.zeros((bs, 3, 3)).float()
    if theta.is_cuda:
        mat = mat.cuda()
    mat[:, 0, 0] = torch.cos(theta).view(-1)
    mat[:, 0, 1] = -torch.sin(theta).view(-1)
    mat[:, 1, 0] = torch.sin(theta).view(-1)
    mat[:, 1, 1] = torch.cos(theta).view(-1)
    mat[:, 2, 2] = 1.
    return mat

def t_get_theta(angles, offsets=None):
    '''Construct a rotation matrix from angles. (This is
    the differentiable version, in PyTorch code.)

    angles should be an nx3 matrix with z,y,z = 0,1,2
    '''

    angles_z = angles[:, 0]
    angles_y = angles[:, 1]
    angles_x = angles[:, 2]

    if offsets is not None:
        trans_x = offsets[:, 0]
        trans_y = offsets[:, 1]
        trans_z = offsets[:, 2]

    thetas = torch.bmm(torch.bmm(t_rot_matrix_z(angles_z),
                                 t_rot_matrix_y(angles_y)),
                       t_rot_matrix_x(angles_x))
    trans = torch.zeros((thetas.size(0), 3, 1))

    if offsets is not None:
        trans[:, 0, :] = trans_x.view(-1, 1)
        trans[:, 1, :] = trans_y.view(-1, 1)
        trans[:, 2, :] = trans_z.view(-1, 1)
    if angles.is_cuda:
        trans = trans.cuda()
    thetas = torch.cat((thetas, trans), dim=2) # add zero padding
    return thetas

# FROM https://github.com/Wallacoloo/printipi/blob/master/util/rotation_matrix.py
def R_to_axis_angle(matrix):
    """Convert the rotation matrix into the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        x = Qzy-Qyz
        y = Qxz-Qzx
        z = Qyx-Qxy
        r = hypot(x,hypot(y,z))
        t = Qxx+Qyy+Qzz
        theta = atan2(r,t-1)
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @return:    The 3D rotation axis and angle.
    @rtype:     numpy 3D rank-1 array, float
    """

    # Axes.
    axis = np.zeros(3, np.float64)
    axis[0] = matrix[2,1] - matrix[1,2]
    axis[1] = matrix[0,2] - matrix[2,0]
    axis[2] = matrix[1,0] - matrix[0,1]

    # Angle.
    r = np.hypot(axis[0], np.hypot(axis[1], axis[2]))
    t = matrix[0,0] + matrix[1,1] + matrix[2,2]
    theta = math.atan2(r, t-1)

    # Normalise the axis.
    axis = axis / r

    # Return the data.
    return axis[None], np.asarray([theta])[None]
