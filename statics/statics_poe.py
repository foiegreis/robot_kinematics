
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STATICS - PRODUCT OF EXPONENTIALS
Code to compute the Torques in Static equilibrium of a Robot, given the Screw Axes, the current Joint Configuration
and the Wrenches vector
#Date Created: april 2024

"""

import numpy as np

def near_zero(z):
    """
    Determines whether a scalar is zero
    :param z: scalar
    :return: bool
    """
    return abs(z) < 1e-6


def axis_angle(p):
    """ Computes the axis angle representation of a 3D vector of exponential coordinates
    :param p: 3D vector of exponential coordinates OmegaTheta = [e1, e2, e3]
    :return: axis-angle representation (omega, theta)
    """
    return p / np.linalg.norm(p), np.linalg.norm(p)


def vec3_to_skew3(p):
    """
    Returns the skew symmetric matrix representation [p] of a 3D  vector
    :param p: 3D vector
    :return: 3x3 matrix
    """
    p_sk = np.array([[0, -p[2], p[1]],
                     [p[2], 0, -p[0]],
                     [-p[1], p[0], 0]])
    return p_sk


def vec6_to_skew4(s):
    """
    Returns the skew symmetric matrix representation [S] of a 6D twist vector
    :param s: 6D twist vector s = [omega, v]
    :return: 4x4 matrix [s] = [[ [omega], v],[0, 0]]
    """
    omega = s[0:3]
    v = s[3:]
    p_sk = vec3_to_skew3(omega)
    twist_sk = np.r_[np.c_[p_sk, v.reshape(3, 1)],[[0, 0, 0, 0]]]
    return twist_sk


def skew3_to_vec3(p_skew):
    """Returns the 3D vector of a 3x3 Skew Symmetric Matrix
    :param p: [p] = skew symmetric matrix
    :return : 3D vector
    """
    p = np.r_[[p_skew[2][1], p_skew[0][2], p_skew[1][0]]]
    return p


def htm_to_rp(T):
    """ Extracts R and p from a homogeneous transformation matrix T
    :param T: 4x4 homogeneous transformation matrix
    :return R: 3x3 rotation matrix
    :return p: 3D position vector
    """
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    return R, p


def htm_adj(T):
    """ Computes the 6X6 skew symmetric adjoint representation of a 4x4 transformation matrix T
    :param T: 4x4 homogeneous transformation matrix in SE(3)
    :return adj_T : 6x6 adjoint representation of T
    """

    R, p = htm_to_rp(T)
    p_sk = vec3_to_skew3(p)
    return np.r_[np.c_[R, np.zeros((3, 3))], np.c_[np.dot(p_sk, R), R]]


def skew3_to_matrix_exp3(p_sk):
    """
    Computes the 3x3 matrix exponential Rot(p_hat, theta) = e^[p]theta of the exp coordinates p_hat*theta, given [p] and theta
    using the Rodrigues formula for rotations
    :param p_sk: 3x3 skew symmetric matrix
    :param theta: angle [rad]
    :return: 3x3 matrix exponential
    """

    ptheta = skew3_to_vec3(p_sk) # exponential coordinates OmegaTheta
    if near_zero(np.linalg.norm(ptheta)):
        mat_exp = np.eye(3)
        return mat_exp
    else:
        theta = axis_angle(ptheta)[1]
        p_sk_pure = p_sk / theta
        mat_exp = np.array(np.eye(3) + np.sin(theta) * p_sk_pure + (1 - np.cos(theta)) * np.dot(p_sk_pure, p_sk_pure))
        res = np.where(near_zero(mat_exp), 0, mat_exp)
        return res


def skew4_to_matrix_exp4(s_sk):
    """Computes the matrix exponential of a 4x4 skew matrix representation of a 6D vector (Screw axis)
    :param s_sk: 4X4 skew symmetric matrix [s] = [[omega], v],[0, 0]]
    :return mat_exp: 4X4 transformation matrix
    """

    omegatheta_sk = s_sk[0:3, 0:3]
    omegatheta = skew3_to_vec3(omegatheta_sk)  # 3D vector of exponential coordinates OmegaTheta
    vtheta = s_sk[0:3, 3]

    # Case Prismatic Joint:
    if near_zero(np.linalg.norm(omegatheta)):
        # return [[I, v*theta], [0, 1]]
        return np.r_[np.c_[np.eye(3), vtheta], [[0, 0, 0, 1]]]

    # Case Revolute Joint
    else:
        # return [[e^[omega]theta, G(theta)v],[0, 1]]
        theta = axis_angle(omegatheta)[1]
        omega_sk = omegatheta_sk / theta
        matexp3 = skew3_to_matrix_exp3(omegatheta_sk)
        G = np.eye(3)*theta + (1 - np.cos(theta))*omega_sk + (theta - np.sin(theta)) * np.dot(omega_sk, omega_sk)
        v = np.dot(G, vtheta)/theta
        matexp4 = np.r_[np.c_[matexp3, v], [[0, 0, 0, 1]]]
        return matexp4


def jacobian_body(b_list, theta_list):
    """Computes the body jacobian given the list of screw axes in body form and the joint configuration
    :param b_list: 6xn matrix of the screw axes in body form (screw axes are the rows)
    :param theta_list: list of the joints configurations
    :return: 6xn jacobian matrix in body form
    """
    # we will compose J by columns
    b_list = np.array(b_list)
    Jb = np.array(b_list.T).copy().astype(float)

    T = np.eye(4)

    for i in range(len(theta_list) - 2, -1, -1):

        b = b_list[i+1, :]
        b_skew = vec6_to_skew4(b * - theta_list[i+1])
        mat_exp = skew4_to_matrix_exp4(b_skew)
        T = np.dot(T, mat_exp)

        adj_T = htm_adj(T)
        J_col = np.dot(adj_T, b_list[i, :])
        Jb[:, i] = J_col
    return Jb


def jacobian_space(s_list, theta_list):
    """Computes the space jacobian given the list of screw axes in space form and the joint configuration
    :param s_list: 6xn matrix of the screw axes in space form (screw axes are the rows)
    :param theta_list: list of the joints configurations
    :return: 6xn jacobian matrix in space form
    """

    s_list = np.array(s_list)
    Js = np.array(s_list.T).copy().astype(float)

    T = np.eye(4)
    for i in range(1, len(theta_list)):

        s = s_list[i - 1, :]
        s_skew = vec6_to_skew4(s * theta_list[i - 1])
        mat_exp = skew4_to_matrix_exp4(s_skew)
        T = np.dot(T, mat_exp)

        adj_T = htm_adj(T)
        J_col = np.dot(adj_T, s_list[i, :])
        Js[:, i] = J_col
    return Js


def is_singularity(J):
    """Evaluates if the Jacobian is at a singularity"""
    return np.linalg.matrix_rank(J) < min(np.shape(J))

def statics(J, F):
    """Computes the statics equation of the joint torques given the Jacobian Matrix and the
     Wrench of end-effector forces
     param: J = jacobian matrix
     param: F = wrench vector 6x1
     returns: tau = joint forces vector n x 1
     """

    # planar case
    if len(F) == 3:
        J = J[2:5, :]

    J_trans = np.transpose(J)
    print(J_trans)
    tau = np.dot(J_trans, F)
    return tau


if __name__ == "__main__":
    # EXAMPLE: PRRRRP Planar Robot

    # current configuration
    thetalist = np.array([0, 0, 0, 0, 0, 0])

    # wrench
    wrench = np.array([0, 1, -1, 1, 0, 0])

    # screw axes in space form
    s_list = np.array([[0, 0, 0, 0, 0, 1],
                       [1, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 0, 0],
                       [0, 0, 1, 1, 1, 0],
                       [0.7071, -0.7071, 0, 0, 0, -0.7071],
                       [0, 0, 0, 0, 1, 0]])

    Js = jacobian_space(s_list, thetalist)
    print(f"\nSpace Jacobian Js: \n{Js}")

    singularity = is_singularity(Js)
    print(f"\nIs the Jacobian Singular at the configuration {thetalist}? \n{'No' if not singularity else 'Yes'}")


    tau = statics(Js, wrench)
    print(f"\nStatics: joint torques and forces for the wrench {wrench}: \n{tau}")

