#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FORWARD KINEMATICS - PRODUCT OF EXPONENTIALS APPROACH
Code to compute the PoE Forward Kinematics of a robot with n joints, given the table of Space Form or Body Form twists
and M matrix

#Author: foiegreis
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

def skew4_to_vec6(s_skew):
    """Returns the 6D vector of a 4x4 Skew Symmetric Matrix
    :param s_skew: [s] = [[ [omega], v],[0, 0]]
    :return: s = [omega, v]
    """
    s = np.r_[[s_skew[2][1], s_skew[0][2], s_skew[1][0]],[s_skew[0][3], s_skew[1][3], s_skew[2][3]]]
    return s


def htm_to_rp(T):
    """ Extracts R and p from a homogeneous transformation matrix T
    :param T: 4x4 homogeneous transformation matrix
    :return R: 3x3 rotation matrix
    :return p: 3D position vector
    """
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    return R, p


def htm_inverse(T):
    """Returns the inverse of a 4x4 homogeneous transformation matrix
    :param T: 4x4 homogeneous transformation matrix in SE(3)
    :return T_trans: 4x4 matrix T-1
    """
    R, p = htm_to_rp(T)
    R_transp = np.array(R).T
    return np.r_[np.c_[R_transp, -np.dot(R_transp, p)], [[0, 0, 0, 1]]]


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
        return np.r_[np.c_[np.eye(3), omegatheta_sk], [[0, 0, 0, 1]]]

    # Case Revolute Joint
    else:
        # return [[e^[omega]theta, G(theta)v],[0, 1]]
        theta = axis_angle(omegatheta)[1]
        omega_sk = omegatheta_sk /theta
        matexp3 = skew3_to_matrix_exp3(omegatheta_sk)
        G = np.eye(3)*theta + (1 - np.cos(theta))*omega_sk + (theta - np.sin(theta)) * np.dot(omega_sk, omega_sk)
        v = np.dot(G, vtheta)/theta
        matexp4 = np.r_[np.c_[matexp3, v], [[0, 0, 0, 1]]]
        return matexp4


def fk_body(M, b_list, theta_list):
    """
    Computes the Forward Kinematics given the M matrix, the list of screw axes in Body Form and the configuration
    :param M: 4x4 M0b homogeneous transformation matrix of the zero configuration
    :param b_list: list of the body screw axes for the robot joints
    :param theta_list: configuration of the robot i.e. list of joint values
    :return: 4x4 homogeneous transformation matrix of the end effector for the configuration
    """
    T = np.array(M)
    for i, b in enumerate(b_list):
        b_skew = vec6_to_skew4(np.array(b) * theta_list[i])
        mat_exp = skew4_to_matrix_exp4(b_skew)
        T = np.dot(T, mat_exp)

    fk = np.round(np.where(near_zero(T), 0, T), 4)
    return fk


def fk_space(M, s_list, theta_list):
    """
    Computes the Forward Kinematics given the M matrix, the list of screw axes in Space Form and the configuration
    :param M: 4x4 M0b homogeneous transformation matrix of the zero configuration
    :param s_list: list of the space screw axes for the robot joints
    :param theta_list: configuration of the robot i.e. list of joint values
    :return: 4x4 homogeneous transformation matrix of the end effector for the configuration
    """

    T = np.eye(4)
    for i, s in enumerate(s_list):
        s_skew = vec6_to_skew4(np.array(s) * theta_list[i])
        mat_exp = skew4_to_matrix_exp4(s_skew)
        T = np.dot(T, mat_exp)

    fk = np.matmul(T, M)
    res = np.where(near_zero(fk), 0, np.round(fk, 4))
    return res


if __name__ == "__main__":

    # EXAMPLE: 3R Spatial Robot

    # Known joint configuration
    theta = [0.92519754, 0.58622516, 0.68427316]

    # M matrix
    M = np.array([[1, 0, 0, 3],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    # screw axes in space form
    s_list = np.array([[0, 0, 1, 0, 0, 0],
                       [0, 0, 1, 0, -1, 0],
                       [0, 0, 1, 0, -2, 0]])

    # screw axes in body form
    b_list = np.array([[0, 0, 1, 0, 3, 0],
                       [0, 0, 1, 0, 2, 0],
                       [0, 0, 1, 0, 1, 0]])

    # FORWARD KINEMATICS applying PoE SPACE FORM of 3R robot
    fk_s = fk_space(M, s_list, theta)
    print(f"\nForward Kinematics of 3R robot T0{s_list.shape[0]} applying PoE Space Form for the configuration {theta}: \n{fk_s}")

    # FORWARD KINEMATICS applying PoE BODY FORM
    fk_b = fk_body(M, b_list, theta)
    print(f"\nForward Kinematics of 3R robot T0{b_list.shape[0]} applying PoE Body Form for the configuration {theta}: \n{fk_b}")


