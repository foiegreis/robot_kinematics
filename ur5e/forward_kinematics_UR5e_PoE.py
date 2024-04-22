#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FORWARD KINEMATICS of UR5e - PRODUCT OF EXPONENTIALS APPROACH
Code to compute the PoE Forward Kinematics of 6 dof UR5e, given the table of Space Form or Body Form twists
and M matrix

#Author: foiegreis
#Date Created: april 2024

"""

import numpy as np


def skew_symmetric3(p):
    """
    Returns the skew symmetric matrix representation [p] of a 3D  vector
    :param p: 3D vector - angular velocity
    :return: 3x3 matrix
    """
    p_sk = np.array([[0, -p[2], p[1]],
                     [p[2], 0, -p[0]],
                     [-p[1], p[0], 0]])
    return p_sk


def skew_symmetric6(p, v):
    """
    Returns the skew symmetric matrix representation [S] of a 6D twist vector
    (Note: we're not going to use this method in the script but is useful to keep it in mind)
    :param p: 3D vector - angular velocity
    :param v: 3D vector - linear velocity
    :return: 4x4 matrix
    """
    p_sk = skew_symmetric3(p)
    twist_sk = np.r_[np.c_[p_sk, v.reshape(3, 1)],[[0, 0, 0, 0]]]

    return twist_sk


def near_zero(z):
    """
    Determines whether a scalar is zero
    :param z: scalar
    :return: bool
    """
    return abs(z) < 1e-6


def rodrigues_exp(p_sk, theta):
    """
    Computes the 3x3 matrix exponential Rot(p_hat, theta) = e^[p]theta of the exp coordinates p_hat*theta, given [p] and theta
    using the Rodrigues formula for rotations
    :param p_sk: 3x3 skew symmetric matrix
    :param theta: angle [rad]
    :return: 3x3 matrix exponential
    """
    mat_exp = np.array(np.eye(3) + np.sin(theta) * p_sk + (1 - np.cos(theta)) * np.dot(p_sk, p_sk))
    return np.where(near_zero(mat_exp), 0, np.round(mat_exp, 4))


def linear_exp(p_sk, v, theta):
    """
    Computes the 3x3 matrix exponential G(theta)v of the exp coordinates given [p], v and theta
    :param p_sk: 3x3 skew symmetric matrix
    :param v: 3x1 lin velocity
    :param theta: angle [rad]
    :return: 3x1 vector
    """
    gv = np.dot(np.eye(3) * theta + (1 - np.cos(theta)) * p_sk \
                + (theta - np.sin(theta)) * np.dot(p_sk, p_sk), v)
    return np.round(gv, 4)


def screw_exp(screw_axis, theta):
    """
    Computes the 4x4 matrix exponential of a screw axis = twist = (omega, v) and an angle theta
    :param screw_axis: 6D vector (omega, v)
    :param theta: angle [rad]
    :return: 4x4 matrix exponential (HTM)
    """

    omega = screw_axis[0:3]
    v = screw_axis[3:]

    # Case Revolute Joint
    if np.linalg.norm(omega) == 1:

        omega_sk = skew_symmetric3(omega)
        rot_exp = rodrigues_exp(omega_sk, theta)
        lin_exp = linear_exp(omega_sk, v, theta)

        s_exp = np.r_[np.c_[rot_exp, lin_exp.reshape(3, 1)],[[0, 0, 0, 1]]]

    # Case Prismatic Joint
    elif np.linalg.norm(omega) == 0 and np.linalg.norm(v) != 1:
        s_exp = np.eye(4)
        s_exp[:3, 3] = v * theta

    else:
        raise AssertionError("Invalid Screw Axis. Screw axis must have zero pitch")

    return s_exp


def forward_kinematics_space(M, s_list, theta_list):
    """
    Computes the Forward Kinematics given the M matrix, the list of screw axes in Space Form and the configuration
    :param M: 4x4 M0b homogeneous transformation matrix of the zero configuration
    :param s_list: list of the space screw axes for the robot joints
    :param theta_list: configuration of the robot i.e. list of joint values
    :return: 4x4 homogeneous transformation matrix of the end effector for the configuration
    """

    fk = np.eye(4)
    for i, s in enumerate(s_list):

        theta = theta_list[i]
        T = screw_exp(s, theta)
        fk = np.matmul(fk, T)

    fk = np.matmul(fk, M)
    res = np.where(near_zero(fk), 0, np.round(fk, 3))
    return res


def forward_kinematics_body(M, b_list, theta_list):
    """
    Computes the Forward Kinematics given the M matrix, the list of screw axes in Body Form and the configuration
    :param M: 4x4 M0b homogeneous transformation matrix of the zero configuration
    :param b_list: list of the body screw axes for the robot joints
    :param theta_list: configuration of the robot i.e. list of joint values
    :return: 4x4 homogeneous transformation matrix of the end effector for the configuration
    """
    fk = M.copy()
    for i, s in enumerate(b_list):

        theta = theta_list[i]
        T = screw_exp(s, theta)
        fk = np.matmul(fk, T)

    res = np.where(near_zero(fk), 0, fk)
    return res


if __name__ == "__main__":

    # UR5e
    # Known joint configuration θ1-θ6
    theta = [0.7854, -0.7854, 0.7854, -1.5708, -1.5708, 0.7854]

    # UR5e specifications
    W1 = 0.109
    W2 = 0.082
    L1 = 0.425
    L2 = 0.392
    H1 = 0.089
    H2 = 0.095

    # M Matrix in Home Configuration
    M = np.array([[-1, 0, 0, (L1 + L2)],
                  [0, 0, 1, (W1 + W2)],
                  [0, 1, 0, (H1 - H2)],
                  [0, 0, 0, 1]])

    # Screw Axes in Space form
    s_list = np.array([[0, 0, 1, 0, 0, 0],
                       [0, 1, 0, -H1, 0, 0],
                       [0, 1, 0, -H1, 0, L1],
                       [0, 1, 0, -H1, 0, (L1 + L2)],
                       [0, 0, -1, -W1, (L1+L2), 0],
                       [0, 1, 0, (H2-H1), 0, (L1+L2)]])

    # Screw Axes in Body form
    b_list = np.array([[0, 1, 0, W1 + W2, 0, L1 + L2],
                       [0, 0, 1, H2, L1 + L2, 0],
                       [0, 0, 1, H2, L2, 0],
                       [0, 0, 1, H2, 0, 0],
                       [0, -1, 0, -W2, 0, 0],
                       [0, 0, 1, 0, 0, 0]])

    print("UR5e 6dof robot arm")
    # FORWARD KINEMATICS applying PoE SPACE FORM
    fk_s = forward_kinematics_space(M, s_list, theta)
    print(f"\nForward Kinematics T0{s_list.shape[0]} applying PoE Space Form for the configuration {theta}: \n{fk_s}")

    # FORWARD KINEMATICS applying PoE BODY FORM
    fk_b = forward_kinematics_body(M, b_list, theta)
    print(f"\nForward Kinematics T0{s_list.shape[0]} applying PoE Body Form for the configuration {theta}: \n{fk_s}")


