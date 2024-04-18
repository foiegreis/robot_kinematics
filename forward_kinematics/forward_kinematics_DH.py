'''
Code to compute the Forward Kinematics of a robot with n joints, knowing the DH parameters of each joint
github @foiegreis
'''

import numpy as np


def dh_to_T(alpha=None, a=None, d=None, phi=None):
    """Returns Link Transformation Matrix given DH parameters joint i-1 - joint i - alpha and phi expressed in radians"""

    T = np.array([[np.cos(phi), -np.sin(phi)*np.cos(alpha), np.sin(phi)*np.sin(alpha), a*np.cos(phi)],
                  [np.sin(phi), np.cos(phi)*np.cos(alpha), -np.cos(phi) * np.sin(alpha), a * np.sin(phi)],
                  [0, np.sin(alpha), np.cos(alpha), d ],
                  [0, 0, 0, 1]])
    return T


def near_zero(z):
    """Determines whether a scalar is small enough to be treated as zero"""
    return abs(z) < 1e-5


def forward_kinematics(list_T):
    """Computes Forward Kinematics given the list of HTM"""

    fk = list_T[0]
    for T in list_T[1:]:
        fk = np.matmul(fk, T)
    res = np.where(near_zero(fk), 0, fk)
    return res


def compute_fk(dh):
    """Given DH Parameters Table returns Forward Kinematics"""

    t_matrices = []
    n = len(dh)

    for joint in dh:
        print(joint)
        alpha, a, d, phi = joint
        tj = dh_to_T(alpha, a, d, phi)
        t_matrices.append(tj)

    res = forward_kinematics(t_matrices)
    print(f"Forward Kinematics = T0{n} =\n{res}")
    return res


if __name__ == "__main__":

    # Known joint configuration θ1-θ6
    theta = [0.7854, -0.7854, 0.7854, -1.5708, -1.5708, 0.7854]

    # DH Parameters Table
    dh = np.array([[np.pi / 2, 0, 0.089, -theta[0] - np.pi / 2],
                   [0, -0.425, 0, theta[1]],
                   [0, -0.3922, 0, theta[2]],
                   [np.pi / 2, 0, 0.109, theta[3]],
                   [-np.pi / 2, 0, 0.0947, theta[4]],
                   [0, 0, 0.0823, theta[5]]])

    compute_fk(dh)


