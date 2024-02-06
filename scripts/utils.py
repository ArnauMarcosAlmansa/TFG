import numpy as np


def lookAt(center, target, up):
    f = (target - center)
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    u = u / np.linalg.norm(u)

    m = np.zeros((4, 4))
    m[0, :-1] = s
    m[1, :-1] = u
    m[2, :-1] = -f
    m[-1, -1] = 1.0

    return m


def scale(pose, factor):
    return pose @ np.array([[factor, 0, 0, 0], [0, factor, 0, 0], [0, 0, factor, 0], [0, 0, 0, factor]])


def rotx(pose, a):
    rot = np.eye(4)
    rot[1, 1] = np.cos(a)
    rot[1, 2] = -np.sin(a)
    rot[2, 1] = np.sin(a)
    rot[2, 2] = np.cos(a)
    return np.dot(rot, pose)


def roty(pose, a):
    rot = np.eye(4)
    rot[0, 0] = np.cos(a)
    rot[0, 2] = np.sin(a)
    rot[2, 0] = -np.sin(a)
    rot[2, 2] = np.cos(a)
    return np.dot(rot, pose)


def rotz(pose, a):
    rot = np.eye(4)
    rot[0, 0] = np.cos(a)
    rot[0, 1] = -np.sin(a)
    rot[1, 0] = np.sin(a)
    rot[1, 1] = np.cos(a)
    return np.dot(rot, pose)