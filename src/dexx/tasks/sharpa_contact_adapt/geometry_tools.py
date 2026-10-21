"""Offline URDF FK for pose inspection, independent of CUDA/Isaac."""
import xml.etree.ElementTree as ET
import numpy as np


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*k + (1-np.cos(angle))*(k@k)


def origin_transform(element):
    t = np.eye(4)
    if element is not None:
        t[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        r, p, y = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        t[:3, :3] = axis_rotation([0, 0, 1], y) @ axis_rotation([0, 1, 0], p) @ axis_rotation([1, 0, 0], r)
    return t


class URDFFK:
    def __init__(self, path):
        self.root = ET.parse(path).getroot()
        self.joints = self.root.findall("joint")
        children = {j.find("child").get("link") for j in self.joints}
        self.root_name = next(l.get("name") for l in self.root.findall("link") if l.get("name") not in children)
        self.limits = {}
        for j in self.joints:
            if j.get("type") == "revolute":
                lim = j.find("limit")
                self.limits[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))

    def frames(self, q, root_position=(0, 0, 0)):
        base = np.eye(4)
        base[:3, 3] = root_position
        poses = {self.root_name: base}
        pending = list(self.joints)
        while pending:
            remaining = []
            for j in pending:
                parent, child = j.find("parent").get("link"), j.find("child").get("link")
                if parent not in poses:
                    remaining.append(j)
                    continue
                t = origin_transform(j.find("origin"))
                if j.get("type") in ("revolute", "continuous"):
                    a = np.eye(4)
                    a[:3, :3] = axis_rotation(np.fromstring(j.find("axis").get("xyz"), sep=" "), q.get(j.get("name"), 0))
                    t = t @ a
                poses[child] = poses[parent] @ t
            if len(remaining) == len(pending):
                raise ValueError("Disconnected or cyclic URDF")
            pending = remaining
        return poses
