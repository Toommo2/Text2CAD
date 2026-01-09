# occ_render.py
import os
import sys
import json
import base64

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone

from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add

from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_VolumeProperties


# ---------------- STEP load / properties ----------------

def _shape_from_step(step_path: str):
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed: {status}")
    reader.TransferRoots()
    shape = reader.OneShape()
    return shape


def _bbox_and_volume(shape):
    box = Bnd_Box()
    # use_triangulation=True can speed up bbox, but bbox must work even if no mesh exists
    brepbndlib_Add(shape, box, True)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()

    props = GProp_GProps()
    brepgprop_VolumeProperties(shape, props)
    vol = float(props.Mass())

    return [float(xmin), float(ymin), float(zmin), float(xmax), float(ymax), float(zmax)], vol


# ---------------- Triangulation extraction (robust) ----------------

def _mesh_shape(shape, linear_deflection=0.5, angular_deflection=0.5):
    """
    Ensure the shape has triangulation. In pythonocc-core, you must call Perform().
    """
    mesher = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    mesher.Perform()


def _tri_nodes_array(tri):
    """
    Return a node accessor:
      - either an Array1OfPnt (with .Lower(), .Upper(), .Value(i))
      - or fall back to tri.Node(i) access
    """
    # Most pythonocc builds expose tri.Nodes() as a method returning TColgp_Array1OfPnt
    try:
        arr = tri.Nodes()  # method
        # sanity check
        _ = arr.Lower()
        return ("array", arr)
    except Exception:
        pass

    # fallback: Node(i)
    try:
        n = tri.NbNodes()
        if n <= 0:
            return ("none", None)
        return ("node", tri)
    except Exception:
        return ("none", None)


def _triangles_from_shape(shape, linear_deflection=0.5, angular_deflection=0.5, max_triangles=None):
    """
    Extract triangles as numpy float32 array (N,3,3).
    """
    _mesh_shape(shape, linear_deflection=linear_deflection, angular_deflection=angular_deflection)

    tris = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)

    while exp.More():
        face = exp.Current()
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)

        if tri is None:
            exp.Next()
            continue

        trsf = loc.Transformation()
        tri_mode, node_src = _tri_nodes_array(tri)
        triangles = tri.Triangles()

        nb_tris = int(tri.NbTriangles())
        if nb_tris <= 0:
            exp.Next()
            continue

        for i in range(1, nb_tris + 1):
            t = triangles.Value(i)
            i1, i2, i3 = t.Get()

            if tri_mode == "array":
                nodes = node_src
                p1 = nodes.Value(i1).Transformed(trsf)
                p2 = nodes.Value(i2).Transformed(trsf)
                p3 = nodes.Value(i3).Transformed(trsf)
            elif tri_mode == "node":
                p1 = tri.Node(i1).Transformed(trsf)
                p2 = tri.Node(i2).Transformed(trsf)
                p3 = tri.Node(i3).Transformed(trsf)
            else:
                # no nodes access
                continue

            tris.append((
                (p1.X(), p1.Y(), p1.Z()),
                (p2.X(), p2.Y(), p2.Z()),
                (p3.X(), p3.Y(), p3.Z()),
            ))

            if max_triangles and len(tris) >= max_triangles:
                break

        if max_triangles and len(tris) >= max_triangles:
            break

        exp.Next()

    if not tris:
        raise RuntimeError("No triangles extracted. Meshing/triangulation failed or shape empty.")

    return np.asarray(tris, dtype=np.float32)  # (N,3,3)


# ---------------- Rendering (fast) ----------------

def _project(tris: np.ndarray, view: str) -> np.ndarray:
    """
    Return projected points as (N,3,2).
    """
    if view == "top":
        return tris[:, :, [0, 1]]  # x,y
    if view == "front":
        return tris[:, :, [0, 2]]  # x,z
    if view == "right":
        return tris[:, :, [1, 2]]  # y,z
    if view == "iso":
        x = tris[:, :, 0]
        y = tris[:, :, 1]
        z = tris[:, :, 2]
        xp = x - y
        yp = z + 0.5 * (x + y)
        return np.stack([xp, yp], axis=-1)
    raise ValueError(f"Unknown view: {view}")


def _plot_edges_fast(proj: np.ndarray, out_path: str, dpi=180):
    """
    proj: (N,3,2)
    Render triangle edges via a LineCollection (fast).
    """
    # build 3 segments per triangle: (p0->p1), (p1->p2), (p2->p0)
    p0 = proj[:, 0, :]
    p1 = proj[:, 1, :]
    p2 = proj[:, 2, :]

    seg01 = np.stack([p0, p1], axis=1)
    seg12 = np.stack([p1, p2], axis=1)
    seg20 = np.stack([p2, p0], axis=1)

    segments = np.concatenate([seg01, seg12, seg20], axis=0)  # (3N,2,2)

    fig = plt.figure(figsize=(8, 6), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    lc = LineCollection(segments, linewidths=0.25)
    ax.add_collection(lc)

    allx = proj[:, :, 0].reshape(-1)
    ally = proj[:, :, 1].reshape(-1)

    xmin, xmax = float(allx.min()), float(allx.max())
    ymin, ymax = float(ally.min()), float(ally.max())

    pad_x = (xmax - xmin) * 0.03 + 1e-6
    pad_y = (ymax - ymin) * 0.03 + 1e-6

    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_path


def _save_projection_png(tris: np.ndarray, view: str, out_path: str, dpi=180):
    proj = _project(tris, view)
    return _plot_edges_fast(proj, out_path, dpi=dpi)


def _b64_file(p: str) -> str:
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# ---------------- CLI ----------------

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"success": False, "error": "usage: occ_render.py <step_path> <out_dir>"}))
        sys.exit(2)

    step_path = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    result = {
        "success": False,
        "error": None,
        "images": {},
        "bbox": None,
        "volume": None,
    }

    try:
        shape = _shape_from_step(step_path)
        bbox, volume = _bbox_and_volume(shape)

        # NOTE: for complex CAD, increase deflection to reduce triangles, faster rendering
        tris = _triangles_from_shape(
            shape,
            linear_deflection=0.8,
            angular_deflection=0.6,
            max_triangles=None,  # optionally cap for speed
        )

        paths = {
            "iso": os.path.join(out_dir, "iso.png"),
            "front": os.path.join(out_dir, "front.png"),
            "right": os.path.join(out_dir, "right.png"),
            "top": os.path.join(out_dir, "top.png"),
        }

        _save_projection_png(tris, "iso", paths["iso"])
        _save_projection_png(tris, "front", paths["front"])
        _save_projection_png(tris, "right", paths["right"])
        _save_projection_png(tris, "top", paths["top"])

        result["images"] = {k: _b64_file(v) for k, v in paths.items()}
        result["bbox"] = bbox
        result["volume"] = volume
        result["success"] = True

        with open(os.path.join(out_dir, "images.json"), "w", encoding="utf-8") as f:
            json.dump(result, f)

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        try:
            with open(os.path.join(out_dir, "images.json"), "w", encoding="utf-8") as f:
                json.dump(result, f)
        except Exception:
            pass

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()