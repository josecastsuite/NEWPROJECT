"""Collect environment info to debug a silent crash."""

import importlib
import importlib.util
import os
import platform
import sys
import traceback
from pathlib import Path


def spec(name: str):
    try:
        return importlib.util.find_spec(name)
    except Exception as exc:
        return f"ERROR: {exc}"


def version(name: str):
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "unknown")
    except Exception as exc:
        return f"ERROR: {exc}"


def check_pyd():
    here = Path(__file__).resolve().parent / "core"
    results = []
    for pyd in here.glob("josecast_core*.pyd"):
        try:
            import ctypes
            _ = ctypes.CDLL(str(pyd))
            results.append(f"{pyd.name}: OK (yuklendi)")
        except Exception as exc:
            results.append(f"{pyd.name}: HATA {exc}")
    return results if results else ["josecast_core.pyd bulunamadi"]


def main():
    log = Path(__file__).resolve().parent / "diagnose_env.log"
    out = []
    out.append(f"OS: {platform.system()} {platform.release()} {platform.machine()}")
    out.append(f"Python: {sys.version}")
    out.append(f"Executable: {sys.executable}")
    out.append("")

    packages = [
        "PyQt6",
        "pyvista",
        "pyvistaqt",
        "vtk",
        "numpy",
        "scipy",
        "trimesh",
        "meshpy",
        "skimage",
        "sklearn",
        "imageio",
        "matplotlib",
    ]
    for pkg in packages:
        name = pkg.replace("-", "_")
        sp = spec(name)
        if sp is None:
            out.append(f"{pkg}: KURULU DEGIL")
        elif isinstance(sp, str) and sp.startswith("ERROR"):
            out.append(f"{pkg}: {sp}")
        else:
            out.append(f"{pkg}: {sp.origin or 'builtin'} (ver: {version(name)})")

    out.append("")
    out.append("josecast_core pyd kontrolu:")
    out.extend(check_pyd())

    out.append("")
    out.append("GPU / OpenGL bilgisi (pyvista varsa):")
    if spec("pyvista") is not None:
        try:
            import pyvista as pv
            out.append(f"pyvista version: {pv.__version__}")
            out.append(f"pyvista GPU support: {pv.system_supports_rendering()}")
            out.append(f"default GPU backend: {pv.global_theme.get('default_gpu_backend', 'N/A')}")
        except Exception as exc:
            out.append(f"pyvista GPU check failed: {exc}\n{traceback.format_exc()}")
    else:
        out.append("pyvista kurulu degil, GPU kontrolu yapilamadi.")

    out.append("")
    out.append("Cevre degiskenleri:")
    for key in ["QT_QPA_PLATFORM", "PYVISTA_OFF_SCREEN", "PYOPENGL_PLATFORM"]:
        out.append(f"  {key}: {os.environ.get(key, '(bos)')}")

    txt = "\n".join(out) + "\n"
    print(txt)
    log.write_text(txt, encoding="utf-8")
    print(f"\n{log} dosyasina yazildi.")


if __name__ == "__main__":
    main()
