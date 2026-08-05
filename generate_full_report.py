#!/usr/bin/env python3
"""Generate JoseCast_Tam_Teknik_Raporu.md: a comprehensive source/formula reference."""
import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "JoseCast_Tam_Teknik_Raporu.md"

PY_FILES = [
    ROOT / "main.py",
    *sorted((ROOT / "core").glob("*.py")),
    *sorted((ROOT / "ui").glob("*.py")),
]

CPP_FILES = [
    *sorted((ROOT / "cpp" / "include" / "josecast").glob("*.h")),
    *sorted((ROOT / "cpp" / "src").glob("*.cpp")),
]

JSON_FILES = [
    ROOT / "core" / "materials_data" / "alloys.json",
    ROOT / "core" / "materials_data" / "molds.json",
]


def code_block(text: str, lang: str = "") -> str:
    return f"```{lang}\n{text.rstrip()}\n```\n\n"


def heading(text: str, level: int = 2) -> str:
    return "#" * level + " " + text + "\n\n"


def dataclass_fields(source: str) -> List[Dict[str, Any]]:
    try:
        tree = ast.parse(source)
    except Exception:
        return []
    out = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            fields = []
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    fields.append({
                        "name": item.target.id,
                        "annotation": ast.unparse(item.annotation) if hasattr(ast, "unparse") else "",
                        "default": ast.unparse(item.value) if item.value and hasattr(ast, "unparse") else "",
                    })
            out.append({"name": node.name, "fields": fields})
    return out


def json_to_table(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    md = heading(f"{path.name} — Özet tablo", level=3)
    keys = list(data.keys())[:20]  # limit to first 20 entries for brevity
    if not keys:
        return ""
    first = data[keys[0]]
    columns = ["key"] + [k for k in first.keys() if not isinstance(first[k], (list, dict))]
    md += "| " + " | ".join(columns) + " |\n"
    md += "| " + " | ".join(["---"] * len(columns)) + " |\n"
    for k in keys:
        row = [k]
        for c in columns[1:]:
            v = data[k].get(c, "")
            if isinstance(v, float):
                v = f"{v:.4g}"
            else:
                v = str(v)[:40]
            row.append(v)
        md += "| " + " | ".join(row) + " |\n"
    md += "\n(Tam JSON dosyası aynı zip'te `core/materials_data/` altındadır.)\n\n"
    return md


def main() -> None:
    md = "# JoseCast Tam Teknik Rapor\n\n"
    md += "Bu rapor, JoseCast analiz motorunun kullandığı tüm formüller, yöntemler, C++/Python bağlantıları, parametreler, UI geri çağrıları (callback) ve veri yapılarını kaynak koduyla birlikte içerir.\n\n"
    md += "## İçindekiler\n\n"
    md += "1. [Malzeme Kütüphanesi](#malzeme-kütüphanesi)\n"
    md += "2. [Veri Yapıları (core/types.py)](#veri-yapıları)\n"
    md += "3. [C++ Bağlantı Katmanı](#c-bağlantı-katmanı)\n"
    md += "4. [Python Çekirdek Modülleri](#python-çekirdek-modülleri)\n"
    md += "5. [UI Modülleri](#ui-modülleri)\n"
    md += "6. [Ana Giriş (main.py)](#ana-giriş)\n\n"

    md += heading("Malzeme Kütüphanesi")
    for jf in JSON_FILES:
        md += heading(str(jf.relative_to(ROOT)), level=3)
        md += json_to_table(jf)

    md += heading("Veri Yapıları")
    types_src = (ROOT / "core" / "types.py").read_text(encoding="utf-8")
    md += code_block(types_src, "python")

    md += heading("C++ Bağlantı Katmanı")
    md += "### C++ Header İmzaları\n\n"
    for h in sorted((ROOT / "cpp" / "include" / "josecast").glob("*.h")):
        md += heading(str(h.relative_to(ROOT)), level=4)
        md += code_block(h.read_text(encoding="utf-8"), "cpp")

    md += "### Nanobind Modülü (bindings.cpp)\n\n"
    bindings = ROOT / "cpp" / "src" / "bindings.cpp"
    if bindings.exists():
        md += code_block(bindings.read_text(encoding="utf-8"), "cpp")

    md += "### C++ Kaynakları\n\n"
    for cf in sorted((ROOT / "cpp" / "src").glob("*.cpp")):
        md += heading(str(cf.relative_to(ROOT)), level=4)
        md += code_block(cf.read_text(encoding="utf-8"), "cpp")

    md += heading("Python Çekirdek Modülleri")
    for pf in sorted((ROOT / "core").glob("*.py")):
        md += heading(str(pf.relative_to(ROOT)), level=3)
        md += code_block(pf.read_text(encoding="utf-8"), "python")

    md += heading("UI Modülleri")
    for uf in sorted((ROOT / "ui").glob("*.py")):
        md += heading(str(uf.relative_to(ROOT)), level=3)
        md += code_block(uf.read_text(encoding="utf-8"), "python")

    md += heading("Ana Giriş")
    main_file = ROOT / "main.py"
    if main_file.exists():
        md += code_block(main_file.read_text(encoding="utf-8"), "python")

    md += "\n---\nRapor sonu. Güncel kaynak kodları aynı zip içinde `core/`, `ui/`, `cpp/` klasörlerindedir.\n"
    OUT.write_text(md, encoding="utf-8")
    print(f"Report written: {OUT} ({OUT.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
