"""Collapse chosen notebook sections and hide their code (title + output stay visible).
Edit HIDE, then run:  python3 hide_sections.py [path/to/notebook.ipynb]

What each viewer reads:
  Colab       metadata.colab.collapsed_sections (heading cell ids) -> section folded on open
              cell cellView="form" + first line "# @title"          -> code hidden, title shown
  JupyterLab  heading cell heading_collapsed=true, code cell jupyter.source_hidden=true
  VS Code     nothing: its ipynb reader ignores all collapse/hidden metadata, so fold by hand
"""
import json
import re
import sys
from pathlib import Path

NB = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path(__file__).resolve().parent / "rtdetrv2_f1tenth_student_ver.ipynb"

# section number -> hide its code? (cells containing STUDENT CODE are never hidden)
HIDE = {
    "0": True,     # Install
    "1": False,     # Configuration
    "2.1": False,   # Helpers
    "2.2": False,   # Backbone: PResNet
    "2.3": False,   # Hybrid encoder
    "2.4": True,    # Multi-scale deformable attention
    "2.5": True,    # Contrastive denoising
    "2.6": False,   # Decoder
    "2.7": True,    # Full model + r18vd config
    "2.8": True,    # Loading the pretrained weights
    "2.9": True,    # Post-processing and preprocessing
    "2.10": False,  # Following one image through the model
    "3": False,     # Run the pretrained (COCO) model
    "4": True,     # Dataset
    "5": False,     # Model for the new class(es)
    "6": True,      # Loss
    "7": True,     # Training and evaluation
    "8": False,     # Inference
}

text = NB.read_bytes().decode("utf-8")
nb = json.loads(text)
section, heading, seen, collapsed = None, "", set(), []
for cell in nb["cells"]:
    if isinstance(cell["source"], str):
        cell["source"] = cell["source"].splitlines(keepends=True)
    src, md = cell["source"], cell["metadata"]
    first = "".join(src).lstrip().split("\n")[0]
    m = re.match(r"#+\s+(\d[\d.]*)", first) if cell["cell_type"] == "markdown" else None
    if m:
        section, heading = m.group(1).rstrip("."), first.lstrip("#").strip()
        seen.add(section)
        if HIDE.get(section, False):
            md["heading_collapsed"] = True             # JupyterLab
            collapsed.append(md.setdefault("id", cell.get("id")))  # Colab
        else:
            md.pop("heading_collapsed", None)
    if cell["cell_type"] != "code" or not src:
        continue
    hide = HIDE.get(section, False)
    if hide and "STUDENT CODE" in "".join(src):
        print(f"section {section}: not hiding a cell that has STUDENT CODE")
        hide = False
    title = f"# @title {heading} (provided)\n"
    if hide:
        if not src[0].startswith("# @title"):
            src.insert(0, "# @title\n")
        if src[0].strip() == "# @title":
            src[0] = title
        md["cellView"] = "form"                                    # Colab
        md.setdefault("jupyter", {})["source_hidden"] = True       # JupyterLab
    else:
        if src[0] == title:          # undo a title this script added earlier
            src.pop(0)
        md.pop("cellView", None)
        md.get("jupyter", {}).pop("source_hidden", None)
        if md.get("jupyter") == {}:
            md.pop("jupyter")

assert set(HIDE) <= seen, f"sections not in notebook: {set(HIDE) - seen}"
nb["metadata"].setdefault("colab", {})["collapsed_sections"] = collapsed
# keep the file's indent, line endings and final newline so the git diff stays small
indent = len(re.match(r"\{\r?\n( *)", text).group(1)) or 1
out = json.dumps(nb, indent=indent, ensure_ascii=False)
if text.endswith("\n"):
    out += "\n"
if "\r\n" in text:
    out = out.replace("\n", "\r\n")
NB.write_bytes(out.encode("utf-8"))
print("hidden:", [s for s, h in HIDE.items() if h])
print("collapsed heading ids:", collapsed)
