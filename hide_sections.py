"""Hide the code of chosen notebook sections in Colab (title + output stay visible).
Edit HIDE, then run:  python3 tools/hide_sections.py"""
import json
import re
from pathlib import Path

NB = Path(__file__).resolve().parent / "rtdetrv2_f1tenth_student_ver.ipynb"

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

nb = json.loads(NB.read_text(encoding="utf-8"))
section, heading, seen = None, "", set()
for cell in nb["cells"]:
    src, md = cell["source"], cell["metadata"]
    first = "".join(src).lstrip().split("\n")[0]
    m = re.match(r"#+\s+(\d[\d.]*)", first) if cell["cell_type"] == "markdown" else None
    if m:
        section, heading = m.group(1).rstrip("."), first.lstrip("#").strip()
        seen.add(section)
    if cell["cell_type"] != "code" or not src:
        continue
    hide = HIDE.get(section, False)
    if hide and "STUDENT CODE" in "".join(src):
        print(f"section {section}: not hiding a cell that has STUDENT CODE")
        hide = False
    if hide:
        if not src[0].startswith("# @title"):
            src.insert(0, "# @title\n")
        if src[0].strip() == "# @title":
            src[0] = f"# @title {heading} (provided)\n"
        md["cellView"] = "form"                  # Colab
        md["jupyter"] = {"source_hidden": True}  # JupyterLab
    else:
        md.pop("cellView", None)
        md.pop("jupyter", None)

assert set(HIDE) <= seen, f"sections not in notebook: {set(HIDE) - seen}"
NB.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
print("hidden:", [s for s, h in HIDE.items() if h])
