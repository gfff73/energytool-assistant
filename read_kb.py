from pathlib import Path

project_dir = Path(__file__).resolve().parent
path = project_dir / "materials/knowledge-base/00-reference/00-product-map.md"
text = path.read_text(encoding="utf-8")
print(text[:500])
