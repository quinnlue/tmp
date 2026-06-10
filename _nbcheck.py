import json, sys
nb = sys.argv[1]
data = json.load(open(nb, encoding="utf-8"))
bad = 0
for i, c in enumerate(data["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    # skip notebook magics / shell escapes
    lines = [("" if l.lstrip().startswith(("%", "!")) else l) for l in src.splitlines()]
    code = "\n".join(lines)
    try:
        compile(code, f"{nb}[cell {i}]", "exec")
    except SyntaxError as e:
        bad += 1
        print(f"SYNTAX ERROR cell {i}: {e}")
print(f"{nb}: {'OK' if bad==0 else str(bad)+' bad cells'}")
