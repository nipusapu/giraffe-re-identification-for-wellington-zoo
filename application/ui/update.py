import json
from pathlib import Path

lock = json.loads(Path("package-lock.json").read_text(encoding="utf-8"))
pkg_path = Path("package.json")
pkg = json.loads(pkg_path.read_text(encoding="utf-8"))

packages = lock.get("packages", {}) or {}

def ver_from_lock(name: str):
    # Prefer lock v2/v3 "packages" entries
    p = packages.get(f"node_modules/{name}")
    if isinstance(p, dict) and isinstance(p.get("version"), str):
        return p["version"]
    # Fallback to lock v1 "dependencies"
    d = lock.get("dependencies", {}).get(name, {})
    if isinstance(d, dict) and isinstance(d.get("version"), str):
        return d["version"]
    return None

need = ["next", "react", "react-dom"]
deps = pkg.setdefault("dependencies", {})
for n in need:
    v = ver_from_lock(n)
    if not v:
        raise SystemExit(f"Could not find {n} version in package-lock.json")
    deps[n] = v  # exact version to match lock

# keep scripts if missing
pkg.setdefault("scripts", {})
pkg["scripts"].setdefault("dev", "next dev")
pkg["scripts"].setdefault("build", "next build")
pkg["scripts"].setdefault("start", "next start")
pkg["scripts"].setdefault("lint", "next lint")

pkg_path.write_text(json.dumps(pkg, indent=2) + "\n", encoding="utf-8")
print("Updated package.json dependencies:", {k: deps[k] for k in need})