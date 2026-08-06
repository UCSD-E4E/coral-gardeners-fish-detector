"""Join staged DB TSVs into a per-fish ground-truth manifest JSON."""
import json, os, re, collections

GT = os.path.join(os.path.dirname(__file__), "gt")
ORF_ROOT = os.environ.get("ORF_ROOT", os.path.expanduser("~/mnt/fishsense_data/REEF/data"))


def load_tsv(name):
    rows = []
    with open(os.path.join(GT, name)) as fh:
        for line in fh:
            rows.append(line.rstrip("\n").split("\t"))
    return rows


# image_id -> (path, camera_id)
imgmeta = {}
for r in load_tsv("testset_imgmeta.tsv"):
    img_id, path, dive, cam = r[0], r[1], r[2], r[3]
    imgmeta[img_id] = (path, int(cam))

def fnum(s):
    if s is None or s == "\\N" or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


# image_id -> list of head/tail dicts (skip rows with any NULL coord)
ht = collections.defaultdict(list)
for r in load_tsv("headtail_live.tsv"):
    img, hx, hy, tx, ty = r
    vals = [fnum(hx), fnum(hy), fnum(tx), fnum(ty)]
    if any(v is None for v in vals):
        continue
    ht[img].append(dict(head_x=vals[0], head_y=vals[1], tail_x=vals[2], tail_y=vals[3]))

# image_id -> list of laser dicts (skip NULL coords)
laser = collections.defaultdict(list)
for r in load_tsv("laser_live.tsv"):
    img, x, y, color = r[0], r[1], r[2], (r[3] if len(r) > 3 else "")
    if fnum(x) is None or fnum(y) is None:
        continue
    laser[img].append(dict(x=fnum(x), y=fnum(y), color=color))

# image_id -> species content
species = {}
for r in load_tsv("species_content.tsv"):
    if len(r) >= 2:
        species[r[0]] = r[1]

SCI = re.compile(r"\(([^)]+)\)\s*$")


def parse_species(content):
    # "Fish, Hogfish (Lachnolaimus maximus)" -> ("Hogfish", "Lachnolaimus maximus")
    if not content or not content.startswith("Fish,"):
        return None
    label = content.split(",", 1)[1].strip()
    m = SCI.search(label)
    sci = m.group(1) if m else None
    common = SCI.sub("", label).strip()
    return dict(common=common, scientific=sci, raw=content)


manifest = []
for img_id, (path, cam) in imgmeta.items():
    sp = parse_species(species.get(img_id, ""))
    if sp is None:  # non-fish content slipped in; skip
        continue
    manifest.append(dict(
        image_id=int(img_id),
        orf_path=os.path.join(ORF_ROOT, path),
        rel_path=path,
        camera_id=cam,
        headtail=ht.get(img_id, []),
        laser=laser.get(img_id, []),
        species=sp,
        n_fish=len(ht.get(img_id, [])),
    ))

out = os.path.join(os.path.dirname(__file__), "manifest.json")
with open(out, "w") as fh:
    json.dump(manifest, fh, indent=2)

print("manifest entries:", len(manifest))
print("with ORF present:", sum(os.path.isfile(m["orf_path"]) for m in manifest))
sp_counts = collections.Counter(m["species"]["scientific"] for m in manifest)
print("species dist:")
for s, c in sp_counts.most_common():
    print(f"  {c:4d}  {s}")
nfish = collections.Counter(m["n_fish"] for m in manifest)
print("n_fish per image:", dict(nfish))
nl = collections.Counter(len(m["laser"]) for m in manifest)
print("n_laser per image:", dict(nl))
print("wrote", out)
