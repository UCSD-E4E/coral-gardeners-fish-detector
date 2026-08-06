"""Full-pipeline stage B: classify SAM3 matched-detection crops under both label
spaces; report detection recall, species-acc-on-matched, and end-to-end accuracy.
"""
import json, os, sys, collections

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
SP = os.path.dirname(__file__)
os.environ.setdefault("HF_HOME", os.path.join(REPO, "models", "hf_cache"))

recs = json.load(open(os.path.join(SP, "detections.json")))
sets = json.load(open(os.path.join(SP, "species_sets.json")))
THIRTEEN, OPEN = sets["little_cayman"], sets["open_union"]

N = len(recs)
matched = [r for r in recs if r["matched"] and r.get("det_crop")]
recall = len(matched) / N
print(f"total {N}  matched {len(matched)}  detection recall {recall:.3f}", flush=True)

from coral_fish_pipeline.classification.bioclip_classifier import BioCLIPClassifier


def run(name, species_list):
    print(f"\n=== full-pipeline condition: {name} ({len(species_list)} classes) ===", flush=True)
    clf = BioCLIPClassifier(species=species_list, region="little_cayman",
                            device="auto", precision="fp16",
                            unknown_threshold=0.0, uncertain_margin=0.0)
    clf.load()
    preds = {}
    for i, r in enumerate(matched):
        pr = clf._predict_image(r["det_crop"])
        preds[r["image_id"]] = dict(pred=pr["species"], top5=[s for s, _ in pr["top5"]])
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(matched)}", flush=True)
    # metrics over ALL N (unmatched -> wrong)
    e2e_top1 = e2e_top5 = 0
    m_top1 = m_top5 = 0
    per = collections.defaultdict(lambda: [0, 0, 0])  # e2e_correct, matched, total
    conf = collections.Counter()
    for r in recs:
        gt = r["gt"]; per[gt][2] += 1
        if r["image_id"] in preds:
            per[gt][1] += 1
            p = preds[r["image_id"]]
            ok1 = (p["pred"] == gt); ok5 = (gt in p["top5"])
            m_top1 += ok1; m_top5 += ok5
            if ok1: e2e_top1 += 1; per[gt][0] += 1
            if ok5: e2e_top5 += 1
            if not ok1: conf[(gt, p["pred"])] += 1
    out = dict(condition=name, n=N, matched=len(matched), recall=recall,
               species_acc_matched_top1=m_top1/len(matched),
               species_acc_matched_top5=m_top5/len(matched),
               e2e_top1=e2e_top1/N, e2e_top5=e2e_top5/N,
               per_species={k: dict(e2e_correct=v[0], matched=v[1], n=v[2]) for k, v in per.items()})
    json.dump(out, open(os.path.join(SP, f"results_fullpipe_{name}.json"), "w"), indent=2)
    print(f"  species acc on matched: top1 {m_top1}/{len(matched)}={m_top1/len(matched):.3f} top5 {m_top5/len(matched):.3f}", flush=True)
    print(f"  END-TO-END (unmatched=wrong): top1 {e2e_top1}/{N}={e2e_top1/N:.3f} top5 {e2e_top5/N:.3f}", flush=True)
    print("  per-species e2e_correct/matched/n:", flush=True)
    for k in sorted(per, key=lambda k: -per[k][2]):
        c, mt, t = per[k]; print(f"    {c:>3}/{mt:>3}/{t:<3}  {k}", flush=True)
    print("  top confusions:", flush=True)
    for (g, pr), v in conf.most_common(6):
        print(f"    {v:3d}  {g} -> {pr}", flush=True)
    return out


which = sys.argv[1] if len(sys.argv) > 1 else "both"
if which in ("prior13", "both"):
    run("prior13", THIRTEEN)
if which in ("open69", "both"):
    run("open69", OPEN)
print(f"\nFULLPIPE DONE ({which})", flush=True)
