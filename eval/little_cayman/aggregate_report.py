import json, os, collections
SP = os.path.dirname(__file__)

clf13 = json.load(open(f"{SP}/results_classifier_prior13.json"))
clf69 = json.load(open(f"{SP}/results_classifier_open69.json"))
fp13 = json.load(open(f"{SP}/results_fullpipe_prior13.json"))
fp69 = json.load(open(f"{SP}/results_fullpipe_open69.json"))
dets = json.load(open(f"{SP}/detections.json"))

COMMON = {
 "Lachnolaimus maximus":"Hogfish","Sparisoma viride":"Stoplight Parrotfish",
 "Mycteroperca bonaci":"Black Grouper","Epinephelus striatus":"Nassau Grouper",
 "Scarus coeruleus":"Blue Parrotfish","Scarus guacamaia":"Rainbow Parrotfish",
 "Ocyurus chrysurus":"Yellowtail Snapper","Scarus coelestinus":"Midnight Parrotfish",
 "Epinephelus itajara":"Goliath Grouper","Lutjanus analis":"Mutton Snapper",
 "Epinephelus morio":"Red Grouper","Lutjanus griseus":"Grey Snapper",
 "Mycteroperca interstitialis":"Yellowmouth Grouper"}

# detection recall per species
drec = collections.defaultdict(lambda:[0,0])
for r in dets:
    drec[r["gt"]][1]+=1
    if r["matched"]: drec[r["gt"]][0]+=1

# order species by support desc
order = sorted(clf13["per_species"].keys(), key=lambda k:-clf13["per_species"][k]["n"])

per=[]
for s in order:
    n = clf13["per_species"][s]["n"]
    per.append(dict(
        sci=s, common=COMMON.get(s,s), n=n,
        clf13=clf13["per_species"][s]["acc"],
        clf69=clf69["per_species"][s]["acc"],
        det_recall=drec[s][0]/drec[s][1] if drec[s][1] else 0,
        fp13=fp13["per_species"][s]["e2e_correct"]/n,
        fp69=fp69["per_species"][s]["e2e_correct"]/n,
    ))

# confusion matrices (gt rows x pred cols) restricted to the 13 label space for the prior13 conditions
LC = json.load(open(f"{SP}/species_sets.json"))["little_cayman"]
def confusion(preds_field_source, is_fullpipe):
    # returns dict gt->pred->count over the 13 classes (+ 'miss' col for undetected in fullpipe)
    M = {g:collections.Counter() for g in order}
    if not is_fullpipe:
        for p in preds_field_source["preds"]:
            M[p["gt"]][p["pred"]]+=1
    return {g:dict(M[g]) for g in order}
conf_clf13 = confusion(clf13, False)

summary = dict(
  n=clf13["n"], det_recall=fp13["recall"], det_matched=fp13["matched"],
  conditions=[
    dict(key="clf13", scope="Classifier-only", space="13-species prior", top1=clf13["top1_acc"], top5=clf13["top5_acc"]),
    dict(key="clf69", scope="Classifier-only", space="69-species open-set", top1=clf69["top1_acc"], top5=clf69["top5_acc"]),
    dict(key="fp13", scope="Full pipeline", space="13-species prior", top1=fp13["e2e_top1"], top5=fp13["e2e_top5"], acc_matched=fp13["species_acc_matched_top1"]),
    dict(key="fp69", scope="Full pipeline", space="69-species open-set", top1=fp69["e2e_top1"], top5=fp69["e2e_top5"], acc_matched=fp69["species_acc_matched_top1"]),
  ])

out = dict(summary=summary, per_species=per, labels=order,
           common=COMMON, confusion_clf13=conf_clf13, lc_order=order)
json.dump(out, open(f"{SP}/report_data.json","w"), indent=2)
print("wrote report_data.json")
print("summary:", json.dumps(summary, indent=2))
