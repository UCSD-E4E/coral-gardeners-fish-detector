#!/usr/bin/env bash
# Regenerate the ground-truth extracts + test-set from a fishsense PostgreSQL
# custom-format dump (pg_restore). Produces gt/*.tsv and testset.ids used by
# build_manifest.py. No local postgres client needed — uses nixpkgs#postgresql.
#
# Usage: extract_db.sh <path-to.dump>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GT="$HERE/gt"; mkdir -p "$GT"
DUMP="${1:?usage: extract_db.sh <fishsense .dump>}"

pg(){ nix shell nixpkgs#postgresql --command pg_restore --data-only -t "$1" -f - "$DUMP" 2>/dev/null \
      | awk '/^COPY public/{f=1;next} /^\\\.$/{f=0} f'; }

# Column layouts (public schema):
#  headtaillabel: id,task,head_x(3),head_y(4),tail_x(5),tail_y(6),image_id(7),user,updated,completed(10),json,proj,superseded(13)
#  laserlabel:    id,task,x(3),y(4),label(5),image_id(6),user,updated,completed(9),json,proj,superseded(12)
#  specieslabel:  id,task,updated,completed(4),json,image_id(6),...,content_of_image(15),...
#  image:         id(1),path(2),taken,checksum(4),is_canonical,dive_id(6),camera_id(7)
#  cameraintrinsics: id,camera_matrix(2),distortion_coefficients(3),camera_id(4)

# --- candidate image sets (completed labels) ---
pg headtaillabel | awk -F'\t' '$10=="t"{print $7}' | sort -u > "$GT/_ht.ids"
pg laserlabel    | awk -F'\t' '$9=="t"{print $6}'  | sort -u > "$GT/_ls.ids"
# real target-fish species labels only (drop Slate / Fish Model / None / Other / Unidentifiable)
pg specieslabel  | awk -F'\t' '$4=="t" && $15 ~ /^Fish,/ && $15 !~ /Other|Unidentifiable/{print $6}' \
                 | sort -u > "$GT/_sp.ids"

comm -12 "$GT/_ht.ids" "$GT/_ls.ids" > "$GT/_htls.ids"
comm -12 "$GT/_htls.ids" "$GT/_sp.ids" > "$HERE/testset.ids"
echo "test-set images: $(wc -l < "$HERE/testset.ids")"

# --- image metadata (id path dive cam), joined to test-set ---
pg image | awk -F'\t' '{print $1"\t"$2"\t"$6"\t"$7}' | sort -t$'\t' -k1,1 > "$GT/_img.tsv"
join -t$'\t' "$HERE/testset.ids" "$GT/_img.tsv" > "$GT/testset_imgmeta.tsv"

# --- live (completed & not superseded) label geometry ---
pg headtaillabel | awk -F'\t' '$10=="t" && $13=="f"{print $7"\t"$3"\t"$4"\t"$5"\t"$6}' > "$GT/headtail_live.tsv"
pg laserlabel    | awk -F'\t' '$9=="t"  && $12=="f"{print $6"\t"$3"\t"$4"\t"$5}'         > "$GT/laser_live.tsv"
pg specieslabel  | awk -F'\t' '$4=="t"{print $6"\t"$15}' | sort -u                        > "$GT/species_content.tsv"

# --- camera intrinsics (cam matrix dist) ---
pg cameraintrinsics | awk -F'\t' '{print $4"\t"$2"\t"$3}' > "$GT/cam_intrinsics.tsv"

rm -f "$GT"/_*.ids "$GT"/_img.tsv "$GT"/_htls.ids
echo "wrote $GT/{testset_imgmeta,headtail_live,laser_live,species_content,cam_intrinsics}.tsv"
