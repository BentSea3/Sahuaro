#!/usr/bin/env bash
# compute_bond_averages.sh
#
# Usage: ./compute_bond_averages.sh [bond_dump_file]
# Defaults to bondInfo.dump if no argument is given.

INPUT=${1:-bondInfo.dump}

if [[ ! -f "$INPUT" ]]; then
  echo "Error: File '$INPUT' not found."
  exit 1
fi

echo "Computing average bond lengths from '$INPUT'..."

awk '
BEGIN {
  # Map bond-type → atom-label pair
  label[1] = "Zn1-O2"
  label[2] = "Zn1-O1"
  label[3] = "C3-H1"
  label[4] = "C3-C3"
  label[5] = "C2-C3"
  label[6] = "C1-C2"
  label[7] = "O2-C1"
}
# Only process lines with exactly 2 fields (type + length)
NF==2 {
  tp = $1     # bond type
  d  = $2     # bond length
  sum[tp]   += d
  cnt[tp]   += 1
}
END {
  # Print in order for types 1 through 7
  for (t = 1; t <= 7; t++) {
    if (cnt[t] > 0) {
      printf "%-8s (type %d) average length = %.6f\n", label[t], t, sum[t]/cnt[t]
    } else {
      printf "%-8s (type %d): no data\n", label[t], t
    }
  }
}
' "$INPUT"

