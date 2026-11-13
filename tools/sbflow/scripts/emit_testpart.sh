#!/usr/bin/env bash
# Emit specific staged parts as base64 with markers.
# The part name is concatenated literally: printf "%02d" 08 fails in bash because a leading zero
# makes it an invalid octal, which silently produced a truncated/duplicated last part.
set -uo pipefail
STAGE=/tmp/v6test
for n in $1; do
  P="$STAGE/part$n"
  if [ ! -f "$P" ]; then
    echo "MISSING $P" >&2
    exit 2
  fi
  echo "###PART${n}###"
  base64 -w0 "$P"
  echo
done
