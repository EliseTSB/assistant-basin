#!/bin/bash

part64_generation_started() {
  local run_root=$1 generated_file=$2 gen_root
  gen_root="$run_root/results/part64/gen64"
  if [ -f "$generated_file" ] || compgen -G "$gen_root/outputs/*.jsonl" >/dev/null; then
    return 0
  fi
  if [ -f "$gen_root/run.json" ] && grep -Eq '"status"[[:space:]]*:[[:space:]]*"complete"' "$gen_root/run.json"; then
    return 0
  fi
  if [ -d "$gen_root/chunks" ] && grep -R -Eq '"state"[[:space:]]*:[[:space:]]*"(reserved|posting|submitted|results_saved|settled|ambiguous|retired|rejected)"' "$gen_root/chunks" 2>/dev/null; then
    return 0
  fi
  return 1
}

part64_scoring_skip_is_failure() {
  local collection_status=$1 run_root=$2 generated_file=$3
  [ "$collection_status" -eq 0 ] && part64_generation_started "$run_root" "$generated_file"
}
