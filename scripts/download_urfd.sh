#!/usr/bin/env bash
# Baixa os 70 vídeos câmera 0 do URF Fall Detection Dataset (30 quedas + 40 ADLs)
# para data/urfd/videos/{fall,adl}/. Idempotente: reexecutar pula arquivos já
# presentes e válidos, sem requisição de rede.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

BASE_URL="https://fenix.ur.edu.pl/~mkepski/ds/data"
VIDEOS_DIR="$REPO_ROOT/data/urfd/videos"
FALL_DIR="$VIDEOS_DIR/fall"
ADL_DIR="$VIDEOS_DIR/adl"

downloaded=0
skipped=0
failed=0
failed_files=()

is_valid_mp4() {
  local path="$1"
  local mime
  mime="$(file --mime-type -b "$path" 2>/dev/null || true)"
  if [[ "$mime" == "video/mp4" ]]; then
    return 0
  fi
  if head -c 16 "$path" 2>/dev/null | grep -aq "ftyp"; then
    return 0
  fi
  return 1
}

looks_like_html_error() {
  local path="$1"
  if head -c 1 "$path" 2>/dev/null | grep -aq "^<"; then
    return 0
  fi
  if head -c 512 "$path" 2>/dev/null | grep -aqi "<html\|<!doctype"; then
    return 0
  fi
  return 1
}

is_valid_download() {
  local path="$1"
  if ! is_valid_mp4 "$path"; then
    return 1
  fi
  if looks_like_html_error "$path"; then
    return 1
  fi
  return 0
}

download_one() {
  local url="$1"
  local dest="$2"

  if [[ -f "$dest" ]] && is_valid_download "$dest"; then
    echo "PULADO (já existe e é válido): $dest"
    skipped=$((skipped + 1))
    return 0
  fi

  local tmp="${dest}.tmp"
  rm -f "$tmp"

  if ! curl -fsS --connect-timeout 15 --max-time 300 --retry 3 --retry-delay 2 -o "$tmp" "$url"; then
    echo "FALHA (download): $url" >&2
    rm -f "$tmp"
    failed=$((failed + 1))
    failed_files+=("$dest")
    return 1
  fi

  if ! is_valid_download "$tmp"; then
    echo "FALHA (conteúdo inválido, provável página de erro): $url" >&2
    rm -f "$tmp"
    failed=$((failed + 1))
    failed_files+=("$dest")
    return 1
  fi

  mv "$tmp" "$dest"
  echo "OK: $dest"
  downloaded=$((downloaded + 1))
  return 0
}

mkdir -p "$FALL_DIR" "$ADL_DIR"

for nn in $(seq -w 1 30); do
  download_one "$BASE_URL/fall-${nn}-cam0.mp4" "$FALL_DIR/fall-${nn}-cam0.mp4" || true
done

for nn in $(seq -w 1 40); do
  download_one "$BASE_URL/adl-${nn}-cam0.mp4" "$ADL_DIR/adl-${nn}-cam0.mp4" || true
done

echo
echo "===== Resumo ====="
echo "Baixados: $downloaded"
echo "Pulados (já presentes e válidos): $skipped"
echo "Falhas: $failed"

if ((failed > 0)); then
  echo "Arquivos com falha:"
  for f in "${failed_files[@]}"; do
    echo "  - $f"
  done
  exit 1
fi

exit 0
