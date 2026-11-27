#!/usr/bin/env bash
# Single-request YuE2 curl test (portable Linux/macOS).
#
# Usage:
#   ./test_yue2_curl.sh                 # full song (about 48s on GB10)
#   ./test_yue2_curl.sh smoke           # tiny request (fast sanity check)
#   ./test_yue2_curl.sh melody          # melody + ABC
#   ./test_yue2_curl.sh harmony_original
#   ./test_yue2_curl.sh harmony_edited
#
# Environment overrides:
#   SERVER=http://127.0.0.1:30067
#   EXAMPLES=/path/to/YuE/examples
#   OUT=`pwd`/curl_out
#   SEED=831001
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER="${SERVER:-http://127.0.0.1:30067}"
OUT="${OUT:-$SCRIPT_DIR/curl_out}"
TASK="${1:-full_song}"
EXAMPLES="${EXAMPLES:-$SCRIPT_DIR/examples}"

# Find a YuE examples checkout when the default is not present.
if [[ ! -f "$EXAMPLES/song.json" ]]; then
    for candidate in \
        "$SCRIPT_DIR/../YuE/examples" \
        "$SCRIPT_DIR/../YuE-latest/examples" \
        "$HOME/workspace/YuE/examples" \
        "$HOME/workspace/Github/YuE/examples"; do
        if [[ -f "$candidate/song.json" ]]; then
            EXAMPLES="$candidate"
            break
        fi
    done
fi
if [[ ! -f "$EXAMPLES/song.json" ]]; then
    echo "ERROR: song.json not found under $EXAMPLES" >&2
    echo "Set EXAMPLES=/path/to/YuE/examples (must contain song.json)." >&2
    exit 2
fi

mkdir -p "$OUT"

if ! curl -sf -m 5 "$SERVER/health" >/dev/null; then
    echo "ERROR: no healthy server at $SERVER (curl $SERVER/health)" >&2
    exit 1
fi
echo "server: $SERVER (health OK)"
echo "examples: $EXAMPLES"

case "$TASK" in
    smoke)             COT=full;   ABC_FILE="";                          SEM=20;   ABCT=32;   FRAMES=256;  STEPS=2  ;;
    full_song)         COT=full;   ABC_FILE="";                          SEM=9000; ABCT=4096; FRAMES=1024; STEPS=32 ;;
    melody)            COT=melody; ABC_FILE="$EXAMPLES/melody.abc";      SEM=9000; ABCT=4096; FRAMES=1024; STEPS=32 ;;
    harmony_original)  COT=full;   ABC_FILE="$EXAMPLES/score.abc";       SEM=9000; ABCT=4096; FRAMES=1024; STEPS=32 ;;
    harmony_edited)    COT=full;   ABC_FILE="$EXAMPLES/score-jazz.abc";  SEM=9000; ABCT=4096; FRAMES=1024; STEPS=32 ;;
    *) echo "unknown task: $TASK (smoke|full_song|melody|harmony_original|harmony_edited)" >&2; exit 2 ;;
esac

# jq -e fails if the field is missing/empty; that is better than sending a
# payload with an empty style or lyrics.
STYLE="$(jq -er '.style // empty' "$EXAMPLES/song.json")" || {
    echo "ERROR: $EXAMPLES/song.json has no .style" >&2
    exit 2
}
LYRICS="$(jq -r '.lyrics // ""' "$EXAMPLES/song.json")"

# full_song / smoke must send abc=null.  Only melody/cover tasks attach ABC,
# and the file must be non-empty.  This prevents the old 500:
#   "External ABC requires nonempty text and cot=melody/full"
ABC_JSON=null
if [[ -n "$ABC_FILE" ]]; then
    if [[ ! -s "$ABC_FILE" ]]; then
        echo "ERROR: ABC file is missing or empty: $ABC_FILE" >&2
        exit 2
    fi
    ABC_JSON="$(jq -Rs . < "$ABC_FILE")"
fi

PAYLOAD="$OUT/${TASK}_payload.json"
jq -n \
    --arg style "$STYLE" --arg lyrics "$LYRICS" --arg cot "$COT" \
    --argjson abc "$ABC_JSON" --argjson seed "${SEED:-831001}" \
    --argjson sem "$SEM" --argjson abct "$ABCT" \
    --argjson frames "$FRAMES" --argjson steps "$STEPS" \
    --arg out "$OUT" --arg fname "${TASK}.wav" \
    '{style:$style, lyrics:$lyrics, cot:$cot, abc:$abc, seed:$seed,
      ode_steps:$steps, semantic_max_tokens:$sem, abc_max_tokens:$abct,
      vae_core_frames:$frames, vae_halo_frames:16,
      output_path:$out, output_file_name:$fname}' > "$PAYLOAD"

echo "POST $SERVER/v1/audio/generations  task=$TASK  cot=$COT  semantic=$SEM  abc=$ABCT  frames=$FRAMES"
META="$OUT/${TASK}_meta.txt"
curl -sS --max-time 600 -o "$OUT/${TASK}_resp.json" \
    -w '%{http_code} %{time_total}\n' \
    -X POST "$SERVER/v1/audio/generations" \
    -H 'Content-Type: application/json' -d @"$PAYLOAD" > "$META"
read -r HTTP_CODE WALL < "$META"
echo "http=$HTTP_CODE  wall=${WALL}s"

if [[ "$HTTP_CODE" -ge 400 ]]; then
    echo "request failed; response:" >&2
    cat "$OUT/${TASK}_resp.json" >&2
    echo >&2
    echo "payload: $PAYLOAD" >&2
    exit 1
fi

jq -r '{inference_time_s, sample_rate, audio_len:(.audio|length)}' \
    "$OUT/${TASK}_resp.json" 2>/dev/null || cat "$OUT/${TASK}_resp.json"
echo "response : $OUT/${TASK}_resp.json"
echo "payload  : $PAYLOAD"
echo "wav      : $OUT/${TASK}.wav"
ls -lh "$OUT/${TASK}.wav" 2>/dev/null || true
