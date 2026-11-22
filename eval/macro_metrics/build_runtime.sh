#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$ROOT/build"
CC_BIN="${CC:-clang}"
mkdir -p "$BUILD"

COMMON=(
  -O2 -fPIC -I"$ROOT" -I"$ROOT/src" -I"$ROOT/ocean/drive" -I"$ROOT/vendor"
  -DPLATFORM_DESKTOP -DDRIVE_AGENT_MAX=512
  -Wno-unused-function -Wno-unused-variable -Wno-unused-but-set-variable
)

if [[ "$(uname -s)" == "Darwin" ]]; then
  RAYLIB="$ROOT/raylib-5.5_macos"
  "$CC_BIN" "${COMMON[@]}" -dynamiclib -I"$RAYLIB/include" \
    "$ROOT/ocean/drive/offline_api.c" "$ROOT/vendor/cJSON.c" \
    "$RAYLIB/lib/libraylib.a" -framework Cocoa -framework IOKit \
    -framework CoreVideo -framework OpenGL -lm \
    -o "$BUILD/libdrive_offline.so"
else
  RAYLIB="$ROOT/raylib-5.5_linux_amd64"
  "$CC_BIN" "${COMMON[@]}" -shared -D_GNU_SOURCE -I"$RAYLIB/include" \
    "$ROOT/ocean/drive/offline_api.c" "$ROOT/vendor/cJSON.c" \
    "$RAYLIB/lib/libraylib.a" -lGL -lm -lpthread -ldl -lrt -lX11 \
    -o "$BUILD/libdrive_offline.so"
fi

"$CC_BIN" -O3 -DPUFFER_DRIVE -I"$ROOT/ocean/drive" -I"$RAYLIB/include" \
  "$ROOT/eval/dreal_highway/policy_act.c" "$ROOT/src/puffercpu.c" -lm \
  -o "$BUILD/i24_policy_act"

echo "$BUILD/libdrive_offline.so"
echo "$BUILD/i24_policy_act"
