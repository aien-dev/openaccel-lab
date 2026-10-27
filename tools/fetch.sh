#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# tools/fetch.sh [all|omega|pocl|hip]
# Fetch the toolchain pinned in toolchain.lock into .oal/ and refuse anything whose sha256 or
# commit id differs. Sources are GitHub only. Writes nothing outside .oal/.
# Exit 0, 1 on a refused or missing file, 2 on a usage error.
# INTERIM host glue (POSIX sh + curl + git). Successor: an Omega fetch-and-verify unit once seeded
# Omega has an HTTPS client word; curl moves bytes, sha256sum decides what is accepted.
set -eu
cd "$(dirname "$0")/.."
. ./toolchain.lock
what=${1:-all}
case $what in all|omega|pocl|hip) ;; *) echo "usage: tools/fetch.sh [all|omega|pocl|hip]" >&2; exit 2;; esac
D=.oal
mkdir -p "$D/tools"

get() {
    url=$1; out=$2; want=$3
    if [ -f "$out" ] && [ "$(sha256sum "$out" | cut -d ' ' -f 1)" = "$want" ]; then
        echo "fetch: cached $want $out" >&2
        return 0
    fi
    curl -fsSL --retry 3 -o "$out.part" "$url"
    got=$(sha256sum "$out.part" | cut -d ' ' -f 1)
    if [ "$got" != "$want" ]; then
        rm -f "$out.part"
        echo "fetch: REFUSED $url sha256 $got, pinned $want" >&2
        exit 1
    fi
    mv "$out.part" "$out"
    echo "fetch: OK $want $out" >&2
}

checkout() {
    repo=$1; commit=$2; dir=$3
    if [ -d "$dir/.git" ] && [ "$(git -C "$dir" rev-parse HEAD)" = "$commit" ]; then
        echo "fetch: cached $repo@$commit" >&2
        return 0
    fi
    rm -rf "$dir"
    git init -q "$dir"
    git -C "$dir" fetch -q --depth 1 "https://github.com/$repo.git" "$commit"
    git -C "$dir" checkout -q FETCH_HEAD
    got=$(git -C "$dir" rev-parse HEAD)
    [ "$got" = "$commit" ] || { echo "fetch: REFUSED $repo at $got, pinned $commit" >&2; exit 1; }
    echo "fetch: OK $repo@$commit" >&2
}

if [ "$what" = all ] || [ "$what" = omega ]; then
    get "https://raw.githubusercontent.com/$OMEGA_MV_REPO/$OMEGA_MV_COMMIT/seed/out/omc" "$D/tools/omc" "$OMC_SHA256"
    chmod +x "$D/tools/omc"
fi

if [ "$what" = all ] || [ "$what" = pocl ]; then
    get "https://github.com/$POCL_REPO/releases/download/$POCL_TAG/$POCL_ASSET" "$D/$POCL_ASSET" "$POCL_SHA256"
    rm -rf "$D/pocl" "$D/icd"
    mkdir -p "$D/pocl" "$D/icd"
    tar -xzf "$D/$POCL_ASSET" -C "$D/pocl" --strip-components=1
    [ -f "$D/pocl/$POCL_LIB" ] || { echo "fetch: $POCL_LIB missing from $POCL_ASSET" >&2; exit 1; }
    echo "$(pwd)/$D/pocl/$POCL_LIB" > "$D/icd/pocl.icd"
    echo "fetch: PoCL at $D/pocl; export OCL_ICD_VENDORS=$(pwd)/$D/icd" >&2
fi

if [ "$what" = all ] || [ "$what" = hip ]; then
    checkout "$HIPCPU_REPO" "$HIPCPU_COMMIT" "$D/hip-cpu"
    get "https://raw.githubusercontent.com/$HIPIFY_REPO/$HIPIFY_COMMIT/bin/hipify-perl" "$D/tools/hipify-perl" "$HIPIFY_PERL_SHA256"
fi
