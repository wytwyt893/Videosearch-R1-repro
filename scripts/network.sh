#!/usr/bin/env bash
# Process-local network policy. Never changes system proxy configuration.
export HF_ENDPOINT="${VIDEOSEARCH_HF_ENDPOINT:-https://hf.co}"
case "${VIDEOSEARCH_NETWORK_MODE:-proxy}" in
  proxy)
    export HTTP_PROXY="${VIDEOSEARCH_HTTP_PROXY:-http://127.0.0.1:7890}"
    export HTTPS_PROXY="${VIDEOSEARCH_HTTPS_PROXY:-$HTTP_PROXY}"
    export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY"
    unset ALL_PROXY all_proxy
    export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,::1}"
    export no_proxy="$NO_PROXY"
    ;;
  direct)
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
    ;;
  inherit) ;;
  *)
    echo 'VIDEOSEARCH_NETWORK_MODE must be proxy, direct, or inherit' >&2
    return 2 2>/dev/null || exit 2
    ;;
esac
