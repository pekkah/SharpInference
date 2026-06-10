#!/bin/bash
# SessionStart hook for SharpInference.
#
# Provisions the .NET 10 SDK so Claude Code on the web can build and test the
# project. The container state is cached after this hook completes, so the SDK
# install + NuGet restore only pay their cost on the first session.
#
# NOTE: This project targets net10.0 (see Directory.Build.props). The .NET SDK
# binaries are downloaded from Microsoft's CDNs (aka.ms / builds.dotnet.microsoft.com).
# Those hosts must be reachable from the environment's network policy. NuGet
# (api.nuget.org) must also be reachable for package restore.
set -euo pipefail

# Only run in Claude Code on the web (remote) sessions; locals already have an SDK.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

DOTNET_DIR="$HOME/.dotnet"
DOTNET_CHANNEL="10.0"

# Idempotent: only install if the SDK isn't already present.
if [ ! -x "$DOTNET_DIR/dotnet" ]; then
  echo "session-start: installing .NET SDK ${DOTNET_CHANNEL} -> ${DOTNET_DIR}" >&2
  TMP_SCRIPT="$(mktemp)"
  # raw.githubusercontent.com hosts the official installer (reachable when GitHub is allowlisted).
  curl -fsSL https://raw.githubusercontent.com/dotnet/install-scripts/main/src/dotnet-install.sh -o "$TMP_SCRIPT"
  chmod +x "$TMP_SCRIPT"
  "$TMP_SCRIPT" --channel "$DOTNET_CHANNEL" --install-dir "$DOTNET_DIR" --no-path
  rm -f "$TMP_SCRIPT"
else
  echo "session-start: .NET SDK already present at ${DOTNET_DIR}" >&2
fi

# Persist environment for this session and every subsequent tool call.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  {
    echo "export DOTNET_ROOT=\"$DOTNET_DIR\""
    echo "export PATH=\"$DOTNET_DIR:\$PATH\""
    echo "export DOTNET_CLI_TELEMETRY_OPTOUT=1"
    echo "export DOTNET_NOLOGO=1"
  } >> "$CLAUDE_ENV_FILE"
fi

export DOTNET_ROOT="$DOTNET_DIR"
export PATH="$DOTNET_DIR:$PATH"
export DOTNET_CLI_TELEMETRY_OPTOUT=1
export DOTNET_NOLOGO=1

echo "session-start: dotnet $("$DOTNET_DIR/dotnet" --version)" >&2

# Warm the NuGet cache so the first in-session build is fast (cached in the container image).
"$DOTNET_DIR/dotnet" restore "$CLAUDE_PROJECT_DIR/SharpInference.slnx" >&2 || \
  echo "session-start: restore failed (non-fatal); will restore on first build" >&2

echo "session-start: setup complete" >&2
