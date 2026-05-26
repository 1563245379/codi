#!/bin/bash
set -e

# Claude Code One-Click Installer for Linux

echo "=== Claude Code Installer ==="

# Step 1: Install Node.js (if not present)
if ! command -v node &> /dev/null; then
    echo "[1/5] Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
else
    echo "[1/5] Node.js already installed: $(node --version)"
fi

# Step 2: Install Claude Code CLI
echo "[2/5] Installing Claude Code CLI..."
npm install -g @anthropic-ai/claude-code

# Step 3: Create settings directory
echo "[3/5] Configuring settings..."
CONFIG_DIR="$HOME/.claude"
mkdir -p "$CONFIG_DIR"

# Step 4: Write settings.json
SETTINGS_FILE="$CONFIG_DIR/settings.json"

# Resolve token: require DEEPSEEK_API_KEY env var
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "Error: DEEPSEEK_API_KEY environment variable is not set."
    echo "Usage:  DEEPSEEK_API_KEY=sk-xxx bash install-claude.sh"
    exit 1
fi
AUTH_TOKEN="$DEEPSEEK_API_KEY"
echo "  -> Using DEEPSEEK_API_KEY from environment variable"

cat > "$SETTINGS_FILE" << EOF
{
  "autoMemoryEnabled": true,
  "enabledPlugins": {
    "claude-md-management@claude-plugins-official": true,
    "code-simplifier@claude-plugins-official": true,
    "remember@claude-plugins-official": true,
    "superpowers@claude-plugins-official": true,
    "skill-creator@claude-plugins-official": true,
    "context7@claude-plugins-official": true
  },
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "${AUTH_TOKEN}",
    "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-v4-flash",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "deepseek-v4-pro[1M]",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "deepseek-v4-flash[1M]",
    "CLAUDE_AUTO_BACKGROUND_TASKS": "1",
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1",
    "ENABLE_TOOL_SEARCH": "true",
    "CLAUDE_CODE_EFFORT_LEVEL": "max"
  },
  "language": "Chinese",
  "model": "opus",
  "permissions": {
    "allow": [],
    "defaultMode": "default"
  },
  "skipDangerousModePermissionPrompt": true,
  "theme": "auto"
}
EOF

# Step 5: Verify installation
echo "[4/5] Verifying installation..."
if command -v claude &> /dev/null; then
    echo "[5/5] Claude Code installed: $(claude --version 2>/dev/null || echo 'CLI ready')"
else
    echo "Warning: claude command not found in PATH"
fi

echo ""
echo "=== Installation Complete ==="
echo "Settings written to: $SETTINGS_FILE"
echo ""
echo "Starting Claude Code:"
IS_SANDBOX=1 claude --dangerously-skip-permissions