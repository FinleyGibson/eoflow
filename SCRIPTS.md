# Config Update Script Usage Guide

## Overview

The `update_config.py` script is designed to update your eoflow configuration file when new configuration options are added to the package. It intelligently merges your existing settings with the latest defaults, ensuring you don't lose any customizations.

## Quick Start

```bash
# From the eoflow project root
python scripts/update_config.py
```

This will:
1. Load your existing config from the default location
2. Merge it with the latest default configuration
3. Create a timestamped backup of your old config
4. Save the updated config with all new keys added

## Command Line Options

### `--config` / `-c`
Specify a custom config file location instead of the default.

```bash
python scripts/update_config.py --config /path/to/custom/config.json
```

### `--dry-run` / `-n`
Preview what changes would be made without actually modifying any files.

```bash
python scripts/update_config.py --dry-run
```

This is useful to see what new configuration options have been added.

### `--quiet` / `-q`
Suppress verbose output, only showing errors.

```bash
python scripts/update_config.py --quiet
```

## Common Use Cases

### 1. After Updating eoflow

When you pull the latest changes from the repository:

```bash
git pull
python scripts/update_config.py
```

### 2. Check for New Config Options

To see what