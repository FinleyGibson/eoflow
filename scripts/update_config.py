#!/usr/bin/env python3
"""
Update eoflow configuration file with latest defaults.

This script updates an existing configuration file to include any new
configuration keys that have been added to the default configuration,
while preserving all existing user-configured values.

A backup of the original configuration is created before updating.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add src to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eoflow.config import Config, get_default_config_dir
from eoflow.log_utils import get_logger

logger = get_logger(__name__)


def backup_config(config_file: Path) -> Path:
    """
    Create a backup of the config file.

    Args:
        config_file: Path to the config file to backup

    Returns:
        Path to the backup file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = (
        config_file.parent
        / f"{config_file.stem}_backup_{timestamp}{config_file.suffix}"
    )
    shutil.copy2(config_file, backup_file)
    return backup_file


def update_config(
    config_file: Path, dry_run: bool = False, verbose: bool = True
) -> bool:
    """
    Update a config file with latest defaults while preserving user values.

    Args:
        config_file: Path to the config file to update
        dry_run: If True, show what would be done without making changes
        verbose: If True, print detailed information

    Returns:
        True if update was successful, False otherwise
    """
    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}")
        return False

    if verbose:
        print(f"Updating config file: {config_file}")

    # Load existing config
    try:
        with open(config_file, "r") as f:
            existing_config = json.load(f)
        if verbose:
            print(
                f"✓ Loaded existing config with {len(existing_config)} sections"
            )
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error: Could not read existing config: {e}")
        return False

    # Get default config
    default_config = Config._default_config()
    if verbose:
        print(f"✓ Loaded default config with {len(default_config)} sections")

    # Merge configs (existing values take precedence)
    merged_config = Config._merge_configs(default_config, existing_config)

    # Analyze changes
    changes = analyze_changes(existing_config, merged_config)

    if verbose:
        print("\n=== Configuration Analysis ===")
        if changes["new_sections"]:
            print(f"\nNew sections to be added: {len(changes['new_sections'])}")
            for section in changes["new_sections"]:
                print(f"  + {section}")

        if changes["new_keys"]:
            print(f"\nNew keys to be added: {len(changes['new_keys'])}")
            for section, keys in changes["new_keys"].items():
                for key in keys:
                    print(f"  + {section}.{key}")

        if changes["preserved_keys"]:
            print(f"\nPreserved user values: {len(changes['preserved_keys'])}")
            if verbose:
                for key, value in changes["preserved_keys"].items():
                    print(f"  ✓ {key} = {value}")

        if not changes["new_sections"] and not changes["new_keys"]:
            print("\n✓ Config is already up to date!")
            return True

    # Create backup and update config
    if not dry_run:
        try:
            backup_file = backup_config(config_file)
            if verbose:
                print(f"\n✓ Created backup: {backup_file}")

            # Write updated config
            with open(config_file, "w") as f:
                json.dump(merged_config, f, indent=2)

            if verbose:
                print(f"✓ Updated config file: {config_file}")
                print("\n✅ Configuration update complete!")

            return True

        except (IOError, OSError) as e:
            print(f"\nError: Could not update config: {e}")
            return False
    else:
        print("\n🔍 DRY RUN - No changes made")
        return True


def analyze_changes(existing: dict, merged: dict) -> dict:
    """
    Analyze the differences between existing and merged configs.

    Args:
        existing: Existing configuration dictionary
        merged: Merged configuration dictionary

    Returns:
        Dictionary with analysis results
    """
    changes = {
        "new_sections": [],
        "new_keys": {},
        "preserved_keys": {},
    }

    # Find new sections
    for section in merged:
        if section not in existing:
            changes["new_sections"].append(section)
        elif isinstance(merged[section], dict):
            # Find new keys within existing sections
            new_keys = []
            for key in merged[section]:
                if key not in existing.get(section, {}):
                    new_keys.append(key)
                elif existing[section][key] != merged[section][key]:
                    # User has customized this value
                    changes["preserved_keys"][f"{section}.{key}"] = existing[
                        section
                    ][key]

            if new_keys:
                changes["new_keys"][section] = new_keys

    return changes


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Update eoflow config file with latest defaults",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update config in default location
  python scripts/update_config.py

  # Update specific config file
  python scripts/update_config.py --config /path/to/config.json

  # Preview changes without updating
  python scripts/update_config.py --dry-run

  # Update quietly
  python scripts/update_config.py --quiet
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to config file (default: use platform default location)",
    )

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be done without making changes",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # Determine config file path
    if args.config:
        config_file = args.config
    else:
        config_file = get_default_config_dir() / "config.json"

    # Run update
    verbose = not args.quiet

    if verbose:
        print("=" * 70)
        print("eoflow Configuration Updater")
        print("=" * 70)
        print()

    success = update_config(config_file, dry_run=args.dry_run, verbose=verbose)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
