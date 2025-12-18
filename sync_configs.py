#!/usr/bin/env python3
"""
Sync model config across celeba32_* or celeba64_* config files.

Usage:
    # Sync celeba32_* files (use celeba32_c.json as source)
    python sync_configs.py --prefix celeba32

    # Sync celeba64_* files
    python sync_configs.py --prefix celeba64

    # Sync specific fields only
    python sync_configs.py --prefix celeba32 --fields hid_channels ch_multipliers

    # Show current configs without syncing
    python sync_configs.py --prefix celeba32 --dry-run
"""

import os
import re
import json
from argparse import ArgumentParser


def load_json_with_comments(filepath):
    """Load JSON file that may contain // comments."""
    with open(filepath, "r") as f:
        content = f.read()
    # Remove // comments
    content = re.sub(r'^\s*//.*$', '', content, flags=re.MULTILINE)
    return json.loads(content)


def save_json_with_comments(filepath, data, original_content):
    """Save JSON while preserving header comments."""
    # Extract header comments
    lines = original_content.split('\n')
    header_lines = []
    for line in lines:
        if line.strip().startswith('//') or line.strip() == '':
            header_lines.append(line)
        else:
            break

    header = '\n'.join(header_lines)
    json_str = json.dumps(data, indent=2)

    with open(filepath, 'w') as f:
        if header:
            f.write(header + '\n')
        f.write(json_str + '\n')


def main():
    parser = ArgumentParser()
    parser.add_argument("--prefix", default="celeba32",
                        help="Config prefix (celeba32 or celeba64)")
    parser.add_argument("--fields", nargs="+",
                        default=["hid_channels", "ch_multipliers", "num_res_blocks", "apply_attn", "drop_rate"],
                        help="Model fields to sync")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show current configs without syncing")
    parser.add_argument("--source", default=None,
                        help="Source config suffix (default: _c)")
    args = parser.parse_args()

    config_dir = os.path.join(os.path.dirname(__file__), "configs")

    # Find all matching configs
    suffixes = ["_c", "_cs", "_cr"]
    source_suffix = args.source or "_c"

    configs = {}
    for suffix in suffixes:
        path = os.path.join(config_dir, f"{args.prefix}{suffix}.json")
        if os.path.exists(path):
            configs[suffix] = {
                "path": path,
                "data": load_json_with_comments(path),
                "original": open(path).read()
            }

    if not configs:
        print(f"No configs found for prefix: {args.prefix}")
        return

    # Show current state
    print(f"\n{'='*60}")
    print(f"  {args.prefix}_* Model Configs")
    print(f"{'='*60}")

    # Header
    header = f"{'Field':<20}"
    for suffix in suffixes:
        if suffix in configs:
            header += f" | {suffix:^15}"
    print(header)
    print("-" * len(header))

    # Data rows
    for field in args.fields:
        row = f"{field:<20}"
        for suffix in suffixes:
            if suffix in configs:
                val = configs[suffix]["data"]["model"].get(field, "N/A")
                val_str = str(val)
                if len(val_str) > 15:
                    val_str = val_str[:12] + "..."
                row += f" | {val_str:^15}"
        print(row)

    if args.dry_run:
        print(f"\n(Dry run - no changes made)")
        return

    if source_suffix not in configs:
        print(f"\nSource config {args.prefix}{source_suffix}.json not found!")
        return

    # Sync from source to others
    source_model = configs[source_suffix]["data"]["model"]
    updated = []

    for suffix, cfg in configs.items():
        if suffix == source_suffix:
            continue

        changed = False
        for field in args.fields:
            if field in source_model:
                old_val = cfg["data"]["model"].get(field)
                new_val = source_model[field]
                if old_val != new_val:
                    cfg["data"]["model"][field] = new_val
                    changed = True

        if changed:
            save_json_with_comments(cfg["path"], cfg["data"], cfg["original"])
            updated.append(suffix)

    if updated:
        print(f"\nSynced from {source_suffix} to: {', '.join(updated)}")
    else:
        print(f"\nAll configs already in sync!")


if __name__ == "__main__":
    main()
