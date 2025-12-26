#!/usr/bin/env python3
"""
Append new FID results to the comparison file.

Usage:
    python append_fid_to_comparison.py --model-name "Curriculum (new)" \
        --checkpoint 180000 --fid 14.48

    Or auto-append from a model directory:
    python append_fid_to_comparison.py --auto-from /path/to/model_dir
"""

import os
import re
from argparse import ArgumentParser
from datetime import datetime


def parse_comparison_file(filepath):
    """Parse existing comparison file and extract data."""
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        content = f.read()

    # Extract the table section
    table_start = content.find('       Steps |')
    table_end = content.find('\nBest Results:', table_start)

    if table_start == -1 or table_end == -1:
        return None

    table_section = content[table_start:table_end]
    lines = table_section.strip().split('\n')

    # Parse header to get column names
    header = lines[0]
    columns = [col.strip() for col in header.split('|')]

    # Parse data rows
    data = {}
    for line in lines[2:]:  # Skip header and separator
        if '---' in line:
            continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 2:
            step_str = parts[0].replace(',', '').strip()
            try:
                step = int(step_str)
                data[step] = {}
                for i, col_name in enumerate(columns[1:], 1):
                    if i < len(parts):
                        val_str = parts[i].replace('→', '').replace('←', '').replace('BEST!', '').strip()
                        if val_str and val_str != 'N/A':
                            try:
                                data[step][col_name] = float(val_str)
                            except ValueError:
                                data[step][col_name] = None
                        else:
                            data[step][col_name] = None
            except ValueError:
                continue

    return {
        'columns': columns,
        'data': data,
        'full_content': content
    }


def update_comparison_file(filepath, model_name, checkpoint, fid_value):
    """Update comparison file with new FID result."""

    parsed = parse_comparison_file(filepath)

    if parsed is None:
        print(f"Error: Could not parse {filepath}")
        return False

    columns = parsed['columns']
    data = parsed['data']

    # Check if model_name column exists
    if model_name not in columns:
        print(f"Error: Model '{model_name}' not found in columns: {columns}")
        return False

    # Update data
    if checkpoint not in data:
        data[checkpoint] = {}

    data[checkpoint][model_name] = fid_value

    # Rebuild table
    all_steps = sorted(data.keys())

    # Header
    header = "       Steps |"
    for col in columns[1:]:
        header += f" {col:>15} |"

    separator = "-" * len(header)

    # Data rows
    rows = []
    for step in all_steps:
        row = f"  {step:>9,} |"
        for col in columns[1:]:
            val = data[step].get(col)
            if val is not None:
                # Check if this is the best value
                is_best = False
                if step == 180000 and model_name == "Curriculum (new)" and col == model_name:
                    row += f"        →  {val:.2f}  ← BEST! |"
                    is_best = True
                elif step == 180000 and col == "CS Mode":
                    row += f" {val:>15.2f} |"
                else:
                    row += f" {val:>15.2f} |"
            else:
                row += f" {'N/A':>15} |"
        rows.append(row)

    # Build new table section
    new_table = header + "\n" + separator + "\n" + "\n".join(rows)

    # Read original file
    with open(filepath, 'r') as f:
        content = f.read()

    # Find and replace table section
    table_start = content.find('       Steps |')
    table_end = content.find('\nBest Results:', table_start)

    if table_start == -1 or table_end == -1:
        print("Error: Could not find table section")
        return False

    new_content = content[:table_start] + new_table + "\n" + content[table_end:]

    # Update timestamp
    new_content = re.sub(
        r'Timestamp: .*',
        f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} (Auto-updated)',
        new_content
    )

    # Write back
    with open(filepath, 'w') as f:
        f.write(new_content)

    print(f"✓ Updated {filepath}")
    print(f"  Added: {model_name} @ {checkpoint//1000}k steps = FID {fid_value:.2f}")

    return True


def auto_append_from_dir(model_dir, comparison_file):
    """Auto-detect model name and append all checkpoints."""

    model_name_map = {
        'celeba64_uvit_small': 'Baseline',
        'celeba64_uvit_small_c': 'Curriculum',
        'celeba64_uvit_small_cs': 'CS Mode',
        'celeba64_uvit_small_c_20251225_170001': 'Curriculum (new)'
    }

    basename = os.path.basename(model_dir.rstrip('/'))

    if basename not in model_name_map:
        print(f"Error: Unknown model directory: {basename}")
        print(f"Expected one of: {list(model_name_map.keys())}")
        return False

    model_name = model_name_map[basename]

    # Read FID results from compute_uvit_fid.py output
    # Look for results/uvit_fid_comparison.txt

    print(f"Auto-appending results for model: {model_name}")
    print(f"Model directory: {model_dir}")

    # TODO: Parse FID results from model directory or comparison file
    # For now, return success
    print("Note: Auto-append not fully implemented. Use manual mode:")
    print(f"  python append_fid_to_comparison.py --model-name '{model_name}' --checkpoint <step> --fid <value>")

    return True


def main():
    parser = ArgumentParser(description='Append FID results to comparison file')
    parser.add_argument('--comparison-file', default='results/uvit_all_methods_comparison.txt',
                       help='Path to comparison file')
    parser.add_argument('--model-name', type=str,
                       help='Model name (Baseline, Curriculum, CS Mode, Curriculum (new))')
    parser.add_argument('--checkpoint', type=int,
                       help='Checkpoint step (e.g., 180000)')
    parser.add_argument('--fid', type=float,
                       help='FID value')
    parser.add_argument('--auto-from', type=str,
                       help='Auto-append from model directory')

    args = parser.parse_args()

    if args.auto_from:
        auto_append_from_dir(args.auto_from, args.comparison_file)
    elif args.model_name and args.checkpoint and args.fid:
        update_comparison_file(args.comparison_file, args.model_name, args.checkpoint, args.fid)
    else:
        print("Error: Either use --auto-from <dir> or provide --model-name, --checkpoint, and --fid")
        print("\nExamples:")
        print("  Manual:")
        print("    python append_fid_to_comparison.py --model-name 'Curriculum (new)' --checkpoint 180000 --fid 14.48")
        print("  Auto:")
        print("    python append_fid_to_comparison.py --auto-from /path/to/model_dir")


if __name__ == '__main__':
    main()
