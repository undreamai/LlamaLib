import os
import re
import sys
from collections import defaultdict
import difflib

def readfile(path):
    with open(path, 'r') as f:
        return [l.rstrip('\n') for l in f.readlines()]


def split_by_original_structure(original_file_contents, patched_lines):
    # Remove includes and pragma once from combined original for fair matching
    def strip_lines(lines):
        return [
            line for line in lines
            if not line.strip().startswith('#include') and line.strip() != '#pragma once'
        ]

    # Find best splitter points in the patched_lines based on original structure
    splits = []
    prev_idx = 0
    for segment in original_file_contents[:-1]:
        segment = strip_lines(segment)
        best_score = 0
        best_split = None
        for last_idx in range(prev_idx+1, min(len(patched_lines), prev_idx + len(segment)*2)):
            window = patched_lines[prev_idx:last_idx]
            score = difflib.SequenceMatcher(None, segment, window).ratio()
            if score > best_score:
                best_score = score
                best_split = last_idx
        if best_split is None:
            raise ValueError("Failed to find match for original segment %d" % i)
        splits.append(patched_lines[prev_idx:best_split])
        prev_idx = best_split

    splits.append(patched_lines[prev_idx:])
    return splits


def reinsert_directives(original_lines, patched_lines, context_lines=5, match_threshold=0.6):
    """
    Reinsert `#include` lines and `#pragma once` from original_lines into patched_lines by matching context.

    Args:
        original_lines: list of lines from the original file
        patched_lines: list of lines from the patched (flattened) file
        context_lines: number of non-include lines to use for context
        match_threshold: min similarity ratio to consider a valid match

    Returns:
        New list of lines with #include and #pragma once reinserted
    """
    # Detect #pragma once
    has_pragma_once = any(line.strip() == '#pragma once' for line in original_lines)

    # Collect all include lines and their context
    include_entries = []

    for i, line in enumerate(original_lines):
        if line.strip().startswith('#include'):
            # Collect forward context (N non-include lines)
            context = []
            for j in range(i + 1, len(original_lines)):
                next_line = original_lines[j].strip()
                if not next_line.startswith('#include') and next_line != '#pragma once':
                    context.append(next_line)
                if len(context) >= context_lines:
                    break
            include_entries.append((line.rstrip('\n'), context))

    # Insertions: (position in patched_lines, line)
    insertions = []
    for include_line, context in include_entries:
        best_score = 0
        best_pos = None
        for i in range(len(patched_lines) - len(context)):
            window = [patched_lines[i + k].strip() for k in range(len(context))]
            score = difflib.SequenceMatcher(None, context, window).ratio()
            if score > best_score:
                best_score = score
                best_pos = i
        if best_pos is not None and best_score >= match_threshold:
            insertions.append((best_pos, include_line))
        else:
            insertions.append((0, include_line))  # fallback to top

    # Sort insertions to apply top-down
    # insertions.sort()
    output_lines = patched_lines[:]
    offset = 0
    inserted_lines = set()

    for pos, line in insertions:
        key = (pos, line)
        if key in inserted_lines:
            continue  # skip duplicates
        output_lines.insert(pos + offset, line)
        inserted_lines.add(key)
        offset += 1

    # Insert #pragma once at the top if needed and not already there
    if has_pragma_once and (len(output_lines) == 0 or output_lines[0].strip() != '#pragma once'):
        output_lines.insert(0, '#pragma once\n')

    return output_lines


concatenated_path = sys.argv[1]  # path to the modified rollup file
input_dir_path = sys.argv[2]  # path to the ggml_cuda dir
output_dir_path = sys.argv[3]  # output path

concat_lines = readfile(concatenated_path)
lines_proc = []
filenames_proc = []

line_no = 0
flag=True
while line_no < len(concat_lines) - 1:
    line_no += 1

    line = concat_lines[line_no].rstrip('\n')
    if '#include' in line or '#pragma once' in line:
        continue

    if '////////////////////////////////////////////////////////////////////////////////' in line:
        line_no += 1
        continue


    if 'ROLLUP' in line:
        filename = line.replace('// ROLLUP ', '')
        line_no += 4
        if len(lines_proc) and len(filenames_proc):
            filenames_proc = filenames_proc[::-1]

            orig_file_lines = [readfile(os.path.join(input_dir_path, filename_proc)) for filename_proc in filenames_proc]
            if (len(filenames_proc) > 1):
                patched_file_lines = split_by_original_structure(orig_file_lines, lines_proc)
            else:
                patched_file_lines = [lines_proc]

            for filename_output, orig_lines, patched_lines in zip(filenames_proc, orig_file_lines, patched_file_lines):
                output_lines = reinsert_directives(orig_lines, patched_lines)
                output_filepath = os.path.join(output_dir_path, filename_output)
                os.makedirs(os.path.dirname(output_filepath), exist_ok = True)
                with open(output_filepath, 'w') as outf:
                    outf.writelines('\n'.join(output_lines))

            filenames_proc = []
            lines_proc = []
        filenames_proc.append(filename)
        continue
        


    if len(filenames_proc):
        lines_proc.append(line)
