import os
import re

INCLUDE_REGEX = re.compile(r'^\s*#\s*include\s*["]([^"]+)["]')

def resolve_include(file_path, include_paths, seen=None):
    if seen is None:
        seen = set()

    if file_path in seen:
        return []
    seen.add(file_path)

    deps = []

    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = INCLUDE_REGEX.match(line)
                if match:
                    header = match.group(1)
                    header_path = find_include(header, include_paths)
                    if header_path:
                        deps.append(header_path)
                        deps += resolve_include(header_path, include_paths, seen)
                    else:
                        print(f"{header} not found")
    except FileNotFoundError:
        pass

    return deps

def find_include(header, include_paths):
    for path in include_paths:
        full_path = os.path.join(path, header)
        if os.path.isfile(full_path):
            return os.path.abspath(full_path)
    return None

# Example usage
if __name__ == '__main__':
    import sys
    source_dir = sys.argv[1]
    include_dirs = sys.argv[2:]
    deps = []
    for source_filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, source_filename)
        deps += resolve_include(source_file, include_dirs)
    for dep in sorted(set(deps)):
        print(dep)
