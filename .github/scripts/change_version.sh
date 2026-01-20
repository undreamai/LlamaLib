#!/bin/bash

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <new_version>"
    exit 1
fi

NEW_VERSION=$1
CURRENT_VERSION=$(cat ./VERSION | tr -d '[:space:]')

echo "Updating version from $CURRENT_VERSION to $NEW_VERSION"

# Update VERSION file
echo "$NEW_VERSION" > ./VERSION

# Files to update
FILES=(
    "./tests/csharp/LlamaLib.Test.csproj"
    "./examples/csharp/agent/LlamaLibExamples.csproj"
    "./examples/csharp/remote_client/LlamaLibExamples.csproj"
    "./examples/csharp/basic_embeddings/LlamaLibExamples.csproj"
    "./examples/csharp/basic/LlamaLibExamples.csproj"
    "./.github/tests/csharp-dotnet/Program.csproj"
    "./csharp/LlamaLib.csproj"
    "./csharp/LlamaLib.targets"
    "./cmake/LlamaLibConfigVersion.cmake"
    "./.github/doxygen/Doxyfile"
)

# Patterns to replace (old -> new)
PATTERNS=(
    "Version=\"${CURRENT_VERSION}\"|Version=\"${NEW_VERSION}\""
    "<PackageVersion>${CURRENT_VERSION}<\/PackageVersion>|<PackageVersion>${NEW_VERSION}<\/PackageVersion>"
    "<LlamaLibVersion>${CURRENT_VERSION}<\/LlamaLibVersion>|<LlamaLibVersion>${NEW_VERSION}<\/LlamaLibVersion>"
    "set(PACKAGE_VERSION \"${CURRENT_VERSION}\")|set(PACKAGE_VERSION \"${NEW_VERSION}\")"
    "PROJECT_NUMBER = v${CURRENT_VERSION}|PROJECT_NUMBER = v${NEW_VERSION}"
)

# Apply all patterns to all files
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        for pattern in "${PATTERNS[@]}"; do
            OLD=$(echo "$pattern" | cut -d'|' -f1)
            NEW=$(echo "$pattern" | cut -d'|' -f2)
            sed -i "s/${OLD}/${NEW}/g" "$file" 2>/dev/null || sed -i '' "s/${OLD}/${NEW}/g" "$file" 2>/dev/null || true
        done
        echo "Updated: $file"
    fi
done

echo "Done!"
