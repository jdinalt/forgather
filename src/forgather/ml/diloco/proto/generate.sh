#!/usr/bin/env bash
# Regenerate the DiLoCo bulk-transport gRPC stubs from bulk.proto (issue #154).
#
# The generated bulk_pb2.py / bulk_pb2_grpc.py are committed/vendored so runtime
# needs only `grpcio` (a hard dependency); `grpcio-tools` is only needed here to
# regenerate. It is declared in the project dependencies, so this just works:
#
#   bash src/forgather/ml/diloco/proto/generate.sh
#
# The grpc stub's import is rewritten to a package-relative import so it works
# regardless of sys.path (protoc emits a bare `import bulk_pb2`).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -m grpc_tools.protoc \
  -I "$here" \
  --python_out="$here" \
  --grpc_python_out="$here" \
  "$here/bulk.proto"

# protoc emits `import bulk_pb2 as ...` (top-level); make it package-relative.
sed -i 's/^import bulk_pb2 as /from . import bulk_pb2 as /' "$here/bulk_pb2_grpc.py"

echo "Regenerated bulk_pb2.py and bulk_pb2_grpc.py in $here"
