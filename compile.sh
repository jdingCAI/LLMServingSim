#!/bin/bash
set -e

# Path
SCRIPT_DIR=$(dirname "$(realpath $0)")

# Intall Chakra (Use Chakra fork in ASTRA-Sim repo).
(
cd ${SCRIPT_DIR}/astra-sim/extern/graph_frontend/chakra
pip3 install .
)

# Compile ASTRA-sim with analytical backend model
# Ensure system protoc is used so its version matches the system libprotobuf.
# Anaconda ships an older protoc whose generated headers are incompatible with
# the system libprotobuf that CMake links against.
(
cd ${SCRIPT_DIR}/astra-sim
if [ -x /usr/bin/protoc ]; then
  export PATH="/usr/bin:$PATH"
fi
bash ./build/astra_analytical/build.sh
)

# Compile ASTRA-sim with ns3 backend model
# (
# cd ${SCRIPT_DIR}/astra-sim
# bash ./build/astra_ns3/build.sh
# )
