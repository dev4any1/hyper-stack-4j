#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# hyper-stack-4j — local dev runner
# Requires: JDK 21+  ·  Maven 3.8+
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MVN="${MVN:-mvn}"     # override: MVN=/path/to/mvn ./run-me.sh test
PORT="${PORT:-8080}"  # coordinator REST port

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info() { echo -e "${CYAN}▶ $*${NC}"; }
ok()   { echo -e "${GREEN}✔ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠ $*${NC}"; }
err()  { echo -e "${RED}✖ $*${NC}" >&2; exit 1; }

# ── Dependency check ──────────────────────────────────────────────────────────
check_deps() {
  command -v java   >/dev/null 2>&1 || err "JDK 21+ not found. Install from https://adoptium.net"
  command -v "$MVN" >/dev/null 2>&1 || err "Maven not found.   brew install maven  or  sudo apt install maven"
  JAVA_VER=$(java -version 2>&1 | awk -F'"' '/version/{print $2}' | cut -d. -f1)
  [[ "${JAVA_VER:-0}" -ge 21 ]] || err "JDK 21+ required (found: $JAVA_VER)"
}

# ── Commands ──────────────────────────────────────────────────────────────────

cmd_test() {
  info "Running all unit tests (skipping integration)..."
  cd "$DIR"
  "$MVN" test -DskipITs --no-transfer-progress
  ok "All unit tests passed"
}

cmd_test_module() {
  local mod="${1:-}"
  [[ -n "$mod" ]] || err "Usage: $0 test-module <module>  e.g. coordinator  health  kvcache"
  [[ -d "$DIR/$mod" ]] || err "Module not found: $mod"
  info "Running tests for module: $mod"
  cd "$DIR"
  "$MVN" test -pl "$mod" -am --no-transfer-progress
  ok "$mod tests passed"
}

cmd_test_fault() {
  info "Running fault tolerance tests only..."
  cd "$DIR"
  "$MVN" test -pl coordinator -am \
    -Dtest="FaultTolerantPipelineTest,HealthReactorTest,RetryPolicyTest" \
    --no-transfer-progress
  ok "Fault tolerance tests passed"
}

cmd_integration() {
  warn "Integration tests fork 3 JVM processes — takes ~30s"
  info "Running full integration suite..."
  cd "$DIR"
  "$MVN" verify -pl integration --no-transfer-progress
  ok "Integration tests passed"
}

cmd_integration_fast() {
  info "Running fast in-process cluster test only (~250ms)..."
  cd "$DIR"
  "$MVN" verify -pl integration -Dit.test=InProcessClusterIT --no-transfer-progress
  ok "InProcessClusterIT passed"
}

cmd_build() {
  info "Compiling all modules (no tests)..."
  cd "$DIR"
  "$MVN" compile -DskipTests --no-transfer-progress
  ok "Build succeeded"
}

cmd_clean() {
  info "Cleaning all build artefacts..."
  cd "$DIR"
  "$MVN" clean --no-transfer-progress
  ok "Clean done"
}

cmd_verify() {
  info "Full verify: compile + unit tests + integration tests..."
  cd "$DIR"
  "$MVN" verify --no-transfer-progress
  ok "Full verify passed"
}

cmd_health_demo() {
  info "Fault tolerance wiring overview"
  cat <<'JAVA'

── What HealthReactor wires together ────────────────────────────────────────────

  FaultTolerantPipeline pipeline = new FaultTolerantPipeline(
      List.of(
          NodePipeline.of("node-1", grpcPipelineToNode1),
          NodePipeline.of("node-2", grpcPipelineToNode2)
      ),
      RetryPolicy.once()  // 2 attempts, 50ms backoff
  );

  HealthReactor reactor = new HealthReactor(
      HealthThresholds.defaults(),   // warning=90%, critical=98%, stale=15s
      pipeline,
      scheduler                      // shut down if all nodes go unavailable
  );

  // Hazelcast IMap listener (each GPU node publishes NodeHealth every 5s):
  nodeHealthMap.addEntryListener(event -> {
      reactor.onHealthProbe(event.getValue());   // <── one call drives everything
  }, true);

  // What happens automatically:
  //   node-1 VRAM reaches 99%    → VRAM_CRITICAL → circuit OPEN
  //                                forward() transparently routes to node-2
  //   node-1 VRAM drops to 60%   → NODE_RECOVERED → circuit reset to CLOSED
  //                                node-1 accepts traffic again
  //   node-2 misses 3 probes     → NODE_STALE → circuit OPEN
  //                                both circuits OPEN → scheduler.shutdown()
  //                                → new requests get HTTP 503

── Circuit states after each event ─────────────────────────────────────────────

  Scenario               node-1 circuit    node-2 circuit
  ──────────────────     ──────────────    ──────────────
  Both healthy           CLOSED            CLOSED
  node-1 VRAM=99%        OPEN              CLOSED   ← route to node-2
  node-1 recovers        CLOSED            CLOSED   ← back to normal
  Both VRAM=99%          OPEN              OPEN     ← scheduler shutdown

── Run the tests to see every scenario live ─────────────────────────────────────

  ./run-me.sh test-fault

JAVA
}

cmd_curl_demo() {
  info "Example REST API commands (coordinator must be running on :${PORT})"
  cat <<CURL

  # Blocking inference
  curl -s -X POST http://localhost:${PORT}/v1/inference \\
       -H 'Content-Type: application/json' \\
       -d '{
             "modelId":  "tinyllama",
             "messages": [{"role": "user", "content": "Hello!"}]
           }' | python3 -m json.tool

  # Streaming inference — tokens arrive as Server-Sent Events
  curl -sN -X POST http://localhost:${PORT}/v1/inference/stream \\
       -H 'Content-Type: application/json' \\
       -d '{
             "modelId":  "tinyllama",
             "messages": [{"role": "user", "content": "Count to five"}]
           }'

  # List registered models
  curl -s http://localhost:${PORT}/v1/models | python3 -m json.tool

  # Cluster health
  curl -s http://localhost:${PORT}/v1/cluster/health | python3 -m json.tool

CURL
}

cmd_watch() {
  local mod="${1:-coordinator}"
  info "Watching $mod for changes (Ctrl-C to stop)..."
  command -v fswatch >/dev/null 2>&1 || err "fswatch not found — brew install fswatch"
  fswatch -o "$DIR/$mod/src" | while read -r; do
    echo ""
    info "Change detected — re-running $mod tests..."
    "$MVN" test -pl "$mod" -am -q --no-transfer-progress 2>&1 | tail -20
    ok "Done at $(date +%H:%M:%S)"
  done
}

usage() {
  echo ""
  echo -e "${CYAN}hyper-stack-4j dev runner${NC}"
  echo ""
  echo "  $0 test                   Unit tests — all modules, skip integration (~10s)"
  echo "  $0 test-module <mod>      Unit tests for one module  e.g. coordinator  health"
  echo "  $0 test-fault             Fault tolerance tests only (FaultTolerantPipeline, HealthReactor)"
  echo "  $0 integration            Full integration suite — forks 3 JVMs (~30s)"
  echo "  $0 integration-fast       InProcessClusterIT only (~250ms)"
  echo "  $0 build                  Compile only, no tests"
  echo "  $0 clean                  Remove all target/ directories"
  echo "  $0 verify                 Full: compile + unit + integration"
  echo "  $0 health-demo            Fault tolerance wiring walkthrough"
  echo "  $0 curl-demo              Example REST API curl commands"
  echo "  $0 watch [mod]            Auto-rerun tests on file changes (requires fswatch)"
  echo ""
  echo "  MVN=/path/to/mvn $0 test  Override Maven binary"
  echo "  PORT=9090 $0 curl-demo    Override coordinator port"
  echo ""
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
check_deps

CMD="${1:-}"
case "$CMD" in
  test)              cmd_test ;;
  test-module)       cmd_test_module "${2:-}" ;;
  test-fault)        cmd_test_fault ;;
  integration)       cmd_integration ;;
  integration-fast)  cmd_integration_fast ;;
  build)             cmd_build ;;
  clean)             cmd_clean ;;
  verify)            cmd_verify ;;
  health-demo)       cmd_health_demo ;;
  curl-demo)         cmd_curl_demo ;;
  watch)             cmd_watch "${2:-coordinator}" ;;
  *)                 usage ;;
esac
