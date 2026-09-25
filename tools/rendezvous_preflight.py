"""Pre-flight reachability check for the multi-node rendezvous port.

The distributed rendezvous (both torchrun's elastic store and c10d's own TCPStore)
is built on a TCPStore served by rank 0's node.

This script checks whether all nodes can reach the rendezvous port on rank 0's
node.

Usage:

    python3 rendezvous_preflight.py --listen 62396          # on rank 0's node
    python3 rendezvous_preflight.py --connect <ip> 62396    # on every node

`--connect` exits non-zero and prints the node's own name, so the job log names the
offending node instead of just timing out.
"""

import argparse
import socket
import sys
import time


def listen(port: int, deadline_s: float) -> int:
    """Bind `port` on all interfaces and accept connections until the deadline."""
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("", port))
    srv.listen(64)
    srv.settimeout(2.0)
    print(f"# preflight listener bound on 0.0.0.0:{port}", flush=True)
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        try:
            conn, _ = srv.accept()
            conn.close()
        except TimeoutError:
            continue
    srv.close()
    return 0


def connect(host: str, port: int, timeout_s: float) -> int:
    """Try one TCP connection to host:port, naming this node either way."""
    node = socket.gethostname()
    try:
        socket.create_connection((host, port), timeout=timeout_s).close()
    except OSError as exc:
        print(f"# preflight FAIL {node}: cannot reach {host}:{port} -> {exc}", flush=True)
        return 1
    print(f"# preflight OK   {node}: {host}:{port} reachable", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--listen", type=int, metavar="PORT", help="serve PORT on all interfaces")
    parser.add_argument("--connect", nargs=2, metavar=("HOST", "PORT"), help="dial HOST PORT")
    parser.add_argument(
        "--deadline", type=float, default=120.0, help="listen seconds (default 120)"
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="connect seconds (default 10)")
    args = parser.parse_args()

    if args.listen:
        return listen(args.listen, args.deadline)
    if args.connect:
        host, port = args.connect
        return connect(host, int(port), args.timeout)
    parser.error("pass either --listen PORT or --connect HOST PORT")
    return 2


if __name__ == "__main__":
    sys.exit(main())
