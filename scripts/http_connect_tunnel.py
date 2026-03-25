#!/usr/bin/env python3
"""Create an HTTP CONNECT tunnel for SSH ProxyCommand or quick reachability checks."""

from __future__ import annotations

import argparse
import os
import selectors
import socket
import sys
from urllib.parse import urlparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open an HTTP CONNECT tunnel through the configured proxy."
    )
    parser.add_argument("target_host", help="Target hostname to CONNECT to.")
    parser.add_argument("target_port", type=int, help="Target TCP port.")
    parser.add_argument(
        "--proxy-url",
        default=None,
        help=(
            "HTTP proxy URL. Defaults to HTTPS_PROXY, https_proxy, HTTP_PROXY, "
            "http_proxy, ALL_PROXY, or all_proxy."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify that CONNECT succeeds, then exit.",
    )
    return parser.parse_args()


def resolve_proxy_url(explicit: str | None) -> str:
    if explicit:
        return explicit
    for key in (
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        value = os.environ.get(key)
        if value:
            return value
    raise SystemExit("No proxy URL found. Pass --proxy-url or export HTTPS_PROXY/HTTP_PROXY.")


def parse_proxy_endpoint(proxy_url: str) -> tuple[str, int]:
    parsed = urlparse(proxy_url)
    if parsed.scheme not in {"http", "https", ""}:
        raise SystemExit(f"Unsupported proxy scheme: {parsed.scheme!r}")
    if not parsed.hostname or not parsed.port:
        raise SystemExit(f"Invalid proxy URL: {proxy_url}")
    return parsed.hostname, parsed.port


def connect_tunnel(proxy_host: str, proxy_port: int, target_host: str, target_port: int) -> socket.socket:
    sock = socket.create_connection((proxy_host, proxy_port), timeout=15)
    request = (
        f"CONNECT {target_host}:{target_port} HTTP/1.1\r\n"
        f"Host: {target_host}:{target_port}\r\n"
        "Proxy-Connection: Keep-Alive\r\n"
        "\r\n"
    ).encode("ascii")
    sock.sendall(request)

    response = bytearray()
    while b"\r\n\r\n" not in response:
        chunk = sock.recv(4096)
        if not chunk:
            raise SystemExit("Proxy closed the connection before returning a CONNECT response.")
        response.extend(chunk)
        if len(response) > 65536:
            raise SystemExit("CONNECT response is unexpectedly large.")

    header, remainder = bytes(response).split(b"\r\n\r\n", 1)
    status_line = header.splitlines()[0].decode("iso-8859-1", errors="replace")
    if " 200 " not in status_line:
        raise SystemExit(f"CONNECT failed: {status_line}")

    sock.setblocking(False)
    if remainder:
        sys.stdout.buffer.write(remainder)
        sys.stdout.buffer.flush()
    return sock


def relay(sock: socket.socket) -> int:
    selector = selectors.DefaultSelector()
    selector.register(sock, selectors.EVENT_READ, "sock")
    selector.register(sys.stdin.buffer, selectors.EVENT_READ, "stdin")

    stdout = sys.stdout.buffer
    stdin = sys.stdin.buffer

    while True:
        for key, _mask in selector.select():
            if key.data == "sock":
                try:
                    data = sock.recv(65536)
                except BlockingIOError:
                    continue
                if not data:
                    return 0
                stdout.write(data)
                stdout.flush()
            else:
                data = stdin.read1(65536)
                if not data:
                    try:
                        sock.shutdown(socket.SHUT_WR)
                    except OSError:
                        pass
                    selector.unregister(stdin)
                    continue
                try:
                    sock.sendall(data)
                except BrokenPipeError:
                    return 0


def main() -> int:
    args = parse_args()
    proxy_url = resolve_proxy_url(args.proxy_url)
    proxy_host, proxy_port = parse_proxy_endpoint(proxy_url)
    sock = connect_tunnel(proxy_host, proxy_port, args.target_host, args.target_port)
    if args.check:
        print(
            f"CONNECT OK: {args.target_host}:{args.target_port} via {proxy_host}:{proxy_port}",
            file=sys.stderr,
        )
        sock.close()
        return 0
    try:
        return relay(sock)
    finally:
        sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
