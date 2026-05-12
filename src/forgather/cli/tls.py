"""`forgather tls` — manage shared TLS state."""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import sys
from pathlib import Path
from typing import Optional

from forgather.tls import load_config
from forgather.tls.ca import (
    cert_info,
    create_ca,
    import_trusted_ca,
    install_server_cert,
    mint_server_cert,
    rebuild_bundle,
)
from forgather.tls.config import save_config
from forgather.tls.discovery import detect_hostnames, detect_ips, merge_san


def tls_cmd(args) -> int:
    sub = getattr(args, "tls_subcommand", None)
    if sub is None:
        print("forgather tls: missing subcommand. Try 'forgather tls --help'.", file=sys.stderr)
        return 2
    handlers = {
        "init": _cmd_init,
        "status": _cmd_status,
        "renew": _cmd_renew,
        "export-ca": _cmd_export_ca,
        "import-ca": _cmd_import_ca,
        "mint": _cmd_mint,
        "install": _cmd_install,
        "deploy": _cmd_deploy,
        "trust-system": _cmd_trust_system,
        "clean": _cmd_clean,
        "enable": _cmd_enable,
        "disable": _cmd_disable,
    }
    handler = handlers.get(sub)
    if handler is None:
        print(f"forgather tls: unknown subcommand: {sub}", file=sys.stderr)
        return 2
    return handler(args) or 0


# ----------------------------------------------------------------------- init


def _cmd_init(args) -> int:
    cfg = load_config()
    hostnames, ips = merge_san(
        detect_hostnames(),
        detect_ips(),
        extra_hostnames=args.hostname,
        extra_ips=args.ip,
    )
    cfg.san_hostnames = hostnames
    cfg.san_ips = ips
    cfg.enabled = True

    ca_existed = cfg.has_ca_authority()
    if ca_existed and not args.force:
        print(f"CA already exists at {cfg.ca_cert} — keeping it.")
    else:
        cn = args.ca_name or f"Forgather CA {platform.node() or socket.gethostname()}"
        create_ca(cfg, common_name=cn, force=args.force)
        print(f"Created CA: {cfg.ca_cert}")

    minted = mint_server_cert(cfg, hostnames=hostnames, ips=ips)
    install_server_cert(cfg, minted)
    print(f"Issued server cert: {cfg.server_cert}")
    print(f"  SAN hostnames: {', '.join(hostnames)}")
    print(f"  SAN IPs:       {', '.join(ips)}")

    rebuild_bundle(cfg)
    save_config(cfg)
    print(f"Wrote config: {cfg.config_file}")
    print()
    print(
        "TLS is now enabled. Every forgather server on this host will\n"
        "serve HTTPS on next start — including loopback binds. Update\n"
        "any bookmarks/scripts that hard-code http:// URLs:\n"
        "  https://localhost:8765/   (forgather server, with ?token=…)\n"
        "  https://localhost:8766/   (dataset_server)\n"
        "  https://localhost:8137/   (inference_server)\n"
        "\n"
        f"State directory: {cfg.root}\n"
        "\n"
        "To opt a single server back to HTTP, pass --no-tls on its\n"
        "command line. To opt out globally without removing the CA,\n"
        "run 'forgather tls disable' (the cert files stay on disk).\n"
        "\n"
        "To set up a peer, mint a cert for it from this host:\n"
        "  forgather tls mint --hostname peer.lan --ip 10.0.0.99 -o /tmp/peer-tls\n"
        "  forgather tls export-ca -o /tmp/forgather-ca.crt"
    )
    return 0


# --------------------------------------------------------------------- status


def _cmd_status(args) -> int:
    cfg = load_config()
    data: dict = {
        "root": str(cfg.root),
        "config_path": str(cfg.config_path) if cfg.config_path else None,
        "enabled": cfg.enabled,
        "auto_on_non_loopback": cfg.auto_on_non_loopback,
        "verify_hostname": cfg.verify_hostname,
        "provisioned": cfg.is_provisioned(),
        "has_ca_authority": cfg.has_ca_authority(),
        "ca_cert": str(cfg.ca_cert),
        "server_cert": str(cfg.server_cert),
        "ca_bundle": str(cfg.ca_bundle),
        "san_hostnames": cfg.san_hostnames,
        "san_ips": cfg.san_ips,
        "trusted": [],
        "ca_info": None,
        "server_info": None,
    }
    if cfg.has_ca_authority():
        try:
            data["ca_info"] = _stringify_serial(cert_info(cfg.ca_cert))
        except Exception as exc:
            data["ca_info"] = {"error": str(exc)}
    if cfg.is_provisioned():
        try:
            data["server_info"] = _stringify_serial(cert_info(cfg.server_cert))
        except Exception as exc:
            data["server_info"] = {"error": str(exc)}
    if cfg.trusted_dir.is_dir():
        for entry in sorted(cfg.trusted_dir.glob("*.crt")):
            try:
                info = _stringify_serial(cert_info(entry))
                info["file"] = str(entry)
                data["trusted"].append(info)
            except Exception as exc:
                data["trusted"].append({"file": str(entry), "error": str(exc)})

    if args.json:
        print(json.dumps(data, indent=2, default=str))
        return 0

    print(f"TLS root        : {data['root']}")
    print(f"Config file     : {data['config_path'] or '<absent>'}")
    print(f"enabled         : {data['enabled']}")
    print(f"provisioned     : {data['provisioned']}")
    print(f"CA authority    : {data['has_ca_authority']}")
    print(
        f"verify_hostname : {cfg.verify_hostname} "
        f"({'strict RFC-6125' if cfg.verify_hostname else 'chain-only — LAN default'})"
    )
    if data["ca_info"]:
        info = data["ca_info"]
        print()
        print("CA cert:")
        print(f"  subject        : {info.get('subject')}")
        print(f"  not_after      : {info.get('not_after')} ({info.get('days_remaining')} days)")
    if data["server_info"]:
        info = data["server_info"]
        print()
        print("Server cert:")
        print(f"  subject        : {info.get('subject')}")
        print(f"  not_after      : {info.get('not_after')} ({info.get('days_remaining')} days)")
        print(f"  SAN hostnames  : {', '.join(info.get('san_dns', []))}")
        print(f"  SAN IPs        : {', '.join(info.get('san_ip', []))}")
    if data["trusted"]:
        print()
        print(f"Trusted CAs ({len(data['trusted'])}):")
        for t in data["trusted"]:
            if "error" in t:
                print(f"  ! {t['file']}: {t['error']}")
            else:
                print(f"  {t['file']}: {t.get('subject')}  ({t.get('days_remaining')}d)")

    # Diagnostic warnings. Listed at the bottom so they're the last
    # thing the operator sees — these are the items most likely to
    # cause a "why isn't this working?" support question.
    warnings: list[str] = []
    if cfg.is_provisioned() and not cfg.enabled:
        warnings.append(
            "Cert is provisioned but enabled=false — servers will only "
            "use TLS when --tls is passed per invocation. Run 'forgather "
            "tls enable' to flip the master switch."
        )
    if data["server_info"] and isinstance(data["server_info"], dict):
        srv = data["server_info"]
        days = srv.get("days_remaining")
        if isinstance(days, int) and days < 30:
            warnings.append(
                f"Server cert expires in {days} days — run 'forgather "
                "tls renew --server' before then."
            )
        # SAN coverage gap: compare cert SAN against currently-detected
        # IPs/hostnames. Anything in the host's current address list
        # that isn't in the cert means peers/clients dialing that
        # address will get a hostname-mismatch error.
        try:
            from forgather.tls.discovery import detect_hostnames, detect_ips

            current_hosts = set(detect_hostnames())
            current_ips = set(detect_ips())
            covered_hosts = {h.lower() for h in srv.get("san_dns", [])}
            covered_ips = set(srv.get("san_ip", []))
            missing_h = sorted(current_hosts - covered_hosts)
            missing_i = sorted(current_ips - covered_ips)
            if missing_h or missing_i:
                msg = "SAN gap — cert does not cover this host's current "
                bits = []
                if missing_h:
                    bits.append(f"hostnames: {', '.join(missing_h)}")
                if missing_i:
                    bits.append(f"IPs: {', '.join(missing_i)}")
                msg += "; ".join(bits) + "."
                msg += (
                    " Re-issue with 'forgather tls renew --server "
                    "--add-hostname <NAME> --add-ip <IP>'."
                )
                warnings.append(msg)
        except Exception:
            pass

    if warnings:
        print()
        print("Warnings:")
        for w in warnings:
            print(f"  ! {w}")
    return 0


def _stringify_serial(info: dict) -> dict:
    # JSON-friendly: serial is an int, but huge — keep as hex string.
    s = info.get("serial")
    if isinstance(s, int):
        info["serial"] = f"0x{s:x}"
    return info


# ---------------------------------------------------------------------- renew


def _cmd_renew(args) -> int:
    cfg = load_config()
    if not cfg.has_ca_authority():
        print(
            "No CA authority on this host — cannot renew. "
            "(Did you mean 'forgather tls install' on a peer?)",
            file=sys.stderr,
        )
        return 1

    hostnames, ips = merge_san(
        cfg.san_hostnames,
        cfg.san_ips,
        extra_hostnames=args.add_hostname,
        extra_ips=args.add_ip,
    )
    cfg.san_hostnames = hostnames
    cfg.san_ips = ips

    if args.ca:
        # CA renewal breaks every peer's trust bundle. Refuse silent
        # nuke — operator has to confirm via stdin.
        print(
            "WARNING: --ca will re-issue this host's CA. Every peer that "
            "trusts the current CA will need the new ca.crt redistributed "
            "before it can validate certs again.",
            file=sys.stderr,
        )
        reply = input("Type 'yes' to continue: ").strip().lower()
        if reply != "yes":
            print("Aborted; CA not renewed.", file=sys.stderr)
            return 1
        cn = f"Forgather CA {platform.node() or socket.gethostname()}"
        create_ca(cfg, common_name=cn, force=True)
        print(f"Re-issued CA: {cfg.ca_cert}")

    minted = mint_server_cert(cfg, hostnames=hostnames, ips=ips)
    install_server_cert(cfg, minted)
    rebuild_bundle(cfg)
    save_config(cfg)
    print(f"Renewed server cert: {cfg.server_cert}")
    return 0


# ------------------------------------------------------------------ export-ca


def _cmd_export_ca(args) -> int:
    cfg = load_config()
    if not cfg.ca_cert.is_file():
        print(f"No CA cert at {cfg.ca_cert}", file=sys.stderr)
        return 1
    data = cfg.ca_cert.read_bytes()
    if args.output == "-":
        sys.stdout.buffer.write(data)
        sys.stdout.flush()
    else:
        out = Path(os.path.expanduser(args.output))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        try:
            os.chmod(out, 0o644)
        except OSError:
            pass
        print(f"Wrote CA cert: {out}")
    return 0


# ------------------------------------------------------------------ import-ca


def _cmd_import_ca(args) -> int:
    cfg = load_config()
    src = Path(os.path.expanduser(args.path))
    if not src.is_file():
        print(f"No such file: {src}", file=sys.stderr)
        return 1
    try:
        dest = import_trusted_ca(cfg, src, name=args.name)
    except ValueError as exc:
        print(f"Refused to import: {exc}", file=sys.stderr)
        return 1
    print(f"Imported CA: {dest}")
    print(f"Bundle: {cfg.ca_bundle}")
    return 0


# ----------------------------------------------------------------------- mint


def _cmd_mint(args) -> int:
    cfg = load_config()
    if not cfg.has_ca_authority():
        print(
            "This host has no CA authority — run 'forgather tls init' "
            "or use the CA holder for minting.",
            file=sys.stderr,
        )
        return 1

    # --hostname / --ip are now optional. On a private LAN with a
    # private CA, the SAN is informational (peer-pull and webui
    # proxies do chain-only validation by default — see
    # docs/operations/tls.md and TLSConfig.verify_hostname). Pass
    # explicit SAN entries only when you also need browsers or
    # external clients to hit the URL by a specific name/IP and
    # have those clients enforce hostname matching.
    extra_hosts = list(args.hostname or [])
    extra_ips = list(args.ip or [])
    if not extra_hosts and not extra_ips:
        # Placeholder SAN that satisfies RFC 5280's "must have at
        # least one SAN entry" requirement without baking in an
        # IP/hostname the operator may not know yet.
        extra_hosts = ["forgather-peer", "localhost"]
        extra_ips = ["127.0.0.1", "::1"]
        print(
            "No --hostname/--ip given; minting a chain-only-trust cert "
            "with placeholder SAN. Peers will validate this cert by CA "
            "chain, ignoring the SAN. Add --hostname/--ip if a browser "
            "or strict-hostname client needs to reach the URL directly.",
            file=sys.stderr,
        )

    hostnames, ips = merge_san([], [], extra_hostnames=extra_hosts, extra_ips=extra_ips)
    minted = mint_server_cert(cfg, hostnames=hostnames, ips=ips)

    out_dir = Path(os.path.expanduser(args.output))
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(out_dir, 0o700)
    except OSError:
        pass
    cert_path = out_dir / "server.crt"
    key_path = out_dir / "server.key"
    cert_path.write_bytes(minted.cert_pem)
    # Atomic 0600 creation for the key — see forgather.tls.ca._write_secret.
    try:
        key_path.unlink()
    except FileNotFoundError:
        pass
    fd = os.open(str(key_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, minted.key_pem)
    finally:
        os.close(fd)
    try:
        os.chmod(cert_path, 0o644)
        os.chmod(key_path, 0o600)
    except OSError as exc:
        print(
            f"Could not set 0600 on {key_path}: {exc}. "
            "Refusing to leave a freshly-minted private key with default permissions.",
            file=sys.stderr,
        )
        return 1

    # Also drop a copy of the CA next to it so the recipient can install
    # everything in one go.
    ca_copy = out_dir / "ca.crt"
    ca_copy.write_bytes(cfg.ca_cert.read_bytes())
    try:
        os.chmod(ca_copy, 0o644)
    except OSError:
        pass

    print(f"Issued server cert: {cert_path}")
    print(f"Issued server key : {key_path}")
    print(f"CA cert (bundle)  : {ca_copy}")
    print(
        "\nDistribute the directory to the peer host, then run there:\n"
        "  forgather tls install --cert server.crt --key server.key --ca ca.crt"
    )
    return 0


# -------------------------------------------------------------------- install


def _cmd_install(args) -> int:
    cfg = load_config()
    cert_src = Path(os.path.expanduser(args.cert))
    key_src = Path(os.path.expanduser(args.key))
    if not cert_src.is_file():
        print(f"--cert: no such file: {cert_src}", file=sys.stderr)
        return 1
    if not key_src.is_file():
        print(f"--key: no such file: {key_src}", file=sys.stderr)
        return 1

    # Cross-validate cert + key + (optional) CA before touching shared
    # state. Catches the "operator merged the wrong files" footgun and
    # the "evil cert + legit CA" supply-chain confusion described in
    # the security audit.
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    try:
        cert_obj = x509.load_pem_x509_certificate(cert_src.read_bytes())
    except Exception as exc:
        print(f"--cert is not a valid PEM certificate: {exc}", file=sys.stderr)
        return 1
    try:
        key_obj = serialization.load_pem_private_key(key_src.read_bytes(), password=None)
    except Exception as exc:
        print(f"--key is not a readable PEM private key: {exc}", file=sys.stderr)
        return 1
    # The cert's public key must match the supplied private key.
    if cert_obj.public_key().public_numbers() != key_obj.public_key().public_numbers():
        print(
            "--cert and --key do not match (cert's public key does not "
            "correspond to the supplied private key). Refusing to install.",
            file=sys.stderr,
        )
        return 1

    ca_obj: Optional["x509.Certificate"] = None
    if args.ca:
        ca_src = Path(os.path.expanduser(args.ca))
        if not ca_src.is_file():
            print(f"--ca: no such file: {ca_src}", file=sys.stderr)
            return 1
        try:
            ca_obj = x509.load_pem_x509_certificate(ca_src.read_bytes())
        except Exception as exc:
            print(f"--ca is not a valid PEM certificate: {exc}", file=sys.stderr)
            return 1
        # Must actually be a CA.
        try:
            from cryptography.x509.oid import ExtensionOID

            bc = ca_obj.extensions.get_extension_for_oid(
                ExtensionOID.BASIC_CONSTRAINTS
            ).value
            if not getattr(bc, "ca", False):
                print(
                    f"--ca {ca_src} is not marked as a CA "
                    "(BasicConstraints CA:FALSE). Refusing to install.",
                    file=sys.stderr,
                )
                return 1
        except x509.ExtensionNotFound:
            print(
                f"--ca {ca_src} has no BasicConstraints extension; "
                "not a CA. Refusing to install.",
                file=sys.stderr,
            )
            return 1
        # Cert must chain to the supplied CA — verify the signature.
        try:
            ca_obj.public_key().verify(
                cert_obj.signature,
                cert_obj.tbs_certificate_bytes,
                padding.PKCS1v15(),
                cert_obj.signature_hash_algorithm,
            )
        except Exception as exc:
            print(
                f"--cert is not signed by --ca (signature verify failed: {exc}). "
                "Refusing to install — did you mix cert and CA from different bundles?",
                file=sys.stderr,
            )
            return 1

    cfg.root.mkdir(parents=True, exist_ok=True)
    cfg.server_cert.parent.mkdir(parents=True, exist_ok=True)
    cfg.server_cert.write_bytes(cert_src.read_bytes())
    # Write the key atomically with 0600 from the start. Mirrors the
    # logic in forgather.tls.ca._write_secret.
    try:
        cfg.server_key.unlink()
    except FileNotFoundError:
        pass
    fd = os.open(str(cfg.server_key), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, key_src.read_bytes())
    finally:
        os.close(fd)
    try:
        os.chmod(cfg.server_cert, 0o644)
        os.chmod(cfg.server_key, 0o600)
    except OSError as exc:
        print(
            f"Could not set 0600 on {cfg.server_key}: {exc}. "
            "Refusing to leave a private key with default permissions.",
            file=sys.stderr,
        )
        return 1
    print(f"Installed cert: {cfg.server_cert}")
    print(f"Installed key : {cfg.server_key}")

    if ca_obj is not None:
        cfg.ca_cert.parent.mkdir(parents=True, exist_ok=True)
        cfg.ca_cert.write_bytes(Path(os.path.expanduser(args.ca)).read_bytes())
        try:
            os.chmod(cfg.ca_cert, 0o644)
        except OSError:
            pass
        print(f"Installed CA  : {cfg.ca_cert} (no key — this host cannot mint)")

    # Populate SAN list from the cert we just installed so status reflects
    # the actual coverage.
    try:
        info = cert_info(cfg.server_cert)
        cfg.san_hostnames = list(info.get("san_dns") or [])
        cfg.san_ips = list(info.get("san_ip") or [])
    except Exception:
        pass

    cfg.enabled = True
    rebuild_bundle(cfg)
    save_config(cfg)
    print(f"Wrote config : {cfg.config_file}")
    return 0


# --------------------------------------------------------------------- deploy


def _cmd_deploy(args) -> int:
    """Mint + scp + install for every cluster peer via ssh.

    Reuses the local forgather server's cluster membership table —
    so the operator never has to type peer hostnames or IPs. Each
    peer gets a freshly minted cert (placeholder SAN, chain-only
    trust); the local CA private key stays on this host.
    """
    import getpass
    import shutil
    import subprocess
    import tempfile

    cfg = load_config()
    if not cfg.has_ca_authority():
        print(
            "This host has no CA authority — run 'forgather tls init' first.",
            file=sys.stderr,
        )
        return 1

    # Build the host-override map from --ssh-host PEER=HOST entries.
    ssh_host_override: dict = {}
    for raw in args.ssh_host or []:
        if "=" not in raw:
            print(f"--ssh-host expects PEER=HOST (got {raw!r})", file=sys.stderr)
            return 2
        peer, host = raw.split("=", 1)
        ssh_host_override[peer.strip()] = host.strip()

    # Two modes of peer discovery:
    #
    #   1. Operator passed positional hostnames/IPs → use those directly
    #      as ssh targets. The peers DON'T have to be in any cluster
    #      membership table — this is the bootstrap path (peer can't
    #      join the cluster yet because it has no TLS, but ssh works
    #      regardless).
    #
    #   2. No positional args → consult the local forgather server's
    #      membership table. Convenient for "deploy to everyone."
    #      Requires the local server to be running with --cluster.
    if args.nodes:
        peers = [
            {"hostname": n, "address": ssh_host_override.get(n, n), "node_id": None}
            for n in args.nodes
        ]
    else:
        try:
            from forgather.cli.server_client import ServerClient

            client = ServerClient(getattr(args, "server", None))
            members_payload = client.cluster_members()
            self_payload = client.cluster_self()
        except Exception as exc:
            print(
                f"could not reach local forgather server: {exc}\n"
                "  Either start it ('forgather server --cluster <name>') so deploy\n"
                "  can read the membership table, or pass peer hostnames explicitly:\n"
                "    forgather tls deploy <host1> <host2> ...",
                file=sys.stderr,
            )
            return 1

        members = members_payload.get("members") or []
        self_id = (self_payload or {}).get("node_id") if self_payload else None
        if not members:
            print(
                "No cluster members visible. Either start the server with\n"
                "--cluster <name>, or pass peer hostnames explicitly:\n"
                "  forgather tls deploy <host1> <host2> ..."
            )
            return 1

        # Drop self; keep "unreachable" peers — reachability in the
        # membership table is driven by HTTPS peer-pull success, and
        # the operator is likely running `deploy` precisely because
        # that pull is broken (TLS state missing or mismatched). ssh
        # is independent of forgather's TLS, so we always try — the
        # per-peer ssh result is the authoritative answer.
        peers = [m for m in members if not (self_id and m.get("node_id") == self_id)]

    if not peers:
        print("No peers to deploy to.")
        return 0

    ssh_user = args.ssh_user or os.environ.get("USER") or getpass.getuser()

    # Plan summary up front so the operator sees what's about to happen
    # before any passwords are prompted for.
    print(f"Deploying TLS state to {len(peers)} peer(s) as ssh user {ssh_user!r}:")
    for p in peers:
        target = ssh_host_override.get(p.get("hostname"), p.get("address"))
        print(f"  • {p.get('hostname')} ({target})")
    print()

    if args.dry_run:
        print("--dry-run: stopping here.")
        return 0

    ssh_base = ["ssh"]
    scp_base = ["scp", "-p"]
    if args.batch:
        ssh_base += ["-o", "BatchMode=yes"]
        scp_base += ["-o", "BatchMode=yes"]

    # Container-target map: per-peer override falls back to --container.
    container_for: dict = {}
    for raw in args.container_host or []:
        if "=" not in raw:
            print(f"--container-host expects PEER=NAME (got {raw!r})", file=sys.stderr)
            return 2
        peer, name = raw.split("=", 1)
        container_for[peer.strip()] = name.strip()

    results: list[tuple] = []
    for peer in peers:
        container = container_for.get(peer.get("hostname"), args.container)
        result = _deploy_to_peer(
            cfg,
            peer,
            ssh_target=ssh_host_override.get(peer.get("hostname"), peer.get("address")),
            ssh_user=ssh_user,
            ssh_base=tuple(ssh_base),
            scp_base=tuple(scp_base),
            force=args.force,
            container=container,
        )
        results.append((peer, result))
        status = "OK" if result is True else f"FAILED — {result}"
        print(f"  {peer.get('hostname')}: {status}")

    failed = [r for _, r in results if r is not True]
    print()
    if failed:
        print(f"{len(failed)} of {len(results)} peer(s) failed.", file=sys.stderr)
        return 1
    print(f"All {len(results)} peer(s) deployed. Restart the forgather server on each peer for TLS to take effect.")
    return 0


def _ca_fingerprint(ca_cert_path: Path) -> str:
    """SHA-256 of the CA cert's on-disk PEM bytes.

    Matches what ``sha256sum`` would produce on the peer's copy, so
    the local and remote hashes are directly comparable.
    """
    import hashlib

    return hashlib.sha256(ca_cert_path.read_bytes()).hexdigest()


def _peer_ca_fingerprint(ssh_base, target, container=None) -> "Optional[str]":
    """Compute the SHA-256 of the peer's CA cert PEM via ssh. None if absent.

    When ``container`` is set, the probe runs inside that Docker
    container and looks at the in-container path (the host's
    ``~/.config/forgather/`` may or may not exist; what we care about
    is what the running forgather server actually sees).
    """
    import subprocess

    in_container_path = "/home/forgather/.config/forgather/tls/ca/ca.crt"
    if container:
        remote_cmd = (
            f"docker exec {container} sh -c "
            f"'sha256sum {in_container_path} 2>/dev/null' "
            "| awk '{print $1}'"
        )
    else:
        remote_cmd = (
            "sha256sum ~/.config/forgather/tls/ca/ca.crt 2>/dev/null "
            "| awk '{print $1}'"
        )
    probe = subprocess.run(
        [*ssh_base, target, remote_cmd],
        capture_output=True,
        text=True,
    )
    fp = probe.stdout.strip()
    if probe.returncode != 0 or not fp:
        return None
    if len(fp) != 64 or not all(c in "0123456789abcdef" for c in fp):
        return None
    return fp


def _deploy_to_peer(
    cfg, peer, *, ssh_target, ssh_user, ssh_base, scp_base, force, container=None
):
    """Mint a cert, ship it via scp, run `tls install` via ssh.

    Returns ``True`` on success or a short error string on failure.
    The local mint goes through the same code path as `tls mint`
    (placeholder SAN by default — peers validate by CA chain).
    """
    import subprocess
    import tempfile

    target = f"{ssh_user}@{ssh_target}"
    # Mint a fresh cert into a temp dir on this host.
    try:
        hostnames, ips = merge_san(
            [], [],
            extra_hostnames=["forgather-peer", "localhost"],
            extra_ips=["127.0.0.1", "::1"],
        )
        from forgather.tls.ca import mint_server_cert as _mint

        minted = _mint(cfg, hostnames=hostnames, ips=ips)
    except Exception as exc:
        return f"mint: {exc}"

    with tempfile.TemporaryDirectory(prefix="forgather-tls-deploy-") as local_tmp:
        local = Path(local_tmp)
        (local / "server.crt").write_bytes(minted.cert_pem)
        # Atomic 0600 creation, matching forgather.tls.ca._write_secret.
        fd = os.open(
            str(local / "server.key"),
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        try:
            os.write(fd, minted.key_pem)
        finally:
            os.close(fd)
        (local / "ca.crt").write_bytes(cfg.ca_cert.read_bytes())

        # CA-aware idempotency check.
        local_ca_fp = _ca_fingerprint(cfg.ca_cert)
        peer_ca_fp = _peer_ca_fingerprint(ssh_base, target, container=container)
        if peer_ca_fp is not None:
            if peer_ca_fp == local_ca_fp:
                return "already deployed (peer's CA matches this host's CA)"
            if not force:
                return (
                    "peer has TLS state from a DIFFERENT CA "
                    f"(local sha256={local_ca_fp[:12]}…, "
                    f"peer sha256={peer_ca_fp[:12]}…). "
                    "Pass --force to overwrite."
                )

        if container:
            # Container path: stage files INTO the running container
            # via a tar pipe through `docker exec -i`. No host-side
            # tempfile required; works regardless of whether the
            # container's state volume is a named volume, a host
            # bind-mount, or unset. The container user owns the
            # extracted files automatically.
            import tarfile, io

            remote_tmp = "/tmp/forgather-tls-deploy"
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                # ``filter`` resets owner/perms so the archive is
                # reproducible — extracted perms are set by --mode 600
                # on the keyfile via a tarinfo tweak below.
                for name in ("server.crt", "server.key", "ca.crt"):
                    info = tar.gettarinfo(str(local / name), arcname=name)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mode = 0o600 if name.endswith(".key") else 0o644
                    with open(local / name, "rb") as f:
                        tar.addfile(info, f)
            buf.seek(0)
            extract_cmd = (
                f"docker exec -i {container} sh -c "
                f"'rm -rf {remote_tmp} && mkdir -m 0700 -p {remote_tmp} "
                f"&& tar -C {remote_tmp} -xpf -'"
            )
            try:
                subprocess.run(
                    [*ssh_base, target, extract_cmd],
                    input=buf.getvalue(),
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as exc:
                err = exc.stderr.decode("utf-8", "replace").strip()
                return f"docker exec tar: {err or exc.returncode}"

            install_cmd = (
                f"docker exec {container} forgather tls install "
                f"--cert {remote_tmp}/server.crt "
                f"--key {remote_tmp}/server.key "
                f"--ca {remote_tmp}/ca.crt"
            )
            try:
                subprocess.run(
                    [*ssh_base, target, install_cmd],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as exc:
                err = exc.stderr.decode("utf-8", "replace").strip()
                return f"remote install (in container {container}): {err or exc.returncode}"
            # Clean up inside the container.
            subprocess.run(
                [*ssh_base, target,
                 f"docker exec {container} rm -rf {remote_tmp}"],
                capture_output=True,
            )
            return True

        # Default path: forgather is installed on the peer host (no container).
        try:
            mk = subprocess.run(
                [*ssh_base, target, "mktemp -d /tmp/forgather-tls-deploy.XXXXXX"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            return f"ssh mktemp: {exc.stderr.strip() or exc.returncode}"
        remote_tmp = mk.stdout.strip()

        try:
            try:
                subprocess.run(
                    [
                        *scp_base,
                        str(local / "server.crt"),
                        str(local / "server.key"),
                        str(local / "ca.crt"),
                        f"{target}:{remote_tmp}/",
                    ],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as exc:
                return f"scp: {exc.stderr.decode('utf-8', 'replace').strip() or exc.returncode}"

            install_cmd = (
                f"forgather tls install "
                f"--cert {remote_tmp}/server.crt "
                f"--key {remote_tmp}/server.key "
                f"--ca {remote_tmp}/ca.crt"
            )
            try:
                subprocess.run(
                    [*ssh_base, target, install_cmd],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as exc:
                err = exc.stderr.decode("utf-8", "replace").strip()
                return f"remote install: {err or exc.returncode}"
        finally:
            subprocess.run(
                [*ssh_base, target, f"rm -rf {remote_tmp}"],
                capture_output=True,
            )

    return True


# ---------------------------------------------------------------- trust-system


def _cmd_trust_system(args) -> int:
    """Brief reminder + commands for THIS host's OS.

    The authoritative procedure lives in ``docs/operations/tls.md``
    under "Trusting the CA from a browser". This subcommand only
    helps when the browser runs on the same machine as forgather
    — common during local development, rare on production. In the
    typical headless-server-plus-remote-laptop deployment, follow
    the doc's per-OS section *on the laptop*, not here.
    """
    cfg = load_config()
    if not cfg.ca_cert.is_file():
        print(f"No CA cert at {cfg.ca_cert}", file=sys.stderr)
        return 1

    print(
        "Primary documentation: docs/operations/tls.md → 'Trusting the CA "
        "from a browser'.\n"
        "It walks through copying the CA from the forgather server to a\n"
        "separate client (macOS / Windows / Linux laptop) over SSH, then\n"
        "the per-OS / per-browser import procedure. Read it there — this\n"
        "summary only covers the case where the browser runs on the same\n"
        "machine as forgather.\n"
    )

    print(f"CA cert on this host: {cfg.ca_cert}")
    print()

    sys_name = platform.system()
    if sys_name == "Linux":
        print("Same-machine install (Linux):")
        print(f"  # Debian/Ubuntu:")
        print(f"  sudo cp {cfg.ca_cert} /usr/local/share/ca-certificates/forgather-ca.crt")
        print(f"  sudo update-ca-certificates")
        print(f"  # Fedora/RHEL:")
        print(f"  sudo cp {cfg.ca_cert} /etc/pki/ca-trust/source/anchors/")
        print(f"  sudo update-ca-trust")
    elif sys_name == "Darwin":
        print("Same-machine install (macOS):")
        print(
            f"  sudo security add-trusted-cert -d -r trustRoot \\\n"
            f"      -k /Library/Keychains/System.keychain {cfg.ca_cert}"
        )
    elif sys_name == "Windows":
        print("Same-machine install (Windows, PowerShell as admin):")
        print(
            f"  Import-Certificate -FilePath {cfg.ca_cert} "
            "-CertStoreLocation Cert:\\LocalMachine\\Root"
        )
    else:
        print(f"Unknown OS ({sys_name}); see the docs for per-OS instructions.")

    print()
    print(
        "Browser running on a different machine? Export the CA and run\n"
        "the install on that machine instead:\n"
        "  forgather tls export-ca > forgather-ca.crt\n"
        "  # then follow docs/operations/tls.md on the client side"
    )
    return 0


# ----------------------------------------------------------- enable / disable


def _cmd_enable(args) -> int:
    cfg = load_config()
    if not cfg.is_provisioned():
        print(
            "No server cert/key on this host — nothing to enable. "
            "Run 'forgather tls init' first.",
            file=sys.stderr,
        )
        return 1
    cfg.enabled = True
    save_config(cfg)
    print(f"TLS enabled in {cfg.config_file}")
    print("Restart any running forgather/dataset/inference servers for the change to take effect.")
    return 0


def _cmd_disable(args) -> int:
    cfg = load_config()
    if cfg.config_path is None:
        print(
            f"No config at {cfg.config_file} — nothing to disable.",
            file=sys.stderr,
        )
        return 1
    cfg.enabled = False
    save_config(cfg)
    print(f"TLS disabled in {cfg.config_file}")
    print(
        "Cert files remain on disk. Re-enable with 'forgather tls enable'.\n"
        "Restart any running servers for the change to take effect.\n"
        "Note: non-loopback binds will be refused unless --insecure is passed."
    )
    return 0


# ---------------------------------------------------------------------- clean


def _cmd_clean(args) -> int:
    cfg = load_config()
    if not args.yes:
        print(
            f"Refusing to wipe {cfg.root}. Pass --yes to confirm.",
            file=sys.stderr,
        )
        return 1
    if cfg.root.exists():
        shutil.rmtree(cfg.root)
        print(f"Removed {cfg.root}")
    else:
        print(f"Nothing to remove at {cfg.root}")
    return 0
