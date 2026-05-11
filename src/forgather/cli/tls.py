"""`forgather tls` — manage shared TLS state."""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import sys
from pathlib import Path

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
        "trust-system": _cmd_trust_system,
        "clean": _cmd_clean,
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
        "TLS is now enabled. Servers binding non-loopback hosts will use "
        "HTTPS automatically;\n"
        "loopback binds default to HTTP unless you pass --tls. "
        "Distribute the CA cert to peers:\n"
        f"  forgather tls export-ca -o /tmp/forgather-ca.crt"
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
    if not args.hostname and not args.ip:
        print("--hostname or --ip is required", file=sys.stderr)
        return 2

    hostnames, ips = merge_san([], [], extra_hostnames=args.hostname, extra_ips=args.ip)
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
    key_path.write_bytes(minted.key_pem)
    try:
        os.chmod(cert_path, 0o644)
        os.chmod(key_path, 0o600)
    except OSError:
        pass

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
        print(f"No such file: {cert_src}", file=sys.stderr)
        return 1
    if not key_src.is_file():
        print(f"No such file: {key_src}", file=sys.stderr)
        return 1

    cfg.root.mkdir(parents=True, exist_ok=True)
    cfg.server_cert.parent.mkdir(parents=True, exist_ok=True)
    cfg.server_cert.write_bytes(cert_src.read_bytes())
    cfg.server_key.write_bytes(key_src.read_bytes())
    try:
        os.chmod(cfg.server_cert, 0o644)
        os.chmod(cfg.server_key, 0o600)
    except OSError:
        pass
    print(f"Installed cert: {cfg.server_cert}")
    print(f"Installed key : {cfg.server_key}")

    if args.ca:
        ca_src = Path(os.path.expanduser(args.ca))
        if not ca_src.is_file():
            print(f"No such file: {ca_src}", file=sys.stderr)
            return 1
        cfg.ca_cert.parent.mkdir(parents=True, exist_ok=True)
        cfg.ca_cert.write_bytes(ca_src.read_bytes())
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


# ---------------------------------------------------------------- trust-system


def _cmd_trust_system(args) -> int:
    cfg = load_config()
    if not cfg.ca_cert.is_file():
        print(f"No CA cert at {cfg.ca_cert}", file=sys.stderr)
        return 1
    sys_name = platform.system()
    print("Add the following CA cert to your OS / browser trust store so")
    print("forgather URLs validate without warnings:")
    print()
    print(f"  CA cert: {cfg.ca_cert}")
    print()
    if sys_name == "Linux":
        print("Linux (Debian/Ubuntu):")
        print(f"  sudo cp {cfg.ca_cert} /usr/local/share/ca-certificates/forgather-ca.crt")
        print("  sudo update-ca-certificates")
        print()
        print("Linux (Fedora/RHEL):")
        print(f"  sudo cp {cfg.ca_cert} /etc/pki/ca-trust/source/anchors/")
        print("  sudo update-ca-trust")
    elif sys_name == "Darwin":
        print("macOS:")
        print(
            "  sudo security add-trusted-cert -d -r trustRoot \\"
        )
        print(f"      -k /Library/Keychains/System.keychain {cfg.ca_cert}")
    elif sys_name == "Windows":
        print("Windows (PowerShell, admin):")
        print(
            f"  Import-Certificate -FilePath {cfg.ca_cert} "
            "-CertStoreLocation Cert:\\LocalMachine\\Root"
        )
    print()
    print("Firefox uses its own trust store: about:preferences#privacy → View Certificates → Import.")
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
