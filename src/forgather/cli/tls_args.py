"""Argument parser for the `forgather tls` command."""

import argparse
from argparse import RawTextHelpFormatter


def create_tls_parser(global_args):
    parser = argparse.ArgumentParser(
        prog="forgather tls",
        description=(
            "Manage shared TLS state for forgather servers.\n"
            "\n"
            "Creates and maintains a local CA + server certificate under\n"
            "~/.config/forgather/tls/ that all three servers\n"
            "(forgather server / dataset_server / inference_server) share."
        ),
        formatter_class=RawTextHelpFormatter,
        epilog=(
            "Typical workflow:\n"
            "\n"
            "  Single host:\n"
            "    forgather tls init                   # one-shot setup\n"
            "    forgather server -H 0.0.0.0          # picks up TLS automatically\n"
            "\n"
            "  Two-node cluster (host A mints, host B serves):\n"
            "    A$ forgather tls init --hostname a.lan --hostname b.lan \\\n"
            "                          --ip 10.0.0.5 --ip 10.0.0.6\n"
            "    A$ forgather tls mint --hostname b.lan --ip 10.0.0.6 -o /tmp/b-tls\n"
            "    A$ forgather tls export-ca -o /tmp/forgather-ca.crt\n"
            "    # scp /tmp/b-tls/* /tmp/forgather-ca.crt to B\n"
            "    B$ forgather tls install --cert /tmp/b-tls/server.crt \\\n"
            "                             --key /tmp/b-tls/server.key \\\n"
            "                             --ca /tmp/forgather-ca.crt\n"
        ),
    )
    sub = parser.add_subparsers(dest="tls_subcommand", help="TLS subcommands")

    init = sub.add_parser(
        "init",
        help="Create CA + server cert for this host (idempotent)",
        formatter_class=RawTextHelpFormatter,
        description=(
            "Provision the shared TLS state.\n"
            "\n"
            "Creates ~/.config/forgather/tls/ca/ca.{crt,key} if absent,\n"
            "mints server.{crt,key} covering auto-detected hostnames\n"
            "and IPs (plus any added via --hostname/--ip), writes\n"
            "config.yaml with enabled: true, and rebuilds ca-bundle.crt."
        ),
    )
    init.add_argument(
        "--hostname",
        action="append",
        default=[],
        metavar="NAME",
        help="Extra SAN hostname (repeatable). Auto-detected names are always added.",
    )
    init.add_argument(
        "--ip",
        action="append",
        default=[],
        metavar="IP",
        help="Extra SAN IP (repeatable). Auto-detected IPs are always added.",
    )
    init.add_argument(
        "--ca-name",
        default=None,
        metavar="CN",
        help="CA common name (default: 'Forgather CA <hostname>').",
    )
    init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing CA. DESTRUCTIVE: existing peer trust breaks.",
    )

    status = sub.add_parser(
        "status",
        help="Show CA + server cert info and effective config",
        formatter_class=RawTextHelpFormatter,
    )
    status.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )

    renew = sub.add_parser(
        "renew",
        help="Re-issue server cert (and optionally the CA) from existing CA",
        formatter_class=RawTextHelpFormatter,
    )
    renew.add_argument(
        "--server", action="store_true", help="Renew the server cert (default)"
    )
    renew.add_argument(
        "--ca",
        action="store_true",
        help="Also re-issue the CA. DESTRUCTIVE: peer trust breaks.",
    )
    renew.add_argument(
        "--add-hostname",
        action="append",
        default=[],
        metavar="NAME",
        help="Add a hostname to the SAN before renewal (persisted).",
    )
    renew.add_argument(
        "--add-ip",
        action="append",
        default=[],
        metavar="IP",
        help="Add an IP to the SAN before renewal (persisted).",
    )

    export = sub.add_parser(
        "export-ca",
        help="Emit this host's CA cert (no key) for distribution",
        formatter_class=RawTextHelpFormatter,
    )
    export.add_argument(
        "-o", "--output", default="-", metavar="PATH", help="Output file (default: stdout)"
    )

    imp = sub.add_parser(
        "import-ca",
        help="Trust a CA cert from another host (rebuilds the bundle)",
        formatter_class=RawTextHelpFormatter,
    )
    imp.add_argument("path", metavar="PATH", help="CA cert file (PEM)")
    imp.add_argument(
        "--name", default=None, help="Label under trusted/<name>.crt (default: subject)"
    )

    mint = sub.add_parser(
        "mint",
        help="Issue a server cert for another host using the local CA",
        formatter_class=RawTextHelpFormatter,
        description=(
            "Mint server cert+key for a different host (typical use: a\n"
            "cluster head issues certs for every peer). Writes\n"
            "<output>/server.crt and <output>/server.key.\n"
        ),
    )
    mint.add_argument(
        "--hostname",
        action="append",
        default=[],
        metavar="NAME",
        help="SAN hostname for the issued cert (repeatable; at least one required)",
    )
    mint.add_argument(
        "--ip",
        action="append",
        default=[],
        metavar="IP",
        help="SAN IP for the issued cert (repeatable)",
    )
    mint.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="DIR",
        help="Directory to write server.crt + server.key into",
    )

    install = sub.add_parser(
        "install",
        help="Install a foreign-minted cert/key + import a CA on this host",
        formatter_class=RawTextHelpFormatter,
        description=(
            "Receiver side of `forgather tls mint`. Copies the supplied\n"
            "cert/key into the shared TLS dir, imports the CA, rebuilds\n"
            "the bundle, and turns on `enabled: true`.\n"
        ),
    )
    install.add_argument(
        "--cert", required=True, metavar="PATH", help="Server certificate (PEM)"
    )
    install.add_argument(
        "--key", required=True, metavar="PATH", help="Server private key (PEM)"
    )
    install.add_argument(
        "--ca",
        default=None,
        metavar="PATH",
        help="CA cert to trust (PEM). Recommended unless already imported.",
    )

    trust = sub.add_parser(
        "trust-system",
        help="Print instructions for adding the CA to OS / browser trust stores",
        formatter_class=RawTextHelpFormatter,
    )

    clean = sub.add_parser(
        "clean",
        help="Remove all TLS state (CA + server cert + bundle + config)",
        formatter_class=RawTextHelpFormatter,
    )
    clean.add_argument(
        "--yes", action="store_true", help="Required: confirm wipe"
    )

    return parser
