"""Argument parser for the `forgather tls` command."""

import argparse
from argparse import RawTextHelpFormatter

from forgather.tls.config import tls_dir


def create_tls_parser(global_args):
    # Resolve the actual TLS root on this host (platformdirs gives us
    # the correct location on macOS/Windows too — not just Linux's
    # ~/.config). Show it in help text so the operator sees the
    # *actual* path they'll be working with.
    _tls_root = str(tls_dir())
    parser = argparse.ArgumentParser(
        prog="forgather tls",
        description=(
            "Manage shared TLS state for forgather servers.\n"
            "\n"
            f"Creates and maintains a local CA + server certificate under\n"
            f"{_tls_root}/ that all three servers\n"
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
            f"Creates {_tls_root}/ca/ca.{{crt,key}} if absent,\n"
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
        "--ca",
        action="store_true",
        help="Also re-issue the CA. DESTRUCTIVE: peer trust breaks until "
        "the new CA is redistributed to every peer.",
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

    deploy = sub.add_parser(
        "deploy",
        help="Mint + scp + install certs on cluster peers via ssh",
        formatter_class=RawTextHelpFormatter,
        description=(
            "Automate the per-peer TLS bring-up. For every cluster peer\n"
            "(discovered via the local forgather server's membership\n"
            "table), mints a cert here, scps it over, and runs\n"
            "`forgather tls install` on the peer via ssh. Requires:\n"
            "\n"
            "  * This host holds the CA (`forgather tls init` already run).\n"
            "  * The local forgather server is running with `--cluster <name>`\n"
            "    so it knows the membership.\n"
            "  * ssh access to every peer. ssh prompts for passwords as\n"
            "    normal — set up ssh-agent or keys if you don't want to\n"
            "    type per-peer (or pass --batch to refuse password\n"
            "    prompts and fail fast on missing keys).\n"
            "  * `forgather` reachable on the remote side: either on the\n"
            "    ssh user's PATH directly, or inside a Docker container\n"
            "    (use --container <name> — the install runs as\n"
            "    `docker exec <name> forgather tls install …` and files\n"
            "    are streamed in via a tar pipe).\n"
            "\n"
            "Idempotent: peers that already have a server cert installed\n"
            "are skipped unless --force is given. The local mint is\n"
            "always fresh — no client state is reused between runs."
        ),
    )
    deploy.add_argument(
        "nodes",
        nargs="*",
        metavar="NODE",
        help=(
            "Hostname or IP of one or more specific peers to deploy to. "
            "If omitted, deploys to every reachable peer in the membership table."
        ),
    )
    deploy.add_argument(
        "--ssh-user",
        default=None,
        metavar="USER",
        help="SSH user on each peer (default: $USER).",
    )
    deploy.add_argument(
        "--ssh-host",
        action="append",
        default=[],
        metavar="PEER=HOST",
        help=(
            "Override the ssh target for a specific peer (repeatable). "
            "Use when the peer's cluster address isn't directly ssh-reachable "
            "by the same name — e.g. `--ssh-host node-b=10.0.0.6` or "
            "`--ssh-host node-b=bastion.lan` if you tunnel through a bastion."
        ),
    )
    deploy.add_argument(
        "--container",
        default=None,
        metavar="NAME",
        help=(
            "Peer is running forgather inside a Docker container; wrap remote "
            "commands in `docker exec <NAME>`. File transfer uses a tar pipe "
            "through `docker exec -i` so the host doesn't need to know where "
            "the container's state volume lives. Same name applies to every "
            "peer in this run — use --container-host PEER=NAME if peers use "
            "different container names."
        ),
    )
    deploy.add_argument(
        "--container-host",
        action="append",
        default=[],
        metavar="PEER=NAME",
        help=(
            "Per-peer override of --container (repeatable). Use when "
            "different peers run different container names."
        ),
    )
    deploy.add_argument(
        "--server",
        default=None,
        metavar="URL",
        help="Local forgather server URL (default: $FORGATHER_SERVER_URL).",
    )
    deploy.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing TLS state on the peer. Off by default.",
    )
    deploy.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Pass BatchMode=yes to ssh/scp: refuses to prompt for "
            "passwords, fails fast if keys aren't set up. Default off "
            "(let ssh prompt as usual)."
        ),
    )
    deploy.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen but don't mint, scp, or install.",
    )

    sub.add_parser(
        "enable",
        help="Set 'enabled: true' in shared config (no cert changes)",
        formatter_class=RawTextHelpFormatter,
        description=(
            "Re-enable TLS for all forgather servers on this host without\n"
            "re-issuing any certs. Existing cert/key files stay in place;\n"
            "this just flips the master switch in config.yaml."
        ),
    )
    sub.add_parser(
        "disable",
        help="Set 'enabled: false' in shared config (no cert changes)",
        formatter_class=RawTextHelpFormatter,
        description=(
            "Disable TLS for all forgather servers on this host without\n"
            "deleting any certs. Useful when troubleshooting connectivity\n"
            "or temporarily reverting. Re-enable with 'forgather tls enable'."
        ),
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
