def control_cmd(args):
    from forgather.trainer_control import get_default_client

    """Handle trainer control commands."""
    try:
        client = get_default_client()

        if args.control_subcommand == "list":
            jobs = client.list_jobs()

            if hasattr(args, "remote") and args.remote:
                # Parse remote host:port
                try:
                    host, port = args.remote.split(":")
                    port = int(port)
                    if hasattr(client, "list_jobs_remote"):
                        remote_jobs = client.list_jobs_remote(host, port)
                        jobs.extend(remote_jobs)
                    else:
                        print(
                            f"Warning: Remote job listing not supported by current client"
                        )
                except ValueError:
                    print(
                        f"Error: Invalid remote format '{args.remote}'. Use HOST:PORT format."
                    )
                    return 1

            if not jobs:
                print("No discoverable training jobs found.")
                return 0

            # Check which jobs are still alive and mark dead ones
            import psutil

            alive_jobs = []
            dead_jobs = []

            for job in jobs:
                try:
                    # Check if process is still running
                    if psutil.pid_exists(job.pid):
                        proc = psutil.Process(job.pid)
                        if proc.is_running():
                            alive_jobs.append((job, "✓"))
                        else:
                            dead_jobs.append((job, "✗"))
                    else:
                        dead_jobs.append((job, "✗"))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    dead_jobs.append((job, "✗"))

            # Display results
            all_jobs = alive_jobs + dead_jobs

            print("Discoverable training jobs:")
            print(
                f"{'Status':<3} {'Job ID':<30} {'Host':<15} {'Port':<6} {'PID':<8} {'Started'}"
            )
            print("-" * 75)
            for job, status in all_jobs:
                import datetime

                started = datetime.datetime.fromtimestamp(job.started_at).strftime(
                    "%m/%d %H:%M"
                )
                print(
                    f"{status:<3} {job.job_id:<30} {job.host:<15} {job.port:<6} {job.pid:<8} {started}"
                )

            if dead_jobs:
                print(f"\n✗ = Process not running ({len(dead_jobs)} dead job(s) found)")
                print("Tip: Use 'forgather control cleanup' to remove dead job files")

        elif args.control_subcommand == "status":
            status = client.get_status(args.job_id)
            print("Job Status:")
            for key, value in status.items():
                if key == "timestamp":
                    import datetime

                    value = datetime.datetime.fromtimestamp(value).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                print(f"  {key}: {value}")

        elif args.control_subcommand == "stop":
            response = client.graceful_stop(args.job_id)
            if response.success:
                print(f"✓ {response.message}")
            else:
                print(f"✗ {response.message}")
                return 1

        elif args.control_subcommand == "save":
            response = client.save_checkpoint(args.job_id)
            if response.success:
                print(f"✓ {response.message}")
            else:
                print(f"✗ {response.message}")
                return 1

        elif args.control_subcommand == "save-stop":
            response = client.save_and_stop(args.job_id)
            if response.success:
                print(f"✓ {response.message}")
            else:
                print(f"✗ {response.message}")
                return 1

        elif args.control_subcommand == "cleanup":
            jobs = client.list_jobs()
            if not jobs:
                print("No job files found.")
                return 0

            # Find dead jobs
            import psutil
            import shutil
            from pathlib import Path

            dead_jobs = []

            for job in jobs:
                try:
                    if not psutil.pid_exists(job.pid):
                        dead_jobs.append(job)
                    else:
                        proc = psutil.Process(job.pid)
                        if not proc.is_running():
                            dead_jobs.append(job)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    dead_jobs.append(job)

            if not dead_jobs:
                print("No dead job files found.")
                return 0

            print(f"Found {len(dead_jobs)} dead job(s):")
            for job in dead_jobs:
                import datetime

                started = datetime.datetime.fromtimestamp(job.started_at).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                print(f"  {job.job_id} (PID {job.pid}, started {started})")

            if not args.force:
                response = input(f"\nRemove {len(dead_jobs)} dead job file(s)? [y/N]: ")
                if response.lower() not in ["y", "yes"]:
                    print("Cleanup cancelled.")
                    return 0

            # Remove dead job directories
            removed_count = 0
            jobs_dir = Path.home() / ".forgather" / "jobs"

            for job in dead_jobs:
                job_dir = jobs_dir / job.job_id
                try:
                    if job_dir.exists():
                        shutil.rmtree(job_dir)
                        removed_count += 1
                        print(f"✓ Removed {job.job_id}")
                except Exception as e:
                    print(f"✗ Failed to remove {job.job_id}: {e}")

            print(f"\nCleanup complete: {removed_count} job file(s) removed.")

        elif args.control_subcommand == "abort":
            # Show warning and ask for confirmation unless --force is used
            if not hasattr(args, "force") or not args.force:
                print(
                    "⚠️  WARNING: Abort will stop training immediately WITHOUT saving!"
                )
                print(
                    "   This action cannot be undone and will lose all unsaved progress."
                )
                response = input(f"\nAbort training job '{args.job_id}'? [y/N]: ")
                if response.lower() not in ["y", "yes"]:
                    print("Abort cancelled.")
                    return 0

            response = client.abort(args.job_id)
            if response.success:
                print(f"✓ {response.message}")
            else:
                print(f"✗ {response.message}")
                return 1

        else:
            print(f"Unknown control subcommand: {args.control_subcommand}")
            return 1

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0
