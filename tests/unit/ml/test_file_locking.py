#!/usr/bin/env python3
"""
Unit tests for file-locking functionality in forgather.ml.construct.

Tests the new file-locking approach that replaces torch.distributed barriers
for synchronizing object construction across multiple processes.
"""

import unittest
import tempfile
import os
import time
import subprocess
import sys
import threading
from unittest.mock import patch, MagicMock
from pathlib import Path

from forgather.ml.construct import (
    file_lock_build,
    build_rule,
    copy_package_files,
    write_file,
)


class TestFileLockBuild(unittest.TestCase):
    """Test the file_lock_build context manager."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.target = os.path.join(self.tmpdir, "test_target")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_file_lock_build_new_target(self):
        """Test file_lock_build when target doesn't exist."""
        with file_lock_build(self.target) as should_build:
            self.assertTrue(should_build)
            # Create the target to simulate construction
            with open(self.target, "w") as f:
                f.write("test content")

    def test_file_lock_build_existing_target(self):
        """Test file_lock_build when target already exists."""
        # Create target first
        with open(self.target, "w") as f:
            f.write("existing content")

        with file_lock_build(self.target) as should_build:
            self.assertFalse(should_build)

    def test_file_lock_build_concurrent_access(self):
        """Test file_lock_build with concurrent access via threading."""
        results = []
        construction_count = 0

        def worker(worker_id):
            nonlocal construction_count
            with file_lock_build(self.target) as should_build:
                if should_build:
                    construction_count += 1
                    time.sleep(0.1)  # Simulate work
                    with open(self.target, "w") as f:
                        f.write(f"Created by worker {worker_id}")
                results.append(worker_id)

        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Only one thread should have done construction
        self.assertEqual(construction_count, 1)
        self.assertEqual(len(results), 5)  # All threads completed
        self.assertTrue(os.path.exists(self.target))

    def test_file_lock_build_timeout(self):
        """Test file_lock_build timeout behavior."""
        # This test is challenging without actual file system contention
        # We'll test that the timeout parameter is accepted
        with file_lock_build(self.target, timeout=1.0) as should_build:
            self.assertTrue(should_build)


class TestBuildRule(unittest.TestCase):
    """Test the updated build_rule function."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.target = os.path.join(self.tmpdir, "build_target")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_build_rule_new_target(self):
        """Test build_rule when target needs to be built."""
        recipe_called = False

        def recipe():
            nonlocal recipe_called
            recipe_called = True
            with open(self.target, "w") as f:
                f.write("built content")

        def loader():
            with open(self.target, "r") as f:
                return f.read()

        result = build_rule(self.target, recipe, loader)

        self.assertTrue(recipe_called)
        self.assertEqual(result, "built content")
        self.assertTrue(os.path.exists(self.target))

    def test_build_rule_existing_target(self):
        """Test build_rule when target already exists."""
        # Create target first
        with open(self.target, "w") as f:
            f.write("existing content")

        recipe_called = False

        def recipe():
            nonlocal recipe_called
            recipe_called = True

        def loader():
            with open(self.target, "r") as f:
                return f.read()

        result = build_rule(self.target, recipe, loader)

        self.assertFalse(recipe_called)
        self.assertEqual(result, "existing content")

    def test_build_rule_with_prerequisites(self):
        """Test build_rule with prerequisite dependency checking."""
        # Create target first
        with open(self.target, "w") as f:
            f.write("old content")

        # Wait to ensure different timestamps
        time.sleep(0.1)

        # Create a prerequisite file that's newer than target
        prereq = os.path.join(self.tmpdir, "prereq")
        with open(prereq, "w") as f:
            f.write("prerequisite")

        # Verify the timestamps
        target_mtime = os.path.getmtime(self.target)
        prereq_mtime = os.path.getmtime(prereq)
        self.assertGreater(
            prereq_mtime, target_mtime, "Prerequisite should be newer than target"
        )

        recipe_called = False

        def recipe():
            nonlocal recipe_called
            recipe_called = True
            with open(self.target, "w") as f:
                f.write("rebuilt content")

        def loader():
            with open(self.target, "r") as f:
                return f.read()

        result = build_rule(self.target, recipe, loader, prerequisites=[prereq])

        self.assertTrue(
            recipe_called, "Recipe should be called when prerequisite is newer"
        )
        self.assertEqual(result, "rebuilt content")

    def test_build_rule_multiprocess(self):
        """Test build_rule with multiple processes."""
        # Create a test script for subprocess execution
        test_script = os.path.join(self.tmpdir, "test_worker.py")
        with open(test_script, "w") as f:
            f.write(
                f"""
import sys
import os
sys.path.insert(0, "{os.path.dirname(os.path.abspath(__file__))}/../../../src")

from forgather.ml.construct import build_rule

target = "{self.target}"
pid = os.getpid()

def recipe():
    import time
    time.sleep(0.1)  # Simulate work
    with open(target, 'w') as f:
        f.write(f"Created by process {{pid}}")

def loader():
    with open(target, 'r') as f:
        return f.read()

result = build_rule(target, recipe, loader)
print(f"{{pid}}:{{result}}")
"""
            )

        # Run multiple processes
        processes = []
        for i in range(3):
            p = subprocess.Popen(
                [sys.executable, test_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            processes.append(p)

        # Collect results
        outputs = []
        for p in processes:
            stdout, stderr = p.communicate()
            if stderr:
                self.fail(f"Process failed with error: {stderr}")
            outputs.append(stdout.strip())

        # All processes should have the same result (from the one that built it)
        # Extract just the result part (after the colon) to compare
        results = [output.split(":", 1)[1] for output in outputs if ":" in output]
        self.assertEqual(len(set(results)), 1, f"Different results: {results}")
        self.assertTrue(os.path.exists(self.target))

        # The result should be from one of the processes (whichever got the lock first)
        with open(self.target, "r") as f:
            content = f.read()
        self.assertTrue(
            content.startswith("Created by process"), f"Unexpected content: {content}"
        )


class TestCopyPackageFiles(unittest.TestCase):
    """Test the updated copy_package_files function."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_copy_package_files_basic(self):
        """Test basic copy_package_files functionality."""

        # Create a simple class instance with a proper module
        class TestClass:
            pass

        obj = TestClass()

        result = copy_package_files(self.tmpdir, obj)

        # Should return the same object
        self.assertIs(result, obj)

        # Should create marker file
        marker = os.path.join(self.tmpdir, ".package_files_copied")
        self.assertTrue(os.path.exists(marker))


class TestWriteFile(unittest.TestCase):
    """Test the updated write_file function."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_write_file_basic(self):
        """Test basic write_file functionality."""
        output_file = os.path.join(self.tmpdir, "test_output.txt")
        test_data = "Hello, World!"

        result = write_file(test_data, output_file=output_file)

        self.assertEqual(result, test_data)
        self.assertTrue(os.path.exists(output_file))

        with open(output_file, "r") as f:
            content = f.read()
        self.assertEqual(content, test_data)

    def test_write_file_with_callable_data(self):
        """Test write_file with callable data."""
        output_file = os.path.join(self.tmpdir, "test_output.txt")

        def data_generator():
            return "Generated content"

        result = write_file(data_generator, output_file=output_file)

        self.assertEqual(result, "Generated content")
        self.assertTrue(os.path.exists(output_file))

    def test_write_file_return_value_override(self):
        """Test write_file with return_value override."""
        output_file = os.path.join(self.tmpdir, "test_output.txt")
        test_data = "Hello, World!"
        override_value = "Override"

        result = write_file(
            test_data, output_file=output_file, return_value=override_value
        )

        self.assertEqual(result, override_value)
        self.assertTrue(os.path.exists(output_file))

        with open(output_file, "r") as f:
            content = f.read()
        self.assertEqual(content, test_data)

    def test_write_file_no_output_file(self):
        """Test write_file when no output_file is specified."""
        test_data = "Hello, World!"

        result = write_file(test_data)

        self.assertEqual(result, test_data)


class TestFileLockingIntegration(unittest.TestCase):
    """Integration tests for file-locking across multiple functions."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_concurrent_build_and_write(self):
        """Test concurrent build_rule and write_file operations."""
        target = os.path.join(self.tmpdir, "concurrent_target")
        write_target = os.path.join(self.tmpdir, "write_target.txt")

        def build_worker():
            def recipe():
                time.sleep(0.1)
                with open(target, "w") as f:
                    f.write("build result")

            def loader():
                with open(target, "r") as f:
                    return f.read()

            return build_rule(target, recipe, loader)

        def write_worker():
            return write_file("write result", output_file=write_target)

        # Run both operations concurrently
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            build_future = executor.submit(build_worker)
            write_future = executor.submit(write_worker)

            build_result = build_future.result()
            write_result = write_future.result()

        self.assertEqual(build_result, "build result")
        self.assertEqual(write_result, "write result")
        self.assertTrue(os.path.exists(target))
        self.assertTrue(os.path.exists(write_target))


if __name__ == "__main__":
    unittest.main()
