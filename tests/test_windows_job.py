"""Windows job lifetime and candidate isolation checks."""
import ctypes
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from src import execute

if os.name == "nt":
    import _winapi
    from src import _windows_job


@unittest.skipUnless(os.name == "nt", "requires Windows job objects")
class WindowsJobTests(unittest.TestCase):
    def start_tree(self, tail, inherit=False):
        grandchild = ("from pathlib import Path; import os,time; "
                      "Path('grandchild').write_text(str(os.getpid())); time.sleep(4)")
        child = (
            "from pathlib import Path; import os,subprocess,sys,time; "
            f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); "
            "Path('child').write_text(str(os.getpid())); time.sleep(4)"
        )
        stdio = "" if inherit else ", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL"
        return (
            "from pathlib import Path; import subprocess,sys,time; "
            f"subprocess.Popen([sys.executable, '-c', {child!r}], stdin=subprocess.DEVNULL{stdio})\n"
            "while not all(Path(p).exists() and Path(p).stat().st_size for p in ('child', 'grandchild')):\n"
            "    time.sleep(.01)\n"
            "print('ready', flush=True)\n" + tail
        )

    def assert_tree_stopped(self, root):
        # The marker is written only after the second generation has launched.
        for name in ("child", "grandchild"):
            pid = int((Path(root) / name).read_text())
            try:
                handle = _winapi.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
            except OSError as error:
                self.assertEqual(error.winerror, 87)  # PID no longer exists.
                continue
            try:
                self.assertEqual(_winapi.WaitForSingleObject(handle, 0), _winapi.WAIT_OBJECT_0)
            finally:
                _winapi.CloseHandle(handle)

    def test_redirected_descendants_stop_after_both_exit_codes(self):
        for exit_code in (0, 7):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as root:
                args = [sys.executable, "-c", self.start_tree(f"raise SystemExit({exit_code})")]
                result = _windows_job.run(args, input="", timeout=2.5, cwd=root)
                self.assertEqual((result.returncode, result.stdout, result.stderr), (exit_code, "ready\n", ""))
                self.assert_tree_stopped(root)

    def test_timeout_closes_inherited_pipes_and_stops_descendants(self):
        with tempfile.TemporaryDirectory() as root:
            args = [sys.executable, "-c", self.start_tree("time.sleep(4)", inherit=True)]
            start = time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):
                _windows_job.run(args, input="", timeout=1, cwd=root)
            self.assertLess(time.monotonic() - start, 3)
            self.assert_tree_stopped(root)

    def test_blocked_stdin_still_times_out(self):
        with tempfile.TemporaryDirectory() as root:
            start = time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):
                _windows_job.run([sys.executable, "-c", "import time; time.sleep(4)"],
                                 input="x" * (1 << 20), timeout=.4, cwd=root)
            self.assertLess(time.monotonic() - start, 2)

    def test_failed_termination_releases_blocked_stdin_and_stops_tree(self):
        with tempfile.TemporaryDirectory() as root:
            args = [sys.executable, "-c", self.start_tree("time.sleep(4)", inherit=True)]
            start = time.monotonic()
            with patch.object(_windows_job, "_kill_job", return_value=0), self.assertRaises(OSError):
                _windows_job.run(args, input="x" * (1 << 20), timeout=1, cwd=root)
            self.assertLess(time.monotonic() - start, 3)
            self.assert_tree_stopped(root)

    def test_member_wait_reports_timeout_and_rejects_unexpected_status(self):
        for status, error_type, diagnostic in (
            (0x102, TimeoutError, r"PID 123.*WAIT_TIMEOUT"),
            (0x80, OSError, r"0x00000080.*PID 123"),
        ):
            close = Mock()
            kill = Mock(return_value=1)
            close_job = Mock()

            def wait_for_member(handle, remaining):
                kill.assert_called_once_with(10, 1)
                self.assertGreaterEqual(remaining, 0)
                self.assertLessEqual(remaining, 1000)
                close.assert_not_called()
                if handle == 20:
                    close_job.assert_not_called()
                    return status
                close_job.assert_called_once_with()
                return _winapi.WAIT_OBJECT_0

            wait = Mock(side_effect=wait_for_member)
            with (
                self.subTest(status=status),
                patch.multiple(_windows_job, _set_job_info=Mock(return_value=1),
                               _member_ids=Mock(return_value=[123, 456]),
                               _open_process=Mock(side_effect=[20, 21]), _kill_job=kill,
                               _wait=wait, _close=close, _query_job=Mock()),
            ):
                with self.assertRaisesRegex(error_type, diagnostic) as caught:
                    _windows_job._stop_job(10, close_job)
                self.assertIs(type(caught.exception), error_type)
                self.assertEqual([entry.args[0] for entry in wait.call_args_list], [20, 21])
                self.assertLessEqual(wait.call_args_list[1].args[1], wait.call_args_list[0].args[1])
                self.assertEqual(close.call_args_list, [call(21), call(20)])
                _windows_job._query_job.assert_not_called()

    def test_wait_failure_after_process_exit_releases_blocked_stdin_and_stops_tree(self):
        real_wait = _windows_job._wait
        real_popen = subprocess.Popen
        observed = []
        launched = []

        def capture_process(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            launched.append(process)
            return process

        def fail_after_exit(handle, remaining):
            result = real_wait(handle, remaining)
            observed.append(result)
            if len(observed) > 1 or result != _winapi.WAIT_OBJECT_0:
                return result
            ctypes.set_last_error(6)  # ERROR_INVALID_HANDLE
            return 0xffffffff  # WAIT_FAILED

        with tempfile.TemporaryDirectory() as root:
            args = [sys.executable, "-c", self.start_tree("time.sleep(4)", inherit=True)]
            start = time.monotonic()
            with (
                patch.object(_windows_job, "_wait", side_effect=fail_after_exit),
                patch("subprocess.Popen", side_effect=capture_process),
                self.assertRaises(OSError) as caught,
            ):
                _windows_job.run(args, input="x" * (1 << 20), timeout=1, cwd=root)
            self.assertLess(time.monotonic() - start, 3)
            self.assertGreaterEqual(len(observed), 2)
            self.assertEqual(set(observed), {_winapi.WAIT_OBJECT_0})
            self.assert_tree_stopped(root)
            self.assertEqual(len(launched), 1)
            process = launched[0]
            self.assertIsNotNone(process.returncode)
            self.assertTrue(all(pipe.closed for pipe in (process.stdin, process.stdout, process.stderr)))
            error = caught.exception
            self.assertNotIsInstance(error, TimeoutError)
            self.assertEqual(error.winerror, 6)
            self.assertRegex(str(error), r"WaitForSingleObject failed for candidate job member PID \d+")

    def test_failed_cleanup_rejects_successful_output(self):
        with patch.object(_windows_job, "_kill_job", return_value=0):
            self.assertEqual(execute.verify(["print('ok')"], ("", "ok"), workers=1), [False])

    def test_text_streams_environment_and_working_directory(self):
        code = ("from pathlib import Path; import os,sys; "
                "Path('written').write_text(input()); "
                "print(os.environ['CTRLFIND_STDIO_FIXTURE']); print('note', file=sys.stderr)")
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, CTRLFIND_STDIO_FIXTURE="inherited"):
            result = _windows_job.run([sys.executable, "-c", code], input="example\n", timeout=1, cwd=root)
            self.assertEqual((result.returncode, result.stdout, result.stderr), (0, "inherited\n", "note\n"))
            self.assertEqual((Path(root) / "written").read_text(), "example")

    def test_exception_and_cancellation_stop_the_tree(self):
        for error in (OSError("capture failed"), KeyboardInterrupt(),):
            with self.subTest(error=type(error)), tempfile.TemporaryDirectory() as root:
                def interrupt(proc, **kwargs):
                    self.assertEqual(proc.stdout.readline(), "ready\n")
                    raise error

                args = [sys.executable, "-c", self.start_tree("time.sleep(4)", inherit=True)]
                with (
                    patch("subprocess.Popen.communicate", autospec=True, side_effect=interrupt),
                    self.assertRaises(type(error)),
                ):
                    _windows_job.run(args, input="", timeout=2.5, cwd=root)
                self.assert_tree_stopped(root)

    def test_assignment_failure_never_runs_the_candidate(self):
        with tempfile.TemporaryDirectory() as root:
            args = [sys.executable, "-c", "from pathlib import Path; Path('unsafe').touch()"]
            with patch.object(_windows_job, "_assign", return_value=0), self.assertRaises(OSError):
                _windows_job.run(args, input="", timeout=1, cwd=root)
            self.assertFalse((Path(root) / "unsafe").exists())

    def test_job_creation_or_configuration_failure_prevents_launch(self):
        for operation in ("_create_job", "_set_job_info"):
            with (
                self.subTest(operation=operation),
                patch.object(_windows_job, operation, return_value=0),
                patch("subprocess.Popen") as launch,
            ):
                self.assertIsNone(execute.run_snippet("print('ok')", ""))
                launch.assert_not_called()

    def test_breakaway_request_is_denied(self):
        source = """import subprocess, sys
try:
    child = subprocess.Popen([sys.executable, '-c', 'pass'], creationflags=0x01000000)
except OSError:
    print('denied')
else:
    child.wait()
    print('escaped')
"""
        # A venv may create an inner job. Exercise the request from the job root.
        with tempfile.TemporaryDirectory() as root:
            args = [sys._base_executable, "-c", source]
            result = _windows_job.run(args, input="", timeout=1, cwd=root)
            self.assertEqual((result.returncode, result.stdout), (0, "denied\n"))


class CandidateIsolationTests(unittest.TestCase):
    def test_non_windows_dispatch_preserves_stock_subprocess_options(self):
        with patch.object(execute, "os", SimpleNamespace(name="posix")), patch("subprocess.run") as run:
            result = execute._run_candidate(["python", "solution.py"], input="case\n", timeout=1, cwd="temp")
            self.assertIs(result, run.return_value)
            run.assert_called_once_with(["python", "solution.py"], input="case\n", timeout=1, cwd="temp",
                                        capture_output=True, text=True)

    def test_sequential_candidates_do_not_share_state(self):
        mutates = """import builtins
import os
from pathlib import Path
builtins._ctrlfind_test_state = True
os.environ['CTRLFIND_LOCAL_TEST_STATE'] = 'mutated'
Path('probe').mkdir()
os.chdir('probe')
Path('state').touch()
print('ok')
"""
        checks = """import builtins
import os
from pathlib import Path
assert not hasattr(builtins, '_ctrlfind_test_state')
assert 'CTRLFIND_LOCAL_TEST_STATE' not in os.environ
assert not Path('probe').exists()
print('ok')
"""
        self.assertEqual(execute.verify([mutates, checks], ("", "ok"), workers=1), [True, True])

    def test_cancelled_batch_shuts_down_pending_futures(self):
        for platform in ("nt", "posix"):
            with (
                self.subTest(platform=platform),
                patch.object(execute, "os", SimpleNamespace(name=platform)),
                patch("src.execute.ThreadPoolExecutor") as executor,
            ):
                pool = executor.return_value.__enter__.return_value
                pool.map.side_effect = KeyboardInterrupt
                with self.assertRaises(KeyboardInterrupt):
                    execute.verify(["pass"], ("", ""), workers=1)
                if platform == "nt":
                    pool.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
                else:
                    pool.shutdown.assert_not_called()
