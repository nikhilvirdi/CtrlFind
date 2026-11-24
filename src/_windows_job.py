"""Run Windows candidates in a kill-on-close job before their first instruction."""
from __future__ import annotations

import ctypes
from contextlib import ExitStack
from ctypes import wintypes as win
import subprocess
import threading
import time


_kernel = ctypes.WinDLL("kernel32", use_last_error=True)
_SIZE = ctypes.c_size_t
_CREATE_SUSPENDED = 0x00000004
_KILL_ON_JOB_CLOSE = 0x00002000
_LIMIT_ACTIVE_PROCESSES = 0x00000008
_PROCESS_JOB_ACCESS = 0x0101  # PROCESS_SET_QUOTA | PROCESS_TERMINATE
_SYNCHRONIZE = 0x00100000
_SNAP_THREADS = 0x00000004
_THREAD_SUSPEND_RESUME = 0x0002
_INVALID_HANDLE = ctypes.c_void_p(-1).value
_DWORD_ERROR = 0xffffffff
_ERROR_INVALID_PARAMETER = 87
_ERROR_MORE_DATA = 234


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("per_process_time", ctypes.c_int64), ("per_job_time", ctypes.c_int64),
        ("flags", win.DWORD), ("min_working_set", _SIZE), ("max_working_set", _SIZE),
        ("active_process_limit", win.DWORD), ("affinity", _SIZE),
        ("priority_class", win.DWORD), ("scheduling_class", win.DWORD),
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("basic", _BasicLimits), ("io", ctypes.c_uint64 * 6),
        ("process_memory_limit", _SIZE), ("job_memory_limit", _SIZE),
        ("peak_process_memory", _SIZE), ("peak_job_memory", _SIZE),
    ]


class _JobAccounting(ctypes.Structure):
    _fields_ = [
        ("times", ctypes.c_int64 * 4), ("page_faults", win.DWORD),
        ("total_processes", win.DWORD), ("active_processes", win.DWORD),
        ("terminated_processes", win.DWORD),
    ]


class _ThreadEntry(ctypes.Structure):
    _fields_ = [
        ("size", win.DWORD), ("usage", win.DWORD), ("thread_id", win.DWORD),
        ("owner_pid", win.DWORD), ("base_priority", win.LONG),
        ("delta_priority", win.LONG), ("flags", win.DWORD),
    ]


def _api(name, result, *arguments):
    function = getattr(_kernel, name)
    function.restype = result
    function.argtypes = arguments
    return function


_create_job = _api("CreateJobObjectW", win.HANDLE, win.LPVOID, win.LPCWSTR)
_set_job_info = _api("SetInformationJobObject", win.BOOL, win.HANDLE, ctypes.c_int, win.LPVOID, win.DWORD)
_query_job = _api("QueryInformationJobObject", win.BOOL, win.HANDLE, ctypes.c_int, win.LPVOID,
                  win.DWORD, win.LPDWORD)
_close = _api("CloseHandle", win.BOOL, win.HANDLE)
_assign = _api("AssignProcessToJobObject", win.BOOL, win.HANDLE, win.HANDLE)
_kill_job = _api("TerminateJobObject", win.BOOL, win.HANDLE, win.UINT)
_open_process = _api("OpenProcess", win.HANDLE, win.DWORD, win.BOOL, win.DWORD)
_snapshot = _api("CreateToolhelp32Snapshot", win.HANDLE, win.DWORD, win.DWORD)
_first_thread = _api("Thread32First", win.BOOL, win.HANDLE, ctypes.POINTER(_ThreadEntry))
_next_thread = _api("Thread32Next", win.BOOL, win.HANDLE, ctypes.POINTER(_ThreadEntry))
_open_thread = _api("OpenThread", win.HANDLE, win.DWORD, win.BOOL, win.DWORD)
_resume = _api("ResumeThread", win.DWORD, win.HANDLE)
_wait = _api("WaitForSingleObject", win.DWORD, win.HANDLE, win.DWORD)


def _checked(value):
    if not value:
        raise ctypes.WinError(ctypes.get_last_error())
    return value


def _resume_primary(pid):
    # Popen closes the primary thread handle. A suspended new process has one
    # thread, which can be reopened via the documented Toolhelp APIs.
    snapshot = _snapshot(_SNAP_THREADS, 0)
    if snapshot == _INVALID_HANDLE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        entry = _ThreadEntry(size=ctypes.sizeof(_ThreadEntry))
        found = _first_thread(snapshot, ctypes.byref(entry))
        while found and entry.owner_pid != pid:
            entry.size = ctypes.sizeof(entry)
            found = _next_thread(snapshot, ctypes.byref(entry))
        if not found:
            raise OSError("suspended candidate thread not found")
        thread = _checked(_open_thread(_THREAD_SUSPEND_RESUME, False, entry.thread_id))
        try:
            if _resume(thread) == _DWORD_ERROR:
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            _close(thread)
    finally:
        _close(snapshot)


def _member_ids(job):
    capacity = 8
    while True:
        size = 2 * ctypes.sizeof(win.DWORD) + capacity * ctypes.sizeof(_SIZE)
        buffer = ctypes.create_string_buffer(size)
        if _query_job(job, 3, buffer, size, None):
            header = (win.DWORD * 2).from_buffer(buffer)
            return (_SIZE * header[1]).from_buffer_copy(buffer, 2 * ctypes.sizeof(win.DWORD))
        error = ctypes.get_last_error()
        if error != _ERROR_MORE_DATA:
            raise ctypes.WinError(error)
        capacity = max(capacity * 2, (win.DWORD * 2).from_buffer(buffer)[0])


def _stop_job(job, close_job):
    with ExitStack() as handles:
        members = []
        termination_error = None
        try:
            # No live member may start another process during teardown.
            limits = _ExtendedLimits()
            limits.basic.flags = _KILL_ON_JOB_CLOSE | _LIMIT_ACTIVE_PROCESSES
            limits.basic.active_process_limit = 1
            _checked(_set_job_info(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
            for pid in _member_ids(job):
                handle = _open_process(_SYNCHRONIZE, False, pid)
                if handle:
                    handles.callback(_close, handle)
                    members.append(handle)
                elif ctypes.get_last_error() != _ERROR_INVALID_PARAMETER:
                    _checked(handle)
        finally:
            if not _kill_job(job, 1):
                termination_error = ctypes.WinError(ctypes.get_last_error())
                close_job()

        # Job accounting can reach zero before pending I/O and cwd handles are released.
        deadline = time.monotonic() + 1.0
        for handle in members:
            remaining = max(0, int((deadline - time.monotonic()) * 1000))
            if _wait(handle, remaining) != 0:
                raise TimeoutError("candidate job member did not terminate")
        if termination_error is not None:
            raise termination_error
        accounting = _JobAccounting()
        while True:
            _checked(_query_job(job, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None))
            if accounting.active_processes == 0:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("candidate job did not terminate")
            time.sleep(0.001)


def run(args, *, input, timeout, cwd):
    """Capture text stdio, with a deadline even if the child blocks stdin."""
    job = process = timer = None
    assigned = False
    expired = threading.Event()
    stop_errors = []
    job_lock = threading.Lock()

    def close_job():
        nonlocal job
        if job is not None:
            _checked(_close(job))
            job = None

    def stop():
        with job_lock:
            if job is not None:
                try:
                    _stop_job(job, close_job)
                except BaseException:
                    # Release the job before waiting on any blocked pipe operation.
                    close_job()
                    raise

    def expire():
        expired.set()
        try:
            stop()
        except Exception as exc:
            stop_errors.append(exc)

    try:
        job = _checked(_create_job(None, None))
        limits = _ExtendedLimits()
        limits.basic.flags = _KILL_ON_JOB_CLOSE
        _checked(_set_job_info(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, cwd=cwd,
                                   creationflags=_CREATE_SUSPENDED)
        handle = _checked(_open_process(_PROCESS_JOB_ACCESS, False, process.pid))
        try:
            _checked(_assign(job, handle))
            assigned = True
        finally:
            _close(handle)
        _resume_primary(process.pid)
        timer = threading.Timer(timeout, expire)
        timer.start()
        stdout, stderr = process.communicate(input=input, timeout=timeout)
        if expired.is_set():
            raise subprocess.TimeoutExpired(args, timeout, output=stdout, stderr=stderr)
        return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
    finally:
        try:
            if timer is not None:
                timer.cancel()
                if timer.ident is not None:
                    timer.join()
            if process is not None:
                try:
                    if assigned:
                        if stop_errors:
                            raise stop_errors[0]
                        if not expired.is_set():
                            stop()
                    else:
                        process.kill()
                finally:
                    try:
                        process.wait(timeout=1)
                    finally:
                        for pipe in (process.stdin, process.stdout, process.stderr):
                            if pipe is not None:
                                pipe.close()
        finally:
            with job_lock:
                close_job()
