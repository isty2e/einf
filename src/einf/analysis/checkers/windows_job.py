"""Windows checker Job ownership and a directly executable stdlib-only launcher."""

import ctypes
import logging
import subprocess
import sys
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Literal, Self
from uuid import uuid4

_LOGGER = logging.getLogger(__name__)
_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_ASSIGN_PROCESS = 0x0001


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _Accounting(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_int64),
        ("TotalKernelTime", ctypes.c_int64),
        ("ThisPeriodTotalUserTime", ctypes.c_int64),
        ("ThisPeriodTotalKernelTime", ctypes.c_int64),
        ("TotalPageFaultCount", ctypes.c_uint32),
        ("TotalProcesses", ctypes.c_uint32),
        ("ActiveProcesses", ctypes.c_uint32),
        ("TotalTerminatedProcesses", ctypes.c_uint32),
    ]


def _load_kernel32() -> ctypes.CDLL:
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    else:
        raise OSError("Windows checker Jobs are unavailable on this platform")
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    kernel32.OpenJobObjectW.argtypes = [
        ctypes.c_uint32,
        ctypes.c_int32,
        ctypes.c_wchar_p,
    ]
    kernel32.OpenJobObjectW.restype = ctypes.c_void_p
    kernel32.SetInformationJobObject.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    kernel32.SetInformationJobObject.restype = ctypes.c_int32
    kernel32.QueryInformationJobObject.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    kernel32.QueryInformationJobObject.restype = ctypes.c_int32
    kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.AssignProcessToJobObject.restype = ctypes.c_int32
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.TerminateJobObject.restype = ctypes.c_int32
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int32
    kernel32.CreateEventW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_wchar_p,
    ]
    kernel32.CreateEventW.restype = ctypes.c_void_p
    kernel32.OpenEventW.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_wchar_p]
    kernel32.OpenEventW.restype = ctypes.c_void_p
    kernel32.SetEvent.argtypes = [ctypes.c_void_p]
    kernel32.SetEvent.restype = ctypes.c_int32
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    return kernel32


def _win_error() -> OSError:
    if sys.platform == "win32":
        return ctypes.WinError(ctypes.get_last_error())
    return OSError("Windows checker API failure")


class WindowsJob(AbstractContextManager["WindowsJob"]):
    """Own a checker Job and its checker-started event until cleanup completes.

    Raises
    ------
    OSError
        If Job creation, kill-on-close configuration, or launch-event creation fails.

    Notes
    -----
    Breakaway is not enabled. The launcher opens its own short-lived handle;
    arbitrary checker descendants never inherit this resource's handle.
    """

    def __init__(self) -> None:
        self._kernel32 = _load_kernel32()
        self._name = f"Local\\einf-checker-{uuid4().hex}"
        self._handle: int | None = None
        self._started_event: int | None = None
        try:
            handle = self._kernel32.CreateJobObjectW(None, self._name)
            if not handle:
                raise _win_error()
            self._handle = handle

            limits = _ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
            if not self._kernel32.SetInformationJobObject(
                self._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
            ):
                raise _win_error()
            started_event = self._kernel32.CreateEventW(
                None, True, False, self._name + "-started"
            )
            if not started_event:
                raise _win_error()
            self._started_event = started_event
        except BaseException:
            try:
                self.close()
            except OSError:
                _LOGGER.warning(
                    "Windows checker Job setup cleanup failed", exc_info=True
                )
            raise

    def launch_command(self, command: tuple[str, ...]) -> tuple[str, ...]:
        """Wrap checker argv in the isolated standard-library launcher.

        Parameters
        ----------
        command : tuple[str, ...]
            Original checker executable and arguments, without a shell.

        Returns
        -------
        tuple[str, ...]
            Python launcher argv carrying the opaque Job name and checker argv.

        Raises
        ------
        OSError
            If the runtime cannot identify its base interpreter executable.
        """
        executable = getattr(sys, "_base_executable", None)
        if not isinstance(executable, str) or not executable:
            raise OSError("Windows checker launcher requires a base Python executable")
        # A venv redirector's exit does not prove that its interpreter has exited.
        return (
            executable,
            "-I",
            "-S",
            str(Path(__file__).resolve()),
            self._name,
            *command,
        )

    def terminate(self) -> None:
        """Request termination of every associated process and nested Job.

        Raises
        ------
        OSError
            If the Job is closed or Windows rejects the termination request.
        """
        if self._handle is None:
            raise OSError("Windows checker Job is closed")
        if not self._kernel32.TerminateJobObject(self._handle, 1):
            raise _win_error()

    def is_empty(self) -> bool:
        """Check native process accounting without assuming that kill means exit.

        Returns
        -------
        bool
            Whether no process remains active in this Job.

        Raises
        ------
        OSError
            If the Job is closed or its accounting cannot be queried.
        """
        if self._handle is None:
            raise OSError("Windows checker Job is closed")
        accounting = _Accounting()
        if not self._kernel32.QueryInformationJobObject(
            self._handle, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None
        ):
            raise _win_error()
        return accounting.ActiveProcesses == 0

    def checker_started(self) -> bool:
        """Read the launcher-owned proof that checker process creation succeeded.

        Returns
        -------
        bool
            Whether the launcher signaled after creating the checker process.

        Raises
        ------
        OSError
            If the event is closed or Windows cannot read its state.
        """
        if self._started_event is None:
            raise OSError("Windows checker launch event is closed")
        status = self._kernel32.WaitForSingleObject(self._started_event, 0)
        if status == 0:
            return True
        if status == 258:
            return False
        raise _win_error()

    def close(self) -> None:
        """Release both native handles; last Job close terminates remaining members.

        Raises
        ------
        OSError
            If Windows rejects closure. Failed handles remain available for retry.
        """
        error = None
        if self._started_event is not None:
            if self._kernel32.CloseHandle(self._started_event):
                self._started_event = None
            else:
                error = _win_error()
        if self._handle is not None:
            if not self._kernel32.CloseHandle(self._handle):
                if error is None:
                    error = _win_error()
            else:
                self._handle = None
        if error is not None:
            raise error

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        try:
            self.close()
        except OSError:
            if exc_value is None:
                raise
            _LOGGER.warning("Windows checker Job handle cleanup failed", exc_info=True)
        return False


def _join_job(name: str) -> None:
    kernel32 = _load_kernel32()
    handle = kernel32.OpenJobObjectW(_JOB_OBJECT_ASSIGN_PROCESS, False, name)
    if not handle:
        raise _win_error()
    try:
        if not kernel32.AssignProcessToJobObject(handle, kernel32.GetCurrentProcess()):
            raise _win_error()
    finally:
        if not kernel32.CloseHandle(handle):
            raise _win_error()


def _main() -> int:
    try:
        _join_job(sys.argv[1])
        kernel32 = _load_kernel32()
        event = kernel32.OpenEventW(0x0002, False, sys.argv[1] + "-started")
        if not event:
            raise _win_error()
        try:
            process = subprocess.Popen(sys.argv[2:])
            if not kernel32.SetEvent(event):
                error = _win_error()
                process.kill()
                process.wait()
                raise error
        finally:
            if not kernel32.CloseHandle(event):
                raise _win_error()
        # SystemExit must fit Windows' signed C long without losing DWORD bits.
        return ctypes.c_int32(process.wait()).value
    except (OSError, IndexError) as error:
        print(f"Windows checker launcher failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())
