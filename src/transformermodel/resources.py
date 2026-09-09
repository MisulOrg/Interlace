"""macOS process footprint from the public libproc rusage_info_v4 ABI."""
import ctypes
import os


def process_footprint(pid=None):
    fields='user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size resident_size phys_footprint proc_start_abstime proc_exit_abstime child_user_time child_system_time child_pkg_idle_wkups child_interrupt_wkups child_pageins child_elapsed_abstime diskio_bytesread diskio_byteswritten cpu_time_qos_default cpu_time_qos_maintenance cpu_time_qos_background cpu_time_qos_utility cpu_time_qos_legacy cpu_time_qos_user_initiated cpu_time_qos_user_interactive billed_system_time serviced_system_time logical_writes lifetime_max_phys_footprint instructions cycles billed_energy serviced_energy interval_max_phys_footprint runnable_time'.split()
    class Usage(ctypes.Structure):
        _fields_=[('uuid',ctypes.c_uint8*16)]+[(name,ctypes.c_uint64) for name in fields]
    try:
        lib=ctypes.CDLL('/usr/lib/libproc.dylib',use_errno=True)
        function=lib.proc_pid_rusage
        function.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
        function.restype=ctypes.c_int
        value=Usage()
        if function(os.getpid() if pid is None else pid,4,ctypes.byref(value)):
            return {'available':False,'errno':ctypes.get_errno()}
        return {'available':True,'current_phys_footprint_bytes':value.phys_footprint,
                'lifetime_max_phys_footprint_bytes':value.lifetime_max_phys_footprint,
                'resident_bytes':value.resident_size,'pageins':value.pageins,
                'disk_read_bytes':value.diskio_bytesread,'disk_written_bytes':value.diskio_byteswritten}
    except OSError as error:return {'available':False,'error':str(error)}
