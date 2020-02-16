#pragma once

/* Concurrent Execution
 *
 * Controls the creation of the command queue for kernels. If it is defined, every Octokernel 
 * object will have its own private command queue, allowing for concurrent execution of
 * queued tasks. If undefined, the command queue will be declared static to the class, effectively
 * creating a single in-order command queue that serializes execution.
 */
#define CONCURRENT_EXECUTION

/* OpenCL Profiler
 *
 * When defined, event profilers will be enabled to capture execution/read/write times. Concurrent
 * execution must be undefined/disabled.
 */
//#define OPENCL_PROFILER_ENABLE

/* Intel Profiler
 *
 * When defined, event profilers will be sent to the Intel profiler to generate an execution trace
 * (profile.mon). Use aocl report to analyze the output. OpenCL Profiler must be enabled.
 */
//#define INTEL_PROFILER_ENABLE

