/* [dec-hostloop] a C host function for CUDA host nodes (like NCCL's hostStreamPlanCallback): records
   CLOCK_MONOTONIC, a call count, the calling thread id and the CPU it ran on. */
#define _GNU_SOURCE
#include <sched.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
void glm53_cb(void *p) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  long *slot = (long *)p;
  slot[0] = (long)t.tv_sec * 1000000000L + t.tv_nsec;
  slot[1] += 1;
  slot[2] = (long)syscall(SYS_gettid);
  slot[3] = (long)sched_getcpu();
}
