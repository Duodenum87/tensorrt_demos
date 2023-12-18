#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <syslog.h>

// Constants and global variables
#define F_NOMINAL 921600000 // Assuming a nominal frequency of 921Mhz for illustration
#define F_MIN 76800000     // Minimum frequency
#define F_MAX 921600000     // Maximum frequency
#define INTERVAL 76800000   // The difference between two freq
#define FREQ_SIZE 12        // 12 frequency options
#define P_THRESHOLD 7000 // Power threshold in watts
#define DELTA_T 10     // Monitoring interval in milliseconds
#define INCREMENT_STEP 100  // Frequency increment step
#define DECREMENT_STEP 100  // Frequency decrement step
#define LOW_WORKLOAD_THRESHOLD 20 // Placeholder value
#define HIGH_WORKLOAD_THRESHOLD 80 // Placeholder value
#define MARGIN 5 // Safety margin for power
#define GPU_PATH "/sys/devices/57000000.gpu/devfreq/57000000.gpu/"

extern void update_frequency_index();
void calculate_power(float);
int read_power();
extern int set_low_bound();
void set_high_bound();

static volatile bool keep_running = true;

typedef struct {
    float bboxes; // #Bounding box 
    float perf; // Processing time of each frame
    float gpu_util; // GPU utiliztion rate 
} dynamic_require;
