#include "DFS.h"

int frequencies[] = {76800000, 153600000, 230400000, 307200000, 384000000, 460800000, 537600000, 614400000, 691200000, 768000000, 844800000, 921600000};
int current_frequency_index = 11;

void calculate_power(float timelapse)
{
    int power = read_power();
    FILE *fp = fopen("./power_consump.txt", "a");
    if (!fp) {
        perror("Failed to create txt");
        return;
    }

    fprintf(fp, "%d\t%f\n", power, timelapse);
    
    fclose(fp);
    return;
}

int read_power() 
{
    FILE *fp;
    char buf[1024];
    int power = 0;

    fp = fopen("/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input", "r");
    if (!fp) {
        perror("Failed to read power");
        exit(1);
    }

    fgets(buf, sizeof(buf), fp);
    sscanf(buf, "%d", &power);

    fclose(fp);
    return power;
} 

int read_frequency()
{
    /* this function read from the filesystem of GPU cur_freq
     * returns the index of the frequency list from 0 to 11 */
    FILE *fp = fopen(GPU_PATH "cur_freq", "r");
    if (!fp) {
        perror("Failed to read frequency: 1");
        exit(1);
    }

    long long number;

    if (fscanf(fp, "%lld", &number) != 1) {
        perror("Failed to read frequency: 2");
        exit(1);
    }
    if (number < F_MIN || number > F_MAX) {
        perror("Failed to read frequency: 3");
        exit(1);
    }

    fclose(fp);
    return (number - F_MIN) / INTERVAL;
}

int read_GPU_usage()
{
    FILE *fp;
    char buf[1024];
    int usage = 0;

    fp = fopen("/sys/devices/platform/host1x/57000000.gpu/load", "r");
    if (!fp) {
        perror("Failed to read usage");
        exit(1);
    }

    fgets(buf, sizeof(buf), fp);
    sscanf(buf, "%d", &usage);

    fclose(fp);
    return usage;
}

void update_frequency_index()
{
    // redundent function
    FILE *fp = fopen("/sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq", "r");
    if (!fp) {
        perror("Failed to open the file");
        return;
    }

    int cur_freq;
    if (fscanf(fp, "%d", &cur_freq) != 1) {
        fprintf(stderr, "Failed to read frequency value\n");
        fclose(fp);
        return;
    }
    fclose(fp);

    // linear search from frequency[]
    bool found = 0;
    for (int i = 0; i < sizeof(frequencies) / sizeof(frequencies[0]); i++) {
        if (frequencies[i] == cur_freq) {
            current_frequency_index = i;
            found = 1;
            break;
        }
    }

    if (!found) {
        fprintf(stderr, "finding frequency from file error\n");
    }
    return;
}

void set_freq(int index)
{
    FILE *fp = fopen(GPU_PATH "userspace/set_freq", "w");
    if (!fp) {
        perror("Error setting frequency");
        return;
    }

    int freq = (index * INTERVAL) + F_MIN;
    fprintf(fp, "%d", freq);
    fclose(fp);

    return;
}

/* void set_freq(int index) */
/* { */
/*     FILE *fileMax, *fileMin; */
/*     fileMax = fopen("/sys/devices/57000000.gpu/devfreq/57000000.gpu/max_freq", "w"); */
/*     fileMin = fopen("/sys/devices/57000000.gpu/devfreq/57000000.gpu/min_freq", "w"); */
/*     if (fileMax == NULL || fileMin == NULL) { */
/*         perror("Error opening file"); */
/*         return; */
/*     } */
/*     int curr_idx = read_frequency(); */

/*     if (index == curr_idx) { */
/*         fclose(fileMax); */
/*         fclose(fileMin); */
/*         return; */
/*     } else if (index > current_frequency_index && index < sizeof(frequencies) / sizeof(frequencies[0])) { */
/*         fprintf(fileMax, "%d", frequencies[index]); */
/*         fprintf(fileMin, "%d", frequencies[index]); */
/*     } else if (index < current_frequency_index && index >= 0) { */
/*         fprintf(fileMin, "%d", frequencies[index]); */
/*         fprintf(fileMax, "%d", frequencies[index]); */
/*     } else { */
/*         perror("Error frequency index"); */
/*         fclose(fileMax); */
/*         fclose(fileMin); */
/*         return; */
/*     } */
/*     fclose(fileMax); */
/*     fclose(fileMin); */
/*     return; */
/* } */

void increase_freq(int steps)
{
    /* increase freq by the input steps */
    int curr_idx = read_frequency();
    int idx = curr_idx + steps;
    if (idx >= FREQ_SIZE) {
        // to the max freq
        set_freq(11);
    } else {
        set_freq(idx);
    }

    return;
}

void decrease_freq(int steps)
{
    int curr_idx = read_frequency();
    int idx = curr_idx - steps;
    if (idx < 0) {
        // to the min freq
        set_freq(0);
    } else {
        set_freq(idx);
    }

    return;
}

int set_low_bound()
{
    /* set to minimal frequency and return the original state index 
     * used in fine-grained scale to the minimal frequency         */
    int from_idx = read_frequency();

    set_freq(0);

    return from_idx;
}

void set_high_bound(int from_idx)
{
    /* return to the original state before set_low_bound()
     * used in fine-grained scale to the minimal frequecny  */
    set_freq(from_idx);

    return;
}

void daemonize() 
{
    /* todo: to measure the power consumption at the background
     * current issue: daemonize will cause CUDA core dump      */
    pid_t pid = fork();

    if (pid < 0) {
        exit(EXIT_FAILURE);
    }

    if (pid > 0) {
        exit(EXIT_SUCCESS);
    }

    if (setsid() < 0) {
        exit(EXIT_FAILURE);
    }

    signal(SIGCHLD, SIG_IGN);
    signal(SIGHUP, SIG_IGN);

    pid = fork();

    if (pid < 0) {
        exit(EXIT_FAILURE);
    }

    if (pid > 0) {
        exit(EXIT_SUCCESS);
    }

    umask(0);

    chdir("/");

    for (int x = sysconf(_SC_OPEN_MAX); x >= 0; x--) {
        close(x);
    }

    openlog("power_daemon", LOG_PID, LOG_DAEMON);
}

void stop_daemon() 
{
    keep_running = false;
}

void power_monitoring_daemon() 
{
    /* daemonize(); */

    while (keep_running) {
        // Check if the target program is still running
        // If not, break the loop and exit

        calculate_power(0.01); // Assuming 1 second intervals
        sleep(0.01); // Sleep for a second or the desired interval
    }

    syslog(LOG_NOTICE, "Power monitoring daemon terminated.");
    closelog();
}

dynamic_require* update_threshold(dynamic_require *thres, dynamic_require *curr)
{
    /* todo: write the c version of three vector weighter */
    // Moving average algorithm
    float alpha = 0.1;
    thres->bboxes = alpha * curr->bboxes + (1 - alpha) * thres->bboxes;
    thres->perf = alpha * curr->perf + (1 - alpha) * thres->perf;
    thres->gpu_util = alpha * curr->gpu_util + (1 - alpha) * thres->gpu_util;
    
    return thres;
}

int gpu_scaling_workload(bool P_THERSHOLD_exceed)
{
    int power = read_power();
    int workload = read_GPU_usage();

    update_frequency_index();

    printf("current power: %d\n", power);
    if (power > P_THRESHOLD) {
        decrease_freq(1);
        P_THERSHOLD_exceed = 1;
    } else if (workload < LOW_WORKLOAD_THRESHOLD) {
        decrease_freq(1);
        P_THERSHOLD_exceed = 0;
    } else if (workload > HIGH_WORKLOAD_THRESHOLD && !P_THERSHOLD_exceed) {
        increase_freq(1);
    }

    return P_THERSHOLD_exceed ? 1 : 0;
}

int main()
{
    /* todo: the initialization of setting GPU scaling governor */
    typedef struct {
        const char* path;
        char info[20];
    } fileOp;

    fileOp ops[3];

    sprintf(ops[1].info, "%d", F_MAX);
    sprintf(ops[2].info, "%d", F_MIN);

    ops[0] = (fileOp){GPU_PATH "governor", "userspace"};
    ops[1].path = GPU_PATH "max_freq";
    ops[2].path = GPU_PATH "min_freq";
    
    int numOps = sizeof(ops) / sizeof(ops[0]);

    for (int i = 0; i < numOps; i++) {
        FILE* fp = fopen(ops[i].path, "w");
        if (!fp) {
            perror("Failed init");
            return -1;
        }

        fprintf(fp, "%s", ops[i].info);
        fclose(fp);
    }
    printf("Successful init GPU scaling governor as userspace");

    return 0;               
}
