#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>

// Constants and global variables
#define F_NOMINAL 921600000 // Assuming a nominal frequency of 921Mhz for illustration
#define F_MIN 76800000     // Minimum frequency
#define F_MAX 921600000     // Maximum frequency
#define P_THRESHOLD 7000 // Power threshold in watts
#define DELTA_T 10     // Monitoring interval in milliseconds
#define INCREMENT_STEP 100  // Frequency increment step
#define DECREMENT_STEP 100  // Frequency decrement step
#define LOW_WORKLOAD_THRESHOLD 20 // Placeholder value
#define HIGH_WORKLOAD_THRESHOLD 80 // Placeholder value
#define MARGIN 5 // Safety margin for power

int frequencies[] = {76800000, 153600000, 230400000, 307200000, 384000000, 460800000, 537600000, 614400000, 691200000, 768000000, 844800000, 921600000};
int current_frequency_index = 11;

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

void set_frequency(int index)
{
    if (index >= 0 && index < sizeof(frequencies) / sizeof(frequencies[0])) {
        current_frequency_index = index;
        printf("Frequency set to: %dHz\n", frequencies[current_frequency_index]);
    } 
}

void increase_freq()
{
    if (current_frequency_index < sizeof(frequencies) / sizeof(frequencies[0]) - 1) {
        set_frequency(current_frequency_index + 1);
    }
}

void decrease_freq()
{
    if (current_frequency_index > 0) {
        set_frequency(current_frequency_index - 1);
    }
}

int main()
{
    while (1)
    {
        sleep(DELTA_T / 1000);

        int power = read_power();
        int workload = read_GPU_usage();

        if (power > P_THRESHOLD) {
            decrease_freq();
        } else if (workload < LOW_WORKLOAD_THRESHOLD) {
            decrease_freq();
        } else if (workload > HIGH_WORKLOAD_THRESHOLD) {
            increase_freq();
        }

    }
    return 0;
}