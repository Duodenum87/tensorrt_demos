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

void increase_freq()
{
    if (current_frequency_index < sizeof(frequencies) / sizeof(frequencies[0]) - 1) {
        int index = current_frequency_index + 1;
        if (index >= 0 && index < sizeof(frequencies) / sizeof(frequencies[0])) {
            FILE *file1, *file2;
            file1 = fopen("/sys/devices/57000000.gpu/devfreq/57000000.gpu/max_freq", "w");
            file2 = fopen("/sys/devices/57000000.gpu/devfreq/57000000.gpu/min_freq", "w");

            if (file1 == NULL || file2 == NULL) {
                perror("Error opening file");
                return;
            }

            fprintf(file1, "%d", frequencies[index]);
            fprintf(file2, "%d", frequencies[index]);
            printf("Frequency set to: %dHz\n", frequencies[index]);

            fclose(file1);
            fclose(file2);
        }
    }
}

void decrease_freq()
{
    if (current_frequency_index > 0) {
        int index = current_frequency_index - 1;
        if (index >= 0 && index < sizeof(frequencies) / sizeof(frequencies[0])) {
            FILE *file1, *file2;
            file1 = fopen("/sys/devices/57000000.gpu/devfreq/57000000.gpu/min_freq", "w");
            file2 = fopen("/sys/devices/57000000.gpu/devfreq/57000000.gpu/max_freq", "w");

            if (file1 == NULL || file2 == NULL) {
                perror("Error opening file");
                return;
            }

            fprintf(file1, "%d", frequencies[index]);
            fprintf(file2, "%d", frequencies[index]);
            printf("Frequency set to: %dHz\n", frequencies[index]);

            fclose(file1);
            fclose(file2);
        }
    }
}

int main(bool P_THERSHOLD_exceed)
{
    // while (1)
    // {
    //     sleep(DELTA_T / 1000);

        int power = read_power();
        int workload = read_GPU_usage();

        update_frequency_index();

        if (power > P_THRESHOLD) {
            decrease_freq();
            P_THERSHOLD_exceed = 1;
        } else if (workload < LOW_WORKLOAD_THRESHOLD) {
            decrease_freq();
            P_THERSHOLD_exceed = 0;
        } else if (workload > HIGH_WORKLOAD_THRESHOLD && !P_THERSHOLD_exceed) {
            increase_freq();
        }

    // }
    return P_THERSHOLD_exceed ? 1 : 0;
}