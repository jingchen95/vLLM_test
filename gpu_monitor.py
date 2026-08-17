#!/usr/bin/env python3
"""
GPU Power and Frequency Monitor/Controller
Usage: 
  Monitor only: python gpu_monitor.py --monitor --duration 60
  Set frequency: python gpu_monitor.py --set-freq --gpu-freq 1500 --mem-freq 6251
  Monitor with frequency change: python gpu_monitor.py --monitor --duration 60 --gpu-freq 1500
"""

import pynvml
import time
import csv
import argparse
import signal
import sys
from datetime import datetime

# Global variable to hold the GPUMonitor instance for the signal handler
monitor_instance = None

class GPUMonitor:
    def __init__(self, gpu_id=0):
        pynvml.nvmlInit()
        self.gpu_id = gpu_id
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        self.monitoring = False
        self.data = []
        
        # Check if energy consumption reading is supported (Volta+)
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self.supports_energy = True
        except pynvml.NVMLError:
            self.supports_energy = False
            print(f"Warning: GPU {gpu_id} does not support Total Energy Consumption reading.")
        
    def get_power(self):
        """Get current power consumption in watts"""
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW to W
            return power
        except:
            return None

    def get_total_energy(self):
        """Get total energy consumption in mJ since driver load"""
        if not self.supports_energy:
            return None
        try:
            # Returns energy in millijoules (mJ)
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except pynvml.NVMLError:
            return None
    
    def get_frequencies(self):
        """Get current GPU and memory frequencies"""
        try:
            gpu_freq = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
            mem_freq = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
            return gpu_freq, mem_freq
        except:
            return None, None
    
    def get_temperature(self):
        """Get GPU temperature"""
        try:
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            return temp
        except:
            return None
    
    def get_utilization(self):
        """Get GPU utilization"""
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            return util.gpu, util.memory
        except:
            return None, None
    
    def set_frequency(self, gpu_freq=None, mem_freq=None):
        """Set GPU and/or memory frequency (requires sudo/admin)"""
        try:
            if gpu_freq and mem_freq:
                print(f"Attempting to set GPU freq to {gpu_freq} MHz and Memory freq to {mem_freq} MHz...")
                pynvml.nvmlDeviceSetApplicationsClocks(self.handle, mem_freq, gpu_freq)
                print("Successfully set both frequencies.")

            elif gpu_freq:
                default_mem_freq = 6251 
                print(f"Attempting to set GPU freq to {gpu_freq} MHz (using default memory freq {default_mem_freq} MHz)...")
                pynvml.nvmlDeviceSetApplicationsClocks(self.handle, default_mem_freq, gpu_freq)
                print("Successfully set GPU frequency.")

            elif mem_freq:
                print(f"Attempting to set Memory freq to {mem_freq} MHz (retaining current GPU freq)...")
                current_gpu, _ = self.get_frequencies() 
                pynvml.nvmlDeviceSetApplicationsClocks(self.handle, mem_freq, current_gpu)
                print("Successfully set memory frequency.")

        except pynvml.NVMLError as e:
            print(f"Error setting frequency: {e}")
            print("Note: Frequency changes may require elevated privileges")
    
    def reset_clocks(self):
        """Reset to default clocks"""
        try:
            pynvml.nvmlDeviceResetApplicationsClocks(self.handle)
            print("Reset to default clocks")
        except pynvml.NVMLError as e:
            print(f"Error resetting clocks: {e}")
    
    def monitor_sample(self):
        """Take one monitoring sample"""
        timestamp = time.time()
        power = self.get_power()
        energy = self.get_total_energy()  # NEW: Get accumulated energy
        gpu_freq, mem_freq = self.get_frequencies()
        temp = self.get_temperature()
        gpu_util, mem_util = self.get_utilization()
        
        sample = {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'power_w': power,
            'total_energy_mj': energy,  # NEW: Add to sample dict
            'gpu_freq_mhz': gpu_freq,
            'mem_freq_mhz': mem_freq,
            'temperature_c': temp,
            'gpu_util_pct': gpu_util,
            'mem_util_pct': mem_util
        }
        
        return sample
    
    def monitor_continuous(self, duration=None, interval=0.1, output_file=None):
        """Monitor continuously, flushing each sample to CSV immediately.

        Writing row-by-row (with flush) means all collected data survives
        even if the process is terminated with SIGKILL (e.g. when a SLURM
        step is cancelled via slurmstepd), which does not invoke the signal
        handler or any atexit/finally cleanup.
        """
        self.monitoring = True
        self.data = []

        print(f"Starting monitoring (interval: {interval}s)")
        if duration:
            print(f"Duration: {duration}s")
        if output_file:
            print(f"Output file: {output_file}")

        start_time = time.time()
        _csv_file = None
        _writer = None

        try:
            while self.monitoring:
                if duration and (time.time() - start_time) >= duration:
                    print("\nMonitoring duration reached.")
                    self.monitoring = False
                    break

                sample = self.monitor_sample()
                self.data.append(sample)

                # Write and flush immediately — survives SIGKILL
                if output_file:
                    if _csv_file is None:
                        _csv_file = open(output_file, 'w', newline='')
                        _writer = csv.DictWriter(_csv_file, fieldnames=sample.keys())
                        _writer.writeheader()
                    _writer.writerow(sample)
                    _csv_file.flush()

                if not self.monitoring:
                    break

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            self.monitoring = False
        finally:
            if _csv_file is not None:
                _csv_file.close()

        print(f"\nCollected {len(self.data)} samples")
    
    def save_data(self, filename):
        """Save monitoring data to CSV"""
        if not self.data:
            print("No data to save")
            return
            
        print(f"Saving {len(self.data)} data points to {filename}...")
        try:
            with open(filename, 'w', newline='') as f:
                if self.data: 
                    writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
                    writer.writeheader()
                    writer.writerows(self.data)
                    print(f"Data saved successfully to {filename}")
                else:
                    print(f"No data available to write to {filename}")
        except Exception as e:
            print(f"Error saving data to {filename}: {e}")
    
    def print_summary(self):
        """Print monitoring summary"""
        if not self.data:
            print("No data for summary.")
            return
            
        powers = [d['power_w'] for d in self.data if d['power_w'] is not None]
        energies = [d['total_energy_mj'] for d in self.data if d['total_energy_mj'] is not None]
        gpu_freqs = [d['gpu_freq_mhz'] for d in self.data if d['gpu_freq_mhz'] is not None]
        
        if powers:
            print(f"\nPower Summary:")
            print(f"  Average: {sum(powers)/len(powers):.1f}W")
            print(f"  Min: {min(powers):.1f}W")
            print(f"  Max: {max(powers):.1f}W")
            
        if energies and len(energies) > 1:
            total_consumed = (energies[-1] - energies[0]) / 1000.0  # mJ to J
            print(f"\nEnergy Consumption (calculated from hardware counter):")
            print(f"  Total: {total_consumed:.3f} J")
        
        if gpu_freqs:
            print(f"\nGPU Frequency Summary:")
            print(f"  Average: {sum(gpu_freqs)/len(gpu_freqs):.0f}MHz")
            print(f"  Min: {min(gpu_freqs)}MHz")
            print(f"  Max: {max(gpu_freqs)}MHz")

def signal_handler(sig, frame):
    global monitor_instance
    signal_name = signal.Signals(sig).name
    print(f'\nSignal {signal_name} received. Stopping monitoring...')
    # Data is already flushed to disk row-by-row — no explicit save needed.
    # Just stop the monitoring loop; the finally block in monitor_continuous
    # will close the CSV file before sys.exit unwinds the stack.
    if monitor_instance:
        monitor_instance.monitoring = False
    pynvml.nvmlShutdown()
    print("Exiting due to signal.")
    sys.exit(0)

def main():
    global monitor_instance
    parser = argparse.ArgumentParser(description='GPU Monitor and Controller')
    parser.add_argument('--monitor', action='store_true', help='Start monitoring')
    parser.add_argument('--duration', type=float, help='Monitoring duration in seconds')
    parser.add_argument('--interval', type=float, default=0.1, help='Sampling interval (default: 0.1s)')
    parser.add_argument('--output', type=str, help='Output CSV file')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID (default: 0)')
    
    # Frequency control
    parser.add_argument('--set-freq', action='store_true', help='Set frequency')
    parser.add_argument('--gpu-freq', type=int, help='GPU frequency in MHz')
    parser.add_argument('--mem-freq', type=int, help='Memory frequency in MHz')
    parser.add_argument('--reset-clocks', action='store_true', help='Reset to default clocks')
    
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler) 
    
    monitor = GPUMonitor(args.gpu_id)
    monitor_instance = monitor
    
    if args.set_freq or args.gpu_freq or args.mem_freq:
        monitor.set_frequency(args.gpu_freq, args.mem_freq)
        if not args.monitor:
            pynvml.nvmlShutdown()
            return
    
    if args.reset_clocks:
        monitor.reset_clocks()
        if not args.monitor:
            pynvml.nvmlShutdown()
            return
    
    if args.monitor:
        try:
            monitor_instance.monitor_continuous(
                duration=args.duration,
                interval=args.interval,
                output_file=args.output,
            )
            monitor_instance.print_summary()
        finally:
            print("Main: Shutting down NVML.")
            pynvml.nvmlShutdown() 
    else:
        sample = monitor.monitor_sample()
        print(f"Current Status:")
        print(f"  Power: {sample['power_w']:.1f}W")
        if sample['total_energy_mj']:
            print(f"  Total Energy (driver lifetime): {sample['total_energy_mj'] / 1000.0:.3f} J")
        print(f"  GPU Freq: {sample['gpu_freq_mhz']}MHz")
        print(f"  Mem Freq: {sample['mem_freq_mhz']}MHz")
        print(f"  Temperature: {sample['temperature_c']}°C")
        print(f"  GPU Utilization: {sample['gpu_util_pct']}%")
        pynvml.nvmlShutdown() 

if __name__ == "__main__":
    main()