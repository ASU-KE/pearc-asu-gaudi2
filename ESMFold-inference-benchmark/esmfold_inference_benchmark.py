#!/usr/bin/env python3
"""
ESMFold Inference Benchmarking Script
Works with both CUDA (A100) and HPU (Gaudi 2) devices

Usage:
Run script and add flags for device and output file location:

    python esmfold_inference_benchmark.py --device cuda --output cuda_results.json
    python esmfold_inference_benchmark.py --device hpu --output hpu_results.json
"""

import argparse
import time
import json
import subprocess
import threading
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import numpy as np

import torch

try:
    import habana_frameworks.torch.hpu.metrics as hpu_metrics
    HPU_METRICS_AVAILABLE = True
except ImportError:
    hpu_metrics = None
    HPU_METRICS_AVAILABLE = False

class PowerMonitor:
    """Monitor power consumption during benchmark"""
    
    def __init__(self, device_type: str, interval: float = 0.1):
        self.device_type = device_type
        self.interval = interval
        self.power_readings = []
        self.monitoring = False
        self.thread = None
        
    def _monitor_cuda(self):
        """Monitor NVIDIA GPU power"""
        while self.monitoring:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=1
                )
                if result.returncode == 0:
                    power = float(result.stdout.strip().split('\n')[0])
                    self.power_readings.append(power)
            except Exception as e:
                pass
            time.sleep(self.interval)


    def _monitor_hpu(self):
        """
            Monitor Gaudi HPU power
            Increased timeout used with hl-smi because of hl-smi output latency
            Power comparison might not be very accurate becase of average power used in Gaudi
        """
        while self.monitoring:
            try:
                result = subprocess.run(
                    ['hl-smi', '-q'],
                    capture_output=True, text=True, timeout=5  # Increased timeout
                )
                if result.returncode == 0:
                    # Collect ALL power readings from all AIPs
                    powers = []
                    for line in result.stdout.split('\n'):
                        if 'Power Draw' in line and 'W' in line:
                            parts = line.split(':')
                            if len(parts) > 1:
                                value_part = parts[1].strip().split()[0]
                                try:
                                    power = float(value_part)
                                    powers.append(power)
                                except:
                                    pass
                
                    # Take max across all AIPs (active one will show spike)
                    if powers:
                        max_power = max(powers)
                        self.power_readings.append(max_power)
            except:
                pass
            time.sleep(self.interval)

    def start(self):
        """Start monitoring power"""
        self.power_readings = []
        self.monitoring = True
        if self.device_type == 'cuda':
            self.thread = threading.Thread(target=self._monitor_cuda, daemon=True)
        else:
            self.thread = threading.Thread(target=self._monitor_hpu, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring and return statistics"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2)
        
        if not self.power_readings:
            return {'avg_watts': 0, 'max_watts': 0, 'min_watts': 0, 'readings_count': 0}
        
        return {
            'avg_watts': float(np.mean(self.power_readings)),
            'max_watts': float(np.max(self.power_readings)),
            'min_watts': float(np.min(self.power_readings)),
            'readings_count': len(self.power_readings)
        }


class ESMFoldBenchmark:
    """Benchmark ESMFold inference on different devices"""
    
    def __init__(self, device: str):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.results = []

        # Base sequence is kept at 350 length. Arbitrary slices of this sequence are used to form shorter or longer sequences
        
        base_seq = 'MGAGASAEEKHSRELEKKLKEDAEKDARTVKLLLLGAGESGKSTIVKQMKIIHQDGYSLEECLEFIAIIYGNTLQSILAIVRAMTTLNIQYGDSARQDDARKLMHMADTIEEGTMPKEMSDIIQRLWKDSGIQACFERASEYQLNDSAGYYLSDLERLVTPGYVPTEQDVLRSRVKTTGIIETQFSFKDLNFRMFDVGGQRSERKKWIHCFEGVTCIIFIAALSAYDMVLVEDDEVNRMHESLHLFNSICNHRYFATTSIVLFLNKKDVFFEKIKKAHLSICFPDYDGPNTYEDAGNYIKVQFLELNMRRDVKEIYSHMTCATDTQNVKFVFDAVTDIIIKENLKDCGLF'

        self.test_sequences = {
            'short_50': base_seq[50:100],
            'short_100': base_seq[31:131],
            'medium_200': base_seq[101:301],
            'medium_350': base_seq,
            'long_400': base_seq + base_seq[20:70],
            'long_500': base_seq[50:300] + base_seq[69:319],
            'longest_700': base_seq + base_seq,
            'longest_1000': base_seq + base_seq + base_seq[25:325],
        }
        
        
    def setup_cuda(self):
        """Setup ESMFold for CUDA device"""
        print("Setting up ESMFold for CUDA...")
        from esm import pretrained
        
        self.model = pretrained.esmfold_v1()
        self.model = self.model.eval().cuda()
        
        # Optionally set chunk size for memory management
        # self.model.set_chunk_size(128)
        
        # Get memory info
        torch.cuda.reset_peak_memory_stats()
        
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
    def setup_hpu(self):
        """Setup ESMFold for HPU device"""
        print("Setting up ESMFold for HPU...")
        
        # Set HPU environment variables
        os.environ["PT_HPU_ENABLE_H2D_DYNAMIC_SLICE"] = "0"
        os.environ["PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES"] = "1"
        
        # Import HPU-specific modules
        import habana_frameworks.torch.core as htcore
        from transformers import AutoTokenizer, EsmForProteinFolding
        from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

        if HPU_METRICS_AVAILABLE:
            self.gc_metric = hpu_metrics.GraphCompilationMetric()
            self.mem_metric = hpu_metrics.DevMemMetric()
            self.cpu_fallback_metric = hpu_metrics.CpuFallbackMetric()
        else:
            self.gc_metric = None
            self.mem_metric = None
            self.cpu_fallback_metric = None

        # Adapt transformers for Gaudi
        adapt_transformers_to_gaudi()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        
        # Load model
        # Set _supports_param_buffer_assignment to False for float16 weights compatibility
        EsmForProteinFolding._supports_param_buffer_assignment = False
        self.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=False)
        self.model = self.model.to(torch.device("hpu"))
        self.model.eval()
        
        # Set chunk size for longer sequences
        self.model.trunk.set_chunk_size(64)
        
        print("HPU Device initialized")
    
    def infer_cuda(self, sequence: str):
        """Run inference on CUDA"""
        with torch.no_grad():
            output = self.model.infer(sequence)
        torch.cuda.synchronize()
        return output
    
    def infer_hpu(self, sequence: str):
        """Run inference on HPU"""
        import habana_frameworks.torch.core as htcore
        
        # Tokenize
        tk = self.tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
        tokenized_input = tk["input_ids"].to(torch.device("hpu"))
        
        with torch.no_grad():
            output = self.model(tokenized_input)
            htcore.mark_step()
        
        return output
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics"""
        if self.device == 'cuda':
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'reserved_gb': torch.cuda.memory_reserved() / 1e9,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9
            }
        return {}
    
    def warmup(self, num_iterations: int = 3):
        """Warmup runs to stabilize performance and compile HPU graphs"""
        print(f"\nRunning {num_iterations} warmup iterations...")
        warmup_seq = 'M' * 100
        
        for i in range(num_iterations):
            start = time.time()
            if self.device == 'cuda':
                _ = self.infer_cuda(warmup_seq)
            else:
                _ = self.infer_hpu(warmup_seq)
            duration = time.time() - start
            if self.device == 'hpu' and self.gc_metric is not None:
                gc_stats = dict(self.gc_metric.stats())
                print(f"  Compilations: {gc_stats['TotalNumber']}, Total time: {gc_stats['TotalTime']:.3f}s, Avg: {gc_stats['AvgTime']:.3f}s")

            print(f"  Warmup {i+1}/{num_iterations}: {duration:.3f}s")
    
    def benchmark_sequence(self, seq_name: str, sequence: str, num_runs: int = 10) -> Dict[str, Any]:
        """Benchmark a single sequence"""
        print(f"\nBenchmarking {seq_name} (length={len(sequence)}, runs={num_runs})...")
        
        latencies = []
        compile_time = None
        power_monitor = PowerMonitor(self.device, interval=2.0 if self.device == 'hpu' else 0.1)

        graph_stats = []
        
        # Reset memory stats
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        
        # Start power monitoring
        power_monitor.start()
        
        # Run benchmark
        for i in range(num_runs):
            start_time = time.time()
            
            if self.device == 'cuda':
                output = self.infer_cuda(sequence)
            else:
                output = self.infer_hpu(sequence)
            
            latency = time.time() - start_time
            latencies.append(latency)
            
            # First run includes compilation for HPU
            if i == 0 and self.device == 'hpu':
                compile_time = latency
                print(f"  Run {i+1}/{num_runs} (with compilation): {latency:.3f}s")
                if self.gc_metric is not None:
                    gc_stats = dict(self.gc_metric.stats())
                    graph_stats.append(gc_stats)
                    print(f"  Compilations: {gc_stats['TotalNumber']}, Total time: {gc_stats['TotalTime']:.3f}s, Avg: {gc_stats['AvgTime']:.3f}s")

            else:
                print(f"  Run {i+1}/{num_runs}: {latency:.3f}s")
                if self.device == 'hpu' and self.gc_metric is not None:
                    gc_stats = dict(self.gc_metric.stats())
                    graph_stats.append(gc_stats)
                    print(f"  Compilations: {gc_stats['TotalNumber']}, Total time: {gc_stats['TotalTime']:.3f}s, Avg: {gc_stats['AvgTime']:.3f}s")

        
        # Stop power monitoring
        power_stats = power_monitor.stop()
        
        # Compute statistics
        latencies_array = np.array(latencies)
        
        # For HPU, also compute stats without first run (compilation)
        if self.device == 'hpu' and len(latencies) > 1:
            inference_only = np.array(latencies[1:])
            results = {
                'sequence_name': seq_name,
                'sequence_length': len(sequence),
                'num_runs': num_runs,
                'compile_time_s': float(compile_time) if compile_time else None,
                'latency_with_compile_mean_s': float(np.mean(latencies_array)),
                'latency_mean_s': float(np.mean(inference_only)),
                'latency_std_s': float(np.std(inference_only)),
                'latency_min_s': float(np.min(inference_only)),
                'latency_max_s': float(np.max(inference_only)),
                'latency_median_s': float(np.median(inference_only)),
                'throughput_seq_per_s': 1.0 / np.mean(inference_only),
                'power_stats': power_stats,
                'memory_stats': self.get_memory_stats(),
                'graph_stats': graph_stats if graph_stats is graph_stats else None
            }
        else:
            results = {
                'sequence_name': seq_name,
                'sequence_length': len(sequence),
                'num_runs': num_runs,
                'latency_mean_s': float(np.mean(latencies_array)),
                'latency_std_s': float(np.std(latencies_array)),
                'latency_min_s': float(np.min(latencies_array)),
                'latency_max_s': float(np.max(latencies_array)),
                'latency_median_s': float(np.median(latencies_array)),
                'throughput_seq_per_s': 1.0 / np.mean(latencies_array),
                'power_stats': power_stats,
                'memory_stats': self.get_memory_stats()
            }
        
        return results
    
    def run_all_benchmarks(self, num_runs: int = 10, output_file: str = None):
        """Run complete benchmark suite"""
        print(f"\n{'='*60}")
        print(f"ESMFold Inference Benchmark")
        print(f"Device: {self.device.upper()}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"{'='*60}")
        
        # Setup model
        if self.device == 'cuda':
            self.setup_cuda()
        elif self.device == 'hpu':
            self.setup_hpu()
        else:
            raise ValueError(f"Unsupported device: {self.device}")
        
        # Warmup
        self.warmup(num_iterations=3)
        
        # Benchmark individual sequences
        all_results = {
            'device': self.device,
            'timestamp': datetime.now().isoformat(),
            'num_runs_per_sequence': num_runs,
            'single_sequence_results': []
        }
        
        for seq_name, sequence in self.test_sequences.items():
            try:
                result = self.benchmark_sequence(seq_name, sequence, num_runs=num_runs)
                all_results['single_sequence_results'].append(result)
            except Exception as e:
                print(f"\nFailed to benchmark {seq_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Print summary
        self.print_summary(all_results)
        
        # Save results
        if output_file is None:
            output_file = f"esmfold_inference_{self.device}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        return all_results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print benchmark summary"""
        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}")
        
        print("\nSingle Sequence Results:")
        if self.device == 'hpu':
            print(f"{'Sequence':<20} {'Length':<10} {'Compile (s)':<12} {'Latency (s)':<15} {'Throughput':<15} {'Power (W)':<12}")
            print("-" * 95)
        else:
            print(f"{'Sequence':<20} {'Length':<10} {'Latency (s)':<15} {'Throughput':<15} {'Power (W)':<12}")
            print("-" * 85)
        
        for result in results['single_sequence_results']:
            if self.device == 'hpu' and 'compile_time_s' in result and result['compile_time_s']:
                print(f"{result['sequence_name']:<20} "
                      f"{result['sequence_length']:<10} "
                      f"{result['compile_time_s']:.3f}       "
                      f"{result['latency_mean_s']:.3f} ± {result['latency_std_s']:.3f}   "
                      f"{result['throughput_seq_per_s']:.3f} seq/s    "
                      f"{result['power_stats'].get('avg_watts', 0):.1f}W")
            else:
                print(f"{result['sequence_name']:<20} "
                      f"{result['sequence_length']:<10} "
                      f"{result['latency_mean_s']:.3f} ± {result['latency_std_s']:.3f}   "
                      f"{result['throughput_seq_per_s']:.3f} seq/s    "
                      f"{result['power_stats'].get('avg_watts', 0):.1f}W")
        
        print(f"\n{'='*80}")
        
        if self.device == 'hpu':
            print("\nNote: HPU latencies exclude first run (compilation time)")


def main():
    parser = argparse.ArgumentParser(description='Benchmark ESMFold inference')
    parser.add_argument('--device', type=str, required=True, choices=['cuda', 'hpu'],
                        help='Device to benchmark on (cuda or hpu)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--num-runs', type=int, default=10,
                        help='Number of runs per sequence (default: 10)')
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = ESMFoldBenchmark(device=args.device)
    results = benchmark.run_all_benchmarks(num_runs=args.num_runs, output_file=args.output)


if __name__ == '__main__':
    main()
