import time
import psutil
import os

process = psutil.Process(os.getpid())


class StageProfiler:
    def __init__(self, out_dir, name):
        self.results = {}
        self.out_dir = out_dir
        self.name = name

    def start(self, stage):
        self.results[stage] = {
            "t0": time.perf_counter(),
            "mem0": process.memory_info().rss,
        }

    def end(self, stage):
        t1 = time.perf_counter()
        mem1 = process.memory_info().rss

        r = self.results[stage]
        r["time"] = t1 - r["t0"]
        r["mem_delta_mb"] = (mem1 - r["mem0"]) / (1024 * 1024)
        r["mem_final_mb"] = mem1 / (1024 * 1024)

    def save(self):
        os.makedirs(self.out_dir, exist_ok=True)

        file_path = os.path.join(self.out_dir, f"{self.name}_profile.txt")

        with open(file_path, "w") as f:
            f.write(f"PROFILE: {self.name}\n")
            f.write("=" * 40 + "\n\n")

            total_time = 0

            for stage, r in self.results.items():
                total_time += r["time"]

                f.write(f"{stage}\n")
                f.write(f"  time        : {r['time']:.4f} s\n")
                f.write(f"  mem delta   : {r['mem_delta_mb']:.2f} MB\n")
                f.write(f"  mem final   : {r['mem_final_mb']:.2f} MB\n\n")

            f.write("=" * 40 + "\n")
            f.write(f"TOTAL TIME: {total_time:.4f} s\n")