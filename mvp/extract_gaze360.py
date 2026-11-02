import os, sys, time, tarfile

class ConcatenatedTarStream:
    def __init__(self, paths):
        self.paths = paths
        self.sizes = [os.path.getsize(p) for p in paths]
        self.total_size = sum(self.sizes)
        self.idx = 0
        self.current_fp = open(paths[0], "rb")
        self.bytes_read = 0

    def read(self, n=-1):
        if self.idx >= len(self.paths):
            return b""
        chunk = self.current_fp.read(n)
        if not chunk and n > 0:
            self.current_fp.close()
            self.idx += 1
            if self.idx < len(self.paths):
                self.current_fp = open(self.paths[self.idx], "rb")
                return self.read(n)
            return b""
        self.bytes_read += len(chunk)
        return chunk

def main():
    base_dir = r"D:\ПРОФ\ВУЗ\ВКР\DATA\Gaze360"
    splits_dir = os.path.join(base_dir, "splits")
    part_names = ["Gaze360.tar.partaa", "Gaze360.tar.partab", "Gaze360.tar.partac", "Gaze360.tar.partad", "Gaze360.tar.partae", "Gaze360.tar.partaf"]
    part_paths = [os.path.join(splits_dir, name) for name in part_names]
    for p in part_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing tar split part: {p}")
    
    stream = ConcatenatedTarStream(part_paths)
    print(f"Concatenated stream initialized: {len(part_paths)} parts, total {stream.total_size / (1024**3):.2f} GB")
    
    t_start = time.time()
    t_last_log = t_start
    extracted_count = 0
    skipped_count = 0
    extracted_bytes = 0
    
    with tarfile.open(mode="r|", fileobj=stream) as tar:
        for member in tar:
            target_path = os.path.join(base_dir, member.name)
            if member.isfile():
                if os.path.exists(target_path) and os.path.getsize(target_path) == member.size:
                    skipped_count += 1
                else:
                    tar.extract(member, path=base_dir)
                    extracted_count += 1
                    extracted_bytes += member.size
            elif member.isdir():
                os.makedirs(target_path, exist_ok=True)
            else:
                tar.extract(member, path=base_dir)
                extracted_count += 1

            total_seen = extracted_count + skipped_count
            now = time.time()
            if now - t_last_log >= 10.0 or total_seen % 20000 == 0:
                elapsed = now - t_start
                rate_mb = (stream.bytes_read / (1024**2)) / (elapsed + 1e-6)
                fps = total_seen / (elapsed + 1e-6)
                progress_pct = (stream.bytes_read / stream.total_size) * 100.0
                print(f"[{progress_pct:5.1f}%] Processed {total_seen:,} files ({extracted_count:,} extracted, {skipped_count:,} skipped) | {rate_mb:6.1f} MB/s | {fps:6.1f} files/s | Elapsed: {elapsed/60:.1f} min | Current: {member.name[:45]}")
                sys.stdout.flush()
                t_last_log = now

    t_total = time.time() - t_start
    print(f"Extraction completed in {t_total:.2f}s ({t_total/60:.2f} min). Total processed: {extracted_count + skipped_count:,} (extracted {extracted_count:,}, skipped {skipped_count:,}, {extracted_bytes / (1024**3):.2f} GB)")

if __name__ == "__main__":
    main()
