import os, sys, time, tarfile

class OffsetTarStream:
    def __init__(self, paths, start_offset):
        self.paths = paths
        self.idx = 0
        self.fp = open(paths[0], "rb")
        self.fp.seek(start_offset)
        self.bytes_read = start_offset
        self.total_size = sum(os.path.getsize(p) for p in paths)

    def read(self, n=-1):
        if self.idx >= len(self.paths):
            return b""
        chunk = self.fp.read(n)
        if not chunk and n > 0:
            self.fp.close()
            self.idx += 1
            if self.idx < len(self.paths):
                self.fp = open(self.paths[self.idx], "rb")
                return self.read(n)
            return b""
        self.bytes_read += len(chunk)
        return chunk

def main():
    base_dir = r"D:\ПРОФ\ВУЗ\ВКР\DATA\Gaze360"
    splits_dir = os.path.join(base_dir, "splits")
    p_e = os.path.join(splits_dir, "Gaze360.tar.partae")
    p_f = os.path.join(splits_dir, "Gaze360.tar.partaf")
    
    first_header_offset = None
    with open(p_e, "rb") as fp:
        off = 0
        while off < 1024 * 1024:
            block = fp.read(512)
            if not block:
                break
            if block[257:262] == b"ustar":
                first_header_offset = off
                break
            off += 512
            
    if first_header_offset is None:
        raise ValueError("Could not find ustar header in partae")
        
    stream = OffsetTarStream([p_e, p_f], first_header_offset)
    print(f"Starting extraction for partae + partaf (offset {first_header_offset}, total {stream.total_size / (1024**3):.2f} GB)...")
    
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
