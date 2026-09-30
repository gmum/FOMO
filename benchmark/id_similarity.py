"""ID-Similarity: ArcFace cosine between generated faces and reference photos.

The protocol is the one T2VUnlearning uses for face erasure. For every video,
each frame is searched for a face, the face is embedded with ArcFace, and the
embedding is compared against the mean embedding of that identity's reference
photographs. The cell value is the mean cosine over the frames where a face was
found.

    ID-sim(erased X, person P, method M)
        = mean over frames of  cos( ArcFace(frame), ref(P) )

Read it the way ESR and PSR are read. On the diagonal, where P is the erased
identity, low is good: the face is gone. Off the diagonal it is the opposite --
a drop there is collateral damage, an identity that nobody asked to remove.

DETECTION RATE IS PART OF THE MEASUREMENT, NOT DIAGNOSTICS.

A method that destroys the face outright leaves nothing to embed. Averaging
only over frames that contain a face would then report the similarity of the
handful of frames where a face survived, which is exactly the frames where
erasure failed -- the number would look high precisely when the method worked
best. So every row carries n_frames and n_faces, and a row whose detection rate
collapsed has to be read as successful erasure regardless of its cosine.

Reference photographs are yours to supply, one directory per identity:

    data/identity_refs/<person slug>/*.jpg

Three to five clear frontal photographs per person is plenty. They are never
committed by this script and never leave the machine.

    python benchmark/id_similarity.py --videos_root videos/identity \\
        --refs_dir data/identity_refs --methods imap_null_lr1e3_000030 \\
        --include_base --out results/identity/id_similarity.csv
"""

import argparse
import csv
import glob
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import slugify  # noqa: E402

# insightface, numpy and the frame reader are imported inside main, after
# --list has had its chance to return, so listing works on a login node where
# the aarch64 environment does not exist.


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def find_cells(videos_root, people, methods, include_base):
    """Every (erased, person, method, directory) the run will score."""
    cells = []
    if include_base:
        for person in people:
            slug = slugify(person)
            path = os.path.join(videos_root, "_shared", "base", slug, "base")
            if os.path.isdir(path):
                cells.append(("none", person, "base", path))
    for erased in people:
        column = os.path.join(videos_root, "unlearn_%s" % slugify(erased))
        if not os.path.isdir(column):
            continue
        for person in people:
            for method in methods:
                path = os.path.join(column, slugify(person), method)
                if os.path.isdir(path) and glob.glob(os.path.join(path, "*.mp4")):
                    cells.append((erased, person, method, path))
    return cells


def build_references(app, refs_dir, people, np, keep_vectors=False):
    """Mean L2-normalised ArcFace embedding per identity.

    Averaging over several photographs is what makes the reference a person
    rather than one lighting condition. A photograph with no detectable face is
    skipped loudly: a silently weaker reference would depress every cell of
    that row and look like successful erasure everywhere.

    With keep_vectors the per-photograph embeddings and filenames come back too,
    which is what --check_refs reports on.
    """
    import imageio.v2 as imageio

    refs = {}
    detail = {}
    for person in people:
        slug = slugify(person)
        paths = sorted(
            p for ext in ("jpg", "jpeg", "png", "webp")
            for p in glob.glob(os.path.join(refs_dir, slug, "*.%s" % ext))
        )
        if not paths:
            print("  %-22s NO REFERENCE IMAGES in %s/%s" % (person, refs_dir, slug))
            continue
        vectors, used = [], []
        for path in paths:
            image = imageio.imread(path)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            faces = app.get(image[:, :, :3][:, :, ::-1])
            if not faces:
                print("  %-22s no face in %s (skipped)" % (person, os.path.basename(path)))
                continue
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            vectors.append(face.normed_embedding)
            used.append(os.path.basename(path))
        if not vectors:
            print("  %-22s NO USABLE REFERENCE" % person)
            continue
        mean = np.mean(vectors, axis=0)
        refs[person] = mean / np.linalg.norm(mean)
        if keep_vectors:
            detail[person] = (vectors, used)
        print("  %-22s %d photo(s), %d with a face" % (person, len(paths), len(vectors)))
    return (refs, detail) if keep_vectors else refs


def check_references(refs, detail, np):
    """Report how well each identity's photographs agree with one another.

    Two photographs of the same face land around 0.6 to 0.9 apart in cosine.
    A pair far below that usually means one of them is a different person, a
    hard profile, or the wrong face picked out of a group shot. Catching it
    here costs a minute; catching it after the evaluation costs a rerun, and
    only if somebody happens to notice that one row is uniformly low.
    """
    print("\nAgreement between reference photographs")
    print("-" * 72)
    for person, (vectors, names) in detail.items():
        if len(vectors) < 2:
            print("%-22s only one usable photo, nothing to cross-check" % person)
            continue
        pairs = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                pairs.append((float(np.dot(vectors[i], vectors[j])), names[i], names[j]))
        low = min(pairs)
        mean = sum(p[0] for p in pairs) / len(pairs)
        flag = "  <-- CHECK" if low[0] < 0.40 else ""
        print("%-22s mean %.3f   worst %.3f  (%s vs %s)%s"
              % (person, mean, low[0], low[1], low[2], flag))
    print("-" * 72)
    print("Below 0.40 on a pair is worth opening both files before trusting the row.")


def score_directory(app, path, reference, stride, np):
    """Per-video cosines and detection counts for one directory.

    Returns a list of (filename, frames_looked_at, [cosine per frame with a
    face]). The per-video breakdown is kept rather than folded away because the
    spread over videos is what the published tables report after the plus-minus
    sign; collapsing to a cell mean here would make it unrecoverable.
    """
    from evaluate import read_frames

    per_video = []
    for video in sorted(glob.glob(os.path.join(path, "*.mp4"))):
        try:
            frames = read_frames(video)
        except Exception as error:
            # A generation interrupted mid-write leaves a zero-stream mp4.
            # One of those used to take the whole run down with it, two hours
            # in, so it is now reported and stepped over: a missing video is a
            # hole in one cell, not a reason to lose everything before it.
            print("  UNREADABLE %s (%s)" % (video, type(error).__name__),
                  flush=True)
            per_video.append((os.path.basename(video), 0, []))
            continue
        looked, hits = 0, []
        for index, frame in enumerate(frames):
            if stride > 1 and index % stride:
                continue
            looked += 1
            faces = app.get(frame[:, :, ::-1])
            if not faces:
                continue
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            hits.append(float(np.dot(face.normed_embedding, reference)))
        per_video.append((os.path.basename(video), looked, hits))
    return per_video


_WORKER = {}


def _init_worker(model, det_size, threads):
    """One analyser per process, with its own thread budget.

    onnxruntime's internal threading stops scaling somewhere around eight
    threads for models this small, so a single session on seventy-two cores
    leaves most of them idle. Several processes with a few threads each keep
    the machine busy instead. The environment variables have to be set before
    onnxruntime is imported, which is why the import lives inside this function.
    """
    import os as _os
    _os.environ["OMP_NUM_THREADS"] = str(threads)
    _os.environ["ORT_NUM_THREADS"] = str(threads)
    _WORKER["app"] = load_analyser(model, det_size, quiet=True)


def _score_cell(task):
    """Score one cell inside a worker. Returns the cell and its per-video rows."""
    import numpy as np
    erased, person, method, path, reference, stride = task
    per_video = score_directory(_WORKER["app"], path, np.asarray(reference), stride, np)
    return erased, person, method, per_video


def load_analyser(model, det_size, quiet=False):
    """Detector plus ArcFace, on whatever device onnxruntime actually has.

    There is no aarch64 GPU build of onnxruntime, so on this cluster the only
    provider is the CPU one. Asking for ctx_id=0 there makes insightface print
    a CUDA warning per model and then fall back anyway; asking for -1 says what
    is happening and keeps the log readable.
    """
    import onnxruntime
    from insightface.app import FaceAnalysis

    # onnxruntime sizes its thread pool from the machine, not from the cpuset
    # Slurm handed us, then tries to pin each thread to a core it does not own.
    # That prints one error per thread -- 288 of them on a GH200 node -- and
    # spends the effort anyway. Naming the count explicitly stops both: the
    # library skips affinity entirely when the number is given.
    try:
        allowed = len(os.sched_getaffinity(0))
    except AttributeError:
        allowed = os.cpu_count() or 8
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK") or allowed)

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    # Affinity errors are logged at ERROR level even though nothing failed.
    onnxruntime.set_default_logger_severity(3)

    providers = onnxruntime.get_available_providers()
    on_gpu = "CUDAExecutionProvider" in providers
    print("onnxruntime providers: %s -> %s, %d thread(s)"
          % (", ".join(providers), "GPU" if on_gpu else "CPU", threads))

    try:
        app = FaceAnalysis(name=model, session_options=options)
    except TypeError:
        # Older insightface does not forward session options; the severity
        # setting above still silences the noise.
        app = FaceAnalysis(name=model)
    app.prepare(ctx_id=0 if on_gpu else -1, det_size=(det_size, det_size))
    return app


def main():
    ap = argparse.ArgumentParser(description="ArcFace ID-Similarity over a video tree.")
    ap.add_argument("--videos_root", required=True,
                    help="Holds unlearn_<erased>/<person>/<method>/NN.mp4.")
    ap.add_argument("--refs_dir", default="data/identity_refs",
                    help="Holds <person slug>/*.jpg reference photographs.")
    ap.add_argument("--prompts_module", default="identity_prompts",
                    help="Module exposing PEOPLE.")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="Method directory names to score. Not needed with "
                         "--check_refs, which touches no videos.")
    ap.add_argument("--include_base", action="store_true",
                    help="Also score _shared/base, the reference row.")
    ap.add_argument("--model", default="buffalo_l", help="insightface model pack.")
    ap.add_argument("--det_size", type=int, default=640)
    ap.add_argument("--frame_stride", type=int, default=4,
                    help="Score every Nth frame. Seventeen frames at stride 4 "
                         "gives five per video, which is plenty for an identity "
                         "that barely moves across two seconds -- and the "
                         "difference matters, because onnxruntime has no aarch64 "
                         "GPU build and this runs on CPU.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--flush_every", type=int, default=20,
                    help="Write both CSVs every N cells, so a job killed at the "
                         "wall clock loses at most that many.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip cells already present in the per-video CSV and "
                         "append to it. A job killed at the wall clock then "
                         "costs nothing: resubmit and it carries on.")
    ap.add_argument("--workers", type=int, default=0,
                    help="Processes scoring cells in parallel. 0 picks a "
                         "sensible number from the cores the job was given. "
                         "1 runs everything in this process, as before.")
    ap.add_argument("--threads_per_worker", type=int, default=4)
    ap.add_argument("--per_video_out", default=None,
                    help="One row per video. Defaults to <out>_per_video.csv. "
                         "make_id_table.py needs it for the spread.")
    ap.add_argument("--list", action="store_true", help="Print the cells and exit.")
    ap.add_argument("--check_refs", action="store_true",
                    help="Embed the reference photographs, report how well they "
                         "agree with each other, and exit. Score nothing.")
    args = ap.parse_args()

    people = importlib.import_module(args.prompts_module).PEOPLE

    if args.check_refs:
        import numpy as np
        app = load_analyser(args.model, args.det_size)
        print("References from %s:" % args.refs_dir)
        refs, detail = build_references(app, args.refs_dir, people, np, keep_vectors=True)
        missing = [p for p in people if p not in refs]
        if missing:
            print("\nNo usable reference for: %s" % ", ".join(missing))
        check_references(refs, detail, np)
        return

    if not args.methods:
        raise SystemExit("--methods is required unless you pass --check_refs")

    cells = find_cells(args.videos_root, people, args.methods, args.include_base)
    if args.list:
        for erased, person, method, path in cells:
            print("%-22s %-22s %-28s %s" % (erased, person, method, path))
        print("\n%d cells" % len(cells))
        return
    if not cells:
        raise SystemExit("No cells found under %s for methods %s"
                         % (args.videos_root, args.methods))

    import numpy as np

    app = load_analyser(args.model, args.det_size)

    print("References from %s:" % args.refs_dir)
    refs = build_references(app, args.refs_dir, people, np)
    missing = [p for p in people if p not in refs]
    if missing:
        print("\nNo reference for: %s" % ", ".join(missing))
        print("Those rows cannot be scored and are skipped.")
    if not refs:
        raise SystemExit("No usable references at all; nothing to compare against.")


    print("\n%-22s %-22s %-28s %7s %7s %8s" %
          ("erased", "person", "method", "frames", "faces", "id_sim"))
    print("-" * 100)
    todo = [(e, p, m, path) for e, p, m, path in cells if p in refs]

    out = args.out or os.path.join("results", "id_similarity.csv")
    per_video_out = args.per_video_out or out.replace(".csv", "_per_video.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    done_rows, done_video_rows = [], []
    if args.resume and os.path.exists(per_video_out):
        with open(per_video_out, newline="", encoding="utf-8") as handle:
            done_video_rows = list(csv.DictReader(handle))
        finished = {(r["erased"], r["person"], r["method"]) for r in done_video_rows}
        before = len(todo)
        todo = [c for c in todo if (c[0], c[1], c[2]) not in finished]
        print("Resuming: %d of %d cells already in %s"
              % (before - len(todo), before, per_video_out))
        if os.path.exists(out):
            with open(out, newline="", encoding="utf-8") as handle:
                done_rows = list(csv.DictReader(handle))
    if not todo:
        print("Nothing left to score.")
        return
    workers = args.workers
    if workers <= 0:
        cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
            else (os.cpu_count() or 8)
        workers = max(1, min(len(todo), cores // args.threads_per_worker))
    print("\nScoring %d cells with %d worker(s) x %d thread(s)"
          % (len(todo), workers, args.threads_per_worker))

    if workers > 1:
        import multiprocessing as mp
        tasks = [(e, p, m, path, refs[p].tolist(), args.frame_stride)
                 for e, p, m, path in todo]
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(workers, initializer=_init_worker,
                        initargs=(args.model, args.det_size, args.threads_per_worker))
        results = pool.imap_unordered(_score_cell, tasks)
    else:
        results = (( e, p, m, score_directory(app, path, refs[p], args.frame_stride, np))
                   for e, p, m, path in todo)
        pool = None

    rows, video_rows = [], []
    for erased, person, method, per_video in results:
        total = sum(looked for _, looked, _ in per_video)
        hits = [c for _, _, cs in per_video for c in cs]
        for name, looked, cosines in per_video:
            mean = float(np.mean(cosines)) if cosines else None
            video_rows.append({
                "erased": erased, "person": person, "method": method,
                "video": name, "n_frames": looked, "n_faces": len(cosines),
                "id_sim": "" if mean is None else "%.6f" % mean,
            })
        value = float(np.mean(hits)) if hits else None
        rows.append({
            "erased": erased,
            "person": person,
            "method": method,
            "n_videos": len(per_video),
            "n_frames": total,
            "n_faces": len(hits),
            "detection_rate": "%.4f" % (len(hits) / total) if total else "",
            "id_sim": "" if value is None else "%.4f" % value,
        })
        print("%-22s %-22s %-28s %7d %7d %8s"
              % (erased, person, method, total, len(hits),
                 "  no face" if value is None else "%.4f" % value), flush=True)

        # Flush periodically. Without this, --resume only ever picks up work
        # from a previous job, and everything the current one did before the
        # wall clock cut it off is lost -- which is the case it exists for.
        if len(rows) % args.flush_every == 0:
            write_csv(out, done_rows + rows)
            write_csv(per_video_out, done_video_rows + video_rows)

    if pool is not None:
        pool.close()
        pool.join()

    # Sorted, so that a run split across two jobs produces comparable files
    # regardless of the order the workers happened to finish in.
    rows = done_rows + rows
    video_rows = done_video_rows + video_rows
    rows.sort(key=lambda r: (r["erased"], r["person"], r["method"]))
    video_rows.sort(key=lambda r: (r["erased"], r["person"], r["method"], r["video"]))

    write_csv(out, rows)
    write_csv(per_video_out, video_rows)
    print("\nWrote %d rows -> %s" % (len(rows), out))
    print("Wrote %d rows -> %s" % (len(video_rows), per_video_out))


if __name__ == "__main__":
    main()
