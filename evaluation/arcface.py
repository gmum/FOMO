"""Identity erasure: ArcFace cosine between generated faces and reference photos.

For every video, each frame is searched for a face, the largest face is
embedded with ArcFace, and the embedding is compared against the mean
embedding of that person's reference photographs. The cell is the mean cosine
over the frames where a face was found.

    ID-sim(erased X, person P) = mean over frames of cos(ArcFace(frame), ref(P))

Read it the way ESR and PSR are read. On the diagonal, where P is the erased
identity, low is good. Off the diagonal it is the opposite: a drop there is
damage to an identity nobody asked to remove.

THE DETECTION RATE IS PART OF THE MEASUREMENT.

A method that destroys the face outright leaves nothing to embed. Averaging
over the frames that do contain a face then reports the similarity of exactly
the frames where erasure failed, so the number looks highest when the method
worked best. Every row therefore carries n_frames and n_faces, and a row whose
detection rate collapsed is successful erasure whatever its cosine says.

Reference photographs are yours to supply, three to five clear frontal shots
per person, one directory each:

    data/identity_refs/<person_slug>/*.jpg

They are never committed and never leave the machine. Published photographs of
public figures are somebody's copyright, which is the other reason they are not
in this repository.

    python -m evaluation.arcface experiments/identities.yaml --check-refs
    python -m evaluation.arcface experiments/identities.yaml --resume
"""

import argparse
import csv
import glob
import os

from experiments import config as expconfig
from experiments import layout


def load_analyser(model, det_size):
    """Detector plus ArcFace, on whatever device onnxruntime actually has.

    There is no aarch64 GPU wheel of onnxruntime, so on an ARM cluster the only
    provider is the CPU one unless it was built locally. Asking for ctx_id=0
    there makes insightface print a CUDA warning per model and fall back
    anyway; asking for -1 says what is happening and keeps the log readable.
    """
    import onnxruntime
    from insightface.app import FaceAnalysis

    # onnxruntime sizes its thread pool from the machine rather than from the
    # cpuset the scheduler handed us, then tries to pin each thread to a core
    # it does not own -- one error per thread, 288 of them on a GH200 node, and
    # the work happens anyway. Naming the count skips affinity entirely.
    try:
        allowed = len(os.sched_getaffinity(0))
    except AttributeError:
        allowed = os.cpu_count() or 8
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK") or allowed)

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    onnxruntime.set_default_logger_severity(3)

    providers = onnxruntime.get_available_providers()
    on_gpu = "CUDAExecutionProvider" in providers
    print("onnxruntime: %s -> %s, %d thread(s)"
          % (", ".join(providers), "GPU" if on_gpu else "CPU", threads))

    try:
        app = FaceAnalysis(name=model, session_options=options)
    except TypeError:
        app = FaceAnalysis(name=model)       # older insightface, no kwarg
    app.prepare(ctx_id=0 if on_gpu else -1, det_size=(det_size, det_size))
    return app


def largest_face(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def build_references(app, refs_dir, people, keep_vectors=False):
    """Mean L2-normalised ArcFace embedding per identity.

    Averaging over several photographs is what makes the reference a person
    rather than one lighting condition. A photograph with no detectable face is
    skipped loudly: a silently weaker reference would depress every cell of
    that row and read as erasure everywhere.
    """
    import imageio.v2 as imageio
    import numpy as np

    refs, detail = {}, {}
    for person in people:
        slug = layout.slugify(person)
        paths = sorted(p for ext in ("jpg", "jpeg", "png", "webp")
                       for p in glob.glob(os.path.join(refs_dir, slug, "*.%s" % ext)))
        if not paths:
            print("  %-22s NO PHOTOGRAPHS in %s/%s" % (person, refs_dir, slug))
            continue
        vectors, used = [], []
        for path in paths:
            image = imageio.imread(path)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            faces = app.get(image[:, :, :3][:, :, ::-1])
            if not faces:
                print("  %-22s no face in %s (skipped)"
                      % (person, os.path.basename(path)))
                continue
            vectors.append(largest_face(faces).normed_embedding)
            used.append(os.path.basename(path))
        if not vectors:
            print("  %-22s NO USABLE PHOTOGRAPH" % person)
            continue
        mean = np.mean(vectors, axis=0)
        refs[person] = mean / np.linalg.norm(mean)
        if keep_vectors:
            detail[person] = (vectors, used)
        print("  %-22s %d photo(s), %d with a face" % (person, len(paths), len(vectors)))
    return (refs, detail) if keep_vectors else refs


def check_references(detail):
    """How well each identity's photographs agree with one another.

    Two photographs of the same face land around 0.6 to 0.9 apart in cosine. A
    pair far below that is usually a different person, a hard profile, or the
    wrong face picked out of a group shot. Catching it here costs a minute;
    catching it afterwards costs a rerun, and only if somebody happens to
    notice that one row is uniformly low.
    """
    import numpy as np

    print("\nAgreement between reference photographs")
    print("-" * 72)
    for person, (vectors, names) in sorted(detail.items()):
        if len(vectors) < 2:
            print("%-22s one usable photo, nothing to cross-check" % person)
            continue
        pairs = [(float(np.dot(vectors[i], vectors[j])), names[i], names[j])
                 for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
        worst = min(pairs)
        mean = sum(p[0] for p in pairs) / len(pairs)
        flag = "   <-- CHECK" if worst[0] < 0.40 else ""
        print("%-22s mean %.3f   worst %.3f  (%s vs %s)%s"
              % (person, mean, worst[0], worst[1], worst[2], flag))
    print("-" * 72)
    print("Below 0.40 on a pair is worth opening both files before trusting the row.")


def score_directory(app, directory, reference, stride):
    """Per video: (filename, frames looked at, cosine for each face found)."""
    import numpy as np
    from evaluation.resnet import read_frames

    out = []
    for video in sorted(glob.glob(os.path.join(directory, "*.mp4"))):
        try:
            frames = read_frames(video)
        except Exception as error:
            print("  UNREADABLE %s (%s)" % (video, type(error).__name__), flush=True)
            out.append((os.path.basename(video), 0, []))
            continue
        looked, hits = 0, []
        for index, frame in enumerate(frames):
            if stride > 1 and index % stride:
                continue
            looked += 1
            faces = app.get(frame[:, :, ::-1])
            if faces:
                hits.append(float(np.dot(largest_face(faces).normed_embedding,
                                         reference)))
        out.append((os.path.basename(video), looked, hits))
    return out


def find_cells(cfg, people, videos_root, include_base):
    """Every (erased, person, directory) the run will score."""
    cells = []
    if include_base:
        base = layout.base_videos(cfg["prompts"], root=videos_root)
        for person in people:
            path = layout.subject_videos(base, person)
            if os.path.isdir(path):
                cells.append(("none", person, path))
    for row in expconfig.rows(cfg):
        column = layout.column_videos(cfg["name"], row.concept, root=videos_root)
        for person in people:
            path = layout.subject_videos(column, person)
            if glob.glob(os.path.join(path, "*.mp4")):
                cells.append((row.concept, person, path))
    return cells


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="ArcFace ID-similarity over a column tree.")
    ap.add_argument("config")
    ap.add_argument("--refs-dir", default="data/identity_refs")
    ap.add_argument("--videos-root", default="videos")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--model", default="buffalo_l", help="insightface model pack.")
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="Score every Nth frame. Above one trades resolution of "
                         "the detection rate for time, which is the metric's own "
                         "denominator, so leave it at one for anything reported.")
    ap.add_argument("--no-base", action="store_true",
                    help="Skip the untouched model. The Original row is the "
                         "scale the other two are read against, so this is for "
                         "topping up an existing file, not for a first run.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip cells already in the per-video CSV and append. "
                         "A job killed at the wall clock then costs nothing.")
    ap.add_argument("--flush-every", type=int, default=10)
    ap.add_argument("--list", action="store_true", help="Print the cells and exit.")
    ap.add_argument("--check-refs", action="store_true",
                    help="Embed the photographs, report their agreement, exit.")
    args = ap.parse_args()

    cfg = expconfig.load(args.config)
    if cfg["kind"] != "identities":
        raise SystemExit("%s is a %s experiment; objects are scored with "
                         "evaluation.resnet." % (args.config, cfg["kind"]))
    people = expconfig.subjects(cfg)

    if args.check_refs:
        app = load_analyser(args.model, args.det_size)
        print("References from %s:" % args.refs_dir)
        refs, detail = build_references(app, args.refs_dir, people, keep_vectors=True)
        missing = [p for p in people if p not in refs]
        if missing:
            print("\nNo usable reference for: %s" % ", ".join(missing))
        check_references(detail)
        return

    cells = find_cells(cfg, people, args.videos_root, not args.no_base)
    if args.list:
        for erased, person, path in cells:
            print("%-22s %-22s %s" % (erased, person, path))
        print("\n%d cells" % len(cells))
        return
    if not cells:
        raise SystemExit("No videos found under %s for %s"
                         % (args.videos_root, cfg["name"]))

    results = layout.experiment_results(cfg["name"], root=args.results_root)
    os.makedirs(results, exist_ok=True)
    out = os.path.join(results, "id_similarity.csv")
    per_video_out = os.path.join(results, "id_similarity_per_video.csv")

    import numpy as np

    app = load_analyser(args.model, args.det_size)
    print("References from %s:" % args.refs_dir)
    refs = build_references(app, args.refs_dir, people)
    if not refs:
        raise SystemExit("No usable references; nothing to compare against.")
    missing = [p for p in people if p not in refs]
    if missing:
        print("\nNo reference for %s -- those rows are skipped."
              % ", ".join(missing))

    todo = [c for c in cells if c[1] in refs]
    done_rows, done_video_rows = [], []
    if args.resume and os.path.exists(per_video_out):
        with open(per_video_out, newline="", encoding="utf-8") as handle:
            done_video_rows = list(csv.DictReader(handle))
        finished = {(r["erased"], r["person"]) for r in done_video_rows}
        before = len(todo)
        todo = [c for c in todo if (c[0], c[1]) not in finished]
        print("Resuming: %d of %d cells already scored." % (before - len(todo), before))
        if os.path.exists(out):
            with open(out, newline="", encoding="utf-8") as handle:
                done_rows = list(csv.DictReader(handle))
    if not todo:
        print("Nothing left to score.")
        return

    print("\n%-22s %-22s %7s %7s %8s"
          % ("erased", "person", "frames", "faces", "id_sim"))
    print("-" * 72)

    rows, video_rows = [], []
    for erased, person, path in todo:
        per_video = score_directory(app, path, refs[person], args.frame_stride)
        total = sum(looked for _, looked, _ in per_video)
        hits = [c for _, _, cosines in per_video for c in cosines]
        for name, looked, cosines in per_video:
            video_rows.append({
                "erased": erased, "person": person, "video": name,
                "n_frames": looked, "n_faces": len(cosines),
                "id_sim": "" if not cosines else "%.6f" % float(np.mean(cosines)),
            })
        value = float(np.mean(hits)) if hits else None
        rows.append({
            "erased": erased, "person": person,
            "n_videos": len(per_video), "n_frames": total, "n_faces": len(hits),
            "detection_rate": "%.4f" % (len(hits) / total) if total else "",
            "id_sim": "" if value is None else "%.4f" % value,
        })
        print("%-22s %-22s %7d %7d %8s"
              % (erased, person, total, len(hits),
                 "  no face" if value is None else "%.4f" % value), flush=True)
        if len(rows) % args.flush_every == 0:
            write_csv(out, done_rows + rows)
            write_csv(per_video_out, done_video_rows + video_rows)

    # Sorted, so a run split across two jobs produces the same file regardless
    # of the order the pieces finished in.
    rows = sorted(done_rows + rows, key=lambda r: (r["erased"], r["person"]))
    video_rows = sorted(done_video_rows + video_rows,
                        key=lambda r: (r["erased"], r["person"], r["video"]))
    write_csv(out, rows)
    write_csv(per_video_out, video_rows)
    print("\nWrote %d rows -> %s" % (len(rows), out))
    print("Wrote %d rows -> %s" % (len(video_rows), per_video_out))


if __name__ == "__main__":
    main()
