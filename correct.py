import sys
import numpy as np
import cv2
import math
import os
import json
from PIL import Image
from tqdm import tqdm

THRESHOLD_RATIO = 2000
MIN_AVG_RED = 60
MAX_HUE_SHIFT = 120
BLUE_MAGIC_VALUE = 1.2
SAMPLE_SECONDS = 2 # Extracts color correction from every N seconds


def _sample_seconds_from_env(default_seconds=SAMPLE_SECONDS):
    """Reads optional sampling interval override from DCC_SAMPLE_SECONDS env var."""
    raw = os.getenv("DCC_SAMPLE_SECONDS")
    if not raw:
        return default_seconds
    try:
        value = int(raw)
        return value if value > 0 else default_seconds
    except ValueError:
        return default_seconds


def _progress_enabled_from_env(default_enabled=False):
    """Reads optional progress toggle from DCC_PROGRESS env var."""
    raw = os.getenv("DCC_PROGRESS")
    if raw is None:
        return default_enabled
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _progress_position_from_env(default_position=0):
    """Reads optional tqdm row position from DCC_PROGRESS_POSITION env var."""
    raw = os.getenv("DCC_PROGRESS_POSITION")
    if raw is None:
        return default_position
    try:
        value = int(raw)
        return value if value >= 0 else default_position
    except ValueError:
        return default_position


def _progress_desc_from_env(video_label):
    """Reads optional progress label from DCC_PROGRESS_DESC env var."""
    raw = os.getenv("DCC_PROGRESS_DESC")
    if raw is None:
        return video_label
    value = raw.strip()
    return value if value else video_label


def _progress_file_from_env():
    """Reads optional machine-readable progress output path from env var."""
    raw = os.getenv("DCC_PROGRESS_FILE")
    if raw is None:
        return None
    value = raw.strip()
    return value if value else None


def _write_progress_file(phase, current, total):
    """Writes progress state for parent batch renderer to consume."""
    progress_path = _progress_file_from_env()
    if progress_path is None:
        return

    payload = {
        "phase": str(phase),
        "current": int(max(0, current)),
        "total": int(max(1, total)),
    }

    temp_path = f"{progress_path}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as progress_file:
            json.dump(payload, progress_file)
        os.replace(temp_path, progress_path)
    except OSError:
        # Progress telemetry is best-effort and should never break processing.
        pass

def hue_shift_red(mat, h):

    U = math.cos(h * math.pi / 180)
    W = math.sin(h * math.pi / 180)

    r = (0.299 + 0.701 * U + 0.168 * W) * mat[..., 0]
    g = (0.587 - 0.587 * U + 0.330 * W) * mat[..., 1]
    b = (0.114 - 0.114 * U - 0.497 * W) * mat[..., 2]

    return np.dstack([r, g, b])

def normalizing_interval(array):

    high = 255
    low = 0
    max_dist = 0

    for i in range(1, len(array)):
        dist = array[i] - array[i-1]
        if(dist > max_dist):
            max_dist = dist
            high = array[i]
            low = array[i-1]

    return (low, high)

def apply_filter(mat, filt):
    # Use OpenCV's native transform for faster per-frame matrix application.
    transform = np.array([
        [filt[0], filt[1], filt[2]],
        [0.0, filt[6], 0.0],
        [0.0, 0.0, filt[12]],
    ], dtype=np.float32)
    offset = np.array([filt[4] * 255, filt[9] * 255, filt[14] * 255], dtype=np.float32)
    transformed = cv2.transform(mat.astype(np.float32), transform)
    transformed += offset
    return np.clip(transformed, 0, 255).astype(np.uint8)

def get_filter_matrix(mat):

    mat = cv2.resize(mat, (256, 256))

    # Get average values of RGB
    avg_mat = np.array(cv2.mean(mat)[:3], dtype=np.uint8)

    # Find hue shift so that average red reaches MIN_AVG_RED
    new_avg_r = avg_mat[0]
    hue_shift = 0
    while(new_avg_r < MIN_AVG_RED):

        shifted = hue_shift_red(avg_mat, hue_shift)
        new_avg_r = np.sum(shifted)
        hue_shift += 1
        if hue_shift > MAX_HUE_SHIFT:
            new_avg_r = MIN_AVG_RED

    # Apply hue shift to whole image and replace red channel
    shifted_mat = hue_shift_red(mat, hue_shift)
    new_r_channel = np.sum(shifted_mat, axis=2)
    new_r_channel = np.clip(new_r_channel, 0, 255)
    mat[..., 0] = new_r_channel

    # Get histogram of all channels
    hist_r = hist = cv2.calcHist([mat], [0], None, [256], [0,256])
    hist_g = hist = cv2.calcHist([mat], [1], None, [256], [0,256])
    hist_b = hist = cv2.calcHist([mat], [2], None, [256], [0,256])

    normalize_mat = np.zeros((256, 3))
    threshold_level = (mat.shape[0]*mat.shape[1])/THRESHOLD_RATIO
    for x in range(256):

        if hist_r[x] < threshold_level:
            normalize_mat[x][0] = x

        if hist_g[x] < threshold_level:
            normalize_mat[x][1] = x

        if hist_b[x] < threshold_level:
            normalize_mat[x][2] = x

    normalize_mat[255][0] = 255
    normalize_mat[255][1] = 255
    normalize_mat[255][2] = 255

    adjust_r_low, adjust_r_high = normalizing_interval(normalize_mat[..., 0])
    adjust_g_low, adjust_g_high = normalizing_interval(normalize_mat[..., 1])
    adjust_b_low, adjust_b_high = normalizing_interval(normalize_mat[..., 2])


    shifted = hue_shift_red(np.array([1, 1, 1]), hue_shift)
    shifted_r, shifted_g, shifted_b = shifted[0][0]

    red_gain = 256 / (adjust_r_high - adjust_r_low)
    green_gain = 256 / (adjust_g_high - adjust_g_low)
    blue_gain = 256 / (adjust_b_high - adjust_b_low)

    redOffset = (-adjust_r_low / 256) * red_gain
    greenOffset = (-adjust_g_low / 256) * green_gain
    blueOffset = (-adjust_b_low / 256) * blue_gain

    adjust_red = shifted_r * red_gain
    adjust_red_green = shifted_g * red_gain
    adjust_red_blue = shifted_b * red_gain * BLUE_MAGIC_VALUE

    return np.array([
        adjust_red, adjust_red_green, adjust_red_blue, 0, redOffset,
        0, green_gain, 0, 0, greenOffset,
        0, 0, blue_gain, 0, blueOffset,
        0, 0, 0, 1, 0,
    ])

def correct(mat):
    original_mat = mat.copy()

    filter_matrix = get_filter_matrix(mat)

    corrected_mat = apply_filter(original_mat, filter_matrix)
    corrected_mat = cv2.cvtColor(corrected_mat, cv2.COLOR_RGB2BGR)

    return corrected_mat

def correct_image(input_path, output_path):
    exif_data = None
    with Image.open(input_path) as image:
        exif_data = image.info.get("exif")
        if image.mode != "RGB":
            image = image.convert("RGB")
        mat = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    rgb_mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    corrected_mat = correct(rgb_mat)

    output_image = Image.fromarray(cv2.cvtColor(corrected_mat, cv2.COLOR_BGR2RGB))
    save_kwargs = {}
    if exif_data:
        save_kwargs["exif"] = exif_data
    output_image.save(output_path, **save_kwargs)

    preview = mat.copy()
    width = preview.shape[1] // 2
    preview[::, width:] = corrected_mat[::, width:]

    preview = cv2.resize(preview, (960, 540))

    return cv2.imencode('.png', preview)[1].tobytes()


def analyze_video(input_video_path, output_video_path):

    # Initialize new video writer
    cap = cv2.VideoCapture(input_video_path)
    fps = max(1, math.ceil(cap.get(cv2.CAP_PROP_FPS)))
    frame_count = max(1, math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    sample_seconds = _sample_seconds_from_env()
    sample_stride = max(1, fps * sample_seconds)

    # Get filter matrices for every 10th frame
    filter_matrix_indexes = []
    filter_matrices = []
    count = 0
    show_progress = _progress_enabled_from_env()
    video_label = _progress_desc_from_env(os.path.basename(input_video_path))
    progress_position = _progress_position_from_env()
    analyze_pbar = tqdm(
        total=frame_count,
        desc=f"{video_label} analyze",
        unit="frame",
        leave=False,
        position=progress_position,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    _write_progress_file("analyze", 0, frame_count)

    if not show_progress:
        print("Analyzing...")
    # Sequential scan is typically faster than random seeks on compressed videos.
    # Use grab/retrieve so we only decode full frames when sampling.
    while cap.isOpened():
        ok = cap.grab()
        if not ok:
            break

        count += 1
        analyze_pbar.update(1)
        if count % 10 == 0 or count == frame_count:
            _write_progress_file("analyze", count, frame_count)
        should_sample = (count % sample_stride == 0) or (count == frame_count)
        if not should_sample:
            continue

        ret, frame = cap.retrieve()
        if not show_progress:
            print(f"{count} frames", end="\r")
        if not ret:
            continue

        mat = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        filter_matrix_indexes.append(count)
        filter_matrices.append(get_filter_matrix(mat))
        yield count

    analyze_pbar.close()
    _write_progress_file("analyze", frame_count, frame_count)
    cap.release()

    # Fallback: if sampled seeks failed, compute from first readable frame.
    if not filter_matrices:
        cap = cv2.VideoCapture(input_video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            mat = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            filter_matrix_indexes = [1]
            filter_matrices = [get_filter_matrix(mat)]
        else:
            filter_matrix_indexes = [1]
            filter_matrices = [np.array([
                1, 0, 0, 0, 0,
                0, 1, 0, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 0, 1, 0,
            ], dtype=np.float32)]

    # Build an interpolation function to get filter matrix at any given frame
    filter_matrices = np.array(filter_matrices)

    yield {
        "input_video_path": input_video_path,
        "output_video_path": output_video_path,
        "fps": fps,
        "frame_count": frame_count,
        "filters": filter_matrices,
        "filter_indices": filter_matrix_indexes
    }

def precompute_filter_matrices(frame_count, filter_indices, filter_matrices):
    filter_matrix_size = len(filter_matrices[0])
    frame_numbers = np.arange(frame_count)
    interpolated_matrices = np.zeros((frame_count, filter_matrix_size))
    for x in range(filter_matrix_size):
        interpolated_matrices[:, x] = np.interp(frame_numbers, filter_indices, filter_matrices[:, x])
    return interpolated_matrices

def process_video(video_data, yield_preview=False):
    cap = cv2.VideoCapture(video_data["input_video_path"])
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_data["fps"]
    frame_count = max(1, int(video_data["frame_count"]))

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    new_video = cv2.VideoWriter(video_data["output_video_path"], fourcc, fps, (frame_width, frame_height))

    # Precompute interpolated filter matrices
    show_progress = _progress_enabled_from_env()
    if not show_progress:
        print("Precomputing filter matrices...")
    interpolated_matrices = precompute_filter_matrices(
        frame_count, video_data["filter_indices"], np.array(video_data["filters"])
    )

    if not show_progress:
        print("Processing...")
    video_label = _progress_desc_from_env(os.path.basename(video_data["input_video_path"]))
    progress_position = _progress_position_from_env()
    process_pbar = tqdm(
        total=frame_count,
        desc=f"{video_label} color",
        unit="frame",
        leave=False,
        position=progress_position,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    _write_progress_file("color", 0, frame_count)
    count = 0
    while cap.isOpened():
        count += 1
        percent = 100 * count / frame_count
        if not show_progress:
            print("{:.2f}%".format(percent), end="\r")
        ret, frame = cap.read()

        if not ret:
            # End video read if we have gone beyond reported frame count
            if count >= frame_count:
                break

            # Failsafe to prevent an infinite loop
            if count >= 1e6:
                break

            # Otherwise this is just a faulty frame read, try reading next
            continue

        # Apply the filter using precomputed matrix
        rgb_mat = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        matrix_idx = min(count - 1, frame_count - 1)
        corrected_mat = apply_filter(rgb_mat, interpolated_matrices[matrix_idx])
        corrected_mat = cv2.cvtColor(corrected_mat, cv2.COLOR_RGB2BGR)
        new_video.write(corrected_mat)
        process_pbar.update(1)
        if count % 10 == 0 or count == frame_count:
            _write_progress_file("color", count, frame_count)

        if yield_preview:
            preview = frame.copy()
            width = preview.shape[1] // 2
            height = preview.shape[0] // 2
            preview[:, width:] = corrected_mat[:, width:]

            preview = cv2.resize(preview, (width, height))

            yield percent, cv2.imencode('.png', preview)[1].tobytes()
        else:
            yield None

    process_pbar.close()
    _write_progress_file("color", frame_count, frame_count)
    cap.release()
    new_video.release()


if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print("Usage")
            print("-"*20)
            print("For image:")
            print("$python correct.py image <source_image_path> <output_image_path>\n")
            print("-"*20)
            print("For video:")
            print("$python correct.py video <source_video_path> <output_video_path>\n")
            exit(0)

        if (sys.argv[1]) == "image":
            mat = cv2.imread(sys.argv[2])
            mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)

            corrected_mat = correct(mat)

            cv2.imwrite(sys.argv[3], corrected_mat)

        else:

            for item in analyze_video(sys.argv[2], sys.argv[3]):

                if type(item) == dict:
                    video_data = item

            [x for x in process_video(video_data, yield_preview=False)]
    except KeyboardInterrupt:
        print("Interrupted.")
        raise SystemExit(130)

