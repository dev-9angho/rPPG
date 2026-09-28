import time
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

# ----------------------------- 설정 -----------------------------
SOURCE = "camera"  # "camera" (웹캠) 또는 "video" (파일 + ground truth 비교)
VIDEO_PATH = "vid.avi"
GT_PATH = "ground_truth.txt"
CAMERA_INDEX = 1
FALLBACK_CAMERA_INDEX = 0
MIRROR_PREVIEW = True  # 웹캠일 때 거울처럼 좌우 반전해서 표시

WINDOW_NAME = "rPPG"
TARGET_WIDTH = 1280  # 카메라 창의 가로 크기(px). 세로는 영상 비율에 맞춰 자동 결정
SHOW_PEAK_LINE = True  # 스펙트럼 그래프의 빨간 피크선(추정 주파수) 표시 여부
FS = 30
PULSE_BAND_HZ = (0.7, 3.0)  # 42~180 BPM
WINDOW_SEC = 10.0
POS_WINDOW_SEC = 1.6
BUTTER_ORDER = 3
NFFT = 8192

MIN_ANALYSIS_TIME = 4.0
MIN_STABILITY_TIME = 8.0
MAX_DEADLINE_TIME = 25.0

ANALYSIS_INTERVAL = 0.5  # 분석 및 그래프 갱신 주기(초)
BPM_SAMPLE_INTERVAL = 1.0
STABILITY_WINDOW = 5
STABILITY_THRES_BPM = 2.5
SNR_MIN_DB = 0.0

DETECT_EVERY = 3
DETECT_SCALE = 0.5
MIN_NEIGHBORS = 5
BOX_EMA_ALPHA = 0.3
FACE_LOST_TIMEOUT = 1.0

ROI_BOXES = {
    "forehead": (0.30, 0.12, 0.40, 0.15),
    "left_cheek": (0.18, 0.55, 0.20, 0.18),
    "right_cheek": (0.62, 0.55, 0.20, 0.18),
}

FONT = cv2.FONT_HERSHEY_DUPLEX


# ----------------------------- 입력 -----------------------------
def open_camera():
    for idx in (CAMERA_INDEX, FALLBACK_CAMERA_INDEX):
        cam = cv2.VideoCapture(idx)
        if cam.isOpened():
            print(f"Camera opened (index {idx}), reported FPS: {cam.get(cv2.CAP_PROP_FPS):.1f}")
            return cam
        cam.release()
    raise RuntimeError("Cannot open camera.")


def open_video(path=VIDEO_PATH):
    cam = cv2.VideoCapture(path)
    if cam.isOpened():
        print(f"Video opened, reported FPS: {cam.get(cv2.CAP_PROP_FPS):.1f}")
        return cam
    cam.release()
    raise RuntimeError("Cannot open video.")


def load_ground_truth(path=GT_PATH, video_duration=None):
    """UBFC 형식: 1행 PPG, 2행 HR(BPM), 3행 시간(초)."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.split() for ln in f if ln.strip()]
    if len(lines) < 2:
        raise ValueError("ground_truth.txt 는 최소 2행(PPG, HR)이 필요합니다.")
    hr = np.array(lines[1], float)
    if len(lines) >= 3 and len(lines[2]) == len(hr):
        t = np.array(lines[2], float)
    else:
        dur = video_duration if video_duration else len(hr) / FS
        t = np.linspace(0, dur, len(hr))
    return t, hr


def truth_in_range(gt_t, gt_hr, t0, t1):
    m = (gt_t >= t0) & (gt_t <= t1)
    if not np.any(m):
        return None
    return float(gt_hr[m].mean())


# ----------------------------- 얼굴/ROI -----------------------------
class FaceTracker:
    def __init__(self):
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.box = None
        self.last_seen = 0.0
        self.frame_count = 0

    # update returns the current face box (x, y, w, h) or None if no face is detected
    def update(self, frame, now):
        # 프레임 마다 얼굴 검출을 수행하면 느리므로, 일정 프레임마다만 수행하고 나머지는 이전 박스를 유지
        self.frame_count += 1
        if self.frame_count % DETECT_EVERY == 1 or self.box is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, None, fx=DETECT_SCALE, fy=DETECT_SCALE)
            faces = self.detector.detectMultiScale(
                small, scaleFactor=1.1, minNeighbors=MIN_NEIGHBORS,
                minSize=(60, 60),
            )
            if len(faces) > 0:
                new = np.array(max(faces, key=lambda f: f[2] * f[3]), float) / DETECT_SCALE
                if self.box is None:
                    self.box = new
                else:
                    self.box = BOX_EMA_ALPHA * new + (1 - BOX_EMA_ALPHA) * self.box
                self.last_seen = now

        if self.box is not None and now - self.last_seen > FACE_LOST_TIMEOUT:
            self.box = None
        return None if self.box is None else self.box.astype(int)


def roi_from_face(face_box, fractions, frame_shape):
    x, y, w, h = face_box
    fx, fy, fw, fh = fractions
    H, W = frame_shape[:2]
    x0 = max(int(x + fx * w), 0)
    y0 = max(int(y + fy * h), 0)
    x1 = min(int(x + (fx + fw) * w), W)
    y1 = min(int(y + (fy + fh) * h), H)
    return x0, y0, x1, y1


# ----------------------------- 신호 처리 -----------------------------
def resample_uniform(timestamps, rgb_values, fs):
    duration = timestamps[-1] - timestamps[0]
    n = max(int(duration * fs), 2)
    t_u = np.linspace(timestamps[0], timestamps[-1], n)
    rgb_u = np.stack(
        [np.interp(t_u, timestamps, rgb_values[:, c]) for c in range(3)], axis=1
    )
    return t_u, rgb_u


def pos_algorithm(rgb, fs, win_sec=POS_WINDOW_SEC):
    """벡터화된 슬라이딩 윈도우 POS (Wang et al., 2017) + overlap-add."""
    n = len(rgb)
    l = int(win_sec * fs)
    if n < l:
        return np.zeros(n)

    idx = np.arange(l)[None, :] + np.arange(n - l + 1)[:, None]  # (W, l)
    c = rgb[idx]  # (W, l, 3)
    mean = c.mean(axis=1, keepdims=True)
    mean[mean == 0] = 1.0
    c = c / mean
    s0 = c[..., 1] - c[..., 2]
    s1 = -2.0 * c[..., 0] + c[..., 1] + c[..., 2]
    h = s0 + (s0.std(axis=1) / (s1.std(axis=1) + 1e-6))[:, None] * s1
    h -= h.mean(axis=1, keepdims=True)
    out = np.zeros(n)
    np.add.at(out, idx, h)
    return out


# butterworth 필터링
_SOS = butter(BUTTER_ORDER, PULSE_BAND_HZ, btype="band", fs=FS, output="sos")

# Bandpass filter
def bandpass(x):
    if len(x) < 3 * (2 * _SOS.shape[0] + 1) + 1:
        return x - np.mean(x)
    return sosfiltfilt(_SOS, x - np.mean(x))


def estimate_bpm(signal, fs):
    # 신호가 들어오면 hanning window 적용
    windowed = signal * np.hanning(len(signal))
    # FFT 길이는 NFFT와 windowed 길이 중 큰 값으로 설정
    n_fft = max(NFFT, len(windowed))
    # FFT 계산 및 주파수 스펙트럼 추출
    spec = np.abs(np.fft.rfft(windowed, n=n_fft))
    # 주파수 벡터 생성
    freqs = np.fft.rfftfreq(n_fft, d=1 / fs)

    # 0.7hz ~ 3hz 
    lo, hi = PULSE_BAND_HZ
    band = (freqs >= lo) & (freqs <= hi)
    # 신호가 pulse band 안에 없으면 ValueError 발생
    if not np.any(band):
        raise ValueError("No FFT bins inside pulse band.")

    # 피크 주파수 추정
    band_idx = np.where(band)[0]
    # 피크 주파수 및 SNR 계산
    peak_idx = band_idx[np.argmax(spec[band_idx])]
    peak_hz = freqs[peak_idx]
    # SNR 계산
    p = spec**2

    # 신호 마스크 생성: 피크 주파수 및 그 배수 주파수
    sig_mask = (np.abs(freqs - peak_hz) <= 0.1) | (np.abs(freqs - 2 * peak_hz) <= 0.1)

    # 신호 전력 및 잡음 전력 계산
    signal_pow = p[sig_mask & (freqs <= 2 * hi)].sum()

    # noise power 계산: 신호 마스크를 제외한 pulse band 내의 전력 합
    noise_pow = p[band & ~sig_mask].sum()

    # SNR 계산 
    snr_db = 10 * np.log10((signal_pow + 1e-12) / (noise_pow + 1e-12))

    # BPM 계산: 피크 주파수를 분당 심박수로 변환
    return peak_hz * 60.0, freqs, spec, snr_db


def analyze_window(ts, rgb):
    ts = np.asarray(ts)
    rgb = np.asarray(rgb)
    if len(ts) < 30 or ts[-1] - ts[0] < MIN_ANALYSIS_TIME:
        return None
    t_u, rgb_u = resample_uniform(ts, rgb, FS)
    pulse = bandpass(pos_algorithm(rgb_u, FS))
    bpm, freqs, spec, snr = estimate_bpm(pulse, FS)
    return {"t": t_u, "pulse": pulse, "bpm": bpm, "freqs": freqs, "spec": spec, "snr": snr}


# ----------------------------- 세션 상태 -----------------------------
class Session:
    STANDBY, MEASURING, DONE = "standby", "measuring", "done"

    def __init__(self):
        self.state = Session.STANDBY
        self.start_time = 0.0
        self.end_time = None
        self.elapsed = 0.0
        self.current_bpm = 0.0
        self.final_bpm = 0.0
        self.snr = 0.0
        self.is_stable = False
        self.truth = None
        self.ts = deque()
        self.rgb = deque()
        self.bpm_history = deque(maxlen=STABILITY_WINDOW)
        self.last_analysis = 0.0
        self.last_bpm_sample = 0.0

    def start(self, now):
        self.__init__()
        self.state = Session.MEASURING
        self.start_time = now

    def finish(self, now):
        self.state = Session.DONE
        self.end_time = now

    def add_sample(self, rgb):
        self.ts.append(self.elapsed)
        self.rgb.append(rgb)
        cutoff = self.elapsed - WINDOW_SEC
        while self.ts and self.ts[0] < cutoff:
            self.ts.popleft()
            self.rgb.popleft()


# ----------------------------- 그래프 -----------------------------
def setup_plot():
    plt.ion()
    fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(8, 6))
    (line_t,) = ax_t.plot([], [], color="red")
    ax_t.set_title("Real-time Pulse Signal (POS + Bandpass)")
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel("Amplitude")
    ax_t.grid(True)

    (line_f,) = ax_f.plot([], [], color="blue", label="Spectrum")
    peak_line = ax_f.axvline(0, color="red", linestyle="--", label="Peak")
    peak_line.set_visible(SHOW_PEAK_LINE)
    ax_f.axvspan(*PULSE_BAND_HZ, color="green", alpha=0.1, label="Pulse Band")
    ax_f.set_xlim(0, 5)
    ax_f.set_title("Real-time Frequency Spectrum")
    ax_f.set_xlabel("Frequency (Hz)")
    ax_f.set_ylabel("Magnitude")
    ax_f.legend(loc="upper right")
    ax_f.grid(True)
    plt.tight_layout()
    return fig, ax_t, ax_f, line_t, line_f, peak_line


def reset_plot(fig, line_t, line_f, peak_line):
    """새 측정을 시작할 때 이전 그래프 내용을 지운다."""
    line_t.set_data([], [])
    line_f.set_data([], [])
    peak_line.set_xdata([0])
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


# ----------------------------- 화면 -----------------------------
def put_centered(img, text, cy, scale, color, thickness):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x = (img.shape[1] - tw) // 2
    cv2.putText(img, text, (x, cy + th // 2), FONT, scale, color, thickness, cv2.LINE_AA)


def render(preview, s, face_found):
    H, W = preview.shape[:2]
    u = H / 480.0  # 해상도에 비례한 크기 단위

    if s.state == Session.STANDBY:
        put_centered(preview, "Press SPACE to start", int(H * 0.5), 1.2 * u, (255, 255, 255), max(2, int(2 * u)))
        put_centered(preview, "Face the camera and stay still", int(H * 0.58), 0.7 * u, (200, 200, 200), max(1, int(u)))

    elif s.state == Session.MEASURING:
        put_centered(preview, "Measuring...", int(H * 0.10), 1.2 * u, (255, 255, 255), max(2, int(2 * u)))
        put_centered(preview, f"{s.elapsed:.0f}s", int(H * 0.19), 0.9 * u, (255, 255, 0), max(1, int(2 * u)))
        if not face_found:
            put_centered(preview, "No face detected", int(H * 0.5), 1.0 * u, (0, 0, 255), max(2, int(2 * u)))
        else:
            put_centered(preview, "Hold still", int(H * 0.92), 0.8 * u, (200, 200, 200), max(1, int(u)))
        # 진행 바
        frac = min(s.elapsed / MAX_DEADLINE_TIME, 1.0)
        bar_h = max(6, int(10 * u))
        cv2.rectangle(preview, (0, H - bar_h), (int(W * frac), H), (0, 255, 0), -1)

    else:  # DONE
        preview[:] = (preview * 0.30).astype(np.uint8)
        if s.final_bpm > 0:
            put_centered(preview, f"{s.final_bpm:.0f}", int(H * 0.45), 9.0 * u, (0, 255, 255), max(4, int(14 * u)))
            put_centered(preview, "BPM", int(H * 0.72), 2.5 * u, (255, 255, 255), max(2, int(4 * u)))
            if not s.is_stable:
                put_centered(preview, "low confidence - try again", int(H * 0.82), 0.8 * u, (0, 165, 255), max(1, int(2 * u)))
            if s.truth is not None:
                put_centered(preview, f"truth {s.truth:.0f} BPM", int(H * 0.89), 0.8 * u, (255, 200, 0), max(1, int(2 * u)))
        else:
            put_centered(preview, "Measurement failed", int(H * 0.45), 1.8 * u, (0, 0, 255), max(2, int(4 * u)))
        put_centered(preview, "SPACE: retry    Q: quit", int(H * 0.95), 0.7 * u, (200, 200, 200), max(1, int(u)))


# ----------------------------- 메인 -----------------------------
def main():
    use_video = SOURCE == "video"
    camera = open_video() if use_video else open_camera()

    video_fps = camera.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 1:
        video_fps = 30.0

    gt_t = gt_hr = None
    if use_video:
        n_frames = camera.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = n_frames / video_fps if n_frames > 0 else None
        gt_t, gt_hr = load_ground_truth(video_duration=duration)

    tracker = FaceTracker()
    fig, ax_t, ax_f, line_t, line_f, peak_line = setup_plot()
    s = Session()

    # 전체화면이 아닌 큰 일반 창 (크기는 첫 프레임을 받은 뒤 영상 비율에 맞춰 설정)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    window_sized = False

    t_origin = time.perf_counter()
    frame_idx = 0
    now = 0.0

    print("SPACE : Start / Retry Q or ESC : Quit")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                if s.state == Session.MEASURING:
                    s.final_bpm = s.current_bpm
                    s.finish(now)
                    s.truth = _truth(s, gt_t, gt_hr)
                if use_video:
                    # 영상 끝: 결과 화면을 유지하려고 마지막 프레임 없이 종료
                    break
                print("Camera read failed.")
                break

            if not window_sized:
                fh_, fw_ = frame.shape[:2]
                cv2.resizeWindow(WINDOW_NAME, TARGET_WIDTH, int(TARGET_WIDTH * fh_ / fw_))
                window_sized = True

            # 웹캠: 벽시계 시간 / 영상 파일: 프레임 번호 기반 영상 시간
            if use_video:
                now = frame_idx / video_fps
            else:
                now = time.perf_counter() - t_origin
            frame_idx += 1

            preview = frame.copy()
            if s.state == Session.MEASURING:
                s.elapsed = now - s.start_time

            face = tracker.update(frame, now)
            face_found = face is not None
            if face_found:
                fx, fy, fw, fh = face
                cv2.rectangle(preview, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

                if s.state == Session.MEASURING:
                    rgbs = []
                    for frac in ROI_BOXES.values():
                        x0, y0, x1, y1 = roi_from_face(face, frac, frame.shape)
                        if x1 <= x0 or y1 <= y0:
                            continue
                        roi = frame[y0:y1, x0:x1]
                        b, g, r = roi.reshape(-1, 3).mean(axis=0)
                        rgbs.append([r, g, b])
                    if rgbs:
                        s.add_sample(np.mean(rgbs, axis=0))

            # 백그라운드 분석 (화면에는 표시하지 않음)
            if (s.state == Session.MEASURING and s.elapsed >= MIN_ANALYSIS_TIME
                    and s.elapsed - s.last_analysis >= ANALYSIS_INTERVAL):
                s.last_analysis = s.elapsed
                try:
                    res = analyze_window(list(s.ts), list(s.rgb))
                except ValueError:
                    res = None

                if res is not None:
                    s.current_bpm = res["bpm"]
                    s.snr = res["snr"]

                    if (s.elapsed >= MIN_STABILITY_TIME # 8초 이후
                            and s.elapsed - s.last_bpm_sample >= BPM_SAMPLE_INTERVAL): # 1초 간격
                        s.last_bpm_sample = s.elapsed
                        # SNR이 0DB 이상
                        if res["snr"] >= SNR_MIN_DB:
                            s.bpm_history.append(res["bpm"])
                        else:
                            s.bpm_history.clear()

                        if len(s.bpm_history) == STABILITY_WINDOW:
                            rng = max(s.bpm_history) - min(s.bpm_history)
                            if rng <= STABILITY_THRES_BPM:
                                s.is_stable = True
                                s.final_bpm = float(np.median(s.bpm_history))

                    # 그래프 갱신
                    sig = res["pulse"]
                    span = max(np.ptp(sig), 1e-6)
                    line_t.set_data(res["t"], sig)
                    ax_t.set_xlim(res["t"][0], max(res["t"][-1], res["t"][0] + 1))
                    ax_t.set_ylim(sig.min() - 0.1 * span, sig.max() + 0.1 * span)

                    m = res["freqs"] <= 5
                    line_f.set_data(res["freqs"][m], res["spec"][m])
                    peak_line.set_xdata([res["bpm"] / 60])
                    ax_f.set_ylim(0, res["spec"][m].max() * 1.2 + 1e-3)
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()

            # 종료 조건
            if s.state == Session.MEASURING:
                if s.is_stable:
                    s.finish(now)
                    s.truth = _truth(s, gt_t, gt_hr)
                    print(f"[STABLE] {s.final_bpm:.1f} BPM at {s.elapsed:.1f}s (SNR {s.snr:.1f} dB)")
                elif s.elapsed >= MAX_DEADLINE_TIME:
                    s.final_bpm = (float(np.median(s.bpm_history)) if s.bpm_history
                                   else s.current_bpm)
                    s.finish(now)
                    s.truth = _truth(s, gt_t, gt_hr)
                    print(f"[TIMEOUT] {s.final_bpm:.1f} BPM (not stable)")

            if MIRROR_PREVIEW and not use_video:
                preview = cv2.flip(preview, 1)  # 텍스트를 그리기 전에 반전
            render(preview, s, face_found)
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if s.state == Session.MEASURING:
                    # 중간 취소: 추정값이 있으면 결과 표시, 없으면 대기 상태로
                    if s.current_bpm > 0:
                        s.final_bpm = s.current_bpm
                        s.finish(now)
                        s.truth = _truth(s, gt_t, gt_hr)
                    else:
                        s = Session()
                else:
                    s.start(now)
                    reset_plot(fig, line_t, line_f, peak_line)
                    print("Started...")
            elif key in (ord("q"), ord("Q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        plt.ioff()

    if s.final_bpm > 0:
        line = f"Final BPM: {s.final_bpm:.1f}"
        if s.truth is not None:
            line += f" | Truth: {s.truth:.1f} | Error: {abs(s.final_bpm - s.truth):.1f}"
        print(line)
    plt.show()  # 마지막 그래프를 확인할 수 있게 창 유지 (닫으면 종료)


def _truth(s, gt_t, gt_hr):
    if gt_t is None or s.end_time is None:
        return None
    t1 = s.end_time
    t0 = max(s.start_time, t1 - WINDOW_SEC)
    return truth_in_range(gt_t, gt_hr, t0, t1)


if __name__ == "__main__":
    main()
