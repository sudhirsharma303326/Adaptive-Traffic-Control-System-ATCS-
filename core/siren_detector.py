"""
=============================================================================
MODULE B: Acoustic Siren Frequency Detection Engine (FFT Signal Processing)
=============================================================================
Implements real-time audio signal processing using Fast Fourier Transform (FFT)
and spectral peak/SNR analysis in SciPy/NumPy to detect emergency vehicle
sirens (Ambulance, Fire Brigade, Police) in the 1000 Hz - 3000 Hz band.
Acts as a fail-safe multi-modal sensor when CCTV visibility is impaired
(e.g., dense fog, torrential rain, occlusion, or complete darkness).
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from scipy import signal
from config.settings import SIREN_CFG, SirenAcousticConfig


@dataclass
class SirenDetectionResult:
    detected: bool
    peak_frequency_hz: float
    snr_db: float
    confidence: float
    pattern_type: str  # "WAIL", "YELP", "HI-LO", "NONE"
    raw_magnitude_peak: float
    siren_band_power: float
    noise_floor_power: float


def detect_siren_frequency(
    audio_buffer: np.ndarray,
    sample_rate: int = 22050,
    target_band: Tuple[float, float] = (1000.0, 3000.0),
    snr_threshold_db: float = 8.5,
    cfg: Optional[SirenAcousticConfig] = None,
) -> SirenDetectionResult:
    """
    Core FFT Siren Detection Filter.
    
    Processes micro-array audio buffers using real Fast Fourier Transform (rfft)
    with a Hann window to analyze spectral energy in the emergency siren frequency
    band (1000 Hz - 3000 Hz) against ambient urban road noise.

    Args:
        audio_buffer: 1D numpy array of audio samples (float32 or int16 normalized to [-1, 1])
        sample_rate: Audio sampling frequency in Hz (default: 22050)
        target_band: Frequency range tuple (min_hz, max_hz) for siren spectrum
        snr_threshold_db: Threshold in decibels required to confirm acoustic siren presence
        cfg: Optional SirenAcousticConfig override

    Returns:
        SirenDetectionResult containing detection boolean, peak Hz, SNR (dB), and confidence.
    """
    if cfg is None:
        cfg = SIREN_CFG

    # 1. Validation & Pre-processing
    if audio_buffer is None or len(audio_buffer) == 0:
        return SirenDetectionResult(
            detected=False,
            peak_frequency_hz=0.0,
            snr_db=0.0,
            confidence=0.0,
            pattern_type="NONE",
            raw_magnitude_peak=0.0,
            siren_band_power=0.0,
            noise_floor_power=0.0,
        )

    # Convert integer PCM to float [-1.0, 1.0] if necessary
    samples = np.asarray(audio_buffer, dtype=np.float32)
    if np.max(np.abs(samples)) > 1.0:
        samples = samples / 32768.0

    # Ensure zero-mean signal (remove DC offset)
    samples = samples - np.mean(samples)
    n_samples = len(samples)

    if n_samples < 256:
        # Not enough samples for a valid FFT
        return SirenDetectionResult(
            detected=False,
            peak_frequency_hz=0.0,
            snr_db=0.0,
            confidence=0.0,
            pattern_type="NONE",
            raw_magnitude_peak=0.0,
            siren_band_power=0.0,
            noise_floor_power=0.0,
        )

    # 2. Windowing to prevent spectral leakage
    window = signal.windows.hann(n_samples)
    windowed_signal = samples * window

    # 3. Compute Real Fast Fourier Transform (rfft)
    fft_spectrum = np.fft.rfft(windowed_signal)
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    magnitude_spectrum = np.abs(fft_spectrum)

    # Power Spectral Density (PSD)
    psd = (magnitude_spectrum ** 2) / (n_samples * sample_rate)

    # 4. Target Siren Band (1000 Hz - 3000 Hz)
    min_hz, max_hz = target_band
    siren_mask = (frequencies >= min_hz) & (frequencies <= max_hz)

    # Reference Noise Bands (ambient road/engine rumble: 200 - 800 Hz and 3500 - 5500 Hz)
    noise_mask_low = (frequencies >= 200.0) & (frequencies <= 800.0)
    noise_mask_high = (frequencies >= 3500.0) & (frequencies <= min(5500.0, sample_rate / 2 - 100))
    noise_mask = noise_mask_low | noise_mask_high

    if not np.any(siren_mask):
        return SirenDetectionResult(
            detected=False,
            peak_frequency_hz=0.0,
            snr_db=0.0,
            confidence=0.0,
            pattern_type="NONE",
            raw_magnitude_peak=0.0,
            siren_band_power=0.0,
            noise_floor_power=0.0,
        )

    # Extract target band power and find peak frequency
    siren_psd = psd[siren_mask]
    siren_freqs = frequencies[siren_mask]

    peak_idx = np.argmax(siren_psd)
    peak_freq = float(siren_freqs[peak_idx])
    peak_power = float(siren_psd[peak_idx])
    siren_total_power = float(np.mean(siren_psd))

    # 5. Full spectrum power & Band Concentration Ratio
    total_power = float(np.mean(psd)) + 1e-12
    siren_energy_ratio = float(np.sum(psd[siren_mask])) / (float(np.sum(psd)) + 1e-12)

    # Calculate baseline ambient noise floor using full audio spectrum median
    noise_floor_power = float(np.median(psd)) + 1e-12

    # 6. Compute Signal-to-Noise Ratio (SNR) in Decibels
    snr_db = 10.0 * np.log10((peak_power + 1e-12) / noise_floor_power)

    # 7. Peak Prominence & Harmonic Concentration check
    spectral_sharpness = peak_power / (siren_total_power + 1e-12)

    # 8. Pattern Identification
    pattern_type = "NONE"
    if 1000.0 <= peak_freq < 1600.0:
        pattern_type = "WAIL"
    elif 1600.0 <= peak_freq <= 2800.0:
        pattern_type = "YELP"
    elif peak_freq > 2800.0:
        pattern_type = "HI-LO"

    # 9. Detection Decision Logic:
    # Requires SNR threshold, spectral peak sharpness, sufficient band energy ratio,
    # and minimal absolute acoustic energy to discard faint silence/leakage.
    is_detected = (
        (snr_db >= snr_threshold_db)
        and (spectral_sharpness >= 2.2)
        and (siren_energy_ratio >= 0.25)
        and (peak_power > 1e-6)
    )
    
    # Calculate confidence score between 0.0 and 1.0
    normalized_snr = np.clip((snr_db - snr_threshold_db) / 20.0, 0.0, 1.0)
    normalized_sharpness = np.clip((spectral_sharpness - 2.2) / 8.0, 0.0, 1.0)
    confidence = float(np.clip(0.6 * normalized_snr + 0.4 * normalized_sharpness, 0.0, 0.99)) if is_detected else 0.0

    return SirenDetectionResult(
        detected=is_detected,
        peak_frequency_hz=round(peak_freq, 1),
        snr_db=round(snr_db, 2),
        confidence=round(confidence, 3),
        pattern_type=pattern_type if is_detected else "NONE",
        raw_magnitude_peak=float(magnitude_spectrum[siren_mask][peak_idx]),
        siren_band_power=round(siren_total_power, 6),
        noise_floor_power=round(noise_floor_power, 6),
    )


class AcousticSirenMonitor:
    """
    Continuous Multi-Lane Acoustic Siren Monitoring Engine.
    Maintains rolling audio ring buffers and detects approaching emergency vehicles.
    """

    def __init__(self, sample_rate: int = 22050, buffer_duration_s: float = 0.5):
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_duration_s)
        self.recent_detections = []
        self.is_overridden = False

    def process_audio_chunk(self, audio_chunk: np.ndarray) -> SirenDetectionResult:
        """Processes real-time audio chunk from acoustic microphone array."""
        result = detect_siren_frequency(
            audio_buffer=audio_chunk,
            sample_rate=self.sample_rate,
        )
        if result.detected:
            self.recent_detections.append(result)
            if len(self.recent_detections) > 10:
                self.recent_detections.pop(0)
            self.is_overridden = True
        else:
            self.is_overridden = False
        return result

    @staticmethod
    def generate_synthetic_siren(
        duration_s: float = 1.0,
        sample_rate: int = 22050,
        siren_type: str = "yelp",
        noise_level: float = 0.08,
    ) -> np.ndarray:
        """
        Synthesizes a realistic emergency siren sound wave with ambient urban noise
        for offline unit testing and automated continuous integration.
        """
        t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
        if siren_type == "yelp":
            # Fast pitch sweep between 1000 Hz and 2500 Hz at ~2.5 Hz rate
            mod_freq = 2.5
            inst_freq = 1750.0 + 750.0 * np.sin(2 * np.pi * mod_freq * t)
            phase = 2 * np.pi * np.cumsum(inst_freq) / sample_rate
            siren_signal = 0.8 * np.sin(phase)
        elif siren_type == "wail":
            # Slower pitch sweep between 800 Hz and 1600 Hz at ~0.3 Hz rate
            mod_freq = 0.35
            inst_freq = 1200.0 + 400.0 * np.sin(2 * np.pi * mod_freq * t)
            phase = 2 * np.pi * np.cumsum(inst_freq) / sample_rate
            siren_signal = 0.8 * np.sin(phase)
        else:
            # Dual-tone Hi-Lo (police/ambulance standard)
            mod = np.where(np.sin(2 * np.pi * 1.5 * t) > 0, 1500.0, 950.0)
            phase = 2 * np.pi * np.cumsum(mod) / sample_rate
            siren_signal = 0.8 * np.sin(phase)

        # Ambient low-frequency road noise
        ambient_noise = np.random.normal(0, noise_level, size=len(t))
        # Filter noise to simulate low-rumble traffic
        b, a = signal.butter(3, 400.0 / (sample_rate / 2), btype="low")
        filtered_traffic_rumble = signal.lfilter(b, a, ambient_noise)

        composite = siren_signal + filtered_traffic_rumble
        return np.asarray(composite, dtype=np.float32)

    @staticmethod
    def generate_ambient_traffic_noise(
        duration_s: float = 1.0,
        sample_rate: int = 22050,
        noise_level: float = 0.20,
    ) -> np.ndarray:
        """Generates standard city traffic noise without any emergency sirens."""
        n_samples = int(sample_rate * duration_s)
        white_noise = np.random.normal(0, noise_level, size=n_samples)
        # Heavy low-pass filtering (< 500 Hz engine rumble)
        b, a = signal.butter(4, 500.0 / (sample_rate / 2), btype="low")
        rumble = signal.lfilter(b, a, white_noise)
        return np.asarray(rumble, dtype=np.float32)
