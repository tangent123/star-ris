
import numpy as np
from config import SimConfig as C
from channels import bs_to_ris_element_gains, ris_to_point_element_gains

class StarRIS:
    def __init__(self, N=None, xi=None, phase=None, rng=None):
        self.N = int(N if N is not None else C.N_RIS)
        self.xi = float(C.XI_INIT if xi is None else xi)
        self.N_tz = int(round(self.xi * self.N))
        self.N_hz = self.N - self.N_tz
        if self.N_hz < 0: self.N_hz = 0
        if self.N_tz < 0: self.N_tz = 0
        self.phase = np.full(self.N_hz, float(C.PHASE_INIT))
        self.rng = np.random.default_rng(C.RNG_SEED if rng is None else rng)

    def repartition(self, xi):
        self.xi = np.clip(xi, 0.0, 1.0)
        self.N_tz = int(round(self.xi * self.N))
        self.N_hz = self.N - self.N_tz
        if len(self.phase) != self.N_hz:
            new_phase = np.zeros(self.N_hz)
            k = min(len(self.phase), self.N_hz)
            new_phase[:k] = self.phase[:k]
            self.phase = new_phase

    def effective_snr_terms(self, bs_pos, ris_pos, users_R, users_T, p_tx_W):
        N_hz = max(self.N_hz, 0)
        if N_hz == 0:
            return np.zeros(len(users_R) + len(users_T))
        g_bs_elem = bs_to_ris_element_gains(bs_pos, ris_pos, N_hz)
        users = list(users_R) + list(users_T)
        snr_terms = []
        for u in users:
            g_ru_elem = ris_to_point_element_gains(ris_pos, u, N_hz)
            amp_elems = np.sqrt(np.maximum(g_bs_elem,0) * np.maximum(g_ru_elem,0))
            coherent_amp = np.sum(amp_elems * np.exp(1j * self.phase))
            power_gain = np.abs(coherent_amp) ** 2
            snr_terms.append(p_tx_W * power_gain)
        return np.array(snr_terms)

    def energy_harvested(self, bs_pos, ris_pos, uav_pos, p_tx_W):
        N_tz = max(self.N_tz, 0)
        if N_tz == 0:
            return 0.0
        g_bs_elem = bs_to_ris_element_gains(bs_pos, ris_pos, N_tz)
        g_ru_elem = ris_to_point_element_gains(ris_pos, uav_pos, N_tz)
        amp_elems = np.sqrt(np.maximum(g_bs_elem,0) * np.maximum(g_ru_elem,0))
        coherent_amp = np.sum(amp_elems)
        power_gain = np.abs(coherent_amp) ** 2
        return C.ETA_EH * p_tx_W * power_gain

    def gradient_phase_step(self, bs_pos, ris_pos, users_R, users_T, p_tx_W, weights, noise_W, step):
        if self.N_hz == 0:
            return 0.0
        eps = 1e-3
        base_terms = self.effective_snr_terms(bs_pos, ris_pos, users_R, users_T, p_tx_W)
        base_rates = np.sum([w * np.log2(1 + s/(noise_W)) for w,s in zip(weights, base_terms)])
        grad = np.zeros_like(self.phase)
        for i in range(self.N_hz):
            old = self.phase[i]
            self.phase[i] = old + eps
            terms = self.effective_snr_terms(bs_pos, ris_pos, users_R, users_T, p_tx_W)
            new_rates = np.sum([w * np.log2(1 + s/(noise_W)) for w,s in zip(weights, terms)])
            grad[i] = (new_rates - base_rates) / eps
            self.phase[i] = old
        self.phase += step * grad
        self.phase = ( (self.phase + np.pi) % (2*np.pi) ) - np.pi
        return float(np.linalg.norm(grad))
