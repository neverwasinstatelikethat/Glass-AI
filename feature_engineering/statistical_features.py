"""
Statistical Feature Engineering for Glass Production Data
Advanced statistical features for predictive analytics and anomaly detection
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
import numpy as np
from scipy import stats, signal
from collections import deque, defaultdict
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalFeatureExtractor:
    """Advanced statistical feature extraction for industrial time series"""
    
    def __init__(
        self,
        window_size: int = 100,
        sampling_rate: float = 1.0,  # Hz
        feature_callback: Optional[Callable] = None
    ):
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.feature_callback = feature_callback
        
        # Data windows for each sensor
        self.data_windows: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        
        # Statistical feature configuration
        self.feature_config = {
            "moments": ["mean", "variance", "skewness", "kurtosis"],
            "quantiles": [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
            "entropy": ["sample_entropy", "approximate_entropy"],
            "hurst": ["hurst_exponent"],
            "fractal": ["fractal_dimension"],
            "wavelet": ["wavelet_energy", "wavelet_entropy"],
            "autocorr": ["autocorrelation_1", "autocorrelation_5", "autocorrelation_10"],
            "spectral": ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff"]
        }
    
    async def update_with_data(self, sensor_name: str, value: float, timestamp: Optional[datetime] = None):
        """Update statistical features with new data point"""
        try:
            if timestamp is None:
                timestamp = datetime.utcnow()
            
            # Add to data window
            self.data_windows[sensor_name].append((timestamp, float(value)))
            
            # Compute statistical features when window is sufficiently filled
            if len(self.data_windows[sensor_name]) >= 20:
                features = await self.compute_statistical_features(sensor_name)
                
                # Call callback if provided
                if self.feature_callback:
                    await self.feature_callback(sensor_name, features)
                
                return features
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Error updating {sensor_name}: {e}")
            return {}
    
    async def compute_statistical_features(self, sensor_name: str) -> Dict[str, float]:
        """Compute comprehensive statistical features for a sensor"""
        try:
            window = self.data_windows[sensor_name]
            if len(window) < 5:
                return {}
            
            # Extract values
            values = np.array([v for t, v in window])
            values = values[~np.isnan(values)]
            
            if len(values) < 3:
                return {}
            
            timestamp = datetime.utcnow()
            features = {
                "sensor_name": sensor_name,
                "timestamp": timestamp.isoformat(),
                "sample_count": len(values)
            }
            
            # Basic moments
            features.update(self._compute_moments(values))
            
            # Quantile features
            features.update(self._compute_quantiles(values))
            
            # Entropy features
            features.update(self._compute_entropy_features(values))
            
            # Hurst exponent
            features.update(self._compute_hurst_exponent(values))
            
            # Fractal dimension
            features.update(self._compute_fractal_dimension(values))
            
            # Wavelet features
            features.update(self._compute_wavelet_features(values))
            
            # Autocorrelation features
            features.update(self._compute_autocorrelation_features(values))
            
            # Spectral features
            features.update(self._compute_spectral_features(values))
            
            # Add metadata
            features["computation_time"] = datetime.utcnow().isoformat()
            features["window_size"] = len(window)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error computing statistical features for {sensor_name}: {e}")
            return {
                "sensor_name": sensor_name,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    def _compute_moments(self, values: np.ndarray) -> Dict[str, float]:
        """Compute statistical moments"""
        features = {}
        
        # Basic moments
        features["mean"] = float(np.mean(values))
        features["variance"] = float(np.var(values))
        features["std"] = float(np.std(values))
        
        # Higher-order moments
        if len(values) >= 4:
            features["skewness"] = float(stats.skew(values))
            features["kurtosis"] = float(stats.kurtosis(values))
        
        # Absolute moments
        abs_values = np.abs(values)
        features["mean_absolute"] = float(np.mean(abs_values))
        features["std_absolute"] = float(np.std(abs_values))
        
        # Root mean square
        features["rms"] = float(np.sqrt(np.mean(values**2)))
        
        # Crest factor
        if features["rms"] != 0:
            features["crest_factor"] = float(np.max(np.abs(values)) / features["rms"])
        
        return features
    
    def _compute_quantiles(self, values: np.ndarray) -> Dict[str, float]:
        """Compute quantile-based features"""
        features = {}
        
        # Standard quantiles
        quantiles = np.quantile(values, self.feature_config["quantiles"])
        for i, q in enumerate(self.feature_config["quantiles"]):
            features[f"quantile_{int(q*100)}"] = float(quantiles[i])
        
        # Interquartile range
        q25 = np.percentile(values, 25)
        q75 = np.percentile(values, 75)
        features["iqr"] = float(q75 - q25)
        
        # Quartile coefficient of dispersion
        if (q75 + q25) != 0:
            features["quartile_dispersion"] = float((q75 - q25) / (q75 + q25))
        
        return features
    
    def _compute_entropy_features(self, values: np.ndarray) -> Dict[str, float]:
        """Compute entropy-based features"""
        features = {}
        
        # Sample entropy
        try:
            samp_en = self._sample_entropy(values, m=2, r=0.2 * np.std(values))
            features["sample_entropy"] = float(samp_en)
        except Exception as e:
            logger.debug(f"⚠️ Sample entropy computation failed: {e}")
        
        # Approximate entropy
        try:
            ap_en = self._approximate_entropy(values, m=2, r=0.2 * np.std(values))
            features["approximate_entropy"] = float(ap_en)
        except Exception as e:
            logger.debug(f"⚠️ Approximate entropy computation failed: {e}")
        
        return features
    
    def _sample_entropy(self, values: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Compute sample entropy"""
        N = len(values)
        if N <= m + 1:
            return 0
        
        # Create templates
        def _maxdist(xi):
            return max([np.abs(xi[:m] - xj[:m]).max() for xj in [values[j:j+m] for j in range(N-m+1)] if not np.array_equal(xi[:m], xj[:m])])
        
        B = (N - m) * (N - m - 1) / 2
        A = 0
        for i in range(N - m):
            xi = values[i:i+m+1]
            for j in range(i+1, N - m):
                xj = values[j:j+m+1]
                if np.abs(xi[:m] - xj[:m]).max() <= r:
                    if np.abs(xi[m] - xj[m]) <= r:
                        A += 1
        
        if A == 0:
            return np.inf
        else:
            return -np.log(A / B)
    
    def _approximate_entropy(self, values: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Compute approximate entropy"""
        N = len(values)
        if N <= m + 1:
            return 0
        
        def _phi(m):
            A = np.zeros(N - m)
            for i in range(N - m):
                template = values[i:i+m]
                A[i] = np.sum([np.abs(template - values[j:j+m]).max() <= r 
                              for j in range(N - m)]) / (N - m)
            return (N - m - 1) * np.mean(np.log(A[A > 0]))
        
        return _phi(m) - _phi(m + 1)
    
    def _compute_hurst_exponent(self, values: np.ndarray) -> Dict[str, float]:
        """Compute Hurst exponent for long-term memory analysis"""
        features = {}
        
        try:
            # Rescaled range analysis
            n = len(values)
            if n < 10:
                return features
            
            # Divide into segments
            max_k = int(np.log2(n))
            rs_values = []
            segment_sizes = []
            
            for k in range(1, max_k):
                segment_size = 2**k
                if segment_size > n // 4:
                    break
                    
                segment_sizes.append(segment_size)
                rs_sum = 0
                num_segments = n // segment_size
                
                for i in range(num_segments):
                    segment = values[i*segment_size:(i+1)*segment_size]
                    if len(segment) == segment_size:
                        # Mean adjusted series
                        mean_adj = segment - np.mean(segment)
                        # Cumulative deviate series
                        cum_dev = np.cumsum(mean_adj)
                        # Range
                        r = np.max(cum_dev) - np.min(cum_dev)
                        # Standard deviation
                        s = np.std(segment)
                        if s > 0:
                            rs_sum += r / s
                
                if num_segments > 0:
                    rs_values.append(rs_sum / num_segments)
            
            # Linear regression to find Hurst exponent
            if len(rs_values) >= 3:
                log_rs = np.log(rs_values)
                log_n = np.log(segment_sizes)
                
                # Remove infinities and NaNs
                valid_mask = np.isfinite(log_rs) & np.isfinite(log_n)
                if np.sum(valid_mask) >= 3:
                    slope, _, _, _, _ = stats.linregress(log_n[valid_mask], log_rs[valid_mask])
                    features["hurst_exponent"] = float(slope)
        
        except Exception as e:
            logger.debug(f"⚠️ Hurst exponent computation failed: {e}")
        
        return features
    
    def _compute_fractal_dimension(self, values: np.ndarray) -> Dict[str, float]:
        """Compute fractal dimension using box-counting method"""
        features = {}
        
        try:
            # Normalize values
            normalized = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-10)
            
            # Box counting method
            box_sizes = [2**i for i in range(1, min(6, int(np.log2(len(normalized))) - 1))]
            counts = []
            
            for box_size in box_sizes:
                # Count non-empty boxes
                num_boxes = int(np.ceil(len(normalized) / box_size))
                box_count = 0
                
                for i in range(num_boxes):
                    start_idx = i * box_size
                    end_idx = min((i + 1) * box_size, len(normalized))
                    box_values = normalized[start_idx:end_idx]
                    
                    if len(box_values) > 0 and np.max(box_values) - np.min(box_values) > 0:
                        box_count += 1
                
                counts.append(box_count)
            
            # Linear regression to find fractal dimension
            if len(counts) >= 3:
                log_counts = np.log(counts)
                log_sizes = np.log([1/bs for bs in box_sizes[:len(counts)]])
                
                valid_mask = np.isfinite(log_counts) & np.isfinite(log_sizes)
                if np.sum(valid_mask) >= 3:
                    slope, _, _, _, _ = stats.linregress(log_sizes[valid_mask], log_counts[valid_mask])
                    features["fractal_dimension"] = float(-slope)
        
        except Exception as e:
            logger.debug(f"⚠️ Fractal dimension computation failed: {e}")
        
        return features
    
    def _compute_wavelet_features(self, values: np.ndarray) -> Dict[str, float]:
        """Compute wavelet-based features"""
        features = {}
        
        try:
            # Simple Haar wavelet transform
            if len(values) >= 8:
                # Decompose using Haar wavelet
                coeffs = pywt.wavedec(values, 'haar', level=min(3, int(np.log2(len(values))) - 1))
                
                # Energy of detail coefficients
                detail_energies = []
                for i, coeff in enumerate(coeffs[1:], 1):  # Skip approximation coefficients
                    energy = np.sum(coeff**2)
                    features[f"wavelet_detail_energy_{i}"] = float(energy)
                    detail_energies.append(energy)
                
                # Total wavelet energy
                if detail_energies:
                    features["wavelet_energy"] = float(np.sum(detail_energies))
                
                # Wavelet entropy
                total_energy = np.sum([e for e in detail_energies if e > 0])
                if total_energy > 0:
                    probabilities = [e / total_energy for e in detail_energies if e > 0]
                    entropy = -np.sum([p * np.log(p) for p in probabilities if p > 0])
                    features["wavelet_entropy"] = float(entropy)
        
        except Exception as e:
            logger.debug(f"⚠️ Wavelet features computation failed: {e}")
        
        return features
    
    def _compute_autocorrelation_features(self, values: np.ndarray) -> Dict[str, float]:
        """Compute autocorrelation-based features"""
        features = {}
        
        try:
            # Compute autocorrelation
            autocorr = np.correlate(values - np.mean(values), values - np.mean(values), mode='full')
            autocorr = autocorr[len(autocorr)//2:] / autocorr[len(autocorr)//2]
            
            # Extract specific lags
            lags = [1, 5, 10]
            for lag in lags:
                if len(autocorr) > lag:
                    features[f"autocorrelation_{lag}"] = float(autocorr[lag])
            
            # First zero crossing
            first_zero = np.where(np.diff(np.sign(autocorr[1:])))[0]
            if len(first_zero) > 0:
                features["first_zero_crossing"] = float(first_zero[0] + 1)
        
        except Exception as e:
            logger.debug(f"⚠️ Autocorrelation features computation failed: {e}")
        
        return features
    
    def _compute_spectral_features(self, values: np.ndarray) -> Dict[str, float]:
        """Compute spectral features using FFT"""
        features = {}
        
        try:
            # Apply FFT
            fft_vals = np.fft.fft(values)
            magnitude = np.abs(fft_vals)[:len(values)//2]
            frequencies = np.fft.fftfreq(len(values), d=1/self.sampling_rate)[:len(values)//2]
            
            if len(magnitude) > 0:
                # Spectral centroid
                if np.sum(magnitude) > 0:
                    centroid = np.sum(frequencies * magnitude) / np.sum(magnitude)
                    features["spectral_centroid"] = float(centroid)
                
                # Spectral bandwidth
                if "spectral_centroid" in features and np.sum(magnitude) > 0:
                    bandwidth = np.sqrt(np.sum(((frequencies - features["spectral_centroid"])**2) * magnitude) / np.sum(magnitude))
                    features["spectral_bandwidth"] = float(bandwidth)
                
                # Spectral rolloff (95% of energy)
                cumsum = np.cumsum(magnitude**2)
                if cumsum[-1] > 0:
                    rolloff_idx = np.where(cumsum >= 0.95 * cumsum[-1])[0]
                    if len(rolloff_idx) > 0:
                        features["spectral_rolloff"] = float(frequencies[rolloff_idx[0]])
                
                # Spectral flux
                if len(magnitude) > 1:
                    flux = np.sum(np.diff(magnitude)**2)
                    features["spectral_flux"] = float(flux)
        
        except Exception as e:
            logger.debug(f"⚠️ Spectral features computation failed: {e}")
        
        return features
    
    async def compute_multivariate_features(self, sensor_data: Dict[str, Any]) -> Dict[str, float]:
        """Compute features across multiple sensors"""
        try:
            # Flatten sensor data
            flat_data = {}
            for sensor_name, value in sensor_data.items():
                if isinstance(value, dict) and "value" in value:
                    flat_data[sensor_name] = value["value"]
                else:
                    flat_data[sensor_name] = value
            
            # Remove None and NaN values
            valid_data = {k: float(v) for k, v in flat_data.items() 
                         if v is not None and not np.isnan(v)}
            
            if len(valid_data) < 2:
                return {}
            
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame([valid_data])
            
            # Correlation matrix features
            features = {}
            
            # Pairwise correlations
            correlations = df.corr()
            for i, col1 in enumerate(correlations.columns):
                for j, col2 in enumerate(correlations.columns):
                    if i < j:
                        corr_value = correlations.iloc[i, j]
                        if not np.isnan(corr_value):
                            features[f"corr_{col1}_{col2}"] = float(corr_value)
            
            # Principal component features
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=min(3, len(valid_data)))
                standardized_data = (df - df.mean()) / (df.std() + 1e-10)
                pca_result = pca.fit_transform(standardized_data)
                
                # Explained variance ratios
                for i, ratio in enumerate(pca.explained_variance_ratio_):
                    features[f"pca_component_{i+1}_variance_ratio"] = float(ratio)
                
                # First component loadings
                for i, (col, loading) in enumerate(zip(df.columns, pca.components_[0])):
                    features[f"pca_first_component_loading_{col}"] = float(loading)
            
            except Exception as e:
                logger.debug(f"⚠️ PCA features computation failed: {e}")
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error computing multivariate features: {e}")
            return {}
    
    def get_feature_importance(self, sensor_name: str) -> Dict[str, float]:
        """Estimate feature importance based on variance and stability"""
        try:
            window = self.data_windows[sensor_name]
            if len(window) < 10:
                return {}
            
            values = np.array([v for t, v in window])
            values = values[~np.isnan(values)]
            
            if len(values) < 5:
                return {}
            
            importance = {}
            
            # Variance-based importance
            importance["variance"] = float(np.var(values))
            
            # Stability (inverse of coefficient of variation)
            mean_val = np.mean(values)
            if mean_val != 0:
                cv = np.std(values) / abs(mean_val)
                importance["stability"] = float(1 / (1 + cv))
            
            # Trend strength
            if len(values) >= 10:
                x = np.arange(len(values))
                slope, _, _, _, _ = stats.linregress(x, values)
                importance["trend_strength"] = float(abs(slope))
            
            return importance
            
        except Exception as e:
            logger.error(f"❌ Error computing feature importance for {sensor_name}: {e}")
            return {}


# Import wavelet library
try:
    import pywt
except ImportError:
    logger.warning("⚠️ PyWavelets not available, wavelet features will be disabled")
    pywt = None


async def main_example():
    """Example usage of Statistical Feature Extractor"""
    
    async def feature_callback(sensor_name, features):
        """Callback for extracted features"""
        print(f"\n📊 Statistical features for {sensor_name}:")
        print(f"   Timestamp: {features.get('timestamp', 'unknown')}")
        print(f"   Sample count: {features.get('sample_count', 0)}")
        print(f"   Features computed: {len(features) - 4}")  # Exclude metadata
        
        # Show some key features
        key_features = ["mean", "std", "skewness", "kurtosis", "hurst_exponent"]
        for feature in key_features:
            if feature in features:
                print(f"   {feature}: {features[feature]:.4f}")
    
    # Create feature extractor
    extractor = StatisticalFeatureExtractor(
        window_size=50,
        sampling_rate=1.0,
        feature_callback=feature_callback
    )
    
    # Simulate sensor data
    print("🔄 Simulating sensor data for statistical feature extraction...")
    
    # Generate different types of signals
    sensors = {
        "furnace_temperature": lambda t: 1500 + 20 * np.sin(0.1 * t) + 10 * np.random.random(),
        "forming_pressure": lambda t: 50 + 5 * np.sin(0.05 * t) + 3 * np.random.random(),
        "quality_score": lambda t: 0.9 + 0.1 * np.sin(0.02 * t) + 0.05 * np.random.random()
    }
    
    # Simulate data stream
    for i in range(100):
        timestamp = datetime.utcnow()
        t = i * 0.1  # Time in seconds
        
        for sensor_name, signal_func in sensors.items():
            value = signal_func(t)
            await extractor.update_with_data(sensor_name, value, timestamp)
        
        # Compute multivariate features occasionally
        if i % 20 == 19:
            sensor_data = {name: signal_func(t) for name, signal_func in sensors.items()}
            multivariate_features = await extractor.compute_multivariate_features(sensor_data)
            if multivariate_features:
                print(f"🔗 Multivariate features: {len(multivariate_features)} computed")
        
        await asyncio.sleep(0.05)
    
    # Show feature importance
    print("\n📈 Feature importance:")
    for sensor_name in sensors.keys():
        importance = extractor.get_feature_importance(sensor_name)
        if importance:
            print(f"   {sensor_name}:")
            for metric, value in importance.items():
                print(f"     {metric}: {value:.4f}")


if __name__ == "__main__":
    import numpy as np
    from scipy import stats
    
    # Try to import PyWavelets for wavelet features
    try:
        import pywt
        has_pywt = True
    except ImportError:
        print("⚠️ PyWavelets not installed, wavelet features will be disabled")
        has_pywt = False
    
    asyncio.run(main_example())