import numpy as np
import scipy.linalg
from typing import Tuple, Optional

class ESKFXYAH:
    """
    Error State Kalman Filter (ESKF) for tracking bounding boxes in image space.
    
    The ESKF maintains two states:
    - Nominal state (x_n): Full state vector (8D) representing the best estimate
    - Error state (δx): Small error state (8D) that follows linear Gaussian assumptions
    
    Key advantages:
    1. Error state is small, so linearization errors are minimized
    2. Better handling of nonlinearities in the system
    3. Numerical stability for orientation-like parameters
    
    State space: (x, y, a, h, vx, vy, va, vh)
    """
    
    def __init__(self):
        """Initialize ESKF with motion and observation uncertainty weights."""
        self.ndim = 4  # Position dimensions (x, y, a, h)
        self.state_dim = 8  # Full state dimension
        self.error_state_dim = 8  # Error state dimension
        self.dt = 1.0
        
        # Motion matrix for nominal state (constant velocity model)
        self._motion_mat = np.eye(2 * self.ndim, 2 * self.ndim)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt
        
        # Update matrix for observation model
        self._update_mat = np.eye(self.ndim, 2 * self.ndim)
        
        # Uncertainty weights
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
        
        # Error state motion matrix (same as nominal for linear dynamics)
        self._error_motion_mat = self._motion_mat.copy()
    
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize nominal state, error state, and error covariance.
        
        Args:
            measurement: Initial measurement (x, y, a, h)
            
        Returns:
            nominal_state: Initial nominal state (8D)
            error_state: Initial error state (8D, zero)
            error_cov: Initial error covariance (8x8)
        """
        # Initialize nominal state
        nominal_pos = measurement
        nominal_vel = np.zeros_like(nominal_pos)
        nominal_state = np.r_[nominal_pos, nominal_vel]
        
        # Initialize error state as zero
        error_state = np.zeros(self.error_state_dim)
        
        # Initialize error covariance
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,  # Aspect ratio has smaller uncertainty
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,  # Aspect ratio velocity has very small uncertainty
            10 * self._std_weight_velocity * measurement[3],
        ]
        error_cov = np.diag(np.square(std))
        
        return nominal_state, error_state, error_cov
    
    def predict(self, 
                nominal_state: np.ndarray, 
                error_state: np.ndarray,
                error_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ESKF prediction step.
        
        Args:
            nominal_state: Current nominal state
            error_state: Current error state
            error_cov: Current error covariance
            
        Returns:
            nominal_state_pred: Predicted nominal state
            error_state_pred: Predicted error state (zero mean)
            error_cov_pred: Predicted error covariance
        """
        # 1. Predict nominal state (nonlinear prediction)
        nominal_state_pred = self._predict_nominal_state(nominal_state)
        
        # 2. Predict error state (always reset to zero after injection)
        error_state_pred = np.zeros_like(error_state)
        
        # 3. Predict error covariance
        # Compute process noise for error state
        std_pos = [
            self._std_weight_position * nominal_state[3],
            self._std_weight_position * nominal_state[3],
            1e-2,
            self._std_weight_position * nominal_state[3],
        ]
        std_vel = [
            self._std_weight_velocity * nominal_state[3],
            self._std_weight_velocity * nominal_state[3],
            1e-5,
            self._std_weight_velocity * nominal_state[3],
        ]
        process_noise_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        # Error covariance prediction
        error_cov_pred = self._error_motion_mat @ error_cov @ self._error_motion_mat.T + process_noise_cov
        
        return nominal_state_pred, error_state_pred, error_cov_pred
    
    def _predict_nominal_state(self, nominal_state: np.ndarray) -> np.ndarray:
        """Predict nominal state using motion model."""
        return self._motion_mat @ nominal_state
    
    def update(self,
               nominal_state: np.ndarray,
               error_state: np.ndarray,
               error_cov: np.ndarray,
               measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ESKF update step.
        
        Args:
            nominal_state: Predicted nominal state
            error_state: Predicted error state (zero)
            error_cov: Predicted error covariance
            measurement: New measurement (x, y, a, h)
            
        Returns:
            nominal_state_updated: Updated nominal state
            error_state_updated: Updated error state (will be reset to zero)
            error_cov_updated: Updated error covariance
        """
        # 1. Compute predicted measurement from nominal state
        H = self._update_mat  # Measurement matrix
        
        # 2. Compute innovation (measurement residual)
        z_pred = H @ nominal_state
        innovation = measurement - z_pred
        
        # 3. Compute innovation covariance
        # Add measurement noise
        std = [
            self._std_weight_position * nominal_state[3],
            self._std_weight_position * nominal_state[3],
            1e-1,
            self._std_weight_position * nominal_state[3],
        ]
        measurement_noise_cov = np.diag(np.square(std))
        
        # Innovation covariance
        S = H @ error_cov @ H.T + measurement_noise_cov
        
        # 4. Compute Kalman gain for error state
        try:
            # Using Cholesky decomposition for numerical stability
            chol_factor, lower = scipy.linalg.cho_factor(S, lower=True, check_finite=False)
            K = scipy.linalg.cho_solve(
                (chol_factor, lower), (H @ error_cov.T).T, check_finite=False
            ).T
        except:
            # Fallback to standard inversion if Cholesky fails
            K = error_cov @ H.T @ np.linalg.inv(S)
        
        # 5. Update error state
        error_state_updated = error_state + K @ innovation
        
        # 6. Update error covariance (Joseph form for numerical stability)
        I = np.eye(self.error_state_dim)
        error_cov_updated = (I - K @ H) @ error_cov @ (I - K @ H).T + K @ measurement_noise_cov @ K.T
        
        # 7. Inject error state into nominal state (reset error state to zero)
        nominal_state_updated = self._inject_error(nominal_state, error_state_updated)
        
        # Reset error state to zero after injection
        error_state_updated = np.zeros_like(error_state_updated)
        
        return nominal_state_updated, error_state_updated, error_cov_updated
    
    def _inject_error(self, nominal_state: np.ndarray, error_state: np.ndarray) -> np.ndarray:
        """
        Inject error state into nominal state.
        
        For linear states, this is simple addition.
        For orientation-like parameters, we would use exponential map.
        Here we use addition since all states are linear.
        """
        return nominal_state + error_state
    
    def project(self, 
                nominal_state: np.ndarray,
                error_state: np.ndarray,
                error_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution to measurement space.
        
        Args:
            nominal_state: Nominal state
            error_state: Error state
            error_cov: Error covariance
            
        Returns:
            mean: Projected mean in measurement space
            cov: Projected covariance in measurement space
        """
        # True state = nominal + error
        true_state = self._inject_error(nominal_state, error_state)
        
        # Projected mean
        mean = self._update_mat @ true_state
        
        # Projected covariance
        std = [
            self._std_weight_position * nominal_state[3],
            self._std_weight_position * nominal_state[3],
            1e-1,
            self._std_weight_position * nominal_state[3],
        ]
        measurement_noise_cov = np.diag(np.square(std))
        
        H = self._update_mat
        cov = H @ error_cov @ H.T + measurement_noise_cov
        
        return mean, cov
    
    def multi_predict(self,
                      nominal_states: np.ndarray,
                      error_states: np.ndarray,
                      error_covs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized prediction for multiple objects.
        
        Args:
            nominal_states: Nx8 array of nominal states
            error_states: Nx8 array of error states
            error_covs: Nx8x8 array of error covariances
            
        Returns:
            nominal_states_pred: Predicted nominal states
            error_states_pred: Predicted error states
            error_covs_pred: Predicted error covariances
        """
        n_objects = len(nominal_states)
        
        # Predict nominal states
        nominal_states_pred = np.dot(nominal_states, self._motion_mat.T)
        
        # Reset error states to zero
        error_states_pred = np.zeros_like(error_states)
        
        # Compute process noise for each object
        std_pos = [
            self._std_weight_position * nominal_states[:, 3],
            self._std_weight_position * nominal_states[:, 3],
            1e-2 * np.ones_like(nominal_states[:, 3]),
            self._std_weight_position * nominal_states[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * nominal_states[:, 3],
            self._std_weight_velocity * nominal_states[:, 3],
            1e-5 * np.ones_like(nominal_states[:, 3]),
            self._std_weight_velocity * nominal_states[:, 3],
        ]
        
        sqr = np.square(np.r_[std_pos, std_vel]).T
        process_noise_covs = np.array([np.diag(sqr[i]) for i in range(n_objects)])
        
        # Predict error covariances
        left = np.dot(self._error_motion_mat, error_covs).transpose((1, 0, 2))
        error_covs_pred = np.dot(left, self._error_motion_mat.T) + process_noise_covs
        
        return nominal_states_pred, error_states_pred, error_covs_pred
    
    def gating_distance(self,
                        nominal_state: np.ndarray,
                        error_state: np.ndarray,
                        error_cov: np.ndarray,
                        measurements: np.ndarray,
                        only_position: bool = False,
                        metric: str = "maha") -> np.ndarray:
        """
        Compute gating distance between state distribution and measurements.
        
        Args:
            nominal_state: Nominal state
            error_state: Error state
            error_cov: Error covariance
            measurements: Nx4 array of measurements
            only_position: If True, use only position for distance
            metric: "maha" for Mahalanobis, "gaussian" for Euclidean
            
        Returns:
            distances: N array of squared distances
        """
        # Project to measurement space
        mean, cov = self.project(nominal_state, error_state, error_cov)
        
        if only_position:
            mean = mean[:2]
            cov = cov[:2, :2]
            measurements = measurements[:, :2]
        
        # Compute distances
        d = measurements - mean
        
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            try:
                chol_factor = np.linalg.cholesky(cov)
                z = scipy.linalg.solve_triangular(
                    chol_factor, d.T, lower=True, check_finite=False, overwrite_b=True
                )
                return np.sum(z * z, axis=0)
            except:
                # Fallback to pseudo-inverse if Cholesky fails
                try:
                    inv_cov = np.linalg.inv(cov)
                    return np.sum(d @ inv_cov * d, axis=1)
                except:
                    # If inversion fails, use diagonal approximation
                    diag_inv = 1.0 / np.diag(cov)
                    return np.sum(d * d * diag_inv, axis=1)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def get_true_state(self, nominal_state: np.ndarray, error_state: np.ndarray) -> np.ndarray:
        """Get the true state by injecting error into nominal state."""
        return self._inject_error(nominal_state, error_state)
    
    def reset_error_state(self, error_state: np.ndarray, error_cov: np.ndarray) -> np.ndarray:
        """
        Reset error state to zero and update covariance.
        
        In ESKF, after injecting error into nominal state, we reset error state to zero.
        For linear states, the covariance doesn't change during reset.
        """
        # For linear states, covariance remains the same
        return np.zeros_like(error_state), error_cov


import numpy as np
import scipy.linalg
from typing import Tuple, Optional

class ESKFXYAH:
    """
    Error State Kalman Filter (ESKF) for tracking bounding boxes in image space.
    
    The ESKF maintains two states:
    - Nominal state (x_n): Full state vector (8D) representing the best estimate
    - Error state (δx): Small error state (8D) that follows linear Gaussian assumptions
    
    State space: (x, y, a, h, vx, vy, va, vh)
    where:
      (x, y): bounding box center position
      a: aspect ratio (width / height)
      h: height
      (vx, vy, va, vh): respective velocities
    """
    
    def __init__(self):
        """Initialize ESKF with motion and observation uncertainty weights."""
        self.ndim = 4  # Position dimensions (x, y, a, h)
        self.state_dim = 8  # Full state dimension
        self.error_state_dim = 8  # Error state dimension
        self.dt = 1.0
        
        # Motion matrix for nominal state (constant velocity model)
        self._motion_mat = np.eye(2 * self.ndim, 2 * self.ndim)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt
        
        # Update matrix for observation model
        self._update_mat = np.eye(self.ndim, 2 * self.ndim)
        
        # Uncertainty weights
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
        
        # Error state motion matrix (same as nominal for linear dynamics)
        self._error_motion_mat = self._motion_mat.copy()
        
        # For aspect ratio (a), we need to handle it specially since it's a ratio
        self._min_aspect_ratio = 0.1  # Minimum allowed aspect ratio
        self._max_aspect_ratio = 10.0  # Maximum allowed aspect ratio
    
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize nominal state, error state, and error covariance.
        
        Args:
            measurement: Initial measurement (x, y, a, h)
            
        Returns:
            nominal_state: Initial nominal state (8D)
            error_state: Initial error state (8D, zero)
            error_cov: Initial error covariance (8x8)
        """
        # Initialize nominal state
        # Ensure aspect ratio is within reasonable bounds
        a = np.clip(measurement[2], self._min_aspect_ratio, self._max_aspect_ratio)
        nominal_pos = np.array([measurement[0], measurement[1], a, measurement[3]])
        nominal_vel = np.zeros_like(nominal_pos)
        nominal_state = np.r_[nominal_pos, nominal_vel]
        
        # Initialize error state as zero
        error_state = np.zeros(self.error_state_dim)
        
        # Initialize error covariance
        std = [
            2 * self._std_weight_position * measurement[3],  # x
            2 * self._std_weight_position * measurement[3],  # y
            1e-2,  # aspect ratio (a) - small initial uncertainty for ratio
            2 * self._std_weight_position * measurement[3],  # height
            10 * self._std_weight_velocity * measurement[3],  # vx
            10 * self._std_weight_velocity * measurement[3],  # vy
            1e-5,  # va - aspect ratio velocity
            10 * self._std_weight_velocity * measurement[3],  # vh
        ]
        error_cov = np.diag(np.square(std))
        
        return nominal_state, error_state, error_cov
    
    def predict(self, 
                nominal_state: np.ndarray, 
                error_state: np.ndarray,
                error_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ESKF prediction step.
        
        Args:
            nominal_state: Current nominal state
            error_state: Current error state
            error_cov: Current error covariance
            
        Returns:
            nominal_state_pred: Predicted nominal state
            error_state_pred: Predicted error state (zero mean)
            error_cov_pred: Predicted error covariance
        """
        # 1. Predict nominal state (nonlinear prediction)
        nominal_state_pred = self._predict_nominal_state(nominal_state)
        
        # 2. Predict error state (always reset to zero after injection)
        error_state_pred = np.zeros_like(error_state)
        
        # 3. Predict error covariance
        # Compute process noise for error state
        # Aspect ratio process noise is independent of height
        std_pos = [
            self._std_weight_position * nominal_state[3],  # x
            self._std_weight_position * nominal_state[3],  # y
            1e-2,  # a (aspect ratio)
            self._std_weight_position * nominal_state[3],  # h
        ]
        std_vel = [
            self._std_weight_velocity * nominal_state[3],  # vx
            self._std_weight_velocity * nominal_state[3],  # vy
            1e-5,  # va (aspect ratio velocity)
            self._std_weight_velocity * nominal_state[3],  # vh
        ]
        process_noise_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        # Error covariance prediction
        error_cov_pred = self._error_motion_mat @ error_cov @ self._error_motion_mat.T + process_noise_cov
        
        return nominal_state_pred, error_state_pred, error_cov_pred
    
    def _predict_nominal_state(self, nominal_state: np.ndarray) -> np.ndarray:
        """Predict nominal state using constant velocity model."""
        return self._motion_mat @ nominal_state
    
    def update(self,
               nominal_state: np.ndarray,
               error_state: np.ndarray,
               error_cov: np.ndarray,
               measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        ESKF update step.
        
        Args:
            nominal_state: Predicted nominal state
            error_state: Predicted error state (zero)
            error_cov: Predicted error covariance
            measurement: New measurement (x, y, a, h)
            
        Returns:
            nominal_state_updated: Updated nominal state
            error_state_updated: Updated error state (will be reset to zero)
            error_cov_updated: Updated error covariance
        """
        # Ensure aspect ratio measurement is within bounds
        a_meas = np.clip(measurement[2], self._min_aspect_ratio, self._max_aspect_ratio)
        measurement_clipped = np.array([measurement[0], measurement[1], a_meas, measurement[3]])
        
        # 1. Compute predicted measurement from nominal state
        H = self._update_mat  # Measurement matrix
        
        # 2. Compute innovation (measurement residual)
        z_pred = H @ nominal_state
        innovation = measurement_clipped - z_pred
        
        # 3. Compute innovation covariance
        # Add measurement noise
        std = [
            self._std_weight_position * nominal_state[3],  # x
            self._std_weight_position * nominal_state[3],  # y
            1e-1,  # a (aspect ratio measurement noise)
            self._std_weight_position * nominal_state[3],  # h
        ]
        measurement_noise_cov = np.diag(np.square(std))
        
        # Innovation covariance
        S = H @ error_cov @ H.T + measurement_noise_cov
        
        # 4. Compute Kalman gain for error state
        try:
            # Using Cholesky decomposition for numerical stability
            chol_factor, lower = scipy.linalg.cho_factor(S, lower=True, check_finite=False)
            K = scipy.linalg.cho_solve(
                (chol_factor, lower), (H @ error_cov.T).T, check_finite=False
            ).T
        except np.linalg.LinAlgError:
            # Fallback to standard inversion if Cholesky fails
            try:
                K = error_cov @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                # If still fails, use diagonal approximation
                diag_S = np.diag(np.diag(S))
                K = error_cov @ H.T @ np.linalg.inv(diag_S)
        
        # 5. Update error state
        error_state_updated = error_state + K @ innovation
        
        # 6. Update error covariance (Joseph form for numerical stability)
        I = np.eye(self.error_state_dim)
        error_cov_updated = (I - K @ H) @ error_cov @ (I - K @ H).T + K @ measurement_noise_cov @ K.T
        
        # 7. Inject error state into nominal state
        nominal_state_updated = self._inject_error(nominal_state, error_state_updated)
        
        # Ensure aspect ratio stays within bounds
        nominal_state_updated[2] = np.clip(
            nominal_state_updated[2], 
            self._min_aspect_ratio, 
            self._max_aspect_ratio
        )
        
        # Reset error state to zero after injection
        error_state_updated = np.zeros_like(error_state_updated)
        
        return nominal_state_updated, error_state_updated, error_cov_updated
    
    def _inject_error(self, nominal_state: np.ndarray, error_state: np.ndarray) -> np.ndarray:
        """
        Inject error state into nominal state.
        
        For linear states, this is simple addition.
        We need to handle aspect ratio specially to keep it positive.
        """
        # Simple addition for all states
        injected_state = nominal_state + error_state
        
        # Ensure aspect ratio stays positive
        if injected_state[2] <= 0:
            injected_state[2] = self._min_aspect_ratio
        
        return injected_state
    
    def project(self, 
                nominal_state: np.ndarray,
                error_state: np.ndarray,
                error_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution to measurement space.
        
        Args:
            nominal_state: Nominal state
            error_state: Error state
            error_cov: Error covariance
            
        Returns:
            mean: Projected mean in measurement space
            cov: Projected covariance in measurement space
        """
        # True state = nominal + error
        true_state = self._inject_error(nominal_state, error_state)
        
        # Projected mean
        mean = self._update_mat @ true_state
        
        # Projected covariance
        std = [
            self._std_weight_position * nominal_state[3],  # x
            self._std_weight_position * nominal_state[3],  # y
            1e-1,  # a (aspect ratio)
            self._std_weight_position * nominal_state[3],  # h
        ]
        measurement_noise_cov = np.diag(np.square(std))
        
        H = self._update_mat
        cov = H @ error_cov @ H.T + measurement_noise_cov
        
        return mean, cov
    
    def multi_predict(self,
                      nominal_states: np.ndarray,
                      error_states: np.ndarray,
                      error_covs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized prediction for multiple objects.
        
        Args:
            nominal_states: Nx8 array of nominal states
            error_states: Nx8 array of error states
            error_covs: Nx8x8 array of error covariances
            
        Returns:
            nominal_states_pred: Predicted nominal states
            error_states_pred: Predicted error states
            error_covs_pred: Predicted error covariances
        """
        n_objects = len(nominal_states)
        
        # Predict nominal states
        nominal_states_pred = np.dot(nominal_states, self._motion_mat.T)
        
        # Reset error states to zero
        error_states_pred = np.zeros_like(error_states)
        
        # Compute process noise for each object
        std_pos = [
            self._std_weight_position * nominal_states[:, 3],  # x
            self._std_weight_position * nominal_states[:, 3],  # y
            1e-2 * np.ones_like(nominal_states[:, 3]),  # a (aspect ratio)
            self._std_weight_position * nominal_states[:, 3],  # h
        ]
        std_vel = [
            self._std_weight_velocity * nominal_states[:, 3],  # vx
            self._std_weight_velocity * nominal_states[:, 3],  # vy
            1e-5 * np.ones_like(nominal_states[:, 3]),  # va (aspect ratio velocity)
            self._std_weight_velocity * nominal_states[:, 3],  # vh
        ]
        
        sqr = np.square(np.r_[std_pos, std_vel]).T
        process_noise_covs = np.array([np.diag(sqr[i]) for i in range(n_objects)])
        
        # Predict error covariances
        left = np.dot(self._error_motion_mat, error_covs).transpose((1, 0, 2))
        error_covs_pred = np.dot(left, self._error_motion_mat.T) + process_noise_covs
        
        return nominal_states_pred, error_states_pred, error_covs_pred
    
    def gating_distance(self,
                        nominal_state: np.ndarray,
                        error_state: np.ndarray,
                        error_cov: np.ndarray,
                        measurements: np.ndarray,
                        only_position: bool = False,
                        metric: str = "maha") -> np.ndarray:
        """
        Compute gating distance between state distribution and measurements.
        
        Args:
            nominal_state: Nominal state
            error_state: Error state
            error_cov: Error covariance
            measurements: Nx4 array of measurements
            only_position: If True, use only position for distance
            metric: "maha" for Mahalanobis, "gaussian" for Euclidean
            
        Returns:
            distances: N array of squared distances
        """
        # Project to measurement space
        mean, cov = self.project(nominal_state, error_state, error_cov)
        
        if only_position:
            mean = mean[:2]
            cov = cov[:2, :2]
            measurements = measurements[:, :2]
        
        # Compute distances
        d = measurements - mean
        
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        elif metric == "maha":
            try:
                chol_factor = np.linalg.cholesky(cov)
                z = scipy.linalg.solve_triangular(
                    chol_factor, d.T, lower=True, check_finite=False, overwrite_b=True
                )
                return np.sum(z * z, axis=0)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if Cholesky fails
                try:
                    inv_cov = np.linalg.inv(cov)
                    return np.sum(d @ inv_cov * d, axis=1)
                except np.linalg.LinAlgError:
                    # If inversion fails, use diagonal approximation
                    diag_inv = 1.0 / np.diag(cov)
                    return np.sum(d * d * diag_inv, axis=1)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def get_true_state(self, nominal_state: np.ndarray, error_state: np.ndarray) -> np.ndarray:
        """Get the true state by injecting error into nominal state."""
        return self._inject_error(nominal_state, error_state)
    
    def get_bounding_box(self, nominal_state: np.ndarray, error_state: np.ndarray) -> np.ndarray:
        """
        Convert state to bounding box format (x, y, w, h).
        
        Args:
            nominal_state: Nominal state
            error_state: Error state
            
        Returns:
            bounding_box: [x, y, width, height]
        """
        true_state = self.get_true_state(nominal_state, error_state)
        
        # Extract values
        x = true_state[0]
        y = true_state[1]
        a = true_state[2]  # aspect ratio = width / height
        h = true_state[3]
        
        # Calculate width from aspect ratio and height
        w = a * h
        
        return np.array([x, y, w, h])
    
    def reset_error_state(self, error_state: np.ndarray, error_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset error state to zero and update covariance.
        
        In ESKF, after injecting error into nominal state, we reset error state to zero.
        For linear states, the covariance doesn't change during reset.
        """
        # For linear states, covariance remains the same
        return np.zeros_like(error_state), error_cov

################################TEST##################################################
# 初始化ESKF
eskf = ESKFXYAH()

# 初始测量
measurement = np.array([100, 200, 1.5, 50])

# 初始化状态
nominal_state, error_state, error_cov = eskf.initiate(measurement)

# 预测
nominal_state_pred, error_state_pred, error_cov_pred = eskf.predict(
    nominal_state, error_state, error_cov
)

# 新测量
new_measurement = np.array([105, 205, 1.45, 52])

# 更新
nominal_state_upd, error_state_upd, error_cov_upd = eskf.update(
    nominal_state_pred, error_state_pred, error_cov_pred, new_measurement
)

# 获取真实状态估计
true_state = eskf.get_true_state(nominal_state_upd, error_state_upd)

################################TEST##################################################
import numpy as np

# 初始化ESKF
eskf = ESKFXYAH()

# 初始测量：x, y, aspect_ratio, height
measurement = np.array([100, 200, 1.5, 50])  # width = 1.5 * 50 = 75

# 初始化状态
nominal_state, error_state, error_cov = eskf.initiate(measurement)

# 预测步骤
nominal_state_pred, error_state_pred, error_cov_pred = eskf.predict(
    nominal_state, error_state, error_cov
)

# 新测量
new_measurement = np.array([105, 205, 1.45, 52])

# 更新步骤
nominal_state_upd, error_state_upd, error_cov_upd = eskf.update(
    nominal_state_pred, error_state_pred, error_cov_pred, new_measurement
)

# 获取边界框
bbox = eskf.get_bounding_box(nominal_state_upd, error_state_upd)
print(f"Bounding box: x={bbox[0]:.2f}, y={bbox[1]:.2f}, w={bbox[2]:.2f}, h={bbox[3]:.2f}")

# 计算门限距离（用于数据关联）
measurements = np.array([[105, 205, 1.45, 52], [110, 210, 1.6, 55]])
distances = eskf.gating_distance(
    nominal_state_upd, error_state_upd, error_cov_upd, 
    measurements, only_position=False, metric="maha"
)
print(f"Gating distances: {distances}")