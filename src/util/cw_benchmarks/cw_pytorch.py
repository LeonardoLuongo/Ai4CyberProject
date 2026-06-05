import logging
from typing import Optional
import numpy as np
import torch
from tqdm.auto import tqdm

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin
from art.utils import check_and_transform_label_format, get_labels_np_array

logger = logging.getLogger(__name__)

class PyTorchCustomAdam:
    """Traduzione 1:1 dell'Adam custom di ART, vettorizzata per GPU."""
    def __init__(self, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        # Inizializziamo a None, creeremo i tensori direttamente sulla GPU al primo step
        self.m_dx = None
        self.v_dx = None

    def update(self, niter: int, x: torch.Tensor, delta_x: torch.Tensor) -> torch.Tensor:
        if self.m_dx is None:
            self.m_dx = torch.zeros_like(delta_x)
            self.v_dx = torch.zeros_like(delta_x)

        self.m_dx = self.beta_1 * self.m_dx + (1 - self.beta_1) * delta_x
        self.v_dx = self.beta_2 * self.v_dx + (1 - self.beta_2) * (delta_x ** 2)
        
        m_dw_corr = self.m_dx / (1 - self.beta_1 ** niter)
        v_dw_corr = self.v_dx / (1 - self.beta_2 ** niter)
        
        x = x - self.alpha * (m_dw_corr / (torch.sqrt(v_dw_corr) + self.epsilon))
        return x

class CarliniLInfMethodPyTorch(EvasionAttack):
    """
    Match perfetto, completamente BATCHATO e parallelizzato per GPU.
    """

    attack_params = EvasionAttack.attack_params + [
        "confidence", "targeted", "learning_rate", "max_iter",
        "decrease_factor", "initial_const", "largest_const",
        "const_factor", "batch_size", "verbose",
    ]
    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(
        self,
        classifier,
        confidence: float = 0.0,
        targeted: bool = False,
        learning_rate: float = 0.01,
        max_iter: int = 10,
        decrease_factor: float = 0.9,
        initial_const: float = 1e-5,
        largest_const: float = 20.0,
        const_factor: float = 2.0,
        batch_size: int = 1,
        verbose: bool = True,
    ) -> None:
        super().__init__(estimator=classifier)

        self._model = self.estimator.model
        self.device = self.estimator.device

        self.confidence = confidence
        self._targeted = targeted
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.decrease_factor = decrease_factor
        self.initial_const = initial_const
        self.largest_const = largest_const
        self.const_factor = const_factor
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()
        
        self._tanh_smoother = 0.999999

    def _forward_model(self, x: torch.Tensor) -> torch.Tensor:
        prep = getattr(self.estimator, 'preprocessing', None)
        if prep is not None:
            if isinstance(prep, tuple) and len(prep) == 2:
                mean, std = prep
            else:
                mean = getattr(prep, 'mean', getattr(prep, '_mean', 0.0))
                std = getattr(prep, 'std', getattr(prep, '_std', 1.0))
                
            mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device)
            std = torch.as_tensor(std, dtype=x.dtype, device=x.device)
            if mean.ndim == 1 and mean.numel() == x.shape[1]:
                mean = mean.view(1, -1, 1, 1)
                std = std.view(1, -1, 1, 1)
                
            return self._model((x - mean) / std)
        return self._model(x)

    def _original_to_tanh(self, x_original: torch.Tensor, clip_min: torch.Tensor, clip_max: torch.Tensor) -> torch.Tensor:
        x_tanh = torch.clamp(x_original, min=clip_min, max=clip_max)
        x_tanh = (x_tanh - clip_min) / (clip_max - clip_min)
        x_tanh = torch.atanh(((x_tanh * 2.0) - 1.0) * self._tanh_smoother)
        return x_tanh

    def _tanh_to_original(self, x_tanh: torch.Tensor, clip_min: torch.Tensor, clip_max: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(x_tanh) + 1.0) / 2.0 * (clip_max - clip_min) + clip_min

    def _loss(self, x_adv: torch.Tensor, target: torch.Tensor, x: torch.Tensor, const: torch.Tensor, tau: torch.Tensor):
        z_predicted = self._forward_model(x_adv)
        z_target = torch.sum(z_predicted * target, dim=1)
        
        min_z = torch.min(z_predicted, dim=1, keepdim=True)[0]
        z_other = torch.max(z_predicted * (1 - target) + (min_z - 1.0) * target, dim=1)[0]

        if self.targeted:
            loss_1 = torch.clamp(z_other - z_target + self.confidence, min=0.0)
        else:
            loss_1 = torch.clamp(z_target - z_other + self.confidence, min=0.0)

        diff_abs = torch.abs(x_adv - x)
        # Fix per batching: tau deve fare broadcast con (B, C, H, W)
        tau_view = tau.view(-1, 1, 1, 1)
        loss_2 = torch.sum(torch.clamp(diff_abs - tau_view, min=0.0).view(x_adv.size(0), -1), dim=1)

        loss = loss_1 * const + loss_2
        return z_predicted, loss, loss_1, loss_2
    
    def _loss_from_logits(self, z_predicted: torch.Tensor, x_adv: torch.Tensor, target: torch.Tensor, x: torch.Tensor, const: torch.Tensor, tau: torch.Tensor):
        """Calcola la loss SENZA fare un nuovo forward pass, usando i logit già calcolati."""
        z_target = torch.sum(z_predicted * target, dim=1)
        min_z = torch.min(z_predicted, dim=1, keepdim=True)[0]
        z_other = torch.max(z_predicted * (1 - target) + (min_z - 1.0) * target, dim=1)[0]

        if self.targeted:
            loss_1 = torch.clamp(z_other - z_target + self.confidence, min=0.0)
        else:
            loss_1 = torch.clamp(z_target - z_other + self.confidence, min=0.0)

        diff_abs = torch.abs(x_adv - x)
        tau_view = tau.view(-1, 1, 1, 1)
        loss_2 = torch.sum(torch.clamp(diff_abs - tau_view, min=0.0).view(x_adv.size(0), -1), dim=1)

        loss = loss_1 * const + loss_2
        return loss

    def _art_manual_gradient(self, z_logits, y_batch, x_adv_det, x_adv_tanh, clip_min, clip_max, x_batch, tau: torch.Tensor):
        min_z = torch.min(z_logits, dim=1, keepdim=True)[0]
        
        if self.targeted:
            i_sub = torch.argmax(y_batch, dim=1)
            i_add = torch.argmax(z_logits * (1 - y_batch) + (min_z - 1.0) * y_batch, dim=1)
        else:
            i_add = torch.argmax(y_batch, dim=1)
            i_sub = torch.argmax(z_logits * (1 - y_batch) + (min_z - 1.0) * y_batch, dim=1)

        logit_diff = z_logits.gather(1, i_add.unsqueeze(1)) - z_logits.gather(1, i_sub.unsqueeze(1))
        loss_grad_1 = torch.autograd.grad(logit_diff.sum(), x_adv_det, retain_graph=False)[0]

        diff = x_adv_det - x_batch
        tau_view = tau.view(-1, 1, 1, 1)
        max_val = torch.clamp(torch.abs(diff) - tau_view, min=0.0)
        loss_grad_2 = torch.sign(max_val) * torch.sign(diff)

        chain_rule = (clip_max - clip_min) * (1.0 - torch.square(torch.tanh(x_adv_tanh))) / (2.0 * self._tanh_smoother)
        
        loss_grad_1 = loss_grad_1 * chain_rule
        loss_grad_2 = loss_grad_2 * chain_rule
        
        return loss_grad_1 + loss_grad_2

    def _generate_single(self, x_batch: torch.Tensor, y_batch: torch.Tensor, clip_min: torch.Tensor, clip_max: torch.Tensor, const: torch.Tensor, tau: torch.Tensor):
        x_adv_batch_tanh = self._original_to_tanh(x_batch, clip_min, clip_max).clone()
        adam = PyTorchCustomAdam(alpha=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        
        for num_iter in range(1, self.max_iter + 1):
            x_adv_batch = self._tanh_to_original(x_adv_batch_tanh, clip_min, clip_max)
            x_adv_det = x_adv_batch.detach().requires_grad_(True)
            
            # 1. Forward Pass con Automatic Mixed Precision (AMP) per velocizzare InceptionResnet
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                z_logits = self._forward_model(x_adv_det)
            
            # Torniamo a float32 puro per la precisione millimetrica dei gradienti ART
            z_logits = z_logits.float()
            
            # 2. Controllo Early Stopping PRIMA dell'update (Risparmiamo 1 intero forward pass!)
            with torch.no_grad():
                loss = self._loss_from_logits(z_logits, x_adv_det, y_batch, x_batch, const, tau)
                if (loss < 0.001).all():
                    break
            
            # 3. Calcolo del gradiente e update
            total_grad = self._art_manual_gradient(z_logits, y_batch, x_adv_det, x_adv_batch_tanh, clip_min, clip_max, x_batch, tau)
            
            with torch.no_grad():
                x_adv_batch_tanh = adam.update(num_iter, x_adv_batch_tanh, total_grad)
                
        with torch.no_grad():
            return self._tanh_to_original(x_adv_batch_tanh, clip_min, clip_max)

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        else:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
            
        clip_min, clip_max = self.estimator.clip_values if self.estimator.clip_values is not None else (np.amin(x), np.amax(x))

        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        clip_min = torch.tensor(clip_min, dtype=torch.float32, device=self.device)
        clip_max = torch.tensor(clip_max, dtype=torch.float32, device=self.device)

        x_adv_out = x_tensor.clone()
        self._model.eval()

        num_batches = int(np.ceil(x_tensor.size(0) / self.batch_size))
        
        # Iteriamo a BLOCCHI, non più campione per campione!
        for batch_idx in tqdm(range(num_batches), desc="C&W L_inf (Batched ART-Match)", disable=not self.verbose):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, x_tensor.size(0))
            
            x_batch = x_tensor[start_idx:end_idx]
            y_batch = y_tensor[start_idx:end_idx]
            B = x_batch.size(0)

            # Tensori di stato per ogni immagine nel batch
            tau = torch.ones(B, device=self.device)
            delta_i_best = torch.ones(B, device=self.device)
            sample_done = torch.zeros(B, dtype=torch.bool, device=self.device) 
            
            # Condizione esterna: continua se c'è almeno un'immagine con tau > limite E che non ha "finito"
            while torch.any((tau > 1.0 / 256.0) & (~sample_done)):
                active_tau_mask = (tau > 1.0 / 256.0) & (~sample_done)
                sample_done = torch.where(active_tau_mask, torch.ones_like(sample_done), sample_done)
                
                const = torch.full((B,), self.initial_const, device=self.device)
                const_found_adv = torch.zeros(B, dtype=torch.bool, device=self.device)
                
                # Condizione interna: continua il grid search del const per chi è ancora attivo
                while torch.any((const < self.largest_const) & active_tau_mask & (~const_found_adv)):
                    active_const_mask = (const < self.largest_const) & active_tau_mask & (~const_found_adv)
                    
                    # Compute perturbazione per l'intero batch
                    x_adv_batch = self._generate_single(x_batch, y_batch, clip_min, clip_max, const, tau)
                    
                    with torch.no_grad():
                        pred_class = torch.argmax(self._forward_model(x_adv_batch), dim=1)
                        target_class = torch.argmax(y_batch, dim=1)
                        delta_i = torch.amax(torch.abs(x_adv_batch - x_batch), dim=(1, 2, 3))
                        
                        if self._targeted:
                            success_mask = (pred_class == target_class) & (delta_i < delta_i_best) & active_const_mask
                        else:
                            success_mask = (pred_class != target_class) & (delta_i < delta_i_best) & active_const_mask
                        
                        if success_mask.any():
                            # Salviamo i migliori successi
                            x_adv_out[start_idx:end_idx][success_mask] = x_adv_batch[success_mask]
                            delta_i_best[success_mask] = delta_i[success_mask]
                            
                            # Immagini di successo escono dal const-loop e rientreranno nel prossimo tau-loop
                            sample_done[success_mask] = False
                            const_found_adv[success_mask] = True
                            
                    # Aumenta const solo per chi non ha ancora avuto successo in questo ciclo
                    const = torch.where(active_const_mask & (~const_found_adv), const * self.const_factor, const)

                # Aggiornamento di tau alla fine del const-loop per i sample attivi
                with torch.no_grad():
                    tau_actual = torch.amax(torch.abs(x_adv_out[start_idx:end_idx] - x_batch), dim=(1, 2, 3))
                    tau = torch.where(active_tau_mask & (tau_actual < tau), tau_actual, tau)
                    tau = torch.where(active_tau_mask, tau * self.decrease_factor, tau)

        return x_adv_out.cpu().numpy()

    def _check_params(self) -> None:
        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")
        if not isinstance(self.decrease_factor, (int, float)) or not 0.0 < self.decrease_factor < 1.0:
            raise ValueError("The decrease factor must be a float between 0 and 1.")
        if not isinstance(self.initial_const, (int, float)) or self.initial_const < 0:
            raise ValueError("The initial constant value must be a positive float.")
        if not isinstance(self.largest_const, (int, float)) or self.largest_const < 0:
            raise ValueError("The largest constant value must be a positive float.")
        if not isinstance(self.const_factor, (int, float)) or self.const_factor < 0:
            raise ValueError("The constant factor value must be a float and greater than 1.")
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("The batch size must be an integer greater than zero.")