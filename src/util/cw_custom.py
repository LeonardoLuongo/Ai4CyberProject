# FILE: src/util/cw_custom.py
import torch
import torch.nn as nn
import torch.optim as optim
import sys

# Abilita i Tensor Cores (TF32) per le GPU Nvidia moderne (Ampere+), 
# fornendo un boost di velocità gratuito senza perdita visibile di precisione.
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

class PyTorchCarliniLInf_FastBinary:
    """
    L'algoritmo definitivo: 
    Architettura a Ricerca Binaria (fissa a 9 step, no loop while lenti)
    + Ottimizzazioni Hardware Estreme (Autocast, Fused Adam, Gather/Scatter, Zero-Grad GPU).
    """
    def __init__(self, model, targeted=False, confidence=0.0,
                 learning_rate=0.01, max_iter=50, search_steps=9, 
                 initial_const=1e-3, largest_const=20.0, loss_converged=0.001): 
        
        self.model = model
        self.model.eval()
        # CONGELAMENTO PESI: Taglia il calcolo dei gradienti inutili
        for param in self.model.parameters():
            param.requires_grad = False

        self.targeted = targeted
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.search_steps = search_steps 
        self.initial_const = initial_const
        self.largest_const = largest_const
        self.loss_converged = loss_converged
        self.device = next(model.parameters()).device

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def forward(self, image, label):
        batch_size = image.size(0)
        image = image.to(self.device)
        # Shape necessaria per gather/scatter
        label = label.to(self.device).view(-1, 1)

        c = torch.full((batch_size,), self.initial_const, device=self.device)
        lower_bound = torch.zeros((batch_size,), device=self.device)
        upper_bound = torch.full((batch_size,), self.largest_const, device=self.device)
        
        tau = torch.ones((batch_size,), device=self.device)

        best_adv_image = image.clone().detach()
        best_Linf = torch.full((batch_size,), float('inf'), device=self.device)

        use_fused_adam = self.device.type == 'cuda' and hasattr(torch.optim.Adam, 'fused')

        # CICLO DI RICERCA BINARIA FISSO
        for search in range(self.search_steps):
            
            x_clamp = torch.clamp(image, 1e-4, 1 - 1e-4)
            w = self.atanh(x_clamp * 2 - 1).clone().detach().requires_grad_(True)

            if use_fused_adam:
                optimizer = optim.Adam([w], lr=self.learning_rate, fused=True)
            else:
                optimizer = optim.Adam([w], lr=self.learning_rate)

            # Prev_loss nativa in VRAM
            prev_loss = torch.tensor(float('inf'), device=self.device)

            # LOOP DI OTTIMIZZAZIONE (MAX_ITER)
            for step in range(self.max_iter):
                
                # AUTOCAST: FP16 per TFLOPS massimi
                with autocast(device_type='cuda', dtype=torch.float16):
                    adv_image = 0.5 * (torch.tanh(w) + 1)
                    logits = self.model(adv_image)

                    # Estrazione Logits iper-veloce (Zero RAM sprecata)
                    real = logits.gather(1, label).squeeze(1)
                    logits_other = logits.clone()
                    logits_other.scatter_(1, label, -10000.0)
                    other = logits_other.max(dim=1)[0]

                    # F.relu batte torch.clamp per velocità a livello CUDA
                    if self.targeted:
                        loss_1 = F.relu(other - real + self.confidence)
                    else:
                        loss_1 = F.relu(real - other + self.confidence)

                    diff = torch.abs(adv_image - image)
                    loss_2 = torch.sum(F.relu(diff - tau.view(-1, 1, 1, 1)), dim=(1, 2, 3))

                    loss = torch.sum(c * loss_1 + loss_2)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()  
                optimizer.step()

                # LOGICA CONVERGENZA FULL GPU
                with torch.no_grad():
                    loss_diff = torch.abs(prev_loss - loss)
                    if loss_diff.item() < self.loss_converged:
                        break
                    prev_loss = loss.detach()

            # --- VALUTAZIONE E AGGIORNAMENTO BINARIO ---
            with torch.no_grad():
                with autocast(device_type='cuda', dtype=torch.float16):
                    eval_img = 0.5 * (torch.tanh(w) + 1)
                    eval_logits = self.model(eval_img)
                    
                eval_pred = torch.argmax(eval_logits, dim=1)
                eval_tau = torch.amax(torch.abs(eval_img - image), dim=(1, 2, 3))
                
                eval_success = (eval_pred == label.squeeze(1)) if self.targeted else (eval_pred != label.squeeze(1))
                
                better_mask = eval_success & (eval_tau < best_Linf)
                if better_mask.any():
                    best_adv_image[better_mask] = eval_img[better_mask].detach()
                    best_Linf[better_mask] = eval_tau[better_mask]

                # Aggiornamento limiti vettorizzato
                upper_bound = torch.where(eval_success, c, upper_bound)
                lower_bound = torch.where(~eval_success, c, lower_bound)
                
                tau = torch.where(eval_success, eval_tau * 0.9, tau)
                
                c_next_binary = (lower_bound + upper_bound) / 2.0
                c_next_exponential = c * 2.0
                c = torch.where(upper_bound < self.largest_const, c_next_binary, c_next_exponential)

        return best_adv_image



class PyTorchCarliniLInf_ParallelGrid:
    """
    State-of-the-Art C&W L_inf tramite Vectorized Hyperparameter Grid.
    ZERO RUMORE AGGIUNTO. Sfrutta la VRAM per lanciare 15 ottimizzazioni 
    parallele dell'immagine pulita, ognuna con una costante 'c' diversa.
    Trova vie d'uscita dai minimi locali semplicemente testando scale di loss differenti.
    """
    def __init__(self, model, targeted=False, confidence=0.0,
                 learning_rate=0.01, max_iter=150, grid_size=15, # 150 iterazioni in un colpo solo su 15 costanti
                 min_const_exp=-3, max_const_exp=2): 
        self.model = model
        self.targeted = targeted
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.grid_size = grid_size
        self.min_const_exp = min_const_exp # da 10^-3
        self.max_const_exp = max_const_exp # a 10^2
        self.device = next(model.parameters()).device

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def forward(self, image, label):
        batch_size = image.size(0)
        
        # 1. Moltiplichiamo l'immagine esatta, SENZA alcun rumore
        image_rep = image.repeat_interleave(self.grid_size, dim=0).to(self.device)
        label_rep = label.repeat_interleave(self.grid_size, dim=0).to(self.device)

        x_clamp = torch.clamp(image_rep, 1e-4, 1 - 1e-4)
        w = self.atanh(x_clamp * 2 - 1).clone().detach()
        w.requires_grad = True

        # 2. Creiamo uno spettro di 15 costanti diverse (da piccole a enormi)
        c_base = torch.logspace(self.min_const_exp, self.max_const_exp, self.grid_size, device=self.device)
        c = c_base.repeat(batch_size) 

        # Budget tau dinamico
        tau = torch.ones(batch_size * self.grid_size, device=self.device)

        best_adv_image = image_rep.clone().detach()
        best_Linf = torch.full((batch_size * self.grid_size,), float('inf'), device=self.device)

        use_fused_adam = self.device.type == 'cuda' and hasattr(torch.optim.Adam, 'fused')
        if use_fused_adam:
            optimizer = optim.Adam([w], lr=self.learning_rate, fused=True)
        else:
            optimizer = optim.Adam([w], lr=self.learning_rate)

        # 3. UNICO CICLO DI OTTIMIZZAZIONE LUNGO E CONTINUO
        for step in range(self.max_iter):
            adv_image = 0.5 * (torch.tanh(w) + 1)
            logits = self.model(adv_image)

            one_hot = torch.eye(logits.shape[1], device=self.device)[label_rep]
            real = torch.max(one_hot * logits, dim=1)[0]
            other = torch.max((1 - one_hot) * logits - one_hot * 10000, dim=1)[0]

            if self.targeted:
                loss_1 = torch.clamp(other - real + self.confidence, min=0.0)
            else:
                loss_1 = torch.clamp(real - other + self.confidence, min=0.0)

            diff = torch.abs(adv_image - image_rep)
            loss_2 = torch.sum(torch.clamp(diff - tau.view(-1, 1, 1, 1), min=0.0), dim=(1, 2, 3))

            loss = torch.sum(c * loss_1 + loss_2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()  
            optimizer.step()

            # Controllo dinamico e early stopping a costo zero
            with torch.no_grad():
                # Valutiamo il L_inf reale
                eval_pred = torch.argmax(logits.detach(), dim=1)
                eval_tau = torch.amax(diff.detach(), dim=(1, 2, 3))
                
                eval_success = (eval_pred == label_rep) if self.targeted else (eval_pred != label_rep)
                better_mask = eval_success & (eval_tau < best_Linf)
                
                if better_mask.any():
                    best_adv_image[better_mask] = adv_image.detach()[better_mask]
                    best_Linf[better_mask] = eval_tau[better_mask]
                    
                    # Appena una costante ha successo, abbassiamo il SUO budget tau del 10% 
                    # spingendola a fare ancora meglio all'iterazione successiva
                    tau[better_mask] = eval_tau[better_mask] * 0.9

        # 4. ESTRAZIONE DEL VINCITORE TRA LE 15 COSTANTI
        final_adv_images = image.clone().detach()
        best_Linf_reshaped = best_Linf.view(batch_size, self.grid_size)
        
        for i in range(batch_size):
            min_idx = torch.argmin(best_Linf_reshaped[i])
            if best_Linf_reshaped[i, min_idx] < float('inf'):
                final_adv_images[i] = best_adv_image[i * self.grid_size + min_idx]
                
        return final_adv_images


class PyTorchCarliniLInf_DLR:
    """
    Carlini & Wagner L_inf ibridato con la DLR Loss (AutoAttack) e Cosine Annealing.
    1. La DLR Loss rende l'attacco intrinsecamente immune alle scale dei logit.
    2. Il Cosine Annealing decresce il Learning Rate permettendo una convergenza 
       millimetrica ai margini del boundary decisionale.
    """
    def __init__(self, model, targeted=False, confidence=0.0,
                 learning_rate=0.01, max_iter=50, search_steps=9, 
                 initial_const=1e-3, largest_const=20.0, loss_converged=1e-4): 
        self.model = model
        self.targeted = targeted
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.search_steps = search_steps 
        self.initial_const = initial_const
        self.largest_const = largest_const
        self.loss_converged = loss_converged
        self.device = next(model.parameters()).device

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def forward(self, image, label):
        batch_size = image.size(0)
        image = image.to(self.device)
        label = label.to(self.device)

        c = torch.full((batch_size,), self.initial_const, device=self.device)
        lower_bound = torch.zeros((batch_size,), device=self.device)
        upper_bound = torch.full((batch_size,), self.largest_const, device=self.device)
        tau = torch.ones((batch_size,), device=self.device)

        best_adv_image = image.clone().detach()
        best_Linf = torch.full((batch_size,), float('inf'), device=self.device)

        use_fused_adam = self.device.type == 'cuda' and hasattr(torch.optim.Adam, 'fused')

        for search in range(self.search_steps):
            x_clamp = torch.clamp(image, 1e-4, 1 - 1e-4)
            w = self.atanh(x_clamp * 2 - 1).clone().detach()
            w.requires_grad = True

            if use_fused_adam:
                optimizer = optim.Adam([w], lr=self.learning_rate, fused=True)
            else:
                optimizer = optim.Adam([w], lr=self.learning_rate)
            
            # Nuova arma: Decadimento del LR per la micro-ottimizzazione
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_iter, eta_min=1e-4)

            prev_loss = float('inf')

            for step in range(self.max_iter):
                adv_image = 0.5 * (torch.tanh(w) + 1)
                logits = self.model(adv_image)
                
                # --- CALCOLO DLR (Difference of Logits Ratio) ---
                # Ordiniamo i logit dal più grande al più piccolo
                logits_sorted, _ = torch.sort(logits, dim=1, descending=True)
                # Il denominatore è la distanza tra il 1° e il 3° logit più probabile
                dlr_denom = (logits_sorted[:, 0] - logits_sorted[:, 2]).clamp_min(1e-4)

                one_hot = torch.eye(logits.shape[1], device=self.device)[label]
                real = torch.max(one_hot * logits, dim=1)[0]
                other = torch.max((1 - one_hot) * logits - one_hot * 10000, dim=1)[0]

                # Dividiamo il margine per il denominatore DLR
                if self.targeted:
                    loss_1 = torch.clamp(other - real + self.confidence, min=0.0) / dlr_denom
                else:
                    loss_1 = torch.clamp(real - other + self.confidence, min=0.0) / dlr_denom

                diff = torch.abs(adv_image - image)
                loss_2 = torch.sum(torch.clamp(diff - tau.view(-1, 1, 1, 1), min=0.0), dim=(1, 2, 3))

                loss = torch.sum(c * loss_1 + loss_2)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()  
                optimizer.step()
                scheduler.step() # Riduciamo morbidamente il Learning Rate ad ogni iterazione

                current_loss = loss.item()
                if abs(prev_loss - current_loss) < self.loss_converged:
                    break
                prev_loss = current_loss

            with torch.no_grad():
                eval_img = 0.5 * (torch.tanh(w) + 1)
                eval_logits = self.model(eval_img) 
                eval_pred = torch.argmax(eval_logits, dim=1)
                
                eval_tau = torch.amax(torch.abs(eval_img - image), dim=(1, 2, 3))
                eval_success = (eval_pred == label) if self.targeted else (eval_pred != label)
                
                better_mask = eval_success & (eval_tau < best_Linf)
                if better_mask.any():
                    best_adv_image[better_mask] = eval_img[better_mask].detach()
                    best_Linf[better_mask] = eval_tau[better_mask]

                upper_bound = torch.where(eval_success, c, upper_bound)
                lower_bound = torch.where(~eval_success, c, lower_bound)
                
                # Se abbiamo successo, stringiamo la morsa di tau del 10%
                tau = torch.where(eval_success, eval_tau * 0.9, tau)
                
                c_next_binary = (lower_bound + upper_bound) / 2.0
                c_next_exponential = c * 2.0
                c = torch.where(upper_bound < self.largest_const, c_next_binary, c_next_exponential)

        return best_adv_image


class PyTorchCarliniLInf_AutoTemp:
    """
    Carlini & Wagner L_infinity SOTA con Temperatura Adattiva (Zero Hyperparameters).
    Invece di usare una temperatura fissa, calcola dinamicamente la deviazione standard
    dei logit a ogni step e la usa per normalizzarli.
    Questo simula i benefici della DLR Loss (AutoAttack) restando fedeli al C&W.
    """
    def __init__(self, model, targeted=False, confidence=0.0,
                 learning_rate=0.01, max_iter=50, search_steps=9, 
                 initial_const=1e-3, largest_const=20.0, loss_converged=0.001): 
        self.model = model
        self.targeted = targeted
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.search_steps = search_steps 
        self.initial_const = initial_const
        self.largest_const = largest_const
        self.loss_converged = loss_converged
        self.device = next(model.parameters()).device

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def forward(self, image, label):
        batch_size = image.size(0)
        image = image.to(self.device)
        label = label.to(self.device)

        c = torch.full((batch_size,), self.initial_const, device=self.device)
        lower_bound = torch.zeros((batch_size,), device=self.device)
        upper_bound = torch.full((batch_size,), self.largest_const, device=self.device)
        tau = torch.ones((batch_size,), device=self.device)

        best_adv_image = image.clone().detach()
        best_Linf = torch.full((batch_size,), float('inf'), device=self.device)

        use_fused_adam = self.device.type == 'cuda' and hasattr(torch.optim.Adam, 'fused')

        # Ricerca Binaria Integrata (I 9 step)
        for search in range(self.search_steps):
            x_clamp = torch.clamp(image, 1e-4, 1 - 1e-4)
            w = self.atanh(x_clamp * 2 - 1).clone().detach()
            w.requires_grad = True

            if use_fused_adam:
                optimizer = optim.Adam([w], lr=self.learning_rate, fused=True)
            else:
                optimizer = optim.Adam([w], lr=self.learning_rate)

            prev_loss = float('inf')

            for step in range(self.max_iter):
                adv_image = 0.5 * (torch.tanh(w) + 1)
                logits = self.model(adv_image)
                
                # ====================================================
                # TEMPERATURA ADATTIVA (AUTO-SCALE)
                # Calcoliamo la deviazione standard dei logit.
                # Lo facciamo senza calcolare i gradienti, usandolo solo come scalare.
                # ====================================================
                with torch.no_grad():
                    # Aggiungiamo 1e-4 per evitare divisioni per zero se i logit sono piatti
                    auto_scale = torch.std(logits, dim=1, keepdim=True).clamp_min(1e-4)
                
                logits_scaled = logits / auto_scale

                # Calcolo Loss classico C&W sui logit normalizzati
                one_hot = torch.eye(logits_scaled.shape[1], device=self.device)[label]
                real = torch.max(one_hot * logits_scaled, dim=1)[0]
                other = torch.max((1 - one_hot) * logits_scaled - one_hot * 10000, dim=1)[0]

                if self.targeted:
                    loss_1 = torch.clamp(other - real + self.confidence, min=0.0)
                else:
                    loss_1 = torch.clamp(real - other + self.confidence, min=0.0)

                diff = torch.abs(adv_image - image)
                loss_2 = torch.sum(torch.clamp(diff - tau.view(-1, 1, 1, 1), min=0.0), dim=(1, 2, 3))

                loss = torch.sum(c * loss_1 + loss_2)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()  
                optimizer.step()

                current_loss = loss.item()
                if abs(prev_loss - current_loss) < self.loss_converged:
                    break
                prev_loss = current_loss

            # Valutazione Fine Step (Non serve scalare qui, la predizione non cambia)
            with torch.no_grad():
                eval_img = 0.5 * (torch.tanh(w) + 1)
                eval_logits = self.model(eval_img) 
                eval_pred = torch.argmax(eval_logits, dim=1)
                
                eval_tau = torch.amax(torch.abs(eval_img - image), dim=(1, 2, 3))
                eval_success = (eval_pred == label) if self.targeted else (eval_pred != label)
                
                better_mask = eval_success & (eval_tau < best_Linf)
                if better_mask.any():
                    best_adv_image[better_mask] = eval_img[better_mask].detach()
                    best_Linf[better_mask] = eval_tau[better_mask]

                upper_bound = torch.where(eval_success, c, upper_bound)
                lower_bound = torch.where(~eval_success, c, lower_bound)
                tau = torch.where(eval_success, eval_tau * 0.9, tau)
                
                c_next_binary = (lower_bound + upper_bound) / 2.0
                c_next_exponential = c * 2.0
                c = torch.where(upper_bound < self.largest_const, c_next_binary, c_next_exponential)

        return best_adv_image


class PyTorchCarliniLInf_TemperatureScaling:
    """
    Carlini & Wagner L_infinity SOTA con Temperature Scaling.
    Mantiene la formula originale del C&W (stile ART), ma scala i logit
    per evitare che reti profonde come FaceNet abbiano gradienti esplosivi.
    Riesce ad abbassare drasticamente L_inf mantenendo alta la percentuale di successo.
    """
    def __init__(self, model, targeted=False, confidence=0.0,
                 learning_rate=0.01, max_iter=50, search_steps=9, 
                 initial_const=1e-3, largest_const=20.0, loss_converged=0.001,
                 temperature=10.0): # <-- Il nostro nuovo "calmante" per i logits
        self.model = model
        self.targeted = targeted
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.search_steps = search_steps 
        self.initial_const = initial_const
        self.largest_const = largest_const
        self.loss_converged = loss_converged
        self.temperature = temperature
        self.device = next(model.parameters()).device

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def forward(self, image, label):
        batch_size = image.size(0)
        image = image.to(self.device)
        label = label.to(self.device)

        c = torch.full((batch_size,), self.initial_const, device=self.device)
        lower_bound = torch.zeros((batch_size,), device=self.device)
        upper_bound = torch.full((batch_size,), self.largest_const, device=self.device)
        tau = torch.ones((batch_size,), device=self.device)

        best_adv_image = image.clone().detach()
        best_Linf = torch.full((batch_size,), float('inf'), device=self.device)

        use_fused_adam = self.device.type == 'cuda' and hasattr(torch.optim.Adam, 'fused')

        for search in range(self.search_steps):
            x_clamp = torch.clamp(image, 1e-4, 1 - 1e-4)
            w = self.atanh(x_clamp * 2 - 1).clone().detach()
            w.requires_grad = True

            if use_fused_adam:
                optimizer = optim.Adam([w], lr=self.learning_rate, fused=True)
            else:
                optimizer = optim.Adam([w], lr=self.learning_rate)

            prev_loss = float('inf')

            for step in range(self.max_iter):
                adv_image = 0.5 * (torch.tanh(w) + 1)
                logits = self.model(adv_image)
                
                # ====================================================
                # TEMPERATURE SCALING (L'unica riga aggiunta alla matematica)
                # Riduce la scala dei logit (es da [-40, 40] a [-4, 4]).
                # Questo bilancia la loss della rete con la loss del rumore!
                # ====================================================
                logits_scaled = logits / self.temperature

                one_hot = torch.eye(logits_scaled.shape[1], device=self.device)[label]
                real = torch.max(one_hot * logits_scaled, dim=1)[0]
                other = torch.max((1 - one_hot) * logits_scaled - one_hot * 10000, dim=1)[0]

                if self.targeted:
                    loss_1 = torch.clamp(other - real + self.confidence, min=0.0)
                else:
                    loss_1 = torch.clamp(real - other + self.confidence, min=0.0)

                diff = torch.abs(adv_image - image)
                loss_2 = torch.sum(torch.clamp(diff - tau.view(-1, 1, 1, 1), min=0.0), dim=(1, 2, 3))

                loss = torch.sum(c * loss_1 + loss_2)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()  
                optimizer.step()

                current_loss = loss.item()
                if abs(prev_loss - current_loss) < self.loss_converged:
                    break
                prev_loss = current_loss

            with torch.no_grad():
                eval_img = 0.5 * (torch.tanh(w) + 1)
                # Per la valutazione finale non serve scalare, argmax(L/10) == argmax(L)
                eval_logits = self.model(eval_img) 
                eval_pred = torch.argmax(eval_logits, dim=1)
                
                eval_tau = torch.amax(torch.abs(eval_img - image), dim=(1, 2, 3))
                eval_success = (eval_pred == label) if self.targeted else (eval_pred != label)
                
                better_mask = eval_success & (eval_tau < best_Linf)
                if better_mask.any():
                    best_adv_image[better_mask] = eval_img[better_mask].detach()
                    best_Linf[better_mask] = eval_tau[better_mask]

                upper_bound = torch.where(eval_success, c, upper_bound)
                lower_bound = torch.where(~eval_success, c, lower_bound)
                tau = torch.where(eval_success, eval_tau * 0.9, tau)
                
                c_next_binary = (lower_bound + upper_bound) / 2.0
                c_next_exponential = c * 2.0
                c = torch.where(upper_bound < self.largest_const, c_next_binary, c_next_exponential)

        return best_adv_image



import torch
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')



class PyTorchCarliniLInf_ParallelGrid:
    """
    State-of-the-Art C&W L_inf tramite Vectorized Hyperparameter Grid.
    Elimina i cicli esterni: usa un'unica passata di ottimizzazione lanciando 
    n repliche dell'immagine pulita, ognuna con un bilanciamento 'C' diverso.
    Riutilizza i logits del forward pass per tagliare il calcolo del 50%.
    """
    def __init__(self, model, targeted=False, confidence=0.0,
                 learning_rate=0.01, max_iter=250, grid_size=20, 
                 min_const_exp=-4, max_const_exp=2): 
        self.model = model
        self.targeted = targeted
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.grid_size = grid_size
        self.min_const_exp = min_const_exp # 10^-4
        self.max_const_exp = max_const_exp # 10^2
        self.device = next(model.parameters()).device

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def forward(self, image, label):
        batch_size = image.size(0)
        
        # 1. Espansione Batch (Uso della VRAM)
        image_rep = image.repeat_interleave(self.grid_size, dim=0).to(self.device)
        label_rep = label.repeat_interleave(self.grid_size, dim=0).to(self.device)

        # 2. Inizializzazione PULITA. Niente rumore gaussiano.
        x_clamp = torch.clamp(image_rep, 1e-4, 1 - 1e-4)
        w = self.atanh(x_clamp * 2 - 1).clone().detach()
        w.requires_grad = True

        # 3. Costruzione della Griglia degli Iperparametri 'C'
        # Crea un tensore logaritmico es. [1e-4, 1e-3, ..., 1e2] direttamente su GPU
        c_base = torch.logspace(self.min_const_exp, self.max_const_exp, self.grid_size, device=self.device)
        c = c_base.repeat(batch_size) 

        # Budget tau dinamico
        tau = torch.ones(batch_size * self.grid_size, device=self.device)

        best_adv_image = image_rep.clone().detach()
        best_Linf = torch.full((batch_size * self.grid_size,), float('inf'), device=self.device)

        use_fused_adam = self.device.type == 'cuda' and hasattr(torch.optim.Adam, 'fused')
        if use_fused_adam:
            optimizer = optim.Adam([w], lr=self.learning_rate, fused=True)
        else:
            optimizer = optim.Adam([w], lr=self.learning_rate)

        # 4. UNICO CICLO DI OTTIMIZZAZIONE INTERNO
        for step in range(self.max_iter):
            adv_image = 0.5 * (torch.tanh(w) + 1)
            logits = self.model(adv_image)

            one_hot = torch.eye(logits.shape[1], device=self.device)[label_rep]
            real = torch.max(one_hot * logits, dim=1)[0]
            other = torch.max((1 - one_hot) * logits - one_hot * 10000, dim=1)[0]

            if self.targeted:
                loss_1 = torch.clamp(other - real + self.confidence, min=0.0)
            else:
                loss_1 = torch.clamp(real - other + self.confidence, min=0.0)

            diff = torch.abs(adv_image - image_rep)
            loss_2 = torch.sum(torch.clamp(diff - tau.view(-1, 1, 1, 1), min=0.0), dim=(1, 2, 3))

            loss = torch.sum(c * loss_1 + loss_2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()  
            optimizer.step()

            # --- VALUTAZIONE E EARLY STOPPING DINAMICO (A COSTO ZERO) ---
            # Sfruttiamo i tensori creati PRIMA dell'aggiornamento pesi. Nessun forward() aggiuntivo!
            with torch.no_grad():
                eval_pred = torch.argmax(logits.detach(), dim=1)
                eval_tau = torch.amax(diff.detach(), dim=(1, 2, 3))
                
                eval_success = (eval_pred == label_rep) if self.targeted else (eval_pred != label_rep)
                better_mask = eval_success & (eval_tau < best_Linf)
                
                if better_mask.any():
                    best_adv_image[better_mask] = adv_image.detach()[better_mask]
                    best_Linf[better_mask] = eval_tau[better_mask]
                    
                    # Dinamica del Tau: appena una costante trova una soluzione, stringiamo il 
                    # suo limite al 90% della soluzione trovata, forzandola a cercare di meglio nel prossimo step!
                    tau[better_mask] = eval_tau[better_mask] * 0.9

        # 5. AGGREGAZIONE FINALE
        final_adv_images = image.clone().detach()
        best_Linf_reshaped = best_Linf.view(batch_size, self.grid_size)
        
        for i in range(batch_size):
            min_idx = torch.argmin(best_Linf_reshaped[i])
            if best_Linf_reshaped[i, min_idx] < float('inf'):
                final_adv_images[i] = best_adv_image[i * self.grid_size + min_idx]
                
        return final_adv_images


class PyTorchCarliniLInf_RestartsSOTA:
    """
    State-of-the-Art C&W L_inf con Batched Random Restarts.
    Sfrutta la vettorizzazione per attaccare l'immagine da punti di partenza multipli
    simultaneamente, scappando dai minimi locali e trovando la distorsione minima assoluta.
    """
    def __init__(self, model, targeted=False, confidence=0.0,
                 learning_rate=0.01, max_iter=50, search_steps=9, 
                 initial_const=1e-3, largest_const=20.0, loss_converged=0.001,
                 restarts=20, restart_noise=0.01): 
        self.model = model
        self.targeted = targeted
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.search_steps = search_steps
        self.initial_const = initial_const
        self.largest_const = largest_const
        self.loss_converged = loss_converged
        
        # Nuovi Iperparametri per i Restarts
        self.restarts = restarts
        self.restart_noise = restart_noise
        self.device = next(model.parameters()).device

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def forward(self, image, label):
        batch_size = image.size(0)
        
        # 1. Espansione Batch: Moltiplichiamo le immagini (es. da 1 a 20 copie vettorizzate)
        image_rep = image.repeat_interleave(self.restarts, dim=0).to(self.device)
        label_rep = label.repeat_interleave(self.restarts, dim=0).to(self.device)

        # 2. Generazione punti di partenza (Restarts)
        clean_clamp = torch.clamp(image_rep, 1e-4, 1 - 1e-4)
        noise = (torch.rand_like(clean_clamp) * 2 - 1) * self.restart_noise
        
        # Azzeriamo il rumore per la PRIMA immagine di ogni gruppo, per garantire
        # che almeno uno dei restart sia la classica ottimizzazione standard senza rumore.
        clean_indices = torch.arange(0, batch_size * self.restarts, self.restarts, device=self.device)
        noise[clean_indices] = 0.0

        noisy_image = torch.clamp(clean_clamp + noise, 1e-4, 1 - 1e-4)
        
        # w_init è il punto di salvataggio. Resterà fisso per tutti i round di Binary Search.
        w_init = self.atanh(noisy_image * 2 - 1)

        # 3. Variabili di Binary Search scalate per il numero di restarts
        c = torch.full((batch_size * self.restarts,), self.initial_const, device=self.device)
        lower_bound = torch.zeros_like(c)
        upper_bound = torch.full_like(c, self.largest_const)
        tau = torch.ones_like(c)

        best_adv_image = image_rep.clone().detach()
        best_Linf = torch.full_like(c, float('inf'))

        use_fused_adam = self.device.type == 'cuda' and hasattr(torch.optim.Adam, 'fused')

        # 4. CICLO DI RICERCA BINARIA SOTA
        for search in range(self.search_steps):
            
            # Ripartiamo dal nostro punto di restart specifico
            w = w_init.clone().detach()
            w.requires_grad = True

            if use_fused_adam:
                optimizer = optim.Adam([w], lr=self.learning_rate, fused=True)
            else:
                optimizer = optim.Adam([w], lr=self.learning_rate)

            prev_loss = float('inf')

            for step in range(self.max_iter):
                adv_image = 0.5 * (torch.tanh(w) + 1)
                logits = self.model(adv_image)

                one_hot = torch.eye(logits.shape[1], device=self.device)[label_rep]
                real = torch.max(one_hot * logits, dim=1)[0]
                other = torch.max((1 - one_hot) * logits - one_hot * 10000, dim=1)[0]

                if self.targeted:
                    loss_1 = torch.clamp(other - real + self.confidence, min=0.0)
                else:
                    loss_1 = torch.clamp(real - other + self.confidence, min=0.0)

                # La distanza va calcolata rispetto all'immagine PULITA (image_rep), non a quella rumorosa iniziale!
                diff = torch.abs(adv_image - image_rep)
                loss_2 = torch.sum(torch.clamp(diff - tau.view(-1, 1, 1, 1), min=0.0), dim=(1, 2, 3))

                loss = torch.sum(c * loss_1 + loss_2)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()  
                optimizer.step()

                current_loss = loss.item()
                if abs(prev_loss - current_loss) < self.loss_converged:
                    break
                prev_loss = current_loss

            with torch.no_grad():
                eval_img = 0.5 * (torch.tanh(w) + 1)
                eval_logits = self.model(eval_img)
                eval_pred = torch.argmax(eval_logits, dim=1)
                
                eval_tau = torch.amax(torch.abs(eval_img - image_rep), dim=(1, 2, 3))
                eval_success = (eval_pred == label_rep) if self.targeted else (eval_pred != label_rep)
                
                better_mask = eval_success & (eval_tau < best_Linf)
                if better_mask.any():
                    best_adv_image[better_mask] = eval_img[better_mask].detach()
                    best_Linf[better_mask] = eval_tau[better_mask]

                upper_bound = torch.where(eval_success, c, upper_bound)
                lower_bound = torch.where(~eval_success, c, lower_bound)
                
                tau = torch.where(eval_success, eval_tau * 0.9, tau)
                
                c_next_binary = (lower_bound + upper_bound) / 2.0
                c_next_exponential = c * 2.0
                c = torch.where(upper_bound < self.largest_const, c_next_binary, c_next_exponential)

        # 5. AGGREGAZIONE FINALE (Estrazione del record assoluto)
        final_adv_images = image.clone().detach()
        best_Linf_reshaped = best_Linf.view(batch_size, self.restarts)
        
        for i in range(batch_size):
            min_idx = torch.argmin(best_Linf_reshaped[i])
            if best_Linf_reshaped[i, min_idx] < float('inf'):
                # Trovato il minimo globale tra tutti i restart per l'immagine 'i'
                final_adv_images[i] = best_adv_image[i * self.restarts + min_idx]
                
        return final_adv_images
    


class PyTorchCarliniLInf_BinarySteps:
    """
    Abbandona i lenti loop esponenziali di ART e utilizza una Ricerca Binaria Vettorizzata.
    Include Fused Adam, TF32, e Loss Convergence Early Stop.
    """
    def __init__(self, model, targeted=False, confidence=0.0,
                 learning_rate=0.01, max_iter=50, search_steps=9, 
                 initial_const=1e-3, largest_const=20.0, loss_converged=0.001): 
        self.model = model
        self.targeted = targeted
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.search_steps = search_steps # Sostituisce i 2 loop while di ART (9 step sono sufficienti)
        self.initial_const = initial_const
        self.largest_const = largest_const
        self.loss_converged = loss_converged
        self.device = next(model.parameters()).device

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def forward(self, image, label):
        batch_size = image.size(0)
        image = image.to(self.device)
        label = label.to(self.device)

        # Ricerca binaria: inizializzazione vettorizzata
        c = torch.full((batch_size,), self.initial_const, device=self.device)
        lower_bound = torch.zeros((batch_size,), device=self.device)
        upper_bound = torch.full((batch_size,), self.largest_const, device=self.device)
        
        # Budget tau dinamico
        tau = torch.ones((batch_size,), device=self.device)

        best_adv_image = image.clone().detach()
        best_Linf = torch.full((batch_size,), float('inf'), device=self.device)

        use_fused_adam = self.device.type == 'cuda' and hasattr(torch.optim.Adam, 'fused')

        # UNICO CICLO ESTERNO (Solo 9 iterazioni al posto delle centinaia di ART!)
        for search in range(self.search_steps):
            
            x_clamp = torch.clamp(image, 1e-4, 1 - 1e-4)
            w = self.atanh(x_clamp * 2 - 1).clone().detach()
            w.requires_grad = True

            if use_fused_adam:
                optimizer = optim.Adam([w], lr=self.learning_rate, fused=True)
            else:
                optimizer = optim.Adam([w], lr=self.learning_rate)

            prev_loss = float('inf')

            # Ottimizzazione C&W Standard
            for step in range(self.max_iter):
                adv_image = 0.5 * (torch.tanh(w) + 1)
                logits = self.model(adv_image)

                one_hot = torch.eye(logits.shape[1], device=self.device)[label]
                real = torch.max(one_hot * logits, dim=1)[0]
                other = torch.max((1 - one_hot) * logits - one_hot * 10000, dim=1)[0]

                if self.targeted:
                    loss_1 = torch.clamp(other - real + self.confidence, min=0.0)
                else:
                    loss_1 = torch.clamp(real - other + self.confidence, min=0.0)

                diff = torch.abs(adv_image - image)
                loss_2 = torch.sum(torch.clamp(diff - tau.view(-1, 1, 1, 1), min=0.0), dim=(1, 2, 3))

                loss = torch.sum(c * loss_1 + loss_2)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()  
                optimizer.step()

                # ART Speedup: Loss convergence
                current_loss = loss.item()
                if abs(prev_loss - current_loss) < self.loss_converged:
                    break
                prev_loss = current_loss

            # --- VALUTAZIONE E AGGIORNAMENTO BINARIO (Senza Loop) ---
            with torch.no_grad():
                eval_img = 0.5 * (torch.tanh(w) + 1)
                eval_logits = self.model(eval_img)
                eval_pred = torch.argmax(eval_logits, dim=1)
                
                eval_tau = torch.amax(torch.abs(eval_img - image), dim=(1, 2, 3))
                eval_success = (eval_pred == label) if self.targeted else (eval_pred != label)
                
                better_mask = eval_success & (eval_tau < best_Linf)
                if better_mask.any():
                    best_adv_image[better_mask] = eval_img[better_mask].detach()
                    best_Linf[better_mask] = eval_tau[better_mask]

                # Logica Binary Search di C&W:
                # Se l'attacco ha successo, la costante 'c' è alta abbastanza -> Abbassiamo l'upper bound
                upper_bound = torch.where(eval_success, c, upper_bound)
                # Se fallisce, 'c' è troppo bassa -> Alziamo il lower bound
                lower_bound = torch.where(~eval_success, c, lower_bound)
                
                # Aggiorniamo 'tau' a un valore leggermente inferiore al miglior L_inf trovato
                tau = torch.where(eval_success, eval_tau * 0.9, tau)
                
                # Calcoliamo la nuova costante 'c' tagliando a metà lo spazio di ricerca
                c_next_binary = (lower_bound + upper_bound) / 2.0
                c_next_exponential = c * 2.0
                c = torch.where(upper_bound < self.largest_const, c_next_binary, c_next_exponential)

        return best_adv_image



class PyTorchCarliniLInf_ARTMatch_Opt:
    """
    Replica esatta della logica L_inf di ART, interamente in PyTorch.
    Ottimizzata al limite per l'hardware:
    1. Usa Fused Adam (se su CUDA)
    2. Usa zero_grad(set_to_none=True)
    3. Abilita TF32 Tensor Cores.
    4. Usa torch.compile su Linux (bypassato su Windows per assenza di Triton).
    """
    def __init__(self, model, targeted=False, confidence=0.0,
                 learning_rate=0.01, max_iter=50,
                 decrease_factor=0.9, initial_const=1e-3,
                 largest_const=20.0, const_factor=2.0, 
                 loss_converged=0.001, verbose=False): 
        self.model = model
        self.targeted = targeted
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.decrease_factor = decrease_factor
        self.initial_const = initial_const
        self.largest_const = largest_const
        self.const_factor = const_factor
        self.loss_converged = loss_converged
        self.verbose = verbose
        self.device = next(model.parameters()).device

        # Gestione sicura del compilatore
        # Su Windows disabilitiamo torch.compile perché il backend Triton non è supportato.
        if hasattr(torch, "compile") and sys.platform != "win32":
            try:
                self._loss_step = torch.compile(self._loss_step_uncompiled, mode="reduce-overhead")
            except Exception:
                self._loss_step = self._loss_step_uncompiled
        else:
            self._loss_step = self._loss_step_uncompiled

    def _loss_step_uncompiled(self, w, image, label, const_val, tau):
        adv_image = 0.5 * (torch.tanh(w) + 1)
        logits = self.model(adv_image)

        one_hot = torch.eye(logits.shape[1], device=self.device)[label]
        real = torch.max(one_hot * logits, dim=1)[0]
        other = torch.max((1 - one_hot) * logits - one_hot * 10000, dim=1)[0]

        if self.targeted:
            loss_1 = torch.clamp(other - real + self.confidence, min=0.0)
        else:
            loss_1 = torch.clamp(real - other + self.confidence, min=0.0)

        diff = torch.abs(adv_image - image)
        loss_2 = torch.sum(torch.clamp(diff - tau.view(-1, 1, 1, 1), min=0.0), dim=(1, 2, 3))

        return torch.sum(const_val * loss_1 + loss_2)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def forward(self, image, label):
        batch_size = image.size(0)
        image = image.to(self.device)
        label = label.to(self.device)

        best_adv_image = image.clone().detach()
        best_Linf = torch.full((batch_size,), float('inf'), device=self.device)
        
        tau = torch.ones(batch_size, device=self.device)
        delta_i_best = torch.ones(batch_size, device=self.device)
        sample_done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        use_fused_adam = self.device.type == 'cuda' and hasattr(torch.optim.Adam, 'fused')

        while (tau > 1.0 / 256.0).any() and not sample_done.all():
            
            sample_done.fill_(True)
            const_val = self.initial_const

            while const_val < self.largest_const:
                x_clamp = torch.clamp(image, 1e-4, 1 - 1e-4)
                w = self.atanh(x_clamp * 2 - 1).clone().detach()
                w.requires_grad = True

                if use_fused_adam:
                    optimizer = optim.Adam([w], lr=self.learning_rate, fused=True)
                else:
                    optimizer = optim.Adam([w], lr=self.learning_rate)

                prev_loss = float('inf')

                for step in range(self.max_iter):
                    loss = self._loss_step(w, image, label, const_val, tau)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()  
                    optimizer.step()

                    # L'arma segreta di ART per risparmiare l'80% del tempo
                    current_loss = loss.item()
                    if abs(prev_loss - current_loss) < self.loss_converged:
                        break
                    prev_loss = current_loss

                with torch.no_grad():
                    eval_img = 0.5 * (torch.tanh(w) + 1)
                    eval_logits = self.model(eval_img)
                    eval_pred = torch.argmax(eval_logits, dim=1)
                    
                    eval_tau = torch.amax(torch.abs(eval_img - image), dim=(1, 2, 3))
                    eval_success = (eval_pred == label) if self.targeted else (eval_pred != label)
                    
                    improved_mask = eval_success & (eval_tau < delta_i_best)
                    
                    if improved_mask.any():
                        best_adv_image[improved_mask] = eval_img[improved_mask].detach()
                        best_Linf[improved_mask] = eval_tau[improved_mask]
                        delta_i_best[improved_mask] = eval_tau[improved_mask]
                        sample_done[improved_mask] = False

                const_val *= self.const_factor

            tau_actual = best_Linf.clone()
            tau_actual[tau_actual == float('inf')] = 1.0 
            
            update_mask = tau_actual < tau
            tau[update_mask] = tau_actual[update_mask]
            
            tau *= self.decrease_factor

        return best_adv_image


import torch.nn.functional as F

from torch.amp import autocast # L'arma segreta per le performance

class PyTorchCarliniLInf_ARTMatch:
    def __init__(self, model, targeted=False, confidence=0.0,
                 learning_rate=0.01, max_iter=50,
                 decrease_factor=0.9, initial_const=1e-3,
                 largest_const=20.0, const_factor=2.0, 
                 loss_converged=0.001, verbose=False): 
        
        self.model = model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.targeted = targeted
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.decrease_factor = decrease_factor
        self.initial_const = initial_const
        self.largest_const = largest_const
        self.const_factor = const_factor
        self.loss_converged = loss_converged
        self.verbose = verbose
        self.device = next(model.parameters()).device

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def forward(self, image, label):
        batch_size = image.size(0)
        image = image.to(self.device)
        label = label.to(self.device).view(-1, 1)

        best_adv_image = image.clone().detach()
        best_Linf = torch.full((batch_size,), float('inf'), device=self.device)
        
        tau = torch.ones(batch_size, device=self.device)
        delta_i_best = torch.ones(batch_size, device=self.device)
        sample_done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        while (tau > 1.0 / 256.0).any() and not sample_done.all():
            
            sample_done.fill_(True)
            const_val = self.initial_const

            while const_val < self.largest_const:
                x_clamp = torch.clamp(image, 1e-4, 1 - 1e-4)
                w = self.atanh(x_clamp * 2 - 1).clone().detach().requires_grad_(True)

                # FUSED ADAM: Usa un solo kernel CUDA per iterazione invece di 5
                try:
                    optimizer = optim.Adam([w], lr=self.learning_rate, fused=True)
                except:
                    optimizer = optim.Adam([w], lr=self.learning_rate)

                prev_loss = torch.tensor(float('inf'), device=self.device)

                for step in range(self.max_iter):
                    
                    # AUTOCAST: Dimezza il peso in memoria e raddoppia i TFLOPS
                    with autocast(device_type='cuda', dtype=torch.float16):
                        adv_image = 0.5 * (torch.tanh(w) + 1)
                        logits = self.model(adv_image)

                        real = logits.gather(1, label).squeeze(1)
                        logits_other = logits.clone()
                        logits_other.scatter_(1, label, -10000.0)
                        other = logits_other.max(dim=1)[0]

                        if self.targeted:
                            loss_1 = F.relu(other - real + self.confidence)
                        else:
                            loss_1 = F.relu(real - other + self.confidence)

                        diff = torch.abs(adv_image - image)
                        loss_2 = torch.sum(F.relu(diff - tau.view(-1, 1, 1, 1)), dim=(1, 2, 3))

                        loss = torch.sum(const_val * loss_1 + loss_2)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()  
                    optimizer.step()

                    # La tua logica Tutto in GPU
                    with torch.no_grad():
                        loss_diff = torch.abs(prev_loss - loss)
                        if loss_diff.item() < self.loss_converged:
                            break
                        prev_loss = loss.detach()

                # Controllo fine loop
                with torch.no_grad():
                    eval_img = 0.5 * (torch.tanh(w) + 1)
                    
                    # Valuta anch'esso al doppio della velocità
                    with autocast(device_type='cuda', dtype=torch.float16):
                        eval_logits = self.model(eval_img)
                        
                    eval_pred = torch.argmax(eval_logits, dim=1)
                    eval_tau = torch.amax(torch.abs(eval_img - image), dim=(1, 2, 3))
                    
                    eval_success = (eval_pred == label.squeeze(1)) if self.targeted else (eval_pred != label.squeeze(1))
                    improved_mask = eval_success & (eval_tau < delta_i_best)
                    
                    if improved_mask.any():
                        best_adv_image[improved_mask] = eval_img[improved_mask].detach()
                        best_Linf[improved_mask] = eval_tau[improved_mask]
                        delta_i_best[improved_mask] = eval_tau[improved_mask]
                        sample_done[improved_mask] = False

                const_val *= self.const_factor

            tau_actual = best_Linf.clone()
            tau_actual[tau_actual == float('inf')] = 1.0 
            
            update_mask = tau_actual < tau
            tau[update_mask] = tau_actual[update_mask]
            tau *= self.decrease_factor

        return best_adv_image


class PyTorchCarliniLInf_Optimized:
    """
    Carlini & Wagner L_infinity SOTA in PyTorch.
    Elimina i cicli annidati utilizzando una Ricerca Binaria Vettorizzata (Binary Search)
    sulla costante 'c' e un aggiornamento dinamico del budget 'tau'.
    """
    def __init__(self, model, targeted=False, confidence=0.0,
                 learning_rate=0.01, max_iter=50, search_steps=9, 
                 initial_const=1e-3, largest_const=20.0): 
        self.model = model
        self.targeted = targeted
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.search_steps = search_steps # Sostituisce i loop while (di default 9 step bastano)
        self.initial_const = initial_const
        self.largest_const = largest_const
        self.device = next(model.parameters()).device

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def forward(self, image, label):
        batch_size = image.size(0)
        image = image.to(self.device)
        label = label.to(self.device)

        # Inizializzazione tensori per la Ricerca Binaria Vettorizzata
        c = torch.full((batch_size,), self.initial_const, device=self.device)
        lower_bound = torch.zeros((batch_size,), device=self.device)
        upper_bound = torch.full((batch_size,), self.largest_const, device=self.device)
        
        # Tau dinamico: parte da 1.0 e si adatta automaticamente ai successi
        tau = torch.ones((batch_size,), device=self.device)

        best_adv_image = image.clone().detach()
        best_Linf = torch.full((batch_size,), float('inf'), device=self.device)

        # UNICO CICLO ESTERNO (Es. 9 iterazioni)
        for search in range(self.search_steps):
            
            x_clamp = torch.clamp(image, 1e-4, 1 - 1e-4)
            w = self.atanh(x_clamp * 2 - 1).clone().detach()
            w.requires_grad = True

            optimizer = optim.Adam([w], lr=self.learning_rate)

            # CICLO DI OTTIMIZZAZIONE INTERNO
            for step in range(self.max_iter):
                adv_image = 0.5 * (torch.tanh(w) + 1)
                logits = self.model(adv_image)

                one_hot = torch.eye(logits.shape[1], device=self.device)[label]
                real = torch.max(one_hot * logits, dim=1)[0]
                other = torch.max((1 - one_hot) * logits - one_hot * 10000, dim=1)[0]

                if self.targeted:
                    loss_1 = torch.clamp(other - real + self.confidence, min=0.0)
                else:
                    loss_1 = torch.clamp(real - other + self.confidence, min=0.0)

                diff = torch.abs(adv_image - image)
                # Il view applica il tau specifico ad ogni immagine nel batch
                loss_2 = torch.sum(torch.clamp(diff - tau.view(-1, 1, 1, 1), min=0.0), dim=(1, 2, 3))

                loss = torch.sum(c * loss_1 + loss_2)

                optimizer.zero_grad()
                loss.backward()  
                optimizer.step()

            # --- FINE DELLO STEP: VALUTAZIONE E AGGIORNAMENTO BINARIO ---
            with torch.no_grad():
                eval_img = 0.5 * (torch.tanh(w) + 1)
                eval_logits = self.model(eval_img)
                eval_pred = torch.argmax(eval_logits, dim=1)
                
                eval_tau = torch.amax(torch.abs(eval_img - image), dim=(1, 2, 3))
                eval_success = (eval_pred == label) if self.targeted else (eval_pred != label)
                
                # 1. Salviamo i migliori risultati trovati
                better_mask = eval_success & (eval_tau < best_Linf)
                if better_mask.any():
                    best_adv_image[better_mask] = eval_img[better_mask].detach()
                    best_Linf[better_mask] = eval_tau[better_mask]

                # 2. Aggiorniamo i bound di Ricerca Binaria usando maschere logiche (Nessun ciclo for di Python!)
                upper_bound = torch.where(eval_success, c, upper_bound)
                lower_bound = torch.where(~eval_success, c, lower_bound)
                
                # 3. Se ha avuto successo, il nuovo target di rumore da battere (tau) è il 90% del risultato attuale
                tau = torch.where(eval_success, eval_tau * 0.9, tau)
                
                # 4. Calcoliamo la 'c' per il prossimo round
                c_next_binary = (lower_bound + upper_bound) / 2.0
                c_next_exponential = c * 2.0
                # Se non abbiamo ancora trovato un upper bound, continuiamo a raddoppiare. Altrimenti dimezziamo.
                c = torch.where(upper_bound < self.largest_const, c_next_binary, c_next_exponential)

        return best_adv_image


class PyTorchCarliniLInf_FullOpt:
    """
    Carlini & Wagner L_infinity nativo in PyTorch vettorizzato (Batch).
    Allineato matematicamente ad ART: utilizza maschere booleane per il 
    tracking indipendente di tau e sample_done per ogni immagine del batch.
    """
    def __init__(self, model, targeted=False, confidence=0.0,
                 learning_rate=0.01, max_iter=50,
                 decrease_factor=0.9, initial_const=1e-3,
                 largest_const=20.0, const_factor=2.0, verbose=False): 
        self.model = model
        self.targeted = targeted
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.decrease_factor = decrease_factor
        self.initial_const = initial_const
        self.largest_const = largest_const
        self.const_factor = const_factor
        self.verbose = verbose
        self.device = next(model.parameters()).device

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def forward(self, image, label):
        batch_size = image.size(0)
        image = image.to(self.device)
        label = label.to(self.device)

        best_adv_image = image.clone().detach()
        best_Linf = torch.full((batch_size,), float('inf'), device=self.device)
        
        # Tau ora è un tensore: ogni immagine ha il suo budget dinamico
        tau = torch.ones(batch_size, device=self.device)
        delta_i_best = torch.ones(batch_size, device=self.device)
        
        # sample_done tiene traccia se un'immagine ha smesso di migliorare
        sample_done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Il ciclo continua finché ci sono tau attivi E almeno un'immagine può ancora migliorare
        while (tau > 1.0 / 256.0).any() and not sample_done.all():
            
            # Assumiamo di aver finito, a meno che non troviamo un miglioramento in questo round
            sample_done.fill_(True)
            const = self.initial_const

            while const < self.largest_const:
                x_clamp = torch.clamp(image, 1e-4, 1 - 1e-4)
                w = self.atanh(x_clamp * 2 - 1).clone().detach()
                w.requires_grad = True

                optimizer = optim.Adam([w], lr=self.learning_rate)

                for step in range(self.max_iter):
                    adv_image = 0.5 * (torch.tanh(w) + 1)
                    logits = self.model(adv_image)

                    one_hot = torch.eye(logits.shape[1], device=self.device)[label]
                    real = torch.max(one_hot * logits, dim=1)[0]
                    other = torch.max((1 - one_hot) * logits - one_hot * 10000, dim=1)[0]

                    if self.targeted:
                        loss_1 = torch.clamp(other - real + self.confidence, min=0.0)
                    else:
                        loss_1 = torch.clamp(real - other + self.confidence, min=0.0)

                    diff = torch.abs(adv_image - image)
                    # view(-1, 1, 1, 1) applica il tau corretto a ciascuna immagine nel batch
                    loss_2 = torch.sum(torch.clamp(diff - tau.view(-1, 1, 1, 1), min=0.0), dim=(1, 2, 3))

                    loss = torch.sum(const * loss_1 + loss_2)

                    optimizer.zero_grad()
                    loss.backward()  
                    optimizer.step()

                # Controllo alla FINE delle max_iter per la costante c attuale
                with torch.no_grad():
                    eval_img = 0.5 * (torch.tanh(w) + 1)
                    eval_logits = self.model(eval_img)
                    eval_pred = torch.argmax(eval_logits, dim=1)
                    
                    eval_tau = torch.amax(torch.abs(eval_img - image), dim=(1, 2, 3))
                    eval_success = (eval_pred == label) if self.targeted else (eval_pred != label)
                    
                    # LOGICA ART: L'attacco ha successo E la perturbazione è minore della migliore trovata finora?
                    improved_mask = eval_success & (eval_tau < delta_i_best)
                    
                    if improved_mask.any():
                        best_adv_image[improved_mask] = eval_img[improved_mask].detach()
                        best_Linf[improved_mask] = eval_tau[improved_mask]
                        delta_i_best[improved_mask] = eval_tau[improved_mask]
                        
                        # Se è migliorata, diciamo all'algoritmo di provare a stringere ancora il tau nel prossimo ciclo!
                        sample_done[improved_mask] = False

                const *= self.const_factor

            # --- AGGIORNAMENTO DI TAU INDIVIDUALE ---
            active_tau_mask = tau > 1.0 / 256.0
            
            # Recuperiamo l'L_inf attuale per le immagini (se infinito, usiamo 1.0 come fallback)
            tau_actual = best_Linf.clone()
            tau_actual[tau_actual == float('inf')] = 1.0 
            
            # Se l'L_inf reale ottenuto è minore del tau imposto, aggiorniamo tau a quel valore
            update_mask = active_tau_mask & (tau_actual < tau)
            tau[update_mask] = tau_actual[update_mask]
            
            # Moltiplichiamo il tau per il decrease_factor (0.9) per stringere il vincolo
            tau[active_tau_mask] *= self.decrease_factor

        return best_adv_image
    

class PyTorchCarliniLInf_EarlyStop:
    """
    Carlini & Wagner L_infinity nativo in PyTorch.
    Supporta l'elaborazione a Batch, Verbose e l'Early Stopping vettorizzato.
    """
    def __init__(self, model, targeted=False, confidence=0.0,
                 learning_rate=0.01, max_iter=50,
                 decrease_factor=0.9, initial_const=1e-3,
                 largest_const=20.0, const_factor=2.0, 
                 early_stop_epsilon=0.10, verbose=False): 
        self.model = model
        self.targeted = targeted
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.decrease_factor = decrease_factor
        self.initial_const = initial_const
        self.largest_const = largest_const
        self.const_factor = const_factor
        self.early_stop_epsilon = early_stop_epsilon
        self.verbose = verbose
        self.device = next(model.parameters()).device

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def forward(self, image, label):
        """
        image: Tensore di shape (Batch, C, H, W)
        label: Tensore di shape (Batch,)
        """
        batch_size = image.size(0)
        image = image.to(self.device)
        label = label.to(self.device)

        # Tensore che conterrà i risultati finali per tutto il batch
        best_adv_image = image.clone().detach()
        best_Linf = torch.full((batch_size,), float('inf'), device=self.device)
        
        # Maschera booleana: tiene traccia di quali immagini hanno già finito (Successo + L_inf <= budget)
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        tau = 1.0 
        # Per i batch, iteriamo finché tutte le immagini non hanno trovato una soluzione
        # o raggiungiamo il limite minimo di tau.
        
        while tau > 1.0 / 256.0 and not is_finished.all():
            const = self.initial_const

            while const < self.largest_const and not is_finished.all():
                x_clamp = torch.clamp(image, 1e-4, 1 - 1e-4)
                w = self.atanh(x_clamp * 2 - 1).clone().detach()
                w.requires_grad = True

                optimizer = optim.Adam([w], lr=self.learning_rate)

                for step in range(self.max_iter):
                    if is_finished.all():
                        break # Se tutto il batch ha finito, usciamo prima!

                    adv_image = 0.5 * (torch.tanh(w) + 1)
                    logits = self.model(adv_image)

                    one_hot = torch.eye(logits.shape[1], device=self.device)[label]
                    
                    real = torch.max(one_hot * logits, dim=1)[0]
                    other = torch.max((1 - one_hot) * logits - one_hot * 10000, dim=1)[0]

                    if self.targeted:
                        loss_1 = torch.clamp(other - real + self.confidence, min=0.0)
                    else:
                        loss_1 = torch.clamp(real - other + self.confidence, min=0.0)

                    # Calcolo penalità L_inf vettorizzato per il batch
                    diff = torch.abs(adv_image - image)
                    loss_2 = torch.sum(torch.clamp(diff - tau, min=0.0), dim=(1, 2, 3))

                    # La loss totale è uno scalare per far funzionare backward() su tutto il batch
                    loss = torch.sum(const * loss_1 + loss_2)

                    optimizer.zero_grad()
                    loss.backward()  
                    optimizer.step()
                    
                    # --- CONTROLLO EARLY STOPPING VETTORIZZATO ---
                    if step % 5 == 0 or step == self.max_iter - 1:
                        with torch.no_grad():
                            eval_img = 0.5 * (torch.tanh(w) + 1)
                            eval_logits = self.model(eval_img)
                            eval_pred = torch.argmax(eval_logits, dim=1)
                            
                            # Calcola L_inf per ogni singola immagine nel batch: shape (Batch,)
                            eval_tau = torch.amax(torch.abs(eval_img - image), dim=(1, 2, 3))
                            eval_success = (eval_pred == label) if self.targeted else (eval_pred != label)
                            
                            # Troviamo quali immagini:
                            # 1. Hanno successo
                            # 2. Sono sotto il budget (0.10)
                            # 3. Non erano GIA' state salvate prima
                            newly_finished = eval_success & (eval_tau <= self.early_stop_epsilon) & (~is_finished)
                            
                            if newly_finished.any():
                                # Salviamo i risultati SOLO per le immagini appena completate
                                best_adv_image[newly_finished] = eval_img[newly_finished].detach()
                                best_Linf[newly_finished] = eval_tau[newly_finished]
                                
                                # Aggiorniamo la maschera globale
                                is_finished |= newly_finished
                                
                                if self.verbose:
                                    print(f"      [C&W] {newly_finished.sum().item()} img concluse allo step {step} (Tot: {is_finished.sum().item()}/{batch_size})")

                # C&W Update Constants (Semplificato per il batch)
                const *= self.const_factor

            # Riduciamo Tau (Se alcune immagini hanno un L_inf < tau, usiamo la media dei successi come nuovo tau)
            if best_Linf[best_Linf < float('inf')].numel() > 0:
                min_found_tau = best_Linf[best_Linf < float('inf')].mean().item()
                if min_found_tau < tau:
                    tau = min_found_tau
            tau *= self.decrease_factor

        if self.verbose:
            print(f"   [C&W Completato] Successi entro il budget: {is_finished.sum().item()}/{batch_size}")
            
        return best_adv_image
