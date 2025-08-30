import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional, Tuple, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_config import NeuratekConfig


class BiologicalLIFNeuron(nn.Module):
    """Biological Leaky Integrate-and-Fire neuron with STDP and homeostasis"""
    
    def __init__(self, config: NeuratekConfig, neuron_type: str = "excitatory"):
        super().__init__()
        self.config = config
        self.neuron_type = neuron_type
        
        # temporal dynamics
        self.dt = 1e-3
        self.tau_mem = config.tau_membrane
        self.tau_syn = config.tau_synapse
        
        # decay coefficients with biological variability
        base_alpha_mem = math.exp(-self.dt / self.tau_mem)
        base_alpha_syn = math.exp(-self.dt / self.tau_syn)
        
        self.alpha_mem_var = nn.Parameter(torch.ones(1) * 0.2)
        self.alpha_syn_var = nn.Parameter(torch.ones(1) * 0.15)
        
        self.base_alpha_mem = base_alpha_mem
        self.base_alpha_syn = base_alpha_syn
        
        # voltage parameters
        self.v_rest = config.v_rest
        self.v_reset = config.v_reset
        self.v_th_base = config.v_threshold_base
        self.v_th_range = config.v_threshold_range
        
        # current injection
        self.current_scale = config.current_injection_scale
        self.current_baseline = config.current_baseline
        self.current_noise_factor = config.current_noise_factor
        
        # noise sources
        self.membrane_noise = nn.Parameter(torch.ones(1) * config.membrane_noise_std)
        self.synaptic_noise = nn.Parameter(torch.ones(1) * config.synaptic_noise_std)
        self.threshold_noise = nn.Parameter(torch.ones(1) * config.threshold_noise_std)
        
        # calibrated variability sources
        self.pink_noise_strength = nn.Parameter(torch.ones(1) * 1.2)
        self.burst_chaos_factor = nn.Parameter(torch.ones(1) * 1.0)
        self.synaptic_jitter = nn.Parameter(torch.ones(1) * 0.8)
        
        # stdp synaptic plasticity
        self.stdp_lr = config.stdp_learning_rate
        self.stdp_tau_plus = config.stdp_tau_plus
        self.stdp_tau_minus = config.stdp_tau_minus
        self.synaptic_weights = nn.Parameter(torch.ones(1) * 1.0)
        
        # homeostatic plasticity
        self.homeostasis_tau = config.homeostasis_tau
        self.target_rate = config.target_firing_rate
        self.homeostasis_strength = config.homeostasis_strength
        self.running_rate = nn.Parameter(torch.ones(1) * self.target_rate)
        self.homeostatic_scaling = nn.Parameter(torch.ones(1) * 1.0)
        
        # biological rhythms
        self.theta_freq = config.theta_frequency
        self.gamma_freq = config.gamma_frequency
        self.background_noise = config.background_noise_factor
        
        # neuron-type specific parameters
        if neuron_type == "excitatory":
            self.threshold_bias = nn.Parameter(torch.randn(1) * 4.0 + 3.0)
            self.current_multiplier = 0.4 + torch.randn(1).item() * 0.15
            self.adaptation_strength = nn.Parameter(torch.ones(1) * 25e-3)
            self.burst_probability = nn.Parameter(torch.ones(1) * 0.005)
            self.burst_duration = 3
            self.burst_chaos = nn.Parameter(torch.ones(1) * 1.5)
        else:  # inhibitory
            self.threshold_bias = nn.Parameter(torch.randn(1) * 4.0 + 2.0)
            self.current_multiplier = 0.5 + torch.randn(1).item() * 0.15
            self.adaptation_strength = nn.Parameter(torch.ones(1) * 30e-3)
            self.rebound_probability = nn.Parameter(torch.ones(1) * 0.008)
            self.rebound_delay = 5
            self.rebound_chaos = nn.Parameter(torch.ones(1) * 1.8)
            
        # individual neuron variability
        self.individual_tau_factor = nn.Parameter(torch.randn(1) * 0.3 + 1.0)
        
        # per-neuron noise factors
        individual_noise_factor_per_neuron = torch.empty(config.snn_neurons_per_layer).uniform_(0.7, 1.3)
        self.register_buffer('individual_noise_factor_per_neuron', individual_noise_factor_per_neuron)
        
        self.individual_rhythm_phase = nn.Parameter(torch.rand(1) * 2 * math.pi)
        self.individual_chaos_seed = nn.Parameter(torch.randn(1) * 1.2)
        
        # per-neuron threshold diversification
        v_th_offset = torch.empty(config.snn_neurons_per_layer).uniform_(-0.5, 0.5) * config.v_threshold_range * config.threshold_diversification_scale
        self.register_buffer('v_th_offset', v_th_offset)
        
        # adaptive noise scaling
        noise_scale_per_neuron = torch.empty(config.snn_neurons_per_layer).uniform_(0.8, 1.2)
        self.register_buffer('noise_scale_per_neuron', noise_scale_per_neuron)
        
        # state tracking
        self.spike_history = []
        self.burst_state = 0
        self.rebound_timer = 0
    
    def _generate_pink_noise(self, shape: Tuple, device: torch.device) -> torch.Tensor:
        """Generate pink noise for biological realism"""
        white1 = torch.randn(shape, device=device) * 1.0
        white2 = torch.randn(shape, device=device) * 0.5
        white3 = torch.randn(shape, device=device) * 0.25
        white4 = torch.randn(shape, device=device) * 0.125
        
        pink_noise = white1 + white2 + white3 + white4
        return pink_noise * 0.1
            
    def forward(self, input_embedding: torch.Tensor, state: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict, Dict]:
        """Forward pass with biological realism"""
        batch_size, seq_len, hidden_size = input_embedding.shape
        device = input_embedding.device
        dtype = input_embedding.dtype
        
        # initialize biological state
        if state is None:
            init_membrane_noise = torch.randn((batch_size, hidden_size), device=device, dtype=dtype) * 3.0
            state = {
                'v_membrane': torch.full((batch_size, hidden_size), self.v_rest, 
                                       device=device, dtype=dtype) + init_membrane_noise,
                'synaptic_current': torch.randn((batch_size, hidden_size), 
                                              device=device, dtype=dtype) * 0.02,
                'refractory_time': torch.zeros((batch_size, hidden_size), 
                                             device=device, dtype=dtype),
                'adaptation': torch.randn((batch_size, hidden_size), 
                                        device=device, dtype=dtype) * 0.02,
                'stdp_trace': torch.zeros((batch_size, hidden_size), device=device, dtype=dtype),
                'burst_counter': torch.zeros((batch_size, hidden_size), device=device, dtype=dtype),
                'rebound_timer': torch.zeros((batch_size, hidden_size), device=device, dtype=dtype),
                'last_spike_time': torch.full((batch_size, hidden_size), -1000.0, device=device, dtype=dtype),
                'firing_rate_history': torch.zeros((batch_size, hidden_size), device=device, dtype=dtype)
            }
        
        # biological current injection
        base_current = input_embedding * self.current_scale * self.current_multiplier
        
        # homeostatic scaling
        homeostatic_factor = torch.clamp(self.homeostatic_scaling, 0.5, 2.0)
        base_current = base_current * homeostatic_factor
        
        # homeostatic current control
        if 'firing_rate_history' in state:
            current_avg_rate = torch.mean(state['firing_rate_history'])
            rate_error = current_avg_rate - self.target_rate
            homeostatic_current_correction = -rate_error * self.homeostasis_strength * 2.0
            base_current = base_current + homeostatic_current_correction
        
        # biological baseline with variability
        baseline_current = self.current_baseline * (1 + torch.randn_like(input_embedding) * self.current_noise_factor)
        input_current = base_current + baseline_current
        
        # biological noise sources
        synaptic_noise_poisson = torch.poisson(torch.ones_like(input_current) * 0.1) * torch.clamp(self.synaptic_noise, 0.1, 1.5) * torch.randn_like(input_current)
        background_noise = torch.randn_like(input_current) * self.background_noise * torch.rand_like(input_current)
        pink_noise = self._generate_pink_noise(input_current.shape, input_current.device) * torch.clamp(self.pink_noise_strength, 0.5, 2.0)
        jitter_noise = torch.randn_like(input_current) * torch.clamp(self.synaptic_jitter, 0.2, 1.2) * torch.sin(torch.randn_like(input_current) * 10)
        
        # biological rhythms
        time_steps = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)
        theta_rhythm = torch.sin(2 * math.pi * self.theta_freq * time_steps * self.dt + self.individual_rhythm_phase) * 0.05
        gamma_rhythm = torch.sin(2 * math.pi * self.gamma_freq * time_steps * self.dt + self.individual_rhythm_phase * 2) * 0.02
        
        # chaotic modulation
        chaos_modulation = torch.sin(self.individual_chaos_seed.clone().detach().to(device) + time_steps * 0.1) * torch.randn_like(input_current) * 0.1
        
        # combine noise sources
        total_noise = (synaptic_noise_poisson + background_noise + pink_noise + jitter_noise + theta_rhythm + gamma_rhythm + chaos_modulation) * self.config.master_variability_scale
        input_current = input_current + total_noise
        
        # process timesteps
        spikes_list = []
        spike_metrics = {'total_spikes': 0, 'spike_times': [], 'variability_factors': [], 'burst_events': 0, 'rebound_events': 0}
        
        # temporal downsampling for performance
        temporal_step = max(1, self.config.temporal_downsampling)
        
        for t in range(0, seq_len, temporal_step):
            current_time = t * self.dt
            I_syn = input_current[:, t, :]
            
            # biological time constants with variability
            alpha_mem_varied = self.base_alpha_mem * (1 + torch.randn(1, device=device) * torch.clamp(self.alpha_mem_var, 0.1, 0.3))
            alpha_syn_varied = self.base_alpha_syn * (1 + torch.randn(1, device=device) * torch.clamp(self.alpha_syn_var, 0.1, 0.25))
            
            # update synaptic current with stdp modulation
            stdp_modulation = 1.0 + 0.5 * torch.tanh(state['stdp_trace'] * 10.0)
            state['synaptic_current'] = (alpha_syn_varied * state['synaptic_current'] + 
                                        (1 - alpha_syn_varied) * I_syn * stdp_modulation)
            
            # biological noise components
            membrane_noise_lognormal = torch.exp(torch.randn_like(state['v_membrane']) * 0.3) * torch.clamp(self.membrane_noise, 1.0, 2.5) - 1.0
            
            # individual neuron noise
            noise_factors_this_layer = self.individual_noise_factor_per_neuron[:hidden_size]
            individual_noise = torch.randn_like(state['v_membrane']) * noise_factors_this_layer
            
            # temporal modulation noise
            temporal_modulation = torch.sin(torch.tensor(current_time * 50.0, device=device, dtype=state['v_membrane'].dtype)) * torch.randn_like(state['v_membrane']) * 0.5
            
            # chaos noise
            chaos_noise = torch.randn_like(state['v_membrane']) * torch.clamp(self.individual_chaos_seed, 0.5, 2.0) * torch.sin(torch.tensor(current_time * 100.0, device=device, dtype=state['v_membrane'].dtype))
            
            # bursting and rebound noise
            burst_noise = torch.zeros_like(state['v_membrane'])
            rebound_noise = torch.zeros_like(state['v_membrane'])
            
            if self.neuron_type == "excitatory":
                burst_trigger = torch.rand_like(state['v_membrane']) < torch.clamp(self.burst_probability, 0.001, 0.01)
                chaotic_burst_intensity = torch.clamp(self.burst_chaos, 0.5, 1.5) * torch.randn_like(state['v_membrane'])
                burst_noise = torch.where(burst_trigger, chaotic_burst_intensity * 1.5, burst_noise)
                state['burst_counter'] = torch.where(burst_trigger, torch.full_like(state['burst_counter'], self.burst_duration), 
                                                   torch.clamp(state['burst_counter'] - 1, 0, float('inf')))
                spike_metrics['burst_events'] += burst_trigger.sum().item()
            else:
                rebound_trigger = torch.rand_like(state['v_membrane']) < torch.clamp(self.rebound_probability, 0.002, 0.02)
                chaotic_rebound_intensity = torch.clamp(self.rebound_chaos, 0.8, 2.0) * torch.randn_like(state['v_membrane'])
                rebound_noise = torch.where(rebound_trigger, chaotic_rebound_intensity * 1.2, rebound_noise)
            
            # total membrane noise
            total_membrane_noise = (membrane_noise_lognormal + individual_noise + temporal_modulation + chaos_noise + burst_noise + rebound_noise) * self.config.master_variability_scale
            
            # adaptive noise injection
            adaptation_state_norm = torch.clamp(state['adaptation'], 0, 1)
            adaptive_noise_std = adaptation_state_norm * self.config.adaptive_noise_sensitivity
            scale_this_layer = self.noise_scale_per_neuron[:hidden_size]
            adaptive_noise = torch.randn_like(state['v_membrane']) * adaptive_noise_std * scale_this_layer
            
            state['v_membrane'] = state['v_membrane'] + adaptive_noise
            
            # membrane dynamics
            not_refractory = state['refractory_time'] <= 0
            tau_individual = self.tau_mem * torch.clamp(self.individual_tau_factor, 0.5, 1.5)
            
            # membrane equation with adaptation
            dv = (-(state['v_membrane'] - self.v_rest) + 
                  state['synaptic_current'] - state['adaptation']) / tau_individual * self.dt
            
            # bursting enhancement
            burst_enhancement = torch.where(state['burst_counter'] > 0, 2.0, 1.0)
            dv = dv * burst_enhancement
            
            state['v_membrane'] = torch.where(
                not_refractory,
                state['v_membrane'] + dv + total_membrane_noise,
                state['v_membrane']
            )
            
            # dynamic threshold with variability
            threshold_noise_beta = torch.distributions.Beta(2, 5).sample(state['v_membrane'].shape).to(device) * torch.clamp(self.threshold_noise, 1.0, 4.0)
            threshold_individual = torch.randn_like(state['v_membrane']) * self.v_th_range
            
            # homeostatic threshold adjustment
            rate_error = state['firing_rate_history'] - self.target_rate
            homeostatic_threshold_adj = rate_error * self.homeostasis_strength
            
            # stdp-dependent threshold modulation
            stdp_threshold_mod = torch.tanh(state['stdp_trace'] * 5.0) * 1.0
            
            # chaotic threshold modulation
            chaos_threshold_mod = torch.sin(self.individual_chaos_seed + torch.tensor(current_time * 75.0, device=device, dtype=state['v_membrane'].dtype)) * torch.randn_like(state['v_membrane']) * 0.5
            
            # burst-dependent threshold variability
            burst_threshold_mod = torch.where(state['burst_counter'] > 0, 
                                            torch.randn_like(state['v_membrane']) * 1.5,
                                            torch.zeros_like(state['v_membrane']))
            
            # combined dynamic threshold with per-neuron diversification
            offset_this_layer = self.v_th_offset[:hidden_size]
            
            v_threshold = (self.v_th_base + self.threshold_bias + 
                          offset_this_layer +
                          threshold_noise_beta + threshold_individual + 
                          homeostatic_threshold_adj + stdp_threshold_mod +
                          chaos_threshold_mod + burst_threshold_mod)
            
            # spike detection
            deterministic_spikes = state['v_membrane'] > v_threshold
            spike_probability = torch.sigmoid((state['v_membrane'] - v_threshold) * 3.0)
            probabilistic_spikes = torch.rand_like(spike_probability) < spike_probability * 0.1
            
            # rebound spikes for inhibitory neurons
            rebound_spikes = torch.zeros_like(deterministic_spikes)
            if self.neuron_type == "inhibitory":
                rebound_trigger = (state['rebound_timer'] <= 0) & (torch.rand_like(state['v_membrane']) < torch.clamp(self.rebound_probability, 0.03, 0.12))
                rebound_spikes = rebound_trigger & (state['v_membrane'] < self.v_th_base - 5.0)
                state['rebound_timer'] = torch.where(rebound_trigger, torch.full_like(state['rebound_timer'], self.rebound_delay), 
                                                   torch.clamp(state['rebound_timer'] - 1, 0, float('inf')))
                spike_metrics['rebound_events'] += rebound_spikes.sum().item()
            
            # combine spike mechanisms
            spikes = ((deterministic_spikes | probabilistic_spikes | rebound_spikes) & not_refractory)
            
            # spatiotemporal dropout during training
            if self.training:
                dropout_mask = torch.rand_like(spikes, dtype=torch.float32) > self.config.spatiotemporal_dropout_p
                spikes = spikes & dropout_mask.bool()
            
            # biological reset with variability
            reset_noise = torch.randn_like(state['v_membrane']) * 3.0
            reset_voltage = self.v_reset + reset_noise
            
            state['v_membrane'] = torch.where(spikes, reset_voltage, state['v_membrane'])
            
            # refractory period with variability
            refractory_variability = torch.randn_like(state['refractory_time']) * 1e-3 + self.config.tau_refractoriness
            state['refractory_time'] = torch.where(spikes,
                                                  torch.clamp(refractory_variability, 1e-3, 8e-3),
                                                  torch.clamp(state['refractory_time'] - self.dt, 0, float('inf')))
            
            # adaptation with stdp influence
            adaptation_base = torch.clamp(self.adaptation_strength, 3e-3, 12e-3)
            adaptation_variability = 1.0 + torch.randn_like(state['adaptation']) * 0.5
            adaptation_increment = adaptation_base * adaptation_variability
            
            state['adaptation'] = torch.where(spikes,
                                            state['adaptation'] + adaptation_increment,
                                            state['adaptation'] * math.exp(-self.dt / self.config.tau_adaptation))
            
            # update biological plasticity
            spike_float = spikes.float()
            state['stdp_trace'] = state['stdp_trace'] * math.exp(-self.dt / self.stdp_tau_plus) + spike_float * self.stdp_lr
            
            # update last spike time for stdp
            state['last_spike_time'] = torch.where(spikes, torch.full_like(state['last_spike_time'], current_time), state['last_spike_time'])
            
            # update firing rate history for homeostasis
            alpha_rate = math.exp(-self.dt / self.homeostasis_tau)
            state['firing_rate_history'] = alpha_rate * state['firing_rate_history'] + (1 - alpha_rate) * spike_float / self.dt
            
            # update synaptic weights via stdp
            dt_spike = current_time - state['last_spike_time']
            stdp_window = torch.exp(-dt_spike / self.stdp_tau_plus) * (dt_spike > 0) * (dt_spike < 0.1)
            self.synaptic_weights.data += torch.mean(stdp_window * spike_float) * self.stdp_lr * 0.001
            self.synaptic_weights.data = torch.clamp(self.synaptic_weights.data, 0.1, 3.0)
            
            # homeostatic scaling update
            rate_error = torch.mean(state['firing_rate_history']) - self.target_rate
            self.homeostatic_scaling.data += -rate_error * self.homeostasis_strength * 0.001
            self.homeostatic_scaling.data = torch.clamp(self.homeostatic_scaling.data, 0.5, 2.0)
            
            # collect spikes and metrics
            spikes_list.append(spike_float)
            
            spike_count = spike_float.sum().item()
            spike_metrics['total_spikes'] += spike_count
            if spike_count > 0:
                spike_metrics['spike_times'].append(t)
                
            # variability tracking
            noise_factors_std = self.individual_noise_factor_per_neuron[:hidden_size].std().item()
            threshold_offsets_std = self.v_th_offset[:hidden_size].std().item()
            adaptive_noise_std = self.noise_scale_per_neuron[:hidden_size].std().item()
            
            variability_factor = {
                'threshold_std': threshold_noise_beta.std().item(),
                'membrane_noise_std': total_membrane_noise.std().item(),
                'adaptation_mean': state['adaptation'].mean().item(),
                'firing_rate_mean': state['firing_rate_history'].mean().item(),
                'stdp_trace_mean': state['stdp_trace'].mean().item(),
                'synaptic_weight': self.synaptic_weights.item(),
                'homeostatic_scaling': self.homeostatic_scaling.item(),
                'per_neuron_noise_diversity': noise_factors_std,
                'per_neuron_threshold_diversity': threshold_offsets_std,
                'per_neuron_adaptive_diversity': adaptive_noise_std,
                'membrane_voltage_std': state['v_membrane'].std().item(),
                'adaptation_state_std': state['adaptation'].std().item(),
                'firing_rate_std': state['firing_rate_history'].std().item()
            }
            spike_metrics['variability_factors'].append(variability_factor)
            
            # handle temporal downsampling
            if temporal_step > 1:
                for _ in range(temporal_step):
                    if len(spikes_list) < seq_len:
                        spikes_list.append(spike_float)
            else:
                spikes_list.append(spike_float)
        
        # ensure correct sequence length
        while len(spikes_list) < seq_len:
            spikes_list.append(torch.zeros_like(spike_float))
        
        # stack spikes
        spike_output = torch.stack(spikes_list[:seq_len], dim=1)
        
        return spike_output, state, spike_metrics


class SNNReflexiveStream(nn.Module):
    """Spiking Neural Network reflexive stream with biological variability"""
    
    def __init__(self, config: NeuratekConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        
        # input projection
        self.input_projection = nn.Linear(config.hidden_size, config.snn_hidden_size)
        self.input_noise = nn.Parameter(torch.ones(1) * 0.3)
        
        # snn layers
        for i in range(config.snn_layers):
            total_neurons = config.snn_neurons_per_layer
            excitatory_neurons = int(total_neurons * config.excitatory_ratio)
            inhibitory_neurons = total_neurons - excitatory_neurons
            
            layer_dict = nn.ModuleDict({
                'excitatory': BiologicalLIFNeuron(config, "excitatory"),
                'inhibitory': BiologicalLIFNeuron(config, "inhibitory"),
                'projection': nn.Linear(config.snn_hidden_size, config.snn_hidden_size),
                'norm': nn.LayerNorm(config.snn_hidden_size),
                'dropout': nn.Dropout(0.25),
            })
            
            layer_dict.register_parameter('layer_noise', nn.Parameter(torch.ones(1) * 0.15))
            self.layers.append(layer_dict)
            
        # output projection
        self.output_projection = nn.Linear(config.snn_hidden_size, config.hidden_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with enhanced variability"""
        batch_size, seq_len, hidden_size = x.shape
        
        # project input with noise injection
        x = self.input_projection(x)
        input_noise = torch.randn_like(x) * torch.clamp(self.input_noise, 0.15, 0.5) * self.config.master_variability_scale
        x = x + input_noise
        
        # initialize metrics
        layer_states = []
        stream_metrics = {
            'total_spikes': 0, 
            'layer_spikes': [], 
            'functional_layers': 0,
            'variability_metrics': []
        }
        
        # process through snn layers
        for i, layer in enumerate(self.layers):
            # add per-layer noise
            layer_noise = torch.randn_like(x) * torch.clamp(layer.layer_noise, 0.08, 0.3) * self.config.master_variability_scale
            x_noisy = x + layer_noise
            
            # split for excitatory/inhibitory
            excitatory_size = int(self.config.snn_neurons_per_layer * self.config.excitatory_ratio)
            inhibitory_size = self.config.snn_neurons_per_layer - excitatory_size
            
            x_exc = x_noisy[:, :, :excitatory_size]
            x_inh = x_noisy[:, :, excitatory_size:excitatory_size + inhibitory_size]
            
            # process through neurons
            spikes_exc, state_exc, metrics_exc = layer['excitatory'](x_exc)
            spikes_inh, state_inh, metrics_inh = layer['inhibitory'](x_inh)
            
            # combine with cross-connections
            if inhibitory_size > 0:
                spikes = torch.cat([spikes_exc, spikes_inh], dim=-1)
            else:
                spikes = spikes_exc
                
            # apply projection
            x = layer['projection'](spikes)
            x = layer['norm'](x)
            x = layer['dropout'](x)
            
            # collect metrics
            exc_spikes = metrics_exc.get('total_spikes', 0)
            inh_spikes = metrics_inh.get('total_spikes', 0)
            layer_spike_count = exc_spikes + inh_spikes
            
            stream_metrics['total_spikes'] += layer_spike_count
            stream_metrics['layer_spikes'].append(layer_spike_count)
            
            if layer_spike_count > 0:
                stream_metrics['functional_layers'] += 1
                
            # variability metrics
            if metrics_exc.get('variability_factors'):
                variability_measures = []
                for vf in metrics_exc['variability_factors']:
                    threshold_var = vf.get('threshold_std', 0)
                    membrane_var = vf.get('membrane_noise_std', 0) 
                    adaptation_var = abs(vf.get('adaptation_mean', 0) - 0.01)
                    rate_var = abs(vf.get('firing_rate_mean', 8.0) - 8.0)
                    stdp_var = abs(vf.get('stdp_trace_mean', 0))
                    
                    per_neuron_noise_div = vf.get('per_neuron_noise_diversity', 0)
                    per_neuron_threshold_div = vf.get('per_neuron_threshold_diversity', 0) 
                    per_neuron_adaptive_div = vf.get('per_neuron_adaptive_diversity', 0)
                    
                    membrane_voltage_std = vf.get('membrane_voltage_std', 0)
                    adaptation_state_std = vf.get('adaptation_state_std', 0)
                    firing_rate_std = vf.get('firing_rate_std', 0)
                    
                    # composite variability metric
                    composite_variability = (
                        threshold_var * 1.0 +
                        membrane_var * 0.3 +
                        adaptation_var * 5.0 +
                        rate_var * 0.1 +
                        stdp_var * 2.0 +
                        per_neuron_noise_div * 10.0 +
                        per_neuron_threshold_div * 8.0 +
                        per_neuron_adaptive_div * 6.0 +
                        membrane_voltage_std * 2.0 +
                        adaptation_state_std * 3.0 +
                        firing_rate_std * 1.5
                    )
                    variability_measures.append(composite_variability)
                
                avg_variability = np.mean(variability_measures) if variability_measures else 0
                stream_metrics['variability_metrics'].append(avg_variability)
                
            layer_states.append({'excitatory': state_exc, 'inhibitory': state_inh})
        
        # output projection
        x = self.output_projection(x)
        
        return x, {'states': layer_states, 'metrics': stream_metrics}
