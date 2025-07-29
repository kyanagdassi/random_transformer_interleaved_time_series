import torch
import torch.nn as nn
from transformers import Mamba2Model, Mamba2Config
from models.lightning_base_model import BaseModel
from core import Config
from linalg_helpers import print_matrix
import pickle

config = Config()


class Mamba2(BaseModel):
    def __init__(self, n_dims_in, n_positions, n_embd, n_layer=12, n_dims_out=5, learning_rate=config.learning_rate, use_pos_emb=config.use_pos_emb):
        super(Mamba2, self).__init__(learning_rate=learning_rate)
        
        self.n_positions = n_positions
        self.n_dims_in = n_dims_in
        self.n_dims_out = n_dims_out
        self.use_pos_emb = use_pos_emb
        
        # Input projection
        self._read_in = nn.Linear(n_dims_in, n_embd)
        
        # Positional embedding (optional)
        if self.use_pos_emb:
            self.pos_embedding = nn.Parameter(torch.randn(1, n_positions, n_embd))
        
        # Mamba-2 configuration - properly sized for our task
        mamba2_config = Mamba2Config(
            hidden_size=n_embd,           # Match our embedding dimension (128)
            state_size=32,               # Smaller state space dimension
            conv_kernel=4,                 # Convolution kernel size
            expand=1,                 # Block expansion factor
            num_heads=8,
            head_dim=16,               # Smaller head dimension
            n_groups=1,                # Number of groups
            chunk_size=256,           # Chunk size for efficient computation
            num_hidden_layers=n_layer,       # Number of layers (12)
            vocab_size=1,        # Set to embedding size to avoid huge embedding layer
            use_cache=True,   
            initializer_range=0.1,   # Weight initialization range
        )

        # Mamba2Model backbone (without its own embedding layer)=
        self.backbone = Mamba2Model(mamba2_config)
        
        # Output projection
        self._read_out = nn.Linear(n_embd, n_dims_out)
        
        self.name = f"mamba2_embd={n_embd}_layer={n_layer}"

    def predict_step(self, input_dict, batch_idx=None):
        current = input_dict["current"]
        embeds = self._read_in(current)
        
        # # Add positional embedding if enabled
        # if self.use_pos_emb:
        #     seq_len = embeds.size(1)
        #     if seq_len <= self.n_positions:
        #         embeds = embeds + self.pos_embedding[:, :seq_len, :]
        #     else:
        #         # Handle sequences longer than max positions
        #         embeds = embeds + self.pos_embedding[:, -seq_len:, :]
        
        output = self.backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        # predict only on xs
        return input_dict, {"preds": prediction}

    def forward(self, input_dict, batch_idx=None, return_intermediate_dict=False):
        input_dict, intermediate_dict = self.predict_step(input_dict, batch_idx)

        # Calculate all loss terms and metrics (scores)
        output_dict = self.calculate_losses_and_metrics(input_dict, intermediate_dict)

        # Calculate optimized loss
        optimized_loss = 0
        for key, loss in output_dict.items():
            if "loss_" in key:
                optimized_loss += loss
        output_dict["optimized_loss"] = optimized_loss
        return (intermediate_dict, output_dict) if return_intermediate_dict else output_dict

    def calculate_losses_and_metrics(self, input_dict, intermediate_dict):
        
        # Calculate loss
        ys = input_dict["target"]

        preds = intermediate_dict["preds"]
        res_sq = (preds - ys) ** 2 #residuals squared

        if config.multi_sys_trace:

            if config.mem_suppress and config.masking:
                #create a mask to identify rows of ys that are all zeros and also all indices from the list of lists input_dict["mask_idx"]
                #ys is of shape [batch_size, seq_len, dims]
                mask_all_zeros = torch.all(ys == 0, dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
                
                mask_selected_indices = torch.zeros_like(mask_all_zeros, dtype=torch.bool)
                for b, idx_list in enumerate(input_dict["mask_idx"]):
                    mask_idx_minus_1 = [int(idx) - 1 for idx in idx_list]
                    mask_selected_indices[b, mask_idx_minus_1, :] = True #since the target has one less entry than the full seqment we need to subtract 1 from the index

                mask = mask_all_zeros | mask_selected_indices #combine the two masks with a logical OR

            else:
                # Create a mask to identify rows of ys that are all zeros
                mask = torch.all(ys == 0, dim=-1, keepdim=True)
            
            # Apply the mask to res_sq to disregard the residuals for rows of ys that are all zeros
            res_sq = res_sq.masked_fill(mask, 0)

            output_dict = {"loss_mse": torch.sum(res_sq) / (~mask).sum()} #mean squared error loss
        else:
            output_dict = {"loss_mse": torch.mean(res_sq)}
            

        # Calculate metrics
        for i in range(ys.shape[1]):
            for j in range(ys.shape[2]):
                output_dict[f"metric_mse_ts{i}_dim_{j}"] = torch.mean(res_sq[:, i, j])

        return output_dict

    def predict_ar(self, ins, fix_window_len=True):
        ins = torch.from_numpy(ins).float().to(self.device)
        one_d = False
        if ins.ndim == 2:
            one_d = True
            ins = ins.unsqueeze(0)
        bsize, points, _ = ins.shape
        d_o = self.n_dims_out
        outs = torch.zeros(bsize, 1, d_o).to(self.device)
        with torch.no_grad():
            for i in range(1, points + 1):
                I = ins[:, :i]
                if fix_window_len and I.shape[1] > self.n_positions:
                    I = I[:, -self.n_positions:]
                _, interm = self.predict_step({"current": I})
                pred = interm["preds"][:, -1:]  # b, 1, d_o
                outs = torch.cat([outs, pred], dim=1)
        outs = outs.detach().cpu().numpy()
        if one_d:
            outs = outs[0]
        return outs
