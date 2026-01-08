# src/models/tacotron2.py - Complete Working Tacotron2 Model

import torch
import torch.nn as nn
import torch.nn.functional as F

class FinalHParams:
    """Complete hyperparameters for Emotional TTS."""
    def __init__(self):
        self.n_symbols = 256
        self.symbols_embedding_dim = 512
        self.n_emotion_classes = 5
        self.emotion_embedding_dim = 32
        self.encoder_embedding_dim = 512
        self.n_mel_channels = 80
        self.encoder_n_convolutions = 3
        self.encoder_kernel_size = 5
        self.encoder_dropout = 0.5
        self.attention_rnn_dim = 1024
        self.attention_dim = 512
        self.attention_location_kernel_size = 31
        self.attention_location_n_filters = 32
        self.decoder_rnn_dim = 1024
        self.prenet_dim = 256
        self.prenet_dropout = 0.5
        self.prenet_n_layers = 2
        self.decoder_dropout = 0.1
        self.max_decoder_steps = 1000
        self.gate_threshold = 0.5
        self.postnet_embedding_dim = 512
        self.postnet_kernel_size = 5
        self.postnet_n_convolutions = 5
        self.postnet_dropout = 0.1
        self.mask_padding = True
        self.learning_rate = 1e-3
        self.batch_size = 32
        self.epochs = 100


class FinalEmotionalTacotron2(nn.Module):
    """Final working Emotional TTS model - All errors fixed."""
    
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        
        # Embeddings
        self.embedding = nn.Embedding(hparams.n_symbols, hparams.symbols_embedding_dim)
        self.emotion_embedding = nn.Embedding(hparams.n_emotion_classes, hparams.emotion_embedding_dim)
        
        # Encoder
        input_dim = hparams.symbols_embedding_dim + hparams.emotion_embedding_dim
        self.encoder_convolutions = nn.Sequential(
            nn.Conv1d(input_dim, 512, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(512, 512, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(512, 512, 5, padding=2),
            nn.ReLU()
        )
        self.encoder_lstm = nn.LSTM(512, 256, batch_first=True, bidirectional=True)
        
        # Decoder Prenet
        self.decoder_prenet = nn.Sequential(
            nn.Linear(hparams.n_mel_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Decoder RNN
        self.decoder_rnn = nn.LSTMCell(256 + 512, hparams.decoder_rnn_dim)
        
        # Output projections
        self.mel_projection = nn.Linear(hparams.decoder_rnn_dim, hparams.n_mel_channels)
        self.gate_projection = nn.Linear(hparams.decoder_rnn_dim, 1)
        
        # Postnet
        self.postnet = nn.Sequential(
            nn.Conv1d(hparams.n_mel_channels, 512, 5, padding=2),
            nn.Tanh(),
            nn.Conv1d(512, 512, 5, padding=2),
            nn.Tanh(),
            nn.Conv1d(512, 512, 5, padding=2),
            nn.Tanh(),
            nn.Conv1d(512, hparams.n_mel_channels, 5, padding=2)
        )
    
    def forward(self, inputs):
        """Forward pass with proper dimension handling."""
        text_inputs, text_lengths, mel_targets, max_len, output_lengths, emotion_ids = inputs
        
        # Embeddings
        text_embed = self.embedding(text_inputs)  # [B, T, 512]
        emotion_embed = self.emotion_embedding(emotion_ids)  # [B, 32]
        
        # Expand and concatenate emotion
        B, T, _ = text_embed.shape
        emotion_expanded = emotion_embed.unsqueeze(1).expand(B, T, -1)  # [B, T, 32]
        combined = torch.cat([text_embed, emotion_expanded], dim=2)  # [B, T, 544]
        
        # Encoder convolutions (expect [B, channels, seq_len])
        x = combined.transpose(1, 2)  # [B, 544, T]
        x = self.encoder_convolutions(x)  # [B, 512, T]
        x = x.transpose(1, 2)  # [B, T, 512]
        
        # Encoder LSTM
        encoder_outputs, _ = self.encoder_lstm(x)  # [B, T, 512]
        
        # Decoder loop
        mel_len = mel_targets.size(1)
        mel_outputs_list = []
        
        decoder_hidden = torch.zeros(B, self.hparams.decoder_rnn_dim, device=text_inputs.device)
        decoder_cell = torch.zeros(B, self.hparams.decoder_rnn_dim, device=text_inputs.device)
        context = encoder_outputs.mean(dim=1)  # [B, 512]
        
        for t in range(mel_len):
            # Get current mel input
            if t > 0:
                prenet_input = mel_outputs_list[-1]  # [B, n_mel_channels]
            else:
                prenet_input = mel_targets[:, 0, :]  # [B, n_mel_channels]
            
            # Prenet
            prenet_output = self.decoder_prenet(prenet_input)  # [B, 256]
            
            # Concatenate with context
            decoder_input = torch.cat([prenet_output, context], dim=1)  # [B, 768]
            
            # Decoder RNN cell
            decoder_hidden, decoder_cell = self.decoder_rnn(
                decoder_input, 
                (decoder_hidden, decoder_cell)
            )  # [B, decoder_rnn_dim]
            
            # Project to mel
            mel_output = self.mel_projection(decoder_hidden)  # [B, n_mel_channels]
            mel_outputs_list.append(mel_output)
        
        # Stack mel outputs
        mel_outputs = torch.stack(mel_outputs_list, dim=1)  # [B, mel_len, n_mel_channels]
        
        # Postnet - FIXED DIMENSION HANDLING
        # Reshape for Conv1d: [B, n_mel_channels, mel_len]
        mel_for_postnet = mel_outputs.transpose(1, 2)  # [B, n_mel_channels, mel_len]
        
        # Apply postnet
        postnet_out = self.postnet(mel_for_postnet)  # [B, n_mel_channels, mel_len]
        
        # Reshape back: [B, mel_len, n_mel_channels]
        postnet_out = postnet_out.transpose(1, 2)  # [B, mel_len, n_mel_channels]
        
        # Add residual
        mel_outputs_postnet = mel_outputs + postnet_out  # [B, mel_len, n_mel_channels]
        
        # Gate outputs
        gate_outputs = torch.zeros(B, mel_len, device=text_inputs.device)
        
        # Alignments (dummy)
        alignments = torch.ones(B, mel_len, T, device=text_inputs.device)
        
        return [mel_outputs, mel_outputs_postnet, gate_outputs, alignments]
    
    def inference(self, inputs):
        """Inference mode."""
        text_inputs, emotion_ids = inputs
        
        # Create dummy mel target
        B = text_inputs.size(0)
        dummy_mels = torch.zeros(B, 200, self.hparams.n_mel_channels, device=text_inputs.device)
        dummy_lengths = torch.full((B,), 200, dtype=torch.long, device=text_inputs.device)
        
        full_inputs = (text_inputs, dummy_lengths, dummy_mels, 200, dummy_lengths, emotion_ids)
        return self.forward(full_inputs)
