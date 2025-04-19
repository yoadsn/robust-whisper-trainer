#!/usr/bin/env python
"""Example script for using a trained robust Whisper encoder."""

import os
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def load_robust_encoder(
    base_model_name: str,
    robust_encoder_path: str,
) -> WhisperForConditionalGeneration:
    """Load a Whisper model with a robust encoder.
    
    Args:
        base_model_name: Name or path of the base Whisper model
        robust_encoder_path: Path to the trained robust encoder
        
    Returns:
        Whisper model with robust encoder
    """
    # Load the base model
    model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
    
    # Load the robust encoder state dict
    encoder_state_dict = torch.load(
        os.path.join(robust_encoder_path, "encoder_state_dict.pt"),
        map_location="cpu",
    )
    
    # Update the model's encoder weights
    model_state_dict = model.state_dict()
    for key, value in encoder_state_dict.items():
        if key in model_state_dict:
            model_state_dict[key] = value
    
    # Load the updated state dict
    model.load_state_dict(model_state_dict)
    
    return model


def transcribe_audio(
    audio_path: str,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    """Transcribe audio using the Whisper model.
    
    Args:
        audio_path: Path to the audio file
        model: Whisper model
        processor: Whisper processor
        device: Device to run the model on
        
    Returns:
        Transcription text
    """
    # Load audio
    import librosa
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Process audio
    input_features = processor(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
    ).input_features.to(device)
    
    # Generate transcription
    model = model.to(device)
    with torch.no_grad():
        generated_ids = model.generate(input_features=input_features)
    
    # Decode the generated ids
    transcription = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]
    
    return transcription


def main():
    """Main function."""
    # Define paths
    base_model_name = "openai/whisper-small"
    robust_encoder_path = "./outputs/robust-whisper"
    audio_path = "path/to/your/audio.wav"  # Replace with your audio file
    
    # Load the model with robust encoder
    model = load_robust_encoder(base_model_name, robust_encoder_path)
    
    # Load the processor
    processor = WhisperProcessor.from_pretrained(base_model_name)
    
    # Transcribe audio
    transcription = transcribe_audio(audio_path, model, processor)
    
    # Print the transcription
    print(f"Transcription: {transcription}")


if __name__ == "__main__":
    main()
