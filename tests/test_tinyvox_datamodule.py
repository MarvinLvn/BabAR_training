import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.hparams import Parameters
from agents.BaseTrainer import BaseTrainer
import time

def test_tinyvox():
    # Parse parameters with TinyVox settings
    import sys
    sys.argv = [
        'test',
        '--custom_dataset', 'True',
        '--use_vad', 'True',
        '--dataset_path', '/scratch2/mlavechin/tinyvox/TinyVox',
        '--language', 'en',
        '--batch_size', '64',
        '--num_proc', '8',
        '--create_dataset', 'False',
        '--debug_dataset', 'False',
    ]

    parameters = Parameters.parse()


    # Setup and test
    agent = BaseTrainer(parameters, run=None)
    pl_model = agent.pl_model
    agent.datamodule.setup("fit")

    train_loader = agent.datamodule.train_dataloader()
    first_batch = next(iter(train_loader))

    print("✅ TinyVox dataset works!")
    print(f"Batch keys: {first_batch.keys()}")
    print(f"Audio shape: {first_batch['array'].shape}")
    print(f"First phonemes: {first_batch['phonemes'][0]}")
    print(f"First sentence: {first_batch['sentence'][0]}")

    # Compute cumulative duration and segment count
    print("\n🔍 Computing dataset statistics...")

    total_duration = 0.0
    total_segments = 0
    batch_count = 0

    # TinyVox dataset sample rate
    sample_rate = 16000

    start_time = time.time()

    # Reset iterator to go through all batches
    train_loader = agent.datamodule.train_dataloader()

    for batch_idx, batch in enumerate(train_loader):
        batch_count += 1

        # Get audio arrays from current batch
        audio_arrays = batch['array']  # Shape: [batch_size, sequence_length]

        # Count segments in this batch
        batch_segments = audio_arrays.shape[0]
        total_segments += batch_segments

        # Calculate duration for this batch
        # Duration = number_of_samples / sample_rate
        for audio in audio_arrays:
            # Handle different tensor types (torch, numpy, etc.)
            if hasattr(audio, 'shape'):
                num_samples = audio.shape[-1]  # Last dimension should be time
            else:
                num_samples = len(audio)

            duration = num_samples / sample_rate
            total_duration += duration

        # Progress update every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"Processed {batch_idx + 1} batches, {total_segments} segments so far...")

    end_time = time.time()
    processing_time = end_time - start_time

    # Convert duration to hours, minutes, seconds
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = total_duration % 60

    print(f"\n📊 Dataset Statistics:")
    print(f"{'=' * 50}")
    print(f"Total batches processed: {batch_count}")
    print(f"Total audio segments: {total_segments:,}")
    print(f"Total duration: {hours:02d}:{minutes:02d}:{seconds:05.2f}")
    print(f"Total duration (seconds): {total_duration:.2f}")
    print(f"Total duration (hours): {total_duration / 3600:.2f}")
    print(f"Average segment length: {total_duration / total_segments:.2f} seconds")
    print(f"Sample rate used: {sample_rate} Hz")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    test_tinyvox()