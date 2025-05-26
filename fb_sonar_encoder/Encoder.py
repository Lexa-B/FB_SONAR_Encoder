# SONAR Translator - Complete Version

import warnings
# Filter torchaudio backend warning from skops
warnings.filterwarnings('ignore', category=UserWarning, module='skops.io._utils')
warnings.filterwarnings('ignore', message='.*Torchaudio\'s I/O functions.*')

from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
import torch
import re
import os
import argparse

class Encoder:
    def __init__(self, LangSource, Verbose):
        self.SONAR_Text2Vec = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=torch.device("cuda"),
            dtype=torch.float32,)
        self.Args = {
            "lang_source": LangSource,
            "verbose": Verbose
        }

    def Encode(self, sentences):
        if self.Args["verbose"]:
            print(f"Embedding Sentence: {sentences}")

        # Ensure sentences is a list
        if isinstance(sentences, str):
            sentences = [sentences]

        embedding = self.SONAR_Text2Vec.predict(sentences, source_lang=self.Args["lang_source"]) 
        if self.Args["verbose"]:
            print(f"embedding.shape: {embedding.shape}")

        return embedding

    def EncodeText(self, InputText, output_file):
        # Load and clean Japanese text
        text = InputText

        # Encode the sentence
        embedding = self.Encode(text)
        print(f"Encoded: {text} → {embedding}")

        # Save as torch tensor
        torch.save(embedding, output_file)
        print(f"Embedding saved as tensor to {output_file}")

# Example usage
if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the command line arguments
    parser = argparse.ArgumentParser(description='Translate text using SONAR.')
    parser.add_argument('-t', '--input_text', type=str, required=True, help='Input text to encode.')
    parser.add_argument('-o', '--output_name', type=str, required=True, help='Name to save the encoded tensor.')
    parser.add_argument('-l', '--lang_source', type=str, required=True, help='Language to encode from (FLORES-200) (e.g., jpn_Jpan, eng_Latn).')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output.')
    args = parser.parse_args()

    if args.verbose:
        print("Verbose mode enabled.")
        print(f"Language Source: {args.lang_source}")
        print(f"Output File: ./fb_sonar_encoder/OutputData/{args.output_name}.pt")

    encoder = Encoder(LangSource=args.lang_source, Verbose=args.verbose)
    encoder.EncodeText(
        args.input_text,
        os.path.join(script_dir, "OutputData", f"{args.output_name}.pt")
    )   

    print(f"Text: \"{args.input_text}\", Language Source: {args.lang_source}, Verbose: {args.verbose}")
