# Transformers for NLP and Chatbots

Preface: this repo follows the book-style journey through the Transformer architecture. It goes from implementing attention by hand, through encoder-only (BERT), decoder-only (GPT) and encoder-decoder models, to fine-tuning and aligning them. Each notebook starts with a short description of what it does.

Hands-on notebooks (PyTorch + Hugging Face) covering the main Transformer families, from building one by hand to fine-tuning and aligning pretrained models.

## Notebooks

| Notebook | What we did |
|---|---|
| `Transformer_from_scratch.ipynb` | Implemented a full Transformer in PyTorch: positional embedding, multi-head attention, encoder/decoder layers, encoder, decoder. |
| `Transformer-English-to-Spanish-Translator.ipynb` | Trained that encoder-decoder (`NmtTransformer`) for English → Spanish translation with a BPE tokenizer. Weights: `models/*.pt`. |
| `Encoder-Only-For-NLU.ipynb` | Trained a BERT from scratch with masked language modeling on WikiText-2 (`my_bert/`), then used pretrained `bert-base-uncased` for embeddings. |
| `distilbert_for_imdbsentiment_analysis.ipynb` | Fine-tuned DistilBERT for sentiment classification on IMDb, 3 epochs with `Trainer` (`bert_distil_ft/`). |
| `Decoder-Only.ipynb` | Text generation with GPT-2 and Mistral-7B (base and instruct), plus a simple chatbot class (`BobTheBot`). |
| `GPT2-shakesperean.ipynb` | Fine-tuned GPT-2 on Shakespeare text (`datasets/shakespeare/`), 3 epochs (`gpt_shakesperean/`). |
| `FineTuneWithTRL.ipynb` | Two-stage alignment of GPT-2 with TRL: SFT on Alpaca (`my_gpt2_sft_alpaca/`), then DPO on Anthropic HH-RLHF (`my_gpt2_sft_alpaca_dpo_hh_rlhf/`). |

## Layout

- `*.ipynb`: the notebooks above
- `datasets/`: local Shakespeare text
- `models/`, `my_bert/`, `bert_distil_ft/`, `gpt_shakesperean/`, `my_gpt2_sft_alpaca*/`: saved weights and checkpoints produced by the notebooks (large; not meant for git)

## Setup

```bash
pip install torch transformers datasets tokenizers trl
```

Then run the notebooks in Jupyter. A GPU is recommended for training and for Mistral-7B.

## License

See `LICENSE`.
