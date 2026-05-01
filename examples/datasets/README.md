# Forgather Dataset Definitions

## Datasets

- **[roneneldan](./roneneldan/)** - TinyStories: synthetically generated short stories (GPT-3.5/4) using a small vocabulary. The default dataset for quick model validation and the tiny_experiments collection.
- **[allenai](./allenai/)** - C4 (Colossal Clean Crawled Corpus): large-scale web text from AllenAI, available in standard and packed variants.
- **[EleutherAI](./EleutherAI/)** - Wikitext-103: document-level Wikipedia text from EleutherAI, commonly used as a language modeling benchmark.
- **[HuggingFaceTB](./HuggingFaceTB/)** - SmolLM-Corpus: curated high-quality educational and synthetic data designed for training small language models.
- **[wikipedia](./wikipedia/)** - Wikimedia Wikipedia dumps (English, November 2023 snapshot).
- **[Open-Orca](./Open-Orca/)** - Open-Orca: augmented FLAN examples distilled through GPT-4/3.5, heavy on chain-of-thought and structured reasoning prompts.
- **[open-thoughts](./open-thoughts/)** - OpenThoughts3-1.2M: 1.2M reasoning conversations (math, code, science) annotated by QwQ-32B, in ShareGPT-style multi-turn format.
- **[OpenAssistant](./OpenAssistant/)** - OpenAssistant: conversational AI dataset from the LAION AI community, constructed from human-annotated conversation trees.
- **[QuixiAI](./QuixiAI/)** - Samantha: Eric Hartford's conversational persona dataset (formerly Cognitive Computations).
- **[ajibawa-2023](./ajibawa-2023/)** - General-Stories-Collection and related datasets by Feynman Innovations.
- **[local_dataset](./local_dataset/)** - Template project for loading local datasets via `datasets.load_from_disk`.


For the dataset project CLI reference — targets, inspection commands, histogram generation, and the `forgather dataset` subcommand — see **[docs/datasets/dataset-cli.md](../../docs/datasets/dataset-cli.md)**.