"""
OpenAssistant Dataset Generator

Loads OpenAssistant conversation trees and generates training examples
by randomly sampling paths through trees with quality-weighted branching.
"""

import gzip
import json
import random
import logging
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import Counter
from dataclasses import dataclass, field, asdict

import numpy as np
import jinja2.sandbox
from datasets.dataset_dict import IterableDatasetDict
from datasets.io.generator import GeneratorDatasetInputStream
from huggingface_hub import hf_hub_download
from forgather.ml.distributed import main_process_first

logger = logging.getLogger(__name__)

# Default ChatML template
chatml_template = """{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""


def deterministic_hash(s: str) -> int:
    """
    Create a deterministic hash from a string.

    Python's built-in hash() is randomized for security,
    so we use CRC32 for fast deterministic hashing instead.
    """
    # CRC32 returns unsigned, but we want signed for compatibility
    return zlib.crc32(s.encode()) & 0xFFFFFFFF


def create_fingerprint(split: str, config: "OpenAssistantConfig") -> str:
    """
    Create a deterministic fingerprint for a dataset split.

    Uses the split name and config parameters to generate a unique identifier
    that avoids expensive pickling of large tree data.

    Args:
        split: Split name ('train', 'validation', or 'test')
        config: OpenAssistantConfig instance

    Returns:
        Hex string fingerprint
    """
    # Convert config to dict (only plain datatypes)
    config_dict = asdict(config)
    # Create deterministic string representation
    config_str = f"{split}_{sorted(config_dict.items())}"
    # Hash it
    hash_val = deterministic_hash(config_str)
    return f"{hash_val:08x}"


@dataclass
class OpenAssistantConfig:
    """Configuration for OpenAssistant dataset generation."""

    # File paths
    input_file_path: Optional[str] = None
    cache_dir: Optional[str] = None

    # Filtering options
    languages: List[str] = field(default_factory=lambda: ["en"])
    min_quality: Optional[float] = None
    min_thread_length: int = 2
    max_thread_length: int = 7
    exclude_deleted: bool = True
    exclude_synthetic: bool = True

    # Sampling options
    branch_temperature: float = 1.0
    seed: int = 42

    # Dataset size
    dataset_length: int = 10000  # -1 for infinite
    val_split: int = 10  # percentage
    test_split: int = 10  # percentage

    def __post_init__(self):
        """Validate and set defaults."""
        if self.input_file_path is None:
            self.input_file_path = "/home/dinalt/rust/datasets/OpenAssistant/2023-11-05_oasst2_ready.trees.jsonl.gz"


def load_trees_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load OpenAssistant trees from a JSONL or JSONL.gz file.

    Args:
        file_path: Path to the JSONL or JSONL.gz file

    Returns:
        List of tree dictionaries
    """
    file_path = Path(file_path)
    trees = []

    logger.info(f"Loading trees from {file_path}")

    if file_path.suffix == ".gz":
        opener = lambda: gzip.open(file_path, mode="rt", encoding="UTF-8")
    else:
        opener = lambda: open(file_path, "r", encoding="UTF-8")

    with opener() as f:
        for line_num, line in enumerate(f, 1):
            try:
                tree = json.loads(line)
                # Validate it's a tree (has message_tree_id and prompt)
                if "message_tree_id" in tree and "prompt" in tree:
                    trees.append(tree)
                else:
                    logger.warning(f"Line {line_num}: Not a tree object, skipping")
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
                continue

    logger.info(f"Loaded {len(trees)} trees")
    return trees


def get_message_quality(message: Dict[str, Any]) -> Optional[float]:
    """Extract quality score from message labels."""
    if not message.get("labels"):
        return None

    quality_label = message["labels"].get("quality")
    if quality_label and quality_label.get("value") is not None:
        return quality_label["value"]

    return None


def quality_weighted_sample(
    replies: List[Dict[str, Any]], temperature: float = 1.0, rng: random.Random = None
) -> Dict[str, Any]:
    """
    Sample a reply using quality-weighted probabilities.

    Args:
        replies: List of reply message dictionaries
        temperature: Temperature for softmax (higher = more random)
        rng: Random number generator for reproducibility

    Returns:
        Selected reply message
    """
    if not replies:
        return None

    if len(replies) == 1:
        return replies[0]

    rng = rng or random

    # Extract quality scores
    qualities = [get_message_quality(reply) for reply in replies]

    # If no quality scores available, use uniform sampling
    if all(q is None for q in qualities):
        return rng.choice(replies)

    # Replace None with median quality (or 0.5 if all None)
    valid_qualities = [q for q in qualities if q is not None]
    median_quality = np.median(valid_qualities) if valid_qualities else 0.5
    qualities = [q if q is not None else median_quality for q in qualities]

    # Apply softmax with temperature
    qualities_array = np.array(qualities, dtype=np.float32)
    exp_qualities = np.exp(qualities_array / temperature)
    probabilities = exp_qualities / np.sum(exp_qualities)

    # Sample based on probabilities
    selected_idx = rng.choices(range(len(replies)), weights=probabilities)[0]
    return replies[selected_idx]


def message_passes_filter(message: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """Check if a message passes the configured filters."""

    # Check language
    if config.get("languages"):
        msg_lang = message.get("lang")
        if msg_lang not in config["languages"]:
            return False

    # Check deleted
    if config.get("exclude_deleted", True):
        if message.get("deleted", False):
            return False

    # Check synthetic
    if config.get("exclude_synthetic", True):
        if message.get("synthetic", False):
            return False

    # Check quality threshold
    min_quality = config.get("min_quality")
    if min_quality is not None:
        quality = get_message_quality(message)
        if quality is not None and quality < min_quality:
            return False

    return True


def extract_random_thread(
    tree: Dict[str, Any], config: Dict[str, Any], rng: random.Random
) -> Optional[List[Dict[str, str]]]:
    """
    Extract a random thread from a tree by following a random path.

    Args:
        tree: Tree dictionary with prompt and nested replies
        config: Configuration dictionary
        rng: Random number generator

    Returns:
        List of message dicts with 'role' and 'content', or None if invalid
    """
    min_length = config.get("min_thread_length", 2)
    max_length = config.get("max_thread_length", 7)
    temperature = config.get("branch_temperature", 1.0)

    messages = []
    current = tree["prompt"]

    if not current:
        return None

    # Traverse tree following random weighted path
    while current:
        # Check if message passes filters
        if not message_passes_filter(current, config):
            # If we hit a filtered message, stop here
            break

        # Map role: 'prompter' -> 'user', 'assistant' -> 'assistant'
        role = "user" if current.get("role") == "prompter" else "assistant"
        content = current.get("text", "").strip()

        if not content:
            break

        messages.append({"role": role, "content": content})

        # For SFT mode, end on assistant messages
        if role == "assistant" and len(messages) >= min_length:
            # This is a valid stopping point
            break

        # Check max length
        if len(messages) >= max_length:
            break

        # Get next message
        replies = current.get("replies", [])
        if not replies:
            break

        # Filter replies
        valid_replies = [r for r in replies if message_passes_filter(r, config)]
        if not valid_replies:
            break

        # Sample next message using quality weighting
        current = quality_weighted_sample(valid_replies, temperature, rng)

    # Validate thread
    if len(messages) < min_length or len(messages) > max_length:
        return None

    # For SFT, must end with assistant
    if messages and messages[-1]["role"] != "assistant":
        return None

    return messages


class TreeDatabase:
    """
    In-memory database of conversation trees with filtering and indexing.
    """

    def __init__(self, trees: List[Dict[str, Any]], config: Dict[str, Any]):
        """
        Initialize tree database with filtering.

        Args:
            trees: List of tree dictionaries
            config: Configuration with filter parameters
        """
        self.config = config
        self.trees = []
        self.tree_ids = []

        # Filter trees
        logger.info("Filtering trees...")
        for tree in trees:
            if self._tree_passes_filter(tree):
                self.trees.append(tree)
                self.tree_ids.append(tree["message_tree_id"])

        logger.info(f"Filtered to {len(self.trees)} trees from {len(trees)} total")

        # Compute statistics
        self._compute_stats()

    def _tree_passes_filter(self, tree: Dict[str, Any]) -> bool:
        """Check if a tree has at least one valid thread."""
        # For now, just check if the tree has a valid prompt
        prompt = tree.get("prompt")
        if not prompt:
            return False

        # Check language at tree level
        if self.config.get("languages"):
            prompt_lang = prompt.get("lang")
            if prompt_lang not in self.config["languages"]:
                return False

        return True

    def _compute_stats(self):
        """Compute statistics about the database."""
        self.stats = {
            "total_trees": len(self.trees),
            "languages": Counter(),
        }

        # Count languages
        for tree in self.trees:
            prompt = tree.get("prompt")
            if prompt and prompt.get("lang"):
                self.stats["languages"][prompt["lang"]] += 1

        logger.info(f"Database stats: {self.stats}")

    def get_random_tree(self, rng: random.Random) -> Dict[str, Any]:
        """Get a random tree from the database."""
        return rng.choice(self.trees)

    @property
    def total_trees(self) -> int:
        return len(self.trees)


class ThreadGenerator:
    """
    Generator that yields random threads from a tree database.

    Generates text examples directly by applying chat template to messages.
    """

    def __init__(
        self,
        tree_db: TreeDatabase,
        length: int,
        config: Dict[str, Any],
        chat_template: Optional[Any] = None,
        template_args: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        """
        Initialize thread generator.

        Args:
            tree_db: TreeDatabase instance
            length: Number of examples to generate (-1 for infinite)
            config: Configuration dictionary
            chat_template: Jinja2 template for formatting conversations (optional)
            template_args: Arguments for chat template (e.g., bos_token, eos_token)
            seed: Random seed
        """
        self.tree_db = tree_db
        self.length = length
        self.config = config
        self.chat_template = chat_template
        self.template_args = template_args or {}
        self.seed = seed
        self.rng = random.Random(seed)
        self.count = 0

    def __iter__(self):
        """Return iterator."""
        self.count = 0
        self.rng = random.Random(self.seed)
        return self

    def __next__(self) -> Dict[str, Any]:
        """Generate next example."""
        # Check length limit
        if self.length != -1 and self.count >= self.length:
            raise StopIteration

        # Keep trying until we get a valid thread
        max_attempts = 100
        for attempt in range(max_attempts):
            tree = self.tree_db.get_random_tree(self.rng)
            thread = extract_random_thread(tree, self.config, self.rng)

            if thread:
                self.count += 1

                # Apply chat template if provided, otherwise return messages
                if self.chat_template:
                    text = self.chat_template.render(
                        messages=thread, **self.template_args
                    )
                    return {"text": text}
                else:
                    return {
                        "messages": thread,
                        "tree_id": tree["message_tree_id"],
                        "thread_length": len(thread),
                    }

        # If we couldn't generate a valid thread after many attempts, raise error
        raise RuntimeError(
            f"Failed to generate valid thread after {max_attempts} attempts"
        )


class OpenAssistantDatasetDict(IterableDatasetDict):
    """
    IterableDatasetDict that generates OpenAssistant splits from configuration.

    Loads trees once and creates splits lazily for efficiency.
    Applies chat template in the generator to produce text examples directly.
    """

    def __init__(
        self,
        tokenizer=None,
        chat_template="",
        map_fn=None,
        map_kwargs=None,
        **config_kwargs,
    ):
        """
        Initialize with configuration parameters.

        Args:
            tokenizer: Tokenizer for extracting special tokens (optional)
            chat_template: Path to chat template file, or empty to use tokenizer's template
            **config_kwargs: Parameters for OpenAssistantConfig
        """
        self.config = OpenAssistantConfig(**config_kwargs)
        self.tokenizer = tokenizer
        self.chat_template_path = chat_template

        # Load and prepare chat template
        self._prepare_chat_template()

        # Load trees once (shared across all splits)
        file_path = self.config.input_file_path

        # Download from HF Hub if needed
        if not Path(file_path).exists():
            logger.info(f"Downloading from HF Hub: {file_path}")
            file_path = hf_hub_download(
                repo_id="OpenAssistant/oasst2",
                filename="2023-11-05_oasst2_ready.trees.jsonl.gz",
                repo_type="dataset",
                cache_dir=self.config.cache_dir,
            )

        self.all_trees = load_trees_from_jsonl(file_path)
        logger.info(f"Loaded {len(self.all_trees)} trees (shared across all splits)")

        # Split trees deterministically based on tree_id hash
        self.split_trees = {}
        for tree in self.all_trees:
            tree_hash = deterministic_hash(tree["message_tree_id"]) % 100

            if tree_hash < self.config.test_split:
                split_name = "test"
            elif tree_hash < (self.config.test_split + self.config.val_split):
                split_name = "validation"
            else:
                split_name = "train"

            if split_name not in self.split_trees:
                self.split_trees[split_name] = []
            self.split_trees[split_name].append(tree)

        for split_name, trees in self.split_trees.items():
            logger.info(f"Split '{split_name}': {len(trees)} trees")

        # Pre-create TreeDatabases for each split (faster than lazy creation)
        logger.info("Creating TreeDatabases for all splits...")
        self.tree_databases = {}
        config_dict = {
            "languages": self.config.languages,
            "min_quality": self.config.min_quality,
            "min_thread_length": self.config.min_thread_length,
            "max_thread_length": self.config.max_thread_length,
            "exclude_deleted": self.config.exclude_deleted,
            "exclude_synthetic": self.config.exclude_synthetic,
            "branch_temperature": self.config.branch_temperature,
        }

        for split_name, trees in self.split_trees.items():
            self.tree_databases[split_name] = TreeDatabase(trees, config_dict)

        # Initialize empty DatasetDict - splits created lazily
        super().__init__()

    def _prepare_chat_template(self):
        """Load and prepare chat template following priority order."""
        template_args = {}

        # Get special tokens from tokenizer if provided
        if self.tokenizer:
            template_args["bos_token"] = self.tokenizer.bos_token
            template_args["eos_token"] = self.tokenizer.eos_token

        # Priority 1: Use path if not empty
        if self.chat_template_path and len(self.chat_template_path) > 0:
            logger.info(f"Using chat template from path: {self.chat_template_path}")
            with open(self.chat_template_path, "r") as f:
                template_str = f.read()
        # Priority 2: Use tokenizer's chat template if it has one
        elif self.tokenizer and self.tokenizer.chat_template:
            logger.info("Using chat template from tokenizer")
            template_str = self.tokenizer.chat_template
        # Priority 3: Fallback to default template and warn
        else:
            logger.warning("No chat template provided, using default ChatML template")
            template_str = chatml_template

        # Create Jinja2 environment and compile template
        environment = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.chat_template = environment.from_string(template_str)
        self.template_args = template_args

    def __getitem__(self, key):
        """
        Lazy split creation - only create dataset when accessed.

        Args:
            key: Split name ('train', 'validation', or 'test')

        Returns:
            IterableDataset for the requested split
        """
        import time

        # Check if already created using parent's data dict
        try:
            return super().__getitem__(key)
        except KeyError:
            pass

        # Create split on-demand
        if key not in self.tree_databases:
            raise KeyError(
                f"Split '{key}' not found. Available splits: {list(self.tree_databases.keys())}"
            )

        # Use pre-created tree database
        tree_db = self.tree_databases[key]

        # Determine dataset length for this split
        if key == "train":
            length = self.config.dataset_length
        else:
            # Val/test splits use 1/10th of train length
            length = (
                max(self.config.dataset_length // 10, 100)
                if self.config.dataset_length > 0
                else 1000
            )

        # Determine seed for this split (consistent across calls)
        split_seed = self.config.seed + deterministic_hash(key) % 1000

        # Prepare config dict for generator
        gen_config = {
            "languages": self.config.languages,
            "min_quality": self.config.min_quality,
            "min_thread_length": self.config.min_thread_length,
            "max_thread_length": self.config.max_thread_length,
            "exclude_deleted": self.config.exclude_deleted,
            "exclude_synthetic": self.config.exclude_synthetic,
            "branch_temperature": self.config.branch_temperature,
        }

        # Create generator factory function that uses gen_kwargs to avoid capturing large objects
        # This prevents dill from having to pickle the entire TreeDatabase during fingerprinting
        def generator_fn(tree_db, length, config, chat_template, template_args, seed):
            """Factory function that creates a new generator instance with the same seed."""
            return ThreadGenerator(
                tree_db=tree_db,
                length=length,
                config=config,
                chat_template=chat_template,
                template_args=template_args,
                seed=seed,
            )

        # Create custom fingerprint to avoid expensive pickling of large tree database
        fingerprint = create_fingerprint(key, self.config)

        # Use GeneratorDatasetInputStream directly with custom fingerprint
        # This bypasses the expensive dill pickling that from_generator() does
        dataset = GeneratorDatasetInputStream(
            generator=generator_fn,
            gen_kwargs={
                "tree_db": tree_db,
                "length": length,
                "config": gen_config,
                "chat_template": self.chat_template,
                "template_args": self.template_args,
                "seed": split_seed,
            },
            fingerprint=fingerprint,
            streaming=True,
        ).read()

        # Store in parent's data dict
        self[key] = dataset
        return dataset


def load_chat_template(chat_template_path=None, tokenizer=None):
    """
    Load and compile a chat template.

    Args:
        chat_template_path: Path to chat template file (optional)
        tokenizer: Tokenizer with chat_template attribute (optional)

    Returns:
        Compiled Jinja2 template
    """
    # Load template string
    if not chat_template_path or len(chat_template_path) == 0:
        if tokenizer and tokenizer.chat_template:
            template_str = tokenizer.chat_template
            logger.info("Using chat template from tokenizer")
        else:
            template_str = chatml_template
            logger.warning("Using default chat template (ChatML)")
    else:
        logger.info(f"Using chat template from file: {chat_template_path}")
        with open(chat_template_path, "r") as f:
            template_str = f.read()

    # Compile template
    environment = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return environment.from_string(template_str)
