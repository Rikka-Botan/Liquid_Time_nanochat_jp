"""
SmolTalk by HuggingFace. Good "general" conversational dataset.
https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
We use the "smol" version, which is more appropriate for smaller models.
"""

import os
from datasets import load_dataset
from tasks.common import Task

# ----------------------------------------------------------------------------- 
class JSmolTalk(Task):
    """ SmolTalk dataset loader with removal of content between tokens A and B. """
    def __init__(self, split, token_start="<think>", token_end="</think>", 
                 remove_between=True, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SmolTalk split must be train|test"

        # Load the dataset directly from HuggingFace
        dataset = load_dataset("RikkaBotan/FineDataset_SFT")
        dataset.shuffle()

        # Split into train/test (approx. 90% / 10%)
        full_train = dataset["train"]
        n_total = 510000
        n_train = 460000

        if split == "train":
            self.data = full_train.select(range(n_train))
        else:
            self.data = full_train.select(range(n_train, n_total))

        self.token_start = token_start
        self.token_end = token_end
        self.remove_between = remove_between

        self.texts = []
        removed_count = 0

        for msg_list in self.data["messages"]:
            processed_messages = []
            for msg in msg_list:
                if msg["role"] == "assistant" and self.remove_between:
                    content = msg["content"]
                    if self.token_start in content and self.token_end in content:
                        start_idx = content.find(self.token_start)
                        end_idx = content.find(self.token_end, start_idx + len(self.token_start))

                        if start_idx != -1 and end_idx != -1:
                            new_content = content[:start_idx] + content[end_idx + len(self.token_end):]
                            processed_messages.append({"role": "assistant", "content": new_content})
                            removed_count += 1
                        else:
                            processed_messages.append(msg)
                    else:
                        processed_messages.append(msg)
                else:
                    processed_messages.append(msg)

            self.texts.append(processed_messages)

        self.length = len(self.texts)
        print(f"Loaded {self.length} examples for {split} split.")
        if remove_between:
            print(f"Removed {removed_count} segments between '{token_start}' and '{token_end}'.")

    def num_examples(self):
        return self.length

    def get_example(self, index):
        if index < 0 or index >= self.length:
            raise IndexError("Index out of range")
        return {"messages": self.texts[index]}
