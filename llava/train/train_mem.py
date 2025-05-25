# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
from llava.train.train import train

if __name__ == "__main__":
    # train(attn_implementation="flash_attention_2")
    train()
