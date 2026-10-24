import os

DATA_ROOT = os.environ.get("UAVNARR_DATA_ROOT", "/root/autodl-tmp/dataset")
LANG_ROOT = os.environ.get("UAVNARR_LANG_ROOT", "/root/autodl-tmp/dataset/Language")
SPLIT_PATH = os.environ.get("UAVNARR_SPLIT", "/root/uav_narr_code/uav_narr_split.json")
MODEL_PATH = os.environ.get("UAVNARR_MODEL", "/root/autodl-tmp/models/VideoLLaMA2-7B")
VL2_PATH = os.environ.get("UAVNARR_VL2", "/root/uav_narr_code/VideoLLaMA2")
OUT_ROOT = os.environ.get("UAVNARR_OUT", "/root/autodl-tmp/uavnarr_v3")

STRUCTURED_PROMPT = (
    "You are watching a UAV aerial video. "
    "Describe what you see in one sentence, focusing on: "
    "1) what the target object is, 2) where it is, 3) what it is doing. "
    "Write in a documentary narration style. Be specific and concise."
)

MISTRAL_SYSTEM = (
    "<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, "
    "while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, "
    "dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive "
    "in nature."
    "\n"
    "If a question does not make any sense, or is not factually coherent, explain why instead of answering "
    "something not correct. If you don't know the answer to a question, please don't share false "
    "information.\n<</SYS>>"
)
