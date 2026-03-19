# src/utils/global_vars.py

# 默认 Attention Mask 的结束位置列表
# 你需要根据实际 batch_size 动态更新这个列表
# 这里只是一个默认的示例占位
ATTN_MASK_END = []

def set_attn_mask_end(mask_end_list):
    """
    设置 ATTN_MASK_END，用于替换输出中最后一个 token 的位置。

    :param mask_end_list: List[int]，每个样本的 attention mask 结束位置
    """
    global ATTN_MASK_END
    ATTN_MASK_END = mask_end_list
