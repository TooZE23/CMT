import torch


class BlockDiagonalMask:
    """
    替代 xformers 的 BlockDiagonalMask 类
    """

    def __init__(self, attn_bias, cat_inputs, split_sizes):
        self.attn_bias = attn_bias
        self.cat_inputs = cat_inputs
        self.split_sizes = split_sizes

    def split(self, tensor):
        """
        按照原始输入的长度分割张量

        Args:
            tensor: 需要分割的张量，形状为 [batch_size, total_seq_len, ...]

        Returns:
            分割后的张量列表
        """
        return list(torch.split(tensor, self.split_sizes, dim=1))

    @classmethod
    def from_tensor_list(cls, tensor_list):
        """
        从张量列表创建 BlockDiagonalMask

        Args:
            tensor_list: 输入张量列表，每个张量形状为 [batch_size, seq_len, dim]

        Returns:
            BlockDiagonalMask 实例和拼接后的输入张量
        """
        # 记录每个张量的序列长度
        split_sizes = [tensor.shape[1] for tensor in tensor_list]

        # 拼接所有输入张量
        cat_inputs = torch.cat(tensor_list, dim=1)  # [batch_size, total_seq_len, dim]

        batch_size = cat_inputs.shape[0]
        total_seq_len = cat_inputs.shape[1]

        # 创建块对角掩码
        attn_bias = torch.full(
            (batch_size, total_seq_len, total_seq_len),
            float("-inf"),
            dtype=cat_inputs.dtype,
            device=cat_inputs.device,
        )

        # 为每个块设置对角线为0（允许注意力）
        start_idx = 0
        for seq_len in split_sizes:
            end_idx = start_idx + seq_len
            attn_bias[:, start_idx:end_idx, start_idx:end_idx] = 0
            start_idx = end_idx

        mask = cls(attn_bias, cat_inputs, split_sizes)
        return mask, cat_inputs
