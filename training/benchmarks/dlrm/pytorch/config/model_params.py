# model configs
# The type of the embedding operation to use. valid options: ["joint", "custom_cuda", "multi_table", "joint_sparse", "joint_fused"]
embedding_type = "custom_cuda"
# Dimensionality of embedding space for categorical features
embedding_dim = 128
#"Linear layer sizes for the top MLP"
top_mlp_sizes = [1024, 1024, 512, 256, 1]
#"Linear layer sizes for the bottom MLP"
bottom_mlp_sizes = [512, 256, 128]
# Type of interaction operation to perform. valid options: ["cuda_dot", "dot", "cat"]
interaction_op = "cuda_dot"