@dataclass
class LIMRArguments:
    """
    LIMR (Learning with Interpretable Mathematical Rewards)的参数
    """
    enabled: bool = field(
        default=False,
        metadata={"help": "是否启用LIMR奖励机制"}
    )
    reward_type: str = field(
        default="rule",
        metadata={"help": "奖励类型: 'rule'(基于规则) 或 'remote'(远程API)"}
    )
    ground_truth_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "包含正确答案的数据集路径，用于rule-based奖励"}
    )
    reward_correct: float = field(
        default=1.0,
        metadata={"help": "正确答案的奖励值"}
    )
    reward_incorrect: float = field(
        default=0.0,
        metadata={"help": "错误答案的奖励值"}
    )
    math_equal_normalize: bool = field(
        default=True,
        metadata={"help": "是否在比较数学表达式时进行规范化"}
    )
    save_samples_path: Optional[str] = field(
        default=None,
        metadata={"help": "保存样本的路径"}
    )
    save_every_n_steps: int = field(
        default=1,
        metadata={"help": "每隔多少步保存一次样本"}
    )
    reward_url: Optional[str] = field(
        default=None,
        metadata={"help": "远程奖励API的URL"}
    )
    
    # 新增参数
    reward_scaling: float = field(
        default=1.0,
        metadata={"help": "奖励缩放因子"}
    )
    reward_shift: float = field(
        default=0.0,
        metadata={"help": "奖励偏移值"}
    )
    normalize_rewards: bool = field(
        default=False,
        metadata={"help": "是否对奖励进行归一化"}
    )
    save_top_k_samples: Optional[int] = field(
        default=None,
        metadata={"help": "每个步骤只保存奖励最高的k个样本"}
    )
    save_bottom_k_samples: Optional[int] = field(
        default=None,
        metadata={"help": "每个步骤只保存奖励最低的k个样本"}
    )