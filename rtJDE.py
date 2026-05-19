class EnhancedPCAChannelCalibration(nn.Module):
    """
    增强版PCA-通道注意力校准
    包含多层感知器增强特征
    """
    
    def __init__(self, input_dim=256, output_dim=128, hidden_dim=192):
        super().__init__()
        
        # 1. 通道重要性评估
        self.importance_evaluator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        # 2. PCA风格投影（两阶段）
        self.pca_stage1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.pca_stage2 = nn.Linear(hidden_dim, output_dim)
        
        # 3. 主成分重要性（可学习特征值）
        self.importance_pca = nn.Parameter(torch.ones(hidden_dim))
        
        # 4. 特征增强MLP
        self.enhancer = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )
        
        # 5. 残差适配器
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
        # 6. 归一化层
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        # 原始特征用于残差
        identity = self.residual(x)
        
        # 步骤1: 评估通道重要性
        importance = self.importance_evaluator(x)
        weighted_input = x * importance
        
        # 步骤2: PCA风格投影
        # 第一阶段降维
        latent = self.pca_stage1(weighted_input)
        
        # 主成分重要性缩放
        importance_scaled = torch.sigmoid(self.importance_pca)
        latent_weighted = latent * importance_scaled.unsqueeze(0).unsqueeze(0)
        
        # 第二阶段重构
        pca_output = self.pca_stage2(latent_weighted)
        
        # 步骤3: 残差连接
        combined = pca_output + identity
        
        # 步骤4: 特征增强
        enhanced = self.enhancer(combined)
        
        # 步骤5: 最终归一化
        output = self.norm2(enhanced)
        
        return output