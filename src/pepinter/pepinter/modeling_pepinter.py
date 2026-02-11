from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn

from astra.models.esmc.modeling_esmc import (
    Pooler,
    ESMCOutput,
    ESMCModel,
    RegressionHead,
    TransformerStack,
    TransformerOutput,
)
from astra.models.esmc.configuration_esmc import ESMCConfig
import math
from astra.loss import FocalLoss
from astra.confidence.mdn import MDNHeader, mdn_loss, MDNInferenceMixin

from .configuration_pepinter import PepInterConfig


class ESMCModel_Adapter(ESMCModel):

    def __init__(self, config: ESMCConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed = nn.Embedding(self.vocab_size, config.hidden_size)
        self.token_type_embed = nn.Embedding(2, config.hidden_size)
        self.transformer = TransformerStack(
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            config.dropout,
        )
        self.init_weights()

    def _embed(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embed(input_ids)
        token_type_embed = self.token_type_embed(token_type_ids)
        x = x + token_type_embed

        return self.transformer(
            x, attention_mask, output_hidden_states=False, output_attentions=False
        ).last_hidden_state

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,  # to play nice with HF adjacent packages
        **kwargs,
    ) -> TransformerOutput:
        if inputs_embeds is None:
            if token_type_ids is None:
                x = self.embed(input_ids)
            else:
                x = self.embed(input_ids) + self.token_type_embed(token_type_ids)
        else:
            x = inputs_embeds
        return self.transformer(
            x, attention_mask, output_hidden_states, output_attentions
        )


class PepInterModelForMaskedLM(ESMCModel_Adapter):
    base_model_prefix = "pepinter"
    supports_gradient_checkpointing = True
    config_class = PepInterConfig

    def __init__(self, config: PepInterConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.sequence_head = RegressionHead(config.hidden_size, self.vocab_size)
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,  # to play nice with HF adjacent packages
        **kwargs,
    ) -> ESMCOutput:
        output = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        x = output.last_hidden_state
        logits = self.sequence_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1))
        return ESMCOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=x,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


class PepInterModelForClassification(ESMCModel_Adapter):
    base_model_prefix = "pepinter"
    supports_gradient_checkpointing = True
    config_class = PepInterConfig

    def __init__(self, config: PepInterConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.num_labels = config.num_labels
        self.problem_type = config.problem_type

        self.classifier = RegressionHead(
            config.hidden_size * 2, self.num_labels, config.hidden_size
        )

        self.pooler = Pooler(["cls", "mean"])
        self.criterion = FocalLoss(
            num_classes=self.num_labels,
            task="binary" if self.num_labels == 1 else "multilabel",
        )
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,  # to play nice with HF adjacent packages
        **kwargs,
    ) -> ESMCOutput:

        output = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        x = output.last_hidden_state
        features = self.pooler(x, attention_mask)
        logits = self.classifier(features)

        loss = None
        if labels is not None:  # 🔥 避免预测推理阶段报错
            if self.num_labels == 1:
                loss = self.criterion(  # 🔥 binary focal
                    logits.view(-1), labels.view(-1)
                )
            else:
                loss = self.criterion(  # 🔥 multiclass focal
                    logits.view(-1, self.num_labels), labels.view(-1)
                )

        return ESMCOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=x,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


class PepInterModelForEnergy(ESMCModel_Adapter):
    base_model_prefix = "pepinter"
    supports_gradient_checkpointing = True
    config_class = PepInterConfig

    def __init__(self, config: PepInterConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.num_labels = config.num_labels
        self.classifier = RegressionHead(config.hidden_size * 2, 1, config.hidden_size)

        self.pooler = Pooler(["cls", "mean"])
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,  # to play nice with HF adjacent packages
        **kwargs,
    ) -> ESMCOutput:

        output = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        x = output.last_hidden_state
        features = self.pooler(x, attention_mask)
        logits = self.classifier(features)

        loss = F.mse_loss(logits, labels)

        return ESMCOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=x,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


class PepInterModelForAffinity(ESMCModel_Adapter, MDNInferenceMixin):
    base_model_prefix = "pepinter"
    supports_gradient_checkpointing = True
    config_class = PepInterConfig

    def __init__(self, config: PepInterConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.num_labels = config.num_labels
        self.header = MDNHeader(
            input_dim=config.hidden_size * 2,
            hidden_dim=config.hidden_size,
            output_dim=config.num_labels,
        )
        
        self.pooler = Pooler(["cls", "mean"])
        self.init_weights()
        self.criterion = mdn_loss

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,  # to play nice with HF adjacent packages
        **kwargs,
    ) -> ESMCOutput:

        output = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        x = output.last_hidden_state
        features = self.pooler(x, attention_mask)

        # labels 安全处理（防止维度被 squeeze）
        if labels is not None:
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            labels = labels.float()

        pi, mu, sigma = self.header(features)

        loss = None
        # ====== ✅ small-loss trick：自适应过滤噪声样本 ======
        if labels is not None:
            if self.training:  # 只在训练阶段启用
                B, K, D = mu.shape
                y = labels.unsqueeze(1).expand(B, K, D)
                z = (y - mu) / sigma
                log_norm = (
                    -0.5 * (z * z).sum(-1)
                    - torch.log(sigma).sum(-1)
                    - 0.5 * D * math.log(2 * math.pi)
                )
                per = -(
                    torch.logsumexp(torch.log(pi.clamp_min(1e-12)) + log_norm, dim=-1)
                )  # [B]

                r = 0.8  # ✅ 仅保留 80% 最可信样本
                threshold = per.quantile(r)  # 选出前 r 小损
                w = (per <= threshold).float()  # 挑选靠谱样本
                w = w / w.mean().clamp_min(1e-6)  # 权重归一化
                loss = (per * w).mean()  # ✅ 最终有选择地更新模型
            else:
                loss = self.criterion(pi, mu, sigma, labels)

        pred = (pi.unsqueeze(-1) * mu).sum(dim=1)  # 更高效，不二次调用 header

        return ESMCOutput(
            loss=loss,
            logits=pred,
            last_hidden_state=x,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

class PepInterModelForAffinityTMP(ESMCModel_Adapter, MDNInferenceMixin):
    base_model_prefix = "pepinter"
    supports_gradient_checkpointing = True
    config_class = PepInterConfig

    def __init__(self, config: PepInterConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.num_labels = config.num_labels
        self.header = MDNHeader(
            input_dim=config.hidden_size * 2,
            hidden_dim=config.hidden_size,
            output_dim=config.num_labels,
        )
        
        self.pooler = Pooler(["cls", "mean"])
        self.init_weights()
        self.criterion = mdn_loss

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,  # to play nice with HF adjacent packages
        **kwargs,
    ) -> ESMCOutput:

        output = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        x = output.last_hidden_state
        features = self.pooler(x, attention_mask)

        # labels 安全处理（防止维度被 squeeze）
        if labels is not None:
            if labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            labels = labels.float()

        pi, mu, sigma = self.header(features)

        loss = None
        # ====== ✅ small-loss trick：自适应过滤噪声样本 ======
        if labels is not None:
            if self.training:  # 只在训练阶段启用
                B, K, D = mu.shape
                y = labels.unsqueeze(1).expand(B, K, D)
                z = (y - mu) / sigma
                log_norm = (
                    -0.5 * (z * z).sum(-1)
                    - torch.log(sigma).sum(-1)
                    - 0.5 * D * math.log(2 * math.pi)
                )
                per = -(
                    torch.logsumexp(torch.log(pi.clamp_min(1e-12)) + log_norm, dim=-1)
                )  # [B]

                r = 0.8  # ✅ 仅保留 80% 最可信样本
                threshold = per.quantile(r)  # 选出前 r 小损
                w = (per <= threshold).float()  # 挑选靠谱样本
                w = w / w.mean().clamp_min(1e-6)  # 权重归一化
                loss = (per * w).mean()  # ✅ 最终有选择地更新模型
            else:
                loss = self.criterion(pi, mu, sigma, labels)

        pred = (pi.unsqueeze(-1) * mu).sum(dim=1)  # 更高效，不二次调用 header

        return ESMCOutput(
            loss=loss,
            logits=pred,
            last_hidden_state=x,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

class PepInterModel(ESMCModel_Adapter):
    base_model_prefix = "pepinter"
    supports_gradient_checkpointing = True
    config_class = PepInterConfig

    def __init__(self, config: PepInterConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.num_labels = config.num_labels
        self.classifier = RegressionHead(
            config.hidden_size * 2, config.num_labels, config.hidden_size * 4
        )

        self.pooler = Pooler(["cls", "mean"])
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,  # to play nice with HF adjacent packages
        **kwargs,
    ) -> ESMCOutput:

        output = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        x = output.last_hidden_state
        features = self.pooler(x, attention_mask)
        logits = self.classifier(features)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = F.mse_loss(logits.flatten(), labels.flatten())
                else:
                    loss = F.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = F.binary_cross_entropy_with_logits(
                    logits.flatten(), labels.flatten()
                )
            elif self.config.problem_type == "multi_label_classification":
                loss = F.cross_entropy(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )

        return ESMCOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=x,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )
