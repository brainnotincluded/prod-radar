# Training Techniques to Improve Russian Sentiment F1-macro Beyond 0.76

**Current baseline:**
- Model: `cointegrated/rubert-tiny2` fine-tuned on 188K Russian social media samples
- F1-macro: **0.758**
- Loss: Weighted cross-entropy (sklearn `compute_class_weight("balanced")`)
- Labels: 3-class (pozitiv / negativ / neytralno)
- Main weaknesses: sarcasm, implicit complaints, emoji-driven sentiment

**Target: F1-macro >= 0.80**

---

## Table of Contents

| # | Technique | Expected Gain | Complexity | Compute Cost | Worth It? |
|---|-----------|---------------|------------|-------------|-----------|
| 1 | Curriculum Learning | +1-2% | Medium | ~1.2x | Yes |
| 2 | Focal / Label Smoothing / Dice Loss | +1-3% | Easy | 1x | Yes (Focal) |
| 3 | R-Drop | +1-2% | Easy | ~1.5x | Yes |
| 4 | Mixup / CutMix for Text | +0.5-1.5% | Medium | ~1.1x | Maybe |
| 5 | Multi-task Learning | +2-4% | Hard | ~1.3x | Yes |
| 6 | Knowledge Distillation | +2-3% | Medium | 2x+ | Yes |
| 7 | Contrastive Learning (SupCon) | +1-3% | Hard | ~2x | Yes |
| 8 | Adversarial Training (FGM/PGD) | +1-2% | Easy | ~1.3x | Yes |
| 9 | LLRD | +0.5-1.5% | Easy | 1x | Yes |
| 10 | Ensemble | +2-4% | Easy | Nx | Yes (if latency allows) |

---

## 1. Curriculum Learning

**Idea:** Train on "easy" examples first (clear positive/negative), then progressively introduce "hard" examples (sarcasm, implicit sentiment, emoji-only). This mirrors how humans learn -- simple before complex.

**Why it helps us:** Our main weaknesses (sarcasm, implicit complaints) are exactly the "hard" examples. Standard training treats them equally from epoch 1, which can confuse early gradient updates. Curriculum learning lets the model build a solid sentiment foundation before tackling ambiguity.

**Expected improvement:** +1-2% F1-macro
**Complexity:** Medium
**Compute cost:** ~1.2x (same epochs, but with sorting/scoring overhead)

**Approach:** Use the model's own loss as a difficulty proxy. After a warm-up epoch, compute per-sample loss, sort by difficulty, and train in easy-to-hard order.

```python
import torch
import numpy as np
from torch.utils.data import DataLoader, Sampler


class CurriculumSampler(Sampler):
    """Samples indices in easy-to-hard order based on per-sample loss."""

    def __init__(self, losses: np.ndarray, epoch: int, total_epochs: int):
        self.indices = np.argsort(losses)  # ascending = easy first
        # Competence: fraction of data to use (grows linearly)
        # epoch 0 -> use easiest 30%, epoch N -> use 100%
        competence = min(1.0, 0.3 + 0.7 * (epoch / max(total_epochs - 1, 1)))
        n_samples = int(len(self.indices) * competence)
        self.indices = self.indices[:n_samples]
        np.random.shuffle(self.indices)  # shuffle within the competence window

    def __iter__(self):
        return iter(self.indices.tolist())

    def __len__(self):
        return len(self.indices)


def compute_per_sample_loss(model, dataset, device="cuda"):
    """Run one pass to get per-sample losses for curriculum scoring."""
    model.eval()
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    all_losses = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Per-sample cross-entropy (no reduction)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, reduction="none"
            )
            all_losses.append(loss.cpu().numpy())
    return np.concatenate(all_losses)


# --- Usage in training loop ---
# After epoch 0 (warm-up with standard random sampling):
per_sample_losses = compute_per_sample_loss(model, train_dataset)

for epoch in range(1, EPOCHS):
    sampler = CurriculumSampler(per_sample_losses, epoch, EPOCHS)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    # ... standard training loop ...
    # Recompute losses at end of each epoch
    per_sample_losses = compute_per_sample_loss(model, train_dataset)
```

**Integration with HF Trainer:** Requires a custom `Trainer` subclass that overrides `get_train_dataloader()` to swap in the curriculum sampler each epoch. Alternatively, implement a manual training loop.

**Worth trying?** **Yes.** Directly addresses our sarcasm/implicit complaint weakness. Start with a warm-up epoch, then introduce curriculum. The overhead is minimal (one extra forward pass per epoch).

---

## 2. Focal Loss vs Label Smoothing vs Dice Loss

### 2a. Focal Loss

**Idea:** Down-weight the loss on well-classified (easy) examples so the model focuses on hard, misclassified ones. Controlled by gamma -- higher gamma = more focus on hard examples.

**Why it helps us:** The neutral class is likely overrepresented, and negative/positive may contain hard sarcastic examples. Focal loss automatically shifts attention to these.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss: focuses on hard-to-classify examples."""

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        """
        Args:
            alpha: per-class weights, tensor of shape (num_classes,).
                   Use compute_class_weight("balanced", ...) values.
            gamma: focusing parameter. 0 = standard CE, 2 = good default.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        # Gather the prob of the true class
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        pt = (probs * targets_one_hot).sum(dim=-1)  # p_t

        # Focal modulating factor
        focal_weight = (1 - pt) ** self.gamma

        # Standard CE
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        return loss


# --- In WeightedTrainer ---
class FocalTrainer(Trainer):
    def __init__(self, *args, class_weights=None, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        alpha = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.focal_loss(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss
```

**Expected improvement:** +1-2% F1-macro (especially on minority classes)
**Best gamma for sentiment:** Start with gamma=2.0, tune in {1.0, 2.0, 3.0}

### 2b. Label Smoothing

**Idea:** Instead of hard targets [0, 0, 1], use soft targets [0.033, 0.033, 0.933]. Prevents overconfidence and can improve calibration.

```python
# Simplest approach: built into HF TrainingArguments
training_args = TrainingArguments(
    # ... other args ...
    label_smoothing_factor=0.1,  # try 0.05, 0.1, 0.15
)

# Or manual implementation for more control:
class LabelSmoothingCE(nn.Module):
    def __init__(self, num_classes=3, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        # True class loss
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        # Smooth loss (uniform over all classes)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
```

**Expected improvement:** +0.5-1% F1-macro
**Note:** Less effective than focal loss for class imbalance, but great for calibration.

### 2c. Dice Loss

**Idea:** Borrowed from image segmentation. Optimizes the F1 score directly (Dice = 2*TP / (2*TP + FP + FN)). Particularly useful when optimizing macro-F1 is the actual goal.

```python
class DiceLoss(nn.Module):
    """Multiclass Dice Loss -- directly optimizes F1-like metric."""

    def __init__(self, smooth=1.0, square_denominator=False):
        super().__init__()
        self.smooth = smooth
        self.square_denominator = square_denominator

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()

        # Per-class dice
        intersection = (probs * targets_one_hot).sum(dim=0)
        if self.square_denominator:
            cardinality = (probs ** 2 + targets_one_hot ** 2).sum(dim=0)
        else:
            cardinality = (probs + targets_one_hot).sum(dim=0)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        # Macro average across classes
        return 1.0 - dice_per_class.mean()


# Often combined: CE + Dice
class CombinedLoss(nn.Module):
    def __init__(self, alpha_ce=0.5, alpha_dice=0.5, class_weights=None):
        super().__init__()
        self.alpha_ce = alpha_ce
        self.alpha_dice = alpha_dice
        self.dice = DiceLoss()
        self.class_weights = class_weights

    def forward(self, logits, targets):
        weight = torch.tensor(self.class_weights, dtype=torch.float32,
                              device=logits.device) if self.class_weights else None
        ce = F.cross_entropy(logits, targets, weight=weight)
        dice = self.dice(logits, targets)
        return self.alpha_ce * ce + self.alpha_dice * dice
```

**Expected improvement:** +1-2% F1-macro (especially for macro-averaging since it treats all classes equally)

### Recommendation for our case

**Use Focal Loss + Label Smoothing together.** Focal loss handles the class imbalance and hard-example focusing; label smoothing regularizes overconfidence on sarcasm cases.

```python
# Combined: Focal Loss with label smoothing
class FocalWithSmoothing(nn.Module):
    def __init__(self, num_classes=3, gamma=2.0, smoothing=0.05, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.alpha = alpha

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        # Smooth targets
        smooth_targets = torch.full_like(probs, self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # Focal modulation on the true-class probability
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma

        # KL-div style loss with smooth targets
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        loss = focal_weight * loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss

        return loss.mean()
```

**Worth trying?** **Yes -- Focal Loss is the single easiest win.** Can be swapped in with 15 lines of code. Try focal first, add label smoothing if F1 plateaus.

---

## 3. R-Drop (Regularized Dropout)

**Idea:** Run each input through the model twice (with different dropout masks). Minimize the KL divergence between the two output distributions. This forces the model to produce consistent predictions regardless of which neurons are dropped, leading to more robust representations.

**Why it helps us:** Sarcasm and implicit complaints are borderline cases where the model's prediction can flip depending on dropout. R-Drop stabilizes these fragile predictions.

**Expected improvement:** +1-2% F1-macro
**Complexity:** Easy
**Compute cost:** ~1.5x (two forward passes per input)

```python
import torch.nn.functional as F


class RDropTrainer(Trainer):
    """R-Drop: Regularized Dropout for consistent predictions."""

    def __init__(self, *args, rdrop_alpha=5.0, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rdrop_alpha = rdrop_alpha
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")

        # Forward pass 1 (dropout mask A)
        outputs1 = model(**inputs)
        logits1 = outputs1.logits

        # Forward pass 2 (dropout mask B) -- different due to stochastic dropout
        outputs2 = model(**inputs)
        logits2 = outputs2.logits

        # Standard CE loss on both
        weight = (torch.tensor(self.class_weights, dtype=torch.float32, device=logits1.device)
                  if self.class_weights is not None else None)
        ce_loss = 0.5 * (
            F.cross_entropy(logits1, labels, weight=weight) +
            F.cross_entropy(logits2, labels, weight=weight)
        )

        # KL divergence between the two predictions (symmetric)
        p1 = F.log_softmax(logits1, dim=-1)
        p2 = F.log_softmax(logits2, dim=-1)
        q1 = F.softmax(logits1, dim=-1)
        q2 = F.softmax(logits2, dim=-1)
        kl_loss = 0.5 * (
            F.kl_div(p1, q2, reduction="batchmean") +
            F.kl_div(p2, q1, reduction="batchmean")
        )

        loss = ce_loss + self.rdrop_alpha * kl_loss

        return (loss, outputs1) if return_outputs else loss


# Usage:
trainer = RDropTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    rdrop_alpha=5.0,        # tune in {1.0, 3.0, 5.0, 7.0}
    class_weights=class_weights.tolist(),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
```

**Key hyperparameter:** `rdrop_alpha` controls the KL regularization strength. Start with 5.0 for sentiment tasks.

**Worth trying?** **Yes.** Very easy to implement (just a custom Trainer). Consistently reported +1-2% in NLU tasks. Especially helpful for our borderline sarcasm cases where dropout variance causes label flipping.

---

## 4. Mixup / CutMix for Text

**Idea:** Interpolate between training examples in embedding space (Mixup) or swap token subsequences between examples (CutMix). Creates synthetic training data that forces the model to learn smoother decision boundaries.

**Expected improvement:** +0.5-1.5% F1-macro
**Complexity:** Medium
**Compute cost:** ~1.1x

### 4a. Embedding Mixup (recommended for text)

```python
import torch
import torch.nn as nn
from transformers import AutoModel


class MixupSentimentModel(nn.Module):
    """rubert-tiny2 with Mixup in embedding space."""

    def __init__(self, model_name, num_labels=3, mixup_alpha=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)
        self.mixup_alpha = mixup_alpha

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (B, hidden)

        if self.training and labels is not None and self.mixup_alpha > 0:
            # Sample mixup coefficient from Beta distribution
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample()
            lam = max(lam, 1 - lam)  # ensure lam >= 0.5

            # Shuffle indices for mixing partners
            batch_size = cls_emb.size(0)
            perm = torch.randperm(batch_size, device=cls_emb.device)

            # Mix embeddings
            mixed_emb = lam * cls_emb + (1 - lam) * cls_emb[perm]
            logits = self.classifier(self.dropout(mixed_emb))

            # Mix labels (soft targets)
            loss_a = nn.functional.cross_entropy(logits, labels)
            loss_b = nn.functional.cross_entropy(logits, labels[perm])
            loss = lam * loss_a + (1 - lam) * loss_b

            return {"loss": loss, "logits": logits}
        else:
            logits = self.classifier(self.dropout(cls_emb))
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
```

### 4b. Token CutMix

```python
def token_cutmix(input_ids_a, input_ids_b, attention_mask_a, attention_mask_b, lam=0.7):
    """Swap a contiguous span of tokens between two sequences."""
    seq_len = input_ids_a.size(1)
    cut_len = int(seq_len * (1 - lam))
    # Random cut position (avoid [CLS] at 0 and [SEP] at end)
    start = torch.randint(1, max(seq_len - cut_len, 2), (1,)).item()
    end = min(start + cut_len, seq_len - 1)

    mixed_ids = input_ids_a.clone()
    mixed_mask = attention_mask_a.clone()
    mixed_ids[:, start:end] = input_ids_b[:, start:end]
    mixed_mask[:, start:end] = attention_mask_b[:, start:end]

    # Adjust lambda based on actual cut
    actual_lam = 1 - (end - start) / seq_len
    return mixed_ids, mixed_mask, actual_lam
```

**Worth trying?** **Maybe.** Embedding Mixup is the safer choice for text. Token-level CutMix can produce nonsensical sequences. This is lower priority than Focal Loss, R-Drop, or Adversarial Training. Try it if you have exhausted the higher-impact techniques.

---

## 5. Multi-task Learning

**Idea:** Train the model on multiple related tasks simultaneously: sentiment + emotion detection + sarcasm detection. Shared representations learn richer features.

**Why it helps us:** Our weaknesses (sarcasm, implicit complaints) are exactly the auxiliary tasks. If the model learns a sarcasm detector jointly, its sentiment predictions on sarcastic text will improve.

**Expected improvement:** +2-4% F1-macro (substantial if auxiliary data is available)
**Complexity:** Hard (requires auxiliary datasets)
**Compute cost:** ~1.3x

```python
import torch
import torch.nn as nn
from transformers import AutoModel


class MultiTaskSentimentModel(nn.Module):
    """
    Multi-task model:
      - Head 1: Sentiment (pozitiv/negativ/neytralno) -- primary
      - Head 2: Emotion (joy/anger/sadness/fear/surprise/neutral)
      - Head 3: Sarcasm (binary)
    """

    def __init__(self, model_name="cointegrated/rubert-tiny2",
                 n_sentiment=3, n_emotion=6, n_sarcasm=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size  # 312 for rubert-tiny2

        self.dropout = nn.Dropout(0.1)

        # Task-specific heads
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_sentiment),
        )
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_emotion),
        )
        self.sarcasm_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_sarcasm),
        )

    def forward(self, input_ids, attention_mask,
                sentiment_labels=None, emotion_labels=None, sarcasm_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = self.dropout(outputs.last_hidden_state[:, 0, :])

        sentiment_logits = self.sentiment_head(cls_emb)
        emotion_logits = self.emotion_head(cls_emb)
        sarcasm_logits = self.sarcasm_head(cls_emb)

        loss = torch.tensor(0.0, device=cls_emb.device)
        if sentiment_labels is not None:
            loss += 1.0 * nn.functional.cross_entropy(sentiment_logits, sentiment_labels)
        if emotion_labels is not None:
            loss += 0.3 * nn.functional.cross_entropy(emotion_logits, emotion_labels)
        if sarcasm_labels is not None:
            loss += 0.5 * nn.functional.cross_entropy(sarcasm_logits, sarcasm_labels)

        return {
            "loss": loss,
            "sentiment_logits": sentiment_logits,
            "emotion_logits": emotion_logits,
            "sarcasm_logits": sarcasm_logits,
        }


# --- Dataset for multi-task ---
class MultiTaskDataset(torch.utils.data.Dataset):
    """Handles missing labels gracefully (not all samples have all labels)."""

    def __init__(self, texts, tokenizer, sentiment_labels=None,
                 emotion_labels=None, sarcasm_labels=None, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length",
                                   max_length=max_length, return_tensors="pt")
        self.sentiment_labels = sentiment_labels
        self.emotion_labels = emotion_labels
        self.sarcasm_labels = sarcasm_labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }
        if self.sentiment_labels is not None:
            item["sentiment_labels"] = torch.tensor(self.sentiment_labels[idx], dtype=torch.long)
        if self.emotion_labels is not None:
            item["emotion_labels"] = torch.tensor(self.emotion_labels[idx], dtype=torch.long)
        if self.sarcasm_labels is not None:
            item["sarcasm_labels"] = torch.tensor(self.sarcasm_labels[idx], dtype=torch.long)
        return item
```

**Auxiliary datasets for Russian:**
- **Emotion:** CEDR (Corpus of Emotions in Russian Dialogs), or RuSentiment with emotion annotations
- **Sarcasm:** Russian Sarcasm Dataset from social media (VK/Telegram), or auto-label using LLM (GPT-4 / Claude) on a subset
- **If no auxiliary data:** Use pseudo-labels from a large model as a bootstrap

**Worth trying?** **Yes, if you can obtain sarcasm labels.** Even pseudo-labeling 10-20K examples for sarcasm with an LLM and using them as an auxiliary task would likely give +1-2% on sarcastic sentiment. The sarcasm head is the single most impactful auxiliary task for our specific weakness.

---

## 6. Knowledge Distillation

**Idea:** Train a large "teacher" model (e.g., `ai-forever/ruBert-large` or `DeepPavlov/rubert-base-cased-sentence`) on the same data, then use its soft predictions to train the smaller rubert-tiny2 "student". The student learns from the teacher's probability distribution (dark knowledge) which contains richer information than hard labels.

**Expected improvement:** +2-3% F1-macro
**Complexity:** Medium (two training runs)
**Compute cost:** 2x+ (train teacher first, then student)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, Trainer


class DistillationTrainer(Trainer):
    """Knowledge distillation: student learns from teacher's soft targets."""

    def __init__(self, *args, teacher_model=None, temperature=4.0,
                 alpha_ce=0.3, alpha_kd=0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.temperature = temperature
        self.alpha_ce = alpha_ce  # weight for hard-label CE loss
        self.alpha_kd = alpha_kd  # weight for distillation KL loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")

        # Student forward
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_logits = teacher_outputs.logits

        # Hard-label CE loss
        ce_loss = F.cross_entropy(student_logits, labels)

        # Soft-label KD loss (KL divergence at temperature T)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
        kd_loss = kd_loss * (self.temperature ** 2)  # scale by T^2

        loss = self.alpha_ce * ce_loss + self.alpha_kd * kd_loss
        return (loss, student_outputs) if return_outputs else loss


# --- Usage ---
# Step 1: Train teacher (rubert-large)
teacher_model_name = "ai-forever/ruBert-large"  # 340M params
# ... train teacher with standard pipeline (same as current but larger model) ...

# Step 2: Distill to student (rubert-tiny2)
student_model = AutoModelForSequenceClassification.from_pretrained(
    "cointegrated/rubert-tiny2", num_labels=3
)
teacher_model = AutoModelForSequenceClassification.from_pretrained(
    "path/to/trained-rubert-large"
)
teacher_model = teacher_model.to("cuda")

trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    teacher_model=teacher_model,
    temperature=4.0,    # try {2, 4, 6, 8}
    alpha_ce=0.3,       # hard label weight
    alpha_kd=0.7,       # soft label weight
)
trainer.train()
```

**Alternative: Offline distillation (cheaper)**

If training the teacher is too expensive, use pre-computed teacher logits:

```python
import numpy as np


def precompute_teacher_logits(teacher, dataset, batch_size=128):
    """Run teacher once, save logits to disk."""
    teacher.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to("cuda") for k, v in batch.items() if k != "labels"}
            logits = teacher(**inputs).logits
            all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits)


# Save: np.save("teacher_logits.npy", teacher_logits)
# Load during student training as additional supervision signal
```

**Teacher model candidates (ranked):**
1. `ai-forever/ruBert-large` -- best quality, expensive
2. `DeepPavlov/rubert-base-cased-sentence` -- good balance
3. `cointegrated/rubert-tiny2` ensemble (5 models) -- cheapest teacher

**Worth trying?** **Yes.** Knowledge distillation is one of the most reliable ways to improve a small model. The key insight: the teacher's probability distribution over classes (e.g., [0.6 neg, 0.3 neutral, 0.1 pos]) for a sarcastic comment carries more information than the hard label alone. The student learns that "this example is primarily negative but somewhat neutral" which is exactly what we need for ambiguous cases.

---

## 7. Contrastive Learning (SimCSE / SupCon)

**Idea:** Learn an embedding space where same-sentiment examples are pulled together and different-sentiment examples are pushed apart. Then fine-tune the classifier on top of these better representations.

**Expected improvement:** +1-3% F1-macro
**Complexity:** Hard (two-stage training)
**Compute cost:** ~2x (contrastive pre-training + fine-tuning)

### 7a. Supervised Contrastive Loss (SupCon)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020).
    Pulls together examples with the same label, pushes apart different labels.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: (B, D) L2-normalized embeddings
            labels: (B,) integer class labels
        """
        device = features.device
        batch_size = features.size(0)

        # Similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # Mask: 1 if same label (positive pair), 0 otherwise
        labels = labels.unsqueeze(0)  # (1, B)
        mask = (labels == labels.T).float().to(device)  # (B, B)

        # Remove self-similarity from the diagonal
        logits_mask = 1.0 - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # Log-softmax over all negatives + positives (excluding self)
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Average log-prob over positive pairs
        n_positives = mask.sum(dim=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / (n_positives + 1e-8)

        loss = -mean_log_prob.mean()
        return loss


class ContrastiveSentimentModel(nn.Module):
    """Two-stage model: contrastive pre-training, then classification."""

    def __init__(self, model_name, hidden_size=312, proj_size=128, num_labels=3):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)

        # Projection head for contrastive learning (discarded after pre-training)
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_size),
        )

        # Classification head (used after contrastive pre-training)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.mode = "contrastive"  # or "classify"

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]

        if self.mode == "contrastive":
            proj = self.projector(cls_emb)
            proj = F.normalize(proj, dim=-1)  # L2 normalize
            return {"embeddings": proj}
        else:
            logits = self.classifier(cls_emb)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}


# --- Training procedure ---
# Stage 1: Contrastive pre-training (5-10 epochs)
model = ContrastiveSentimentModel("cointegrated/rubert-tiny2")
model.mode = "contrastive"
supcon = SupConLoss(temperature=0.07)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
for epoch in range(10):
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids, attention_mask)
        loss = supcon(out["embeddings"], labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Stage 2: Freeze BERT, train classifier (or fine-tune all with small LR)
model.mode = "classify"
# Optionally freeze BERT for a few epochs:
# for param in model.bert.parameters():
#     param.requires_grad = False
# ... then unfreeze and fine-tune with small LR
```

### 7b. SimCSE-style (unsupervised pre-training)

If you want to improve representations without labels first:

```python
# SimCSE: use dropout as augmentation
# Pass same input twice -> two different dropout masks -> positive pair
# All other examples in batch -> negative pairs

class SimCSELoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """z1, z2: (B, D) embeddings from two forward passes."""
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Cosine similarity matrix
        sim = torch.matmul(z1, z2.T) / self.temperature  # (B, B)

        # Positive pairs are on the diagonal
        labels = torch.arange(z1.size(0), device=z1.device)
        loss = F.cross_entropy(sim, labels)
        return loss
```

**Worth trying?** **Yes, but as a second-round optimization.** SupCon is particularly powerful for sentiment because it explicitly structures the embedding space by sentiment polarity. However, it requires careful temperature tuning and a two-stage pipeline. Try simpler techniques (Focal Loss, LLRD, Adversarial) first.

---

## 8. Adversarial Training (FGM / PGD)

**Idea:** Add small, worst-case perturbations to the input embeddings during training. This forces the model to be robust to slight variations, which helps with sarcasm and subtle linguistic cues.

**Expected improvement:** +1-2% F1-macro
**Complexity:** Easy
**Compute cost:** ~1.3x (FGM) to ~1.6x (PGD with 3 steps)

### 8a. FGM (Fast Gradient Method) -- recommended starting point

```python
import torch


class FGM:
    """Fast Gradient Method for adversarial training.
    Perturbs word embeddings by epsilon in the gradient direction.
    """

    def __init__(self, model, epsilon=1.0, emb_name="word_embeddings"):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        """Add adversarial perturbation to embeddings."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    perturbation = self.epsilon * param.grad / norm
                    param.data.add_(perturbation)

    def restore(self):
        """Restore original embeddings."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# --- Integration with HF Trainer ---
class AdversarialTrainer(Trainer):
    """Trainer with FGM adversarial training on embedding layer."""

    def __init__(self, *args, fgm_epsilon=1.0, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fgm = None
        self.fgm_epsilon = fgm_epsilon
        self.class_weights = class_weights

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Initialize FGM on first call
        if self.fgm is None:
            self.fgm = FGM(model, epsilon=self.fgm_epsilon)

        model.train()
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")

        # Step 1: Normal forward + backward
        outputs = model(**inputs)
        weight = (torch.tensor(self.class_weights, dtype=torch.float32, device=outputs.logits.device)
                  if self.class_weights else None)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        self.accelerator.backward(loss)

        # Step 2: Adversarial forward + backward (on perturbed embeddings)
        self.fgm.attack()
        outputs_adv = model(**inputs)
        loss_adv = torch.nn.functional.cross_entropy(outputs_adv.logits, labels, weight=weight)
        self.accelerator.backward(loss_adv)
        self.fgm.restore()

        return (loss + loss_adv).detach()
```

### 8b. PGD (Projected Gradient Descent) -- stronger but slower

```python
class PGD:
    """Projected Gradient Descent -- iterative adversarial attack.
    Stronger than FGM but requires K forward-backward passes.
    """

    def __init__(self, model, epsilon=1.0, alpha=0.3, steps=3, emb_name="word_embeddings"):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha  # step size
        self.steps = steps
        self.emb_name = emb_name
        self.backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    perturbation = self.alpha * param.grad / norm
                    param.data.add_(perturbation)
                    # Project back to epsilon-ball
                    delta = param.data - self.backup[name]
                    delta_norm = torch.norm(delta)
                    if delta_norm > self.epsilon:
                        delta = self.epsilon * delta / delta_norm
                    param.data = self.backup[name] + delta

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                param.data = self.backup[name]
        self.backup = {}

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name in self.grad_backup:
                    param.grad = self.grad_backup[name]


# PGD training loop (manual):
pgd = PGD(model, epsilon=1.0, alpha=0.3, steps=3)
for batch in train_loader:
    # Normal forward-backward
    outputs = model(**batch)
    loss = criterion(outputs.logits, batch["labels"])
    loss.backward()
    pgd.backup_grad()

    # K steps of PGD
    for step in range(pgd.steps):
        pgd.attack(is_first_attack=(step == 0))
        if step != pgd.steps - 1:
            model.zero_grad()
        outputs_adv = model(**batch)
        loss_adv = criterion(outputs_adv.logits, batch["labels"])
        loss_adv.backward()

    pgd.restore_grad()
    pgd.restore()
    optimizer.step()
    optimizer.zero_grad()
```

**Worth trying?** **Yes -- FGM is a top-3 easy win.** Start with FGM (epsilon=1.0), which adds only ~30% compute overhead. If you see gains, try PGD with 3 steps. Adversarial training is especially effective for our sarcasm weakness because sarcasm often involves subtle word choices that small perturbations can simulate.

---

## 9. Layer-wise Learning Rate Decay (LLRD)

**Idea:** Use a higher learning rate for the classifier head and upper transformer layers, and progressively lower learning rates for deeper layers. The intuition: lower layers capture general linguistic features (keep them stable), upper layers capture task-specific features (adapt them aggressively).

**Expected improvement:** +0.5-1.5% F1-macro
**Complexity:** Easy
**Compute cost:** 1x (same training time, just different optimizer config)

```python
from transformers import AutoModelForSequenceClassification


def get_llrd_optimizer(model, base_lr=2e-5, decay_factor=0.9, weight_decay=0.01):
    """Create AdamW optimizer with layer-wise learning rate decay.

    For rubert-tiny2 (3 transformer layers):
      - Embeddings: base_lr * decay^3 = 2e-5 * 0.729 = 1.46e-5
      - Layer 0:    base_lr * decay^2 = 2e-5 * 0.81  = 1.62e-5
      - Layer 1:    base_lr * decay^1 = 2e-5 * 0.9   = 1.80e-5
      - Layer 2:    base_lr * decay^0 = 2e-5 * 1.0   = 2.00e-5
      - Classifier: base_lr * 5       = 1e-4          (5x higher)
    """
    opt_params = []
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}

    # Get number of transformer layers
    # rubert-tiny2 has model.bert.encoder.layer[0..2]
    num_layers = model.config.num_hidden_layers  # 3 for rubert-tiny2

    # Classifier head (highest LR)
    classifier_params = []
    for name, param in model.named_parameters():
        if "classifier" in name:
            classifier_params.append(param)
    opt_params.append({
        "params": classifier_params,
        "lr": base_lr * 5,
        "weight_decay": 0.0,
    })

    # Transformer layers (decaying LR from top to bottom)
    for layer_idx in range(num_layers - 1, -1, -1):
        layer_params_decay = []
        layer_params_no_decay = []

        for name, param in model.named_parameters():
            if f"encoder.layer.{layer_idx}." in name:
                if any(nd in name for nd in no_decay):
                    layer_params_no_decay.append(param)
                else:
                    layer_params_decay.append(param)

        depth = num_layers - 1 - layer_idx  # 0 for top layer, 2 for bottom
        lr = base_lr * (decay_factor ** depth)

        if layer_params_decay:
            opt_params.append({
                "params": layer_params_decay,
                "lr": lr,
                "weight_decay": weight_decay,
            })
        if layer_params_no_decay:
            opt_params.append({
                "params": layer_params_no_decay,
                "lr": lr,
                "weight_decay": 0.0,
            })

    # Embeddings (lowest LR)
    emb_params_decay = []
    emb_params_no_decay = []
    for name, param in model.named_parameters():
        if "embeddings" in name:
            if any(nd in name for nd in no_decay):
                emb_params_no_decay.append(param)
            else:
                emb_params_decay.append(param)

    emb_lr = base_lr * (decay_factor ** num_layers)
    if emb_params_decay:
        opt_params.append({
            "params": emb_params_decay,
            "lr": emb_lr,
            "weight_decay": weight_decay,
        })
    if emb_params_no_decay:
        opt_params.append({
            "params": emb_params_no_decay,
            "lr": emb_lr,
            "weight_decay": 0.0,
        })

    optimizer = torch.optim.AdamW(opt_params)
    return optimizer


# --- Integration with HF Trainer ---
class LLRDTrainer(Trainer):
    def __init__(self, *args, decay_factor=0.9, **kwargs):
        self.decay_factor = decay_factor
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        self.optimizer = get_llrd_optimizer(
            self.model,
            base_lr=self.args.learning_rate,
            decay_factor=self.decay_factor,
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer


# Usage:
trainer = LLRDTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    decay_factor=0.9,  # try {0.8, 0.85, 0.9, 0.95}
)
```

**Worth trying?** **Yes -- zero-cost improvement.** This changes nothing about training time or memory. It only changes how the optimizer is configured. For rubert-tiny2 with only 3 transformer layers, the effect is smaller than for BERT-base (12 layers), but still consistently positive. Try `decay_factor=0.85` for our small model.

---

## 10. Ensemble Methods

**Idea:** Train multiple models (different seeds, architectures, or hyperparameters) and combine their predictions at inference time.

**Expected improvement:** +2-4% F1-macro
**Complexity:** Easy (training is trivial, inference cost is the concern)
**Compute cost:** Nx training, Nx inference (N = number of models)

### 10a. Simple Averaging (start here)

```python
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentEnsemble:
    """Ensemble of N sentiment models. Averages softmax probabilities."""

    def __init__(self, model_paths: list, device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_paths[0])
        self.models = []
        for path in model_paths:
            model = AutoModelForSequenceClassification.from_pretrained(path)
            model.to(device)
            model.eval()
            self.models.append(model)

    @torch.no_grad()
    def predict(self, texts: list, batch_size=64):
        """Average softmax probabilities across all models."""
        all_probs = []
        for model in self.models:
            model_probs = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_texts, truncation=True, padding=True,
                    max_length=128, return_tensors="pt"
                ).to(self.device)
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                model_probs.append(probs.cpu().numpy())
            all_probs.append(np.concatenate(model_probs))

        # Average across models
        avg_probs = np.mean(all_probs, axis=0)
        predictions = np.argmax(avg_probs, axis=-1)
        return predictions, avg_probs

    @torch.no_grad()
    def predict_with_confidence(self, texts: list, batch_size=64):
        """Returns predictions + disagreement score for uncertainty estimation."""
        preds, avg_probs = self.predict(texts, batch_size)
        # Disagreement: std of predicted class across models
        all_preds = []
        for model in self.models:
            model_preds = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_texts, truncation=True, padding=True,
                    max_length=128, return_tensors="pt"
                ).to(self.device)
                logits = model(**inputs).logits
                model_preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
            all_preds.append(np.concatenate(model_preds))
        all_preds = np.array(all_preds)  # (N_models, N_samples)

        # Disagreement ratio: fraction of models that disagree with majority
        from scipy.stats import mode
        majority, _ = mode(all_preds, axis=0)
        disagreement = (all_preds != majority).mean(axis=0)

        return preds, avg_probs, disagreement


# --- Train N models with different seeds ---
def train_ensemble(n_models=5, seeds=None):
    """Train N models with different random seeds."""
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]

    model_paths = []
    for i, seed in enumerate(seeds[:n_models]):
        print(f"Training model {i+1}/{n_models} with seed={seed}")
        training_args = TrainingArguments(
            output_dir=f"models/ensemble/model_{i}",
            seed=seed,
            # ... same hyperparameters as baseline ...
        )
        # ... train and save ...
        model_paths.append(f"models/ensemble/model_{i}/final")

    return model_paths


# Usage:
ensemble = SentimentEnsemble([
    "models/ensemble/model_0/final",
    "models/ensemble/model_1/final",
    "models/ensemble/model_2/final",
])
predictions, probabilities = ensemble.predict(["Отличный продукт!", "Ужасный сервис"])
```

### 10b. Diverse Ensemble (different architectures/configs)

For maximum diversity, combine models with different:
- Seeds (cheapest diversity)
- Max sequence lengths (64, 128, 256)
- Loss functions (CE, Focal, Dice)
- Learning rates

```python
# More diverse = better ensemble
ensemble_configs = [
    {"seed": 42,  "loss": "focal", "lr": 2e-5, "max_len": 128},
    {"seed": 123, "loss": "ce",    "lr": 3e-5, "max_len": 128},
    {"seed": 456, "loss": "focal", "lr": 2e-5, "max_len": 256},
    {"seed": 789, "loss": "dice",  "lr": 1e-5, "max_len": 128},
    {"seed": 42,  "loss": "focal", "lr": 2e-5, "max_len": 128, "adversarial": True},
]
```

### 10c. Stacking (learned ensemble)

```python
from sklearn.linear_model import LogisticRegression


def train_stacker(model_paths, val_texts, val_labels):
    """Train a logistic regression stacker on model probabilities."""
    all_probs = []
    for path in model_paths:
        model = AutoModelForSequenceClassification.from_pretrained(path).to("cuda").eval()
        tokenizer = AutoTokenizer.from_pretrained(path)
        probs = []
        with torch.no_grad():
            for i in range(0, len(val_texts), 64):
                batch = val_texts[i:i+64]
                inputs = tokenizer(batch, truncation=True, padding=True,
                                   max_length=128, return_tensors="pt").to("cuda")
                logits = model(**inputs).logits
                probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        all_probs.append(np.concatenate(probs))

    # Stack features: (N_samples, N_models * N_classes)
    X_stack = np.concatenate(all_probs, axis=1)

    stacker = LogisticRegression(max_iter=1000, C=1.0)
    stacker.fit(X_stack, val_labels)
    return stacker
```

**Worth trying?** **Yes -- ensembles almost always help.** The question is whether inference latency/cost is acceptable. For our case:
- If this is a batch pipeline (process social media posts periodically): ensemble of 3-5 models is feasible.
- If this is a real-time API: consider distilling the ensemble into a single model (see technique #6).
- Minimum viable ensemble: 3 models with different seeds + Focal Loss, average probabilities. Expected: +2-3% F1-macro.

---

## Recommended Implementation Order

Based on effort-to-impact ratio for our specific case (rubert-tiny2, 188K samples, sarcasm weakness):

### Phase 1: Quick wins (1-2 days, expected +2-4% cumulative)

1. **LLRD** (Section 9) -- zero-cost, 30 min to implement
2. **Focal Loss** (Section 2a) -- swap loss function, 1 hour
3. **FGM Adversarial Training** (Section 8a) -- 1 hour, reliable +1-2%
4. **R-Drop** (Section 3) -- 1 hour, +1-2%

### Phase 2: Medium effort (3-5 days, expected +2-3% additional)

5. **Knowledge Distillation** (Section 6) -- train rubert-base teacher first
6. **Curriculum Learning** (Section 1) -- implement difficulty scoring
7. **Ensemble of 3 models** (Section 10a) -- train 3 seeds with best Phase 1 config

### Phase 3: High effort, high reward (1-2 weeks)

8. **Multi-task with sarcasm detection** (Section 5) -- need sarcasm labels
9. **Contrastive pre-training** (Section 7) -- SupCon before fine-tuning
10. **Mixup** (Section 4) -- marginal gains at this point

### Combined Recipe (all Phase 1 techniques together)

```python
class UltraTrainer(Trainer):
    """Combines LLRD + Focal Loss + FGM + R-Drop in one Trainer."""

    def __init__(self, *args, class_weights=None, focal_gamma=2.0,
                 fgm_epsilon=1.0, rdrop_alpha=5.0, llrd_decay=0.85, **kwargs):
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        self.fgm_epsilon = fgm_epsilon
        self.rdrop_alpha = rdrop_alpha
        self.llrd_decay = llrd_decay
        self.fgm = None
        self.focal_loss = FocalLoss(
            alpha=torch.tensor(class_weights) if class_weights else None,
            gamma=focal_gamma
        )
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        """LLRD optimizer."""
        self.optimizer = get_llrd_optimizer(
            self.model,
            base_lr=self.args.learning_rate,
            decay_factor=self.llrd_decay,
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Focal Loss + R-Drop."""
        labels = inputs.pop("labels")

        # R-Drop: two forward passes
        outputs1 = model(**inputs)
        outputs2 = model(**inputs)

        # Focal loss on both
        focal1 = self.focal_loss(outputs1.logits, labels)
        focal2 = self.focal_loss(outputs2.logits, labels)
        ce_loss = 0.5 * (focal1 + focal2)

        # KL divergence (R-Drop)
        p1 = torch.nn.functional.log_softmax(outputs1.logits, dim=-1)
        p2 = torch.nn.functional.log_softmax(outputs2.logits, dim=-1)
        q1 = torch.nn.functional.softmax(outputs1.logits, dim=-1)
        q2 = torch.nn.functional.softmax(outputs2.logits, dim=-1)
        kl_loss = 0.5 * (
            torch.nn.functional.kl_div(p1, q2, reduction="batchmean") +
            torch.nn.functional.kl_div(p2, q1, reduction="batchmean")
        )

        loss = ce_loss + self.rdrop_alpha * kl_loss
        return (loss, outputs1) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Add FGM adversarial perturbation."""
        if self.fgm is None:
            self.fgm = FGM(model, epsilon=self.fgm_epsilon)

        model.train()
        inputs = self._prepare_inputs(inputs)

        # Normal forward + backward (includes R-Drop + Focal)
        loss = self.compute_loss(model, dict(inputs))
        self.accelerator.backward(loss)

        # Adversarial step
        self.fgm.attack()
        loss_adv = self.compute_loss(model, dict(inputs))
        self.accelerator.backward(loss_adv)
        self.fgm.restore()

        return loss.detach()


# --- Full usage ---
model = AutoModelForSequenceClassification.from_pretrained(
    "cointegrated/rubert-tiny2", num_labels=3
)

training_args = TrainingArguments(
    output_dir="models/ultra_sentiment",
    num_train_epochs=5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=42,
)

trainer = UltraTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    class_weights=class_weights.tolist(),
    focal_gamma=2.0,
    fgm_epsilon=1.0,
    rdrop_alpha=5.0,
    llrd_decay=0.85,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()
```

---

## Expected Cumulative Improvement

| Stage | Techniques | Estimated F1-macro |
|-------|-----------|-------------------|
| Baseline | Weighted CE | 0.758 |
| + Phase 1 | LLRD + Focal + FGM + R-Drop | 0.78 - 0.80 |
| + Phase 2 | + Distillation + Curriculum + Ensemble(3) | 0.81 - 0.83 |
| + Phase 3 | + Multi-task (sarcasm) + SupCon | 0.83 - 0.86 |

**Note:** Gains are not strictly additive. Each technique has diminishing returns when combined with others. The estimates above account for this overlap.

---

## Key References

1. **Focal Loss:** Lin et al., "Focal Loss for Dense Object Detection" (2017)
2. **R-Drop:** Wu et al., "R-Drop: Regularized Dropout for Neural Networks" (2021)
3. **Curriculum Learning:** Bengio et al., "Curriculum Learning" (2009)
4. **SupCon:** Khosla et al., "Supervised Contrastive Learning" (2020)
5. **SimCSE:** Gao et al., "SimCSE: Simple Contrastive Learning of Sentence Embeddings" (2021)
6. **FGM/PGD for NLP:** Miyato et al., "Adversarial Training Methods for Semi-Supervised Text Classification" (2017)
7. **LLRD:** Howard & Ruder, "Universal Language Model Fine-tuning for Text Classification" (2018)
8. **Knowledge Distillation:** Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
9. **Mixup:** Zhang et al., "mixup: Beyond Empirical Risk Minimization" (2018)
10. **Dice Loss for NLU:** Li et al., "Dice Loss for Data-imbalanced NLP Tasks" (2020)
