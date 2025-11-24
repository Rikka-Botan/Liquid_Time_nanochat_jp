# 🌸 Liquid-Time-nanochat ver Japanese & English

このリポジトリは、**nanochat** をより高速に学習・推論するために、**Liquid Time-Constant Networks (LTCs)** および **Liquid Foundation Models (LFM2)** から着想を得た **SLC2** というモジュールを使用しています。
**SEA Model series Op.0: Saint Iberis** は、学習時間を削減しながら、同等の性能を実現します。
このリポジトリでは日本語・英語バイリンガルなモデルを構築することができます。

# 🌸 Saint Iberis Architecture

<img width="4400" height="1595" alt="Saint_Iberis" src="https://github.com/user-attachments/assets/9edba3df-0bc6-43b0-aedd-f57ad1929ee3" />

| Property              | Saint Iberis d12              | Remarks                                               |
| --------------------- | ----------------------------- |------------------------------------------------------ |
| **Total parameters**  | 376,240,128 (376M)            | n_layer: 16, n_head: 16, n_kv_head: 16, n_embd: 1024  |
| **Layers**            | 16 (9 slc2 + 7 attn)          | attn layers: 1, 4, 7, 10, 11, 14, 15                  |
| **Vocabulary size**   | 65,536                        | -                                                     |
| **License**           | Apache                        | -                                                    |

# 🌸 SLC2 Formulation

```markdown
y = B ⋅ ∏ᵢ₌ⱼ⁽ʲ⁺ᵏ⁾ Aᵢ ⋅ xᵢ
```

# 🌸 SLC2 pseudo code

```python
----------------------------------------
Algorithm: SLC2
----------------------------------------
Input: x: (B, S, E)
Output: y: (B, S, E)
    1: alpha, A, B, x₁ <- Linear(x)
    2: x₂: (B, S, E) <- Convolution1D(E, E)(SiLU(alpha)*A*x₁)
    3: x₃: (B, S, E) <- B*SiLU(x₂)
    4: y: (B, S, E) <- Linear(x₃)
    5: return y
----------------------------------------
```

# 🌸 Performance

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.1501   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2491   | 0.2807   | -        |
| ARC-Easy        | -        | 0.2563   | 0.2673   | -        |
| GSM8K           | -        | 0.0167   | 0.0250   | -        |
| HumanEval       | -        | 0.0305   | 0.0122   | -        |
| MMLU            | -        | 0.2714   | 0.2735   | -        |
| ChatCORE        | -        | 0.1785   | 0.1875   | -        |

# 🌸 Training result

## Base Training
- Minimum validation bpb: 0.8436
- Final validation bpb: 0.8436

## Mid Training
- Minimum validation bpb: 0.4561

## SFT Training
- Training loss: 1.3444
- Validation loss: 1.1934

# 🌸 クイックスタート

最速で実行する方法は、speedrun スクリプト [speedrun.sh](speedrun.sh) を実行することです。
このスクリプトは nanochat の **$100** の学習と推論を行います。
8×H100 ノード（$24/時）だと、合計実行時間は約 **4 時間** です。

まず好きなプロバイダ（例: [Lambda](https://lambda.ai/service/gpu-cloud)）で 8×H100 GPU サーバーを起動し、以下の手順を実行します。

まず、リポジトリをインストールします。

```bash
git clone https://github.com/Rikka-Botan/Liquid_Time_nanochat_jp.git
```

次に、学習を開始します。

```bash
cd
cd Liquid_Time_nanochat_jp
pwd
bash speedrun.sh
```

4 時間待ちます。完了すると ChatGPT のような Web UI で LLM と会話できます。
uv の仮想環境を再度有効化して（`source .venv/bin/activate`）、Web サーバーを起動します。

```bash
python -m scripts.chat_web
```

表示される URL にアクセスします。たとえば Lambda ならノードのパブリック IP にポートをつけてアクセスします。

例:
`http://209.20.xxx.xxx:8000/`

あとは ChatGPT と同じように話しかければ OK です。
物語を書かせたり、詩を書かせたり、「私は誰？」と聞いて幻覚を見せたり、空がなぜ青いのか、なぜ緑なのか聞いてみてください。
speedrun モデルは 4e19 FLOPs の能力なので、幼稚園児と話している感じです。

# 🌸 重み

[RikkaBotan/nanochat_saint_iberis_jp](https://huggingface.co/RikkaBotan/nanochat_saint_iberis_jp)

# 🌸　データセット

事前学習用：
https://huggingface.co/datasets/RikkaBotan/FineDataset_13B_JpEn

チューニング用：
https://huggingface.co/datasets/RikkaBotan/Cute_Synthetic_smoltalk_jp_sft


# 🌸 About us

## 六花牡丹（りっかぼたん）
おっとりで甘えん坊な研究者見習い

<img width="4405" height="2480" alt="RikkaBotan_Logo" src="https://github.com/user-attachments/assets/2a7e48cc-9b96-42f9-b63c-053d9a5312ca" />


# 🌸 引用

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

# 🌸 ライセンス

MIT
