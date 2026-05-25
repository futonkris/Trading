import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

INPUT_PATH  = 'data/news_aligned_2021_2023.parquet'
OUTPUT_PATH = 'data/news_scored_2021_2023.parquet'
MODEL_NAME  = 'ProsusAI/finbert'
BATCH_SIZE  = 1024 
MAX_LENGTH  = 512 

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"using {device}")

    print(f"({MODEL_NAME})")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    print(f"class order {model.config.id2label}")

    print(f"loading news {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"{len(df)} articles to score")

    df['article_title'] = df['article_title'].fillna('').astype(str)

    titles = df['article_title'].tolist()
    all_probs = []

    for i in tqdm(range(0, len(titles), BATCH_SIZE)):
        batch = titles[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)

    df['sent_pos']   = all_probs[:, 0]
    df['sent_neg']   = all_probs[:, 1]
    df['sent_neu']   = all_probs[:, 2]
    df['sent_score'] = df['sent_pos'] - df['sent_neg'] 

    print(f"scored {len(df)} articles.")
    print(df['sent_score'].describe())

    print("top 5 postive headlines")
    for _, row in df.nlargest(5, 'sent_score').iterrows():
        print(f"[{row['sent_score']:+.3f}] {row['article_title'][:120]}")

    print("top 5 negative headlines")
    for _, row in df.nsmallest(5, 'sent_score').iterrows():
        print(f"[{row['sent_score']:+.3f}] {row['article_title'][:120]}")

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"saved to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()