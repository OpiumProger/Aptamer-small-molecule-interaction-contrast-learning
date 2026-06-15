"""Export diverse rerank pairs for manual RSAPred submission."""
import pandas as pd

df = pd.read_csv("rsapred_rerank_all.csv")
div = df[df["passes_motif_filter"] == True].sort_values(["molecule_index", "rank_diverse"])

rows = []
for _, r in div.iterrows():
    dna = r["sequence"]
    rows.append(
        {
            "pair_id": f"mol{int(r['molecule_index'])}_diverse_{int(r['rank_diverse'])}",
            "molecule_index": int(r["molecule_index"]),
            "rank_diverse": int(r["rank_diverse"]),
            "smiles": r["smiles"],
            "dna_sequence": dna,
            "rna_sequence": dna.replace("T", "U"),
            "sequence_sim": round(float(r["sequence_sim"]), 4),
            "motif_penalty": float(r["motif_penalty"]),
            "decoded_sim": round(float(r["decoded_sim"]), 4),
        }
    )

out = pd.DataFrame(rows)
out.to_csv("rsapred_pairs_to_submit.csv", index=False)
print(f"Saved {len(out)} pairs -> rsapred_pairs_to_submit.csv")
print(out.to_string(index=False))
