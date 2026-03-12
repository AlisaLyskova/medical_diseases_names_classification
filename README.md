# Medical diseases names classification
**Work in progress**


### Overview
This project trains a **text classification model** to map **ICD‑11 Russian disease titles** to an **ICD‑11 identifier** derived from the ICD‑11 hierarchy in the source table.

It:
- loads the ICD‑11 tabulation file (TSV),
- cleans disease titles,
- computes `NEW_ID` using your rule (leaf / level logic),
- builds a dictionary `TitleNEW -> NEW_ID`,
- tokenizes titles with a BERT tokenizer,
- trains a Keras model (Embedding + (Bi)LSTM + Dense) to predict the class.

### Data
You need the ICD‑11 tabulation file (TSV), e.g.:
- [SimpleTabulation-ICD-11-MMS-ru.txt](SimpleTabulation-ICD-11-MMS-ru.txt)

Expected columns used by the script:
- `Title` (raw title with `- ` prefixes indicating hierarchy)
- `Code`
- `BlockId`
- `Grouping1`
- `isLeaf`

### How `NEW_ID` is computed
`NEW_ID` is created from the table using this logic:

```python
df["NEW_ID"] = np.where(
    ~(df["Title"].str.startswith("- -")) & df["Title"].str.startswith("- ") & (df["isLeaf"] == True),
    df["Code"],
    np.where(
        ~(df["Title"].str.startswith("- -")) & df["Title"].str.startswith("- ") & (df["Code"].isna()),
        df["BlockId"],
        np.where(
            ~(df["Title"].str.startswith("- -")) & df["Title"].str.startswith("- "),
            df["Code"],
            df["Grouping1"],
        ),
    ),
)
```

### Outputs
After running, you get:
-  dictionary mapping `NEW_ID` → cleaned disease title
- `icd11_text_classifier.h5`: trained Keras model
- `label_forward_mapping.npy` / `label_backward_mapping.npy`: label ↔ id mappings used by the classifier


