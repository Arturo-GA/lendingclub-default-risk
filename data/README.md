# Datos

Fuente: **LendingClub Loan Data 2007–2018Q4**, publicado en Kaggle por *wordsforthewise*:
https://www.kaggle.com/datasets/wordsforthewise/lending-club

Descargar `accepted_2007_to_2018Q4.csv.gz` (~1.6 GB) y colocarlo en `data/raw/`.

```bash
# con la CLI de Kaggle (requiere token en ~/.kaggle/kaggle.json)
kaggle datasets download -d wordsforthewise/lending-club -f accepted_2007_to_2018Q4.csv.gz -p data/raw/
```

`scripts/make_dataset.py` lee solo las ~70 columnas necesarias y escribe `data/processed/loans.parquet`.
Ninguno de los dos archivos se versiona.

Diccionario de datos oficial: `LCDataDictionary.xlsx` en el mismo dataset de Kaggle.
