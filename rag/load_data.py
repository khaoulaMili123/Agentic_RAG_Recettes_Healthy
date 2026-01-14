"""
RÔLE :
Ce fichier est responsable du chargement des données brutes depuis Kaggle.

CE QU'IL FAIT :
- Télécharge le dataset Food.com (ou un autre dataset Kaggle)
- Charge le fichier principal (ex: recipes.csv) dans un DataFrame pandas
- Ne fait AUCUN nettoyage ni transformation

QUAND IL EST UTILISÉ :
- Au début du projet
- Lorsqu'on change ou met à jour le dataset source

SORTIE :
- Un DataFrame pandas contenant les recettes brutes
"""

from kagglehub import KaggleDatasetAdapter
import kagglehub

DATASET_SLUG = "irkaal/foodcom-recipes-and-reviews"
DEFAULT_FILE_PATH = "recipes.csv"

def load_foodcom_recipes(file_path: str = DEFAULT_FILE_PATH):
    """
    Charge les recettes Food.com depuis Kaggle dans un DataFrame pandas.
    """
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        DATASET_SLUG,
        DEFAULT_FILE_PATH,
    )
    return df
def main():
    df = load_foodcom_recipes()
    print("Dataset chargé")
    print("Shape:", df.shape)
    print("Colonnes:", list(df.columns))
    print("\nFirst 5 records:\n", df.head())


if __name__ == "__main__":
    main()


