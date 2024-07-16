###- IMPORTATION DES LIBRAIRIES -### Début

# - Manipulation de données
import pandas as pd
import numpy as np

# - API et visualisation des données
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# - Logging
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# - Pré-traitement et évaluation
from scipy import stats
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold,
    cross_val_score,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, learning_curve

# - Modèles de régression
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# - Modèles de classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score
from sklearn.model_selection import learning_curve


default_hyperparameter_spaces = {
    "LinearRegression": {},
    "Ridge": {"alpha": [0.1, 1.0]},
    "KNeighborsRegressor": {"n_neighbors": [3, 5]},
    "XGBRegressor": {"n_estimators": [50, 100]},
}


def import_dataset(DATA, delimiter=None, decimal="."):
    """
    Importer des données à partir d'un fichier CSV dans un DataFrame.

    Paramètres:
    - data: str ou objet fichier
        Le chemin du fichier ou l'objet fichier contenant les données CSV.
    - delimiter: str, optionnel
        La chaîne utilisée pour séparer les valeurs. Si non spécifié, `sep` sera utilisé.
    - decimal: str, défaut='.'
        Caractère à utiliser comme séparateur décimal.

    Renvoie:
    - df: pandas.DataFrame
        Les données du fichier CSV sous forme de DataFrame.
    """
    try:
        df = pd.read_csv(DATA, delimiter=delimiter, decimal=decimal)
        logging.info("Données importées avec succès.")
        return df[sorted(df.columns)]
    except Exception as e:
        logging.error(f"Erreur lors de l'importation des données: {e}")
        return None


def basic_inspection(
    df, visualize_nulls=True, detailed_summary=True, check_duplicates=True
):
    """
    Effectue une inspection de base d'un DataFrame.

    Paramètres:
    - df: pandas.DataFrame
        Le DataFrame à inspecter.
    - visualize_nulls: bool, défaut=True
        Visualiser les valeurs manquantes avec une heatmap.
    - detailed_summary: bool, défaut=True
        Afficher un résumé détaillé des colonnes du DataFrame.
    - check_duplicates: bool, défaut=True
        Vérifier et afficher le nombre de lignes en double.

    Retourne:
    - None
    """
    st.write(f"Nombre de lignes: {df.shape[0]}")
    st.write(f"Nombre de colonnes: {df.shape[1]}")

    st.write("Types des données:\n", df.dtypes)

    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        st.success("Aucune valeur manquante dans votre dataset")
    else:
        st.write(
            "\nColonnes avec des valeurs manquantes:\n", null_counts[null_counts > 0]
        )

    if check_duplicates:
        duplicate_rows = df.duplicated().sum()
        st.write(f"\nNombre de lignes en double: {duplicate_rows}\n")

    if visualize_nulls:
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.heatmap(df.isna(), cbar=True, cmap="viridis", ax=ax)
        ax.set_title("Heatmap des valeurs manquantes")
        st.pyplot(fig)

    if detailed_summary:
        summary_table = pd.DataFrame(
            {
                "Valeurs_uniques": df.nunique(),
                "Type_de_donnée": df.dtypes,
                "Nombre_de_null": df.isnull().sum(),
                "Pourcentage_de_null": (df.isnull().sum() / df.shape[0] * 100).round(2),
            }
        )
        st.write(summary_table)


def check_outliers(df, make_plot=True):
    """
    Détecter et visualiser les valeurs aberrantes dans un DataFrame.

    Paramètres:
    - df: pandas.DataFrame
        Le DataFrame à analyser.
    - make_plot: bool
        Visualiser les valeurs aberrantes avec des boxplots.

    Retourne:
    - outlier_summary: dict
        Un dictionnaire contenant le nombre et les indices des valeurs aberrantes pour chaque colonne numérique.
    """
    outlier_summary = {}

    for column in df.select_dtypes(
        include=np.number
    ).columns:  # on ne garde que les colonnes numériques
        try:
            # Calcul des Z-scores
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            z_outliers = np.where(z_scores > 3)[0]

            # Calcul des intervalles interquartiles
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = df[
                (df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))
            ].index

            # Combiner les outliers des deux méthodes
            combined_outliers = np.union1d(z_outliers, iqr_outliers)

            if make_plot:
                st.write(column)
                fig, ax = plt.subplots(figsize=(4, 2))
                sns.boxplot(x=df[column])
                plt.title(f"Boxplot de {column} (valeurs aberrantes surlignées)")
                st.pyplot(fig)

            # Résumé des valeurs aberrantes
            outlier_summary[column] = {
                "num_outliers": len(combined_outliers),
                "outliers_index": combined_outliers,
            }
        except Exception as e:
            st.write(f"Erreur lors du traitement de la colonne {column}: {e}")

    return outlier_summary


def deal_w_null(
    df,
    strategy_num="median",
    strategy_cat="mode",
    fill_value_num=0,
    fill_value_cat="missing",
):
    """
    Nettoyer les données en remplissant les valeurs manquantes.

    Paramètres:
    - df: pandas.DataFrame
        Le DataFrame à nettoyer.
    - strategy_num: str, défaut='median'
        Stratégie pour remplir les valeurs manquantes des colonnes numériques ('mean', 'median', 'mode', 'constant', 'drop').
    - strategy_cat: str, défaut='mode'
        Stratégie pour remplir les valeurs manquantes des colonnes non numériques ('mode', 'constant', 'drop').
    - fill_value_num: int/float, défaut=0
        Valeur à utiliser pour remplir les valeurs manquantes des colonnes numériques si la stratégie est 'constant'.
    - fill_value_cat: str, défaut='missing'
        Valeur à utiliser pour remplir les valeurs manquantes des colonnes non numériques si la stratégie est 'constant'.

    Retourne:
    - df_clean: pandas.DataFrame
        Le DataFrame nettoyé.
    """
    # Traiter les colonnes numériques
    for col in df.select_dtypes(include=np.number).columns:
        if strategy_num == "mean":
            df[col].fillna(df[col].mean(), inplace=True)
        elif strategy_num == "median":
            df[col].fillna(df[col].median(), inplace=True)
        elif strategy_num == "mode":
            df[col].fillna(df[col].mode()[0], inplace=True)
        elif strategy_num == "constant":
            df[col].fillna(fill_value_num, inplace=True)
        elif strategy_num == "drop":
            df = df.dropna(subset=[col])
        else:
            raise ValueError(
                "Stratégie inconnue pour les colonnes numériques. Utilisez 'mean', 'median', 'mode', 'constant', ou 'drop'."
            )

    # Traiter les colonnes non numériques
    for col in df.select_dtypes(exclude=np.number).columns:
        if strategy_cat == "mode":
            df[col].fillna(df[col].mode()[0], inplace=True)
        elif strategy_cat == "constant":
            df[col].fillna(fill_value_cat, inplace=True)
        elif strategy_cat == "drop":
            df = df.dropna(subset=[col])
        else:
            raise ValueError(
                "Stratégie inconnue pour les colonnes non numériques. Utilisez 'mode', 'constant', ou 'drop'."
            )

    st.success("Take a look at your clean Data below:")
    st.write(df)
    return df


def encode_categorical(df, method="onehot"):
    """
    Encode categorical variables using the specified method.

    Parameters:
    - df: pandas.DataFrame
        The DataFrame containing the data.
    - method: str
        The encoding method to use. Options are 'onehot', 'ordinal', 'frequency'.

    Returns:
    - df_encoded: pandas.DataFrame
        The DataFrame with encoded categorical variables.
    """
    if method == "onehot":
        df_encoded = pd.get_dummies(df, drop_first=True)
    elif method == "ordinal":
        encoder = OrdinalEncoder()
        df_encoded = df.copy()
        for col in df.select_dtypes(exclude=np.number).columns:
            df_encoded[col] = encoder.fit_transform(df[[col]])
    elif method == "frequency":
        df_encoded = df.copy()
        for col in df.select_dtypes(exclude=np.number).columns:
            freq = df[col].value_counts()
            df_encoded[col] = df[col].map(freq)
    else:
        raise ValueError(
            "Méthode d'encodage inconnue. Utilisez 'onehot', 'ordinal', ou 'frequency'."
        )

    st.success("Votre DataFrame avec variables catégorielles encodées est ci-dessous:")
    st.write(df_encoded)

    return df_encoded


def do_grid_search(
    data,
    target,
    regressors,
    hyperparameter_spaces=None,
    scoring=["neg_mean_squared_error"],
    cv=5,
    refit_score="neg_mean_squared_error",
    verbose=1,
    n_jobs=-1,
    return_train_score=False,
):
    """
    Perform Grid Search to train models and determine the best model.

    Parameters:
    - data: pandas.DataFrame
        The input data.
    - target: str
        The target variable.
    - regressors: dict
        Dictionary of regressors to evaluate.
    - hyperparameter_spaces: dict, default=None
        Dictionary of hyperparameter spaces for each regressor. If None, default spaces are used.
    - scoring: list, default=['neg_mean_squared_error']
        List of scoring metrics to evaluate.
    - cv: int, default=5
        Number of cross-validation folds.
    - refit_score: str, default='neg_mean_squared_error'
        The score to use for refitting the best model.
    - verbose: int, default=1
        Verbosity level.
    - n_jobs: int, default=-1
        Number of jobs to run in parallel.
    - return_train_score: bool, default=False
        Whether to return the training score.

    Returns:
    - best_models: dict
        Dictionary of best models for each regressor.
    - best_scores: dict
        Dictionary of best scores for each regressor.
    - detailed_results: dict
        Detailed results of the grid search.
    - best_params: dict
        Dictionary of best parameters for each regressor.
    - best_model: object
        The best model based on the refit score.
    """
    if hyperparameter_spaces is None:
        hyperparameter_spaces = default_hyperparameter_spaces

    best_models = {}
    best_scores = {}
    detailed_results = {}
    best_params = {}

    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for model_name, params in hyperparameter_spaces.items():
        logger.info(f"Performing Grid Search for {model_name}")
        reg = regressors[model_name]
        cv_split = KFold(n_splits=cv) if isinstance(cv, int) else cv

        grid_search = GridSearchCV(
            reg,
            params,
            cv=cv_split,
            scoring=scoring,
            refit=refit_score,
            verbose=verbose,
            n_jobs=n_jobs,
            return_train_score=return_train_score,
        )
        grid_search.fit(X_train, y_train)

        best_models[model_name] = grid_search.best_estimator_
        best_scores[model_name] = {
            metric: grid_search.cv_results_["mean_test_%s" % metric][
                grid_search.best_index_
            ]
            for metric in scoring
        }
        detailed_results[model_name] = grid_search.cv_results_
        best_params[model_name] = grid_search.best_params_

        logger.info(f"Best Parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best Scores for {model_name}: {best_scores[model_name]}")

    if refit_score:
        best_model_name = max(best_scores, key=lambda x: best_scores[x][refit_score])
        logger.info(
            f"Best model based on {refit_score}: {best_model_name} with score {best_scores[best_model_name][refit_score]}"
        )
        best_model = best_models[best_model_name]
    else:
        best_model = None

    st.write(best_models, best_scores, detailed_results, best_params, best_model)

    return best_models, best_scores, detailed_results, best_params, best_model


def get_numeric_columns(df):
    """
    This function returns a list of the column names in the given DataFrame that have numeric data types.

    Parameters:
    df (pd.DataFrame): The DataFrame from which to extract numeric column names.

    Returns:
    list: A list of numeric column names.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("The input must be a valid DataFrame.")

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_columns


###- FONCTIONS UTILES -### Fin
#################################################################################################


def train_and_evaluate(df_encoded, target):
    # Séparer les features et la target
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]
    num_classes = len(np.unique(y))

    if num_classes == 2:

        # Séparation en train et test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Modèle XGBoost
        xgb_model = XGBClassifier(use_label_encoder=True, eval_metric="logloss")
        xgb_model.fit(X_train, y_train)
        xgb_y_pred = xgb_model.predict(X_test)
        xgb_y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]

        # Affichage des métriques
        st.write("Modèle XGBOOST : Métriques ")
        st.write("Accuracy:", accuracy_score(y_test, xgb_y_pred))
        st.write("F1 Score:", f1_score(y_test, xgb_y_pred))

        # Courbes ROC
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_y_pred_prob)

        plt.figure(figsize=(14, 7))
        plt.subplot(1, 2, 1)
        plt.plot(
            fpr_xgb,
            tpr_xgb,
            label="XGBoost (area = %0.2f)" % roc_auc_score(y_test, xgb_y_pred_prob),
        )
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")

        # Courbes d'apprentissage
        plt.subplot(1, 2, 2)
        train_sizes, train_scores_xgb, test_scores_xgb = learning_curve(
            XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
            X,
            y,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
        )

        train_scores_mean_xgb = np.mean(train_scores_xgb, axis=1)
        test_scores_mean_xgb = np.mean(test_scores_xgb, axis=1)

        plt.plot(
            train_sizes,
            train_scores_mean_xgb,
            "o-",
            color="r",
            label="XGBoost Training score",
        )
        plt.plot(
            train_sizes,
            test_scores_mean_xgb,
            "o-",
            color="g",
            label="XGBoost Cross-validation score",
        )

        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.title("Learning Curves")
        plt.legend(loc="best")
        plt.grid()

        st.pyplot(plt)
        return xgb_model
    else:
        st.error("Please choose a binary target")


#######################################################   Fonction Main de l'Application   ##############################################################""""
