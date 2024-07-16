import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import utils


###- MAIN -### Début


def main():

    url = "https://www.linkedin.com/in/rimey-aboky-25603a20b/"

    st.title("APPLICATION D'AUTO MACHINE LEARNING")
    st.subheader(
        "Do you have a regression or a classification task and you don't want to code? You are in the right place"
    )
    st.sidebar.write("[Author : Rimey ABOKY & Aminatou DIALLO](%s)" % url)
    st.sidebar.markdown(
        "**This application allows you to do your Regression and Classification tasks without writing a single line of code** \n"
        "1. Load your Data file in CSV, \n"
        "1. Choose your ML Task , \n"
        "1. Do your own Feature engineering , \n"
        "1. Run the training  Button , \n"
        "1. See the performance of your model  \n"
    )

    # Step 1: Upload your data
    fichier = st.file_uploader("Choose a file", type=["csv"])
    if fichier is not None:
        DELIMETER = st.text_input("Choose a delimiter", value=";")
        DECIMAL = st.text_input("Choose a decimal", value=".")
        df = utils.import_dataset(fichier, delimiter=DELIMETER, decimal=DECIMAL)

        if df is not None:
            st.success("Dataset successfully loaded! See what is in your data")
        else:
            st.error(
                "Failed to load dataset. Please check your CSV and choose the right delimiter."
            )

        if st.checkbox("Display raw Data", False):
            st.write(df.head(100))

    ## Inspection base de données

    if fichier is not None:
        st.header("Step 2: Basic Data Inspection")
        VISUALIZE_NULLS = st.checkbox("Display null data", True)
        DETAILED_SUMMARY = st.checkbox("Display detailed summary", True)
        CHECK_DUPLICATES = st.checkbox(
            "Display duplicates", True
        )  ### trouver un truc plus cool
        MAKE_PLOT = st.checkbox("Display box plot", True)
        if st.button("Run Basic Inspection"):
            utils.basic_inspection(
                df,
                visualize_nulls=VISUALIZE_NULLS,
                detailed_summary=DETAILED_SUMMARY,
                check_duplicates=CHECK_DUPLICATES,
            )
            utils.check_outliers(df, make_plot=MAKE_PLOT)
            st.success(
                "Now you will need to clean your data, you can proceed and go to Step 2"
            )

        #  STEP 2:  MISSING VALUES

        st.header("Step 2: Handle Missing Values")
        STRATEGY_NUM = st.selectbox(
            "Select metric for Numeric Columns",
            ["mean", "median", "mode", "constant", "drop"],
        )
        FILL_VALUE_NUM = st.number_input(
            "Fill Value for Numeric Columns (if constant)", value=0.0
        )
        STRATEGY_CAT = st.selectbox(
            "Select Strategy for Categorical Columns", ["mode", "constant", "drop"]
        )
        FILL_VALUE_CAT = st.text_input(
            "Fill Value for Categorical Columns (if constant)", value="missing"
        )

        df_clean = utils.deal_w_null(
            df,
            strategy_num=STRATEGY_NUM,
            strategy_cat=STRATEGY_CAT,
            fill_value_num=FILL_VALUE_NUM,
            fill_value_cat=FILL_VALUE_CAT,
        )

        # STEP 3: ENCODING

        st.header("Step 4 : Encoding Choice")
        ENCODING_METHOD = st.selectbox(
            "Choose the encoding technique", options=["ordinal", "onehot", "frequency"]
        )
        df_encoded = utils.encode_categorical(df_clean, method=ENCODING_METHOD)

    if fichier is not None:
        st.sidebar.markdown("**Machine Learning**")

        modelisation = st.sidebar.checkbox("Modelisation")

        if modelisation:
            choix_tache = st.sidebar.radio(
                "OPTIONS", ["Choose your task", "Regression", "Classification"]
            )

            if choix_tache == "Regression":
                # Modèles et hyperparamètres pour la régression
                st.header("Hyperparameter tuning and modelisation - Regression")

                target_column = st.selectbox(
                    "Choose the target column", df_encoded.columns
                )

                default_hyperparameter_spaces = {
                    "LinearRegression": {},
                    "Ridge": {"alpha": [0.1, 1.0]},
                    "KNeighborsRegressor": {"n_neighbors": [3, 5]},
                    "XGBRegressor": {"n_estimators": [50, 100]},
                }
                regressors = {
                    "LinearRegression": LinearRegression(),
                    "Ridge": Ridge(),
                    "KNeighborsRegressor": KNeighborsRegressor(),
                    "XGBRegressor": XGBRegressor(),
                }
                hyperparameter_spaces = {}

                hyperparam = st.checkbox(
                    "Do you want to select the hyperparameter spaces ?", False
                )

                if hyperparam:
                    for model_name in default_hyperparameter_spaces:
                        hyperparameters = st.text_input(
                            f"Enter the hyper parameter spaces for {model_name} (format: param1=[v1,v2], param2=[v3,v4])",
                            value=str(default_hyperparameter_spaces[model_name]),
                        )
                        try:
                            hyperparameter_spaces[model_name] = eval(hyperparameters)
                        except:
                            st.error(
                                f"Invalid hyperparameter format for{model_name}. Please check your input."
                            )
                else:
                    hyperparameter_spaces = default_hyperparameter_spaces

                # Exécution de la recherche de grille pour la régression
                scoring_metrics = st.multiselect(
                    "Please select the scoring metrics (The first metric will be used for refit)",
                    ["neg_mean_squared_error", "r2", "neg_mean_absolute_error"],
                    default=["neg_mean_squared_error"],
                )
                if st.checkbox("Start the grid search", True):
                    (
                        best_models,
                        best_scores,
                        detailed_results,
                        best_params,
                        best_model,
                    ) = utils.do_grid_search(
                        df_encoded,
                        target_column,
                        regressors,
                        hyperparameter_spaces=hyperparameter_spaces,
                        scoring=scoring_metrics,
                        refit_score=scoring_metrics[0],
                    )
                    st.write("Best Models:", best_models)
                    st.write("Best Scores:", best_scores)
                    st.write("Detailed Results:", detailed_results)
                    st.write("Best Parameters:", best_params)
                    st.write("Best Model:", best_model)

            elif choix_tache == "Classification":
                target_column = st.selectbox(
                    "Choose the target column", df_encoded.columns
                )

                #        best_model = utils.train_and_evaluate(df_encoded, target_column)

                if st.checkbox("Train your model", True):
                    best_model = utils.train_and_evaluate(df_encoded, target_column)

            if st.sidebar.checkbox("Prediction"):
                st.header("PREDICTIONS ON NEW DATA")
                new_file = st.file_uploader(
                    "Upload new data for predictions", type=["csv"]
                )

                if new_file is not None:

                    DELIMETER2 = st.text_input("Choose a delimiter ", value=";")
                    DECIMAL2 = st.text_input("Choose a decimal ", value=".")
                    df_2 = utils.import_dataset(
                        new_file, delimiter=DELIMETER2, decimal=DECIMAL2
                    )

                    if df_2 is not None:
                        st.success(
                            "Dataset successfully loaded! Ssee what is in your data"
                        )
                    else:
                        st.error(
                            "Failed to load dataset. Please check your CSV and choose the right delimiter."
                        )

                    if st.checkbox("Display raw Data ", True):
                        st.write(df_2.head(100))

                    df_clean2 = utils.deal_w_null(
                        df_2,
                        strategy_num=STRATEGY_NUM,
                        strategy_cat=STRATEGY_CAT,
                        fill_value_num=FILL_VALUE_NUM,
                        fill_value_cat=FILL_VALUE_CAT,
                    )

                    df2_encoded = utils.encode_categorical(
                        df_clean2, method=ENCODING_METHOD
                    )
                    variable_tampon = set(df2_encoded.columns)
                    variable_tampon.add(target_column)
                    if set(df_encoded.columns) != variable_tampon:
                        st.error(
                            "Votre nouveau dataset n'a pas les memes colonnes que le dataset d'entrainement"
                        )
                    else:
                        predictions = best_model.predict(df2_encoded)
                        id_columns = st.multiselect(
                            "Choisissez la variable à conserver avec les predictions",
                            df2_encoded.columns,
                        )

                        if id_columns:
                            df_predictions = pd.DataFrame(
                                {
                                    **{col: df2_encoded[col] for col in id_columns},
                                    "Predictions": predictions,
                                }
                            )

                            st.write(
                                "Here is your dataframe with selected columns and the predictions:"
                            )
                            st.write(df_predictions)
                            csv = df_predictions.to_csv(index=False)
                            st.download_button(
                                label="Télécharger le CSV",
                                data=csv,
                                file_name="output.csv",
                                mime="text/csv",
                            )
                        else:
                            st.write("Veuillez sélectionner au moins une colonne.")


if __name__ == "__main__":
    main()

###- MAIN -### Fin
