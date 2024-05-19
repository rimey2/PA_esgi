import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



def main() :


    url = "https://www.linkedin.com/in/rimey-aboky-25603a20b/"

    st.title("APPLICATION D'AUTO MACHINE LEARNING")
    st.subheader("Do you have a regression or a classification task and you don't want to code? You are in the right place")
    st.sidebar.write("[Author : Rimey ABOKY & Aminatou DIALLO](%s)" % url)
    st.sidebar.markdown(

                    "**This application allows you to do your Regression and Classification tasks without writing a single line of code** \n" 
                    "1. Load your Data file in CSV; \n"
                    "1. Choose your ML Task ; \n"
                    "1. Do your own Feature engineering; \n"
                    "1. Choose your model; \n"
                    "1. Run the training  Button; \n"
                    "1. See the performance of your model; \n"
    )

   ## Importation de nos données
   
    
    def importer(data):    
        df = pd.read_csv(data) 
        return df
             
    fichier = st.file_uploader("Upload your data in csv ", type = ['csv'])
    
    if fichier is not None :
        donnees = importer(fichier)
        ##st.write(donnees)
 
    if st.checkbox("Display raw Data", False):
           st.dataframe(donnees.head(100))
           st.subheader("Sample of 100 observations")   
        
    
                 #################### Fonction qui permet de faire une inspection de la base de données ################################################

    def basic_inspection(df, visualize_nulls=True, detailed_summary=True, check_duplicates=True):
    
        st.write(f'Number of rows: {df.shape[0]}\nNumber of columns: {df.shape[1]}\n')
        st.write('Data types:\n', df.dtypes)
    
        null_counts = df.isnull().sum()
        if null_counts.sum()==0:
             st.write("No missing values in your dataset")
        else :
            st.write('\nColumns with null values:\n', null_counts[null_counts > 0])
    
        if check_duplicates:
            duplicate_rows = df.duplicated().sum()
            st.write(f'\nNumber of duplicate rows: {duplicate_rows}\n')

        if visualize_nulls:
            fig, ax = plt.subplots(figsize=(15, 7))  
            sns.heatmap(df.isna(), cbar=True, cmap='viridis', ax = ax)
            ax.set_title('Heatmap of Missing Values')
            st.pyplot(fig)
        
        
        summary_table = pd.DataFrame({
            "Unique_values": df.nunique(),
            "Data_type": df.dtypes,
            "Null_count": df.isnull().sum(),
            "Null_percentage": (df.isnull().sum() / df.shape[0] * 100).round(2)
    })

        st.write (summary_table)

           ##############  Fonction d'identification des valeurs aberrantes dans la base de données   #############################################

    def val_aberrantes(df, plot=True):
   
        outlier_summary = {}
    
        for column in df.select_dtypes(include=np.number).columns:  # on ne garde que les colonnes numériques
        
            # Calcule des Z score
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            z_outliers = np.where(z_scores > 3)[0]
        
            # Calcul des intervalles interquartiles
        
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))].index
        
            # Combiner les outliers des deux méthodes 
            combined_outliers = np.union1d(z_outliers, iqr_outliers)
        
            if plot:
                st.write(column)
                plt.figure(figsize=(8, 4))
                sns.boxplot(x=df[column])
                plt.title(f'Boxplot of {column} (Outliers highlighted)')
                st.pyplot()

        # Summary
            outlier_summary[column] = {
                'num_outliers': len(combined_outliers),
                'outliers_index': combined_outliers
        }
    
        return outlier_summary   
         

    inspect = st.sidebar.button('Inspect your data')
    outliers = st.button('You might want to see if you have trash in your data')

    if inspect :
        basic_inspection(donnees, visualize_nulls=True, detailed_summary=True, check_duplicates=True)
  
    if outliers:
        val_aberrantes(donnees, plot=True)    


    
    ### Fonction d'analyse exploratoire et de statistiques 

''''
    def analyse_exploratoire(df, col_name, visualize=True, **kwargs):
    
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not found in the DataFrame.")

     ### Statistiques descriptives

        descriptive_stats = df[col_name].describe()
        print("\n"*2, col_name, "\n"*2, descriptive_stats)

        if visualize:
        
            custom_colors = ['red', 'yellow', 'black']  # Red, Yellow, Black
            color = kwargs.get('hist_color', custom_colors[0])  ## couleur rouge par défaut
        
            # 2. Histogram with KDE
            plt.figure(figsize=kwargs.get('figsize', (12, 6)))
            sns.histplot(df[col_name], kde=True, bins=kwargs.get('bins', 30),
                     color=color,
                     kde_kws={'bw_adjust': kwargs.get('bw_adjust', 1)},
                     line_kws={'color': custom_colors[2], 'lw': 2})  # Use black for KDE line
            plt.title(f'Histogram and KDE of {col_name}')
            plt.xlabel(col_name)
            plt.ylabel('')
            plt.grid(True, linestyle='--', linewidth=0.5, color=custom_colors[1])  # Use yellow for grid lines
            plt.show()

        # 3. Boxplot
            plt.figure(figsize=kwargs.get('figsize', (12, 6)))
            sns.boxplot(x=df[col_name], color=kwargs.get('boxplot_color', custom_colors[1]))  # Use yellow for boxplot
            plt.title(f'Boxplot of {col_name}')
            plt.xlabel(col_name)
            plt.grid(True, linestyle='--', linewidth=0.5, color=custom_colors[2])  # Use black for grid lines
            plt.show()

        return descriptive_stats

         ##############  Correlation entre les variables  ####################

         
    correlation_cols = [df.columns(include :'float', 'int64')]
    correlation = df[correlation_cols].corr(method='pearson')

    ####   Matrice de correlation

    fig, ax = plt.subplots()

    ax.figure.set_size_inches(10, 10)
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(correlation, cmap=cmap, mask=mask, square=True, linewidths=.5, 
            annot=True, annot_kws={'size':14})
    plt.show() 
            
        '''
    ##############  Correlation entre les variables  ####################
    
    
if __name__ == '__main__' :
    main()

  
  



    




