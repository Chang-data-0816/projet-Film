import streamlit as st
#import unicodedata
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.neighbors import NearestNeighbors



class MultiApp:
    
    def __init__(self):
        self.apps = []
        self.app_dict = {}
   
    def add_app(self, title, func):
        if title not in self.apps:
            self.apps.append(title)
            self.app_dict[title] = func

    def run(self):
        title = st.radio(
            'Choisir la visualisation : ',
            self.apps,
            format_func=lambda title: str(title))
        self.app_dict[title]()

@st.experimental_memo   # Add the caching decorator
def load_data(url):
    df = pd.read_csv(url, sep="\t", na_values=['NA','\\N'])
    return df



#@st.experimental_singleton(suppress_st_warning=True)


def kpi():
    tab1, tab2 = st.tabs(["Movie", "Participants"])
    
    with tab1:
        st.title("Statistiques des Movies")
        st.markdown("""
                <style>
                .big-font {
                    font-size:24px !important;
                    color : Chocolate
                }
                </style>
                """, unsafe_allow_html=True)

        st.markdown('<p class="big-font">Durée moyenne des films par année</p>', unsafe_allow_html=True)
        
        
        #st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
        df_film = load_data("https://raw.githubusercontent.com/Chang-data-0816/projet-Film/main/genre_div.tsv")
        #Isolation des données pour les 'movie' et 'tvMovie'
        #df_film = df_genre[(df_genre['titleType']=='movie')|(df_genre['titleType']=='tvMovie')]
        #df_film_mean =  df_film.loc[(df_film['runtimeMinutes'] < 1000) & (df_film['runtimeMinutes'] > 0) & (df_film['startYear'] != 0)]
        #df_film['runtimeMinutes'].mean()
        #KPI n°1 -       

        # Group the DataFrame by year and calculate the mean run time for each year
        mean_movie_year = df_film.groupby('startYear')['runtimeMinutes'].mean().reset_index()

        # Plot the results
        line_chart = px.line(mean_movie_year, x='startYear', y='runtimeMinutes')
        line_chart.update_traces(line_color='olive')
        st.plotly_chart(line_chart)
        #KPI n°2
        st.markdown('<p class="big-font">Nombre de films réalisés par année</p>', unsafe_allow_html=True)
        # count the number of movies released in each year
        movie_year = df_film.groupby(df_film['startYear']).count()

        # plot the chart using Plotly Express
        fig_bar = px.bar(movie_year, x=movie_year.index, y='tconst',color_discrete_sequence=["orange"]
            ,labels={'x':'Year', 'y':'Number of Movies'})
        #line_chart.update_traces(bar_color='brown')
        st.plotly_chart(fig_bar)
        #KPI n°3
        st.markdown('<p class="big-font">Répartition des films par genre</p>', unsafe_allow_html=True)
        # First, get the count of each genre in the dataframe
        genre_counts = df_film['genres'].value_counts()

        # Filter out the genres that are below 2% of the total number of movies
        other_genre = genre_counts[genre_counts < 0.03 * len(df_film)].index
        df_film.loc[df_film['genres'].isin(other_genre), 'genres'] = 'Other'
        df_film.drop_duplicates(inplace=True)
        labels=df_film['genres'].value_counts().index
        # Plot the pie chart with the filtered genres
        
        pie_chart = px.pie(df_film, values = df_film['genres'].value_counts(), names = labels,color_discrete_sequence=px.colors.sequential.Blues_r)
        #ax_pie.axis('equal')
        #plt.title('Genre Distribution')
        
        st.plotly_chart(pie_chart)



    with tab2:
        df_perso_fin = load_data("https://raw.githubusercontent.com/Chang-data-0816/projet-Film/main/df_perso_fin.tsv")
        st.title("Statistiques des Participants")
        #st.subheader('')
        st.markdown('<p class="big-font">Quels sont les Meilleurs Réalisateurs / Acteurs / Actrices?</p>', unsafe_allow_html=True)
        st.markdown('**Pairplots** : correlations entre **:blue[_nb_Production_] & :blue[_aver_Note_] & :blue[_sum_numVotes_]**.')            
        #st.dataframe(df_perso_fin)
        fig_pair = sns.pairplot(df_perso_fin, hue='categorie', palette=['tab:blue', 'tab:grey', 'tab:orange'])
        st.pyplot(fig_pair)

        st.markdown('**Lmplots** : correlations entre **:blue[_aver_Note_] & :blue[_sum_numVotes_]** pour les 200 premières personnes.')
        x = df_perso_fin[df_perso_fin['categorie']=='director'].iloc[:200]
        y = df_perso_fin[df_perso_fin['categorie']=='actor'].iloc[:200]
        z = df_perso_fin[df_perso_fin['categorie']=='actress'].iloc[:200]
        data_top_perso = pd.concat([x,y,z])
        
        fig_lm = sns.lmplot(x='sum_numVotes',y='aver_Note', data=data_top_perso,hue='categorie',
                palette=['tab:blue', 'tab:grey', 'tab:orange'],markers=['o','+','x'], col='categorie')
        st.pyplot(fig_lm)

        #st.caption('**Les Meilleurs Réalisateurs / Acteur / Actresse**')
        st.markdown('<p class="big-font">Les Meilleurs Réalisateurs / Acteurs / Actrices</p>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        nb = col1.slider('Choisissez un nombre', 6, 20, 12)
       

        role = col2.selectbox('Rôle de participant',('director', 'actor', 'actress'))
        top_perso = df_perso_fin[df_perso_fin['categorie'] == role].iloc[:nb]
        fig, ax = plt.subplots(figsize=(10,4))
        sns.set_style('white')
        ax1 = sns.barplot(x='nom_Person', y='sum_numVotes', palette = 'Greens_r', data=top_perso,saturation=.5,dodge=False) #, hue='categorie', dodge=False
        ax2 = ax1.twinx()

        line2, = ax2.plot('nom_Person','aver_Note', data=top_perso,color = sns.xkcd_rgb["light blue"],linestyle = 'dashed')
        p2 = ax2.scatter('nom_Person','aver_Note', data=top_perso,color = sns.xkcd_rgb["pale red"],marker = 'o',s = 50)
        ax1.set_xlabel('')
        ax1.set_ylabel('Sum_nb_Votes',fontsize=14)
        ax2.set_ylabel('Note_moyen_Pondérée',fontsize=14)
        ax1.set_ylim([0,50000000])
        ax2.set_ylim([0,10])
        ax1.set_xticklabels(ax1.get_xticklabels(),rotation = 80,fontsize=10)
        st.pyplot(fig)
        


#@st.experimental_singleton(suppress_st_warning=False)
def system():
    #Ajouter une image comme backgrand
    #@st.experimental_memo   
    #def add_bg_from_url():
        #st.markdown(
            #f"""
            #<style>
            #.stApp {{
            #    background-image: url("https://cdn.pixabay.com/photo/2019/04/24/11/27/flowers-4151900_960_720.jpg");
            #    background-attachment: fixed;
            #    background-size: cover
            #}}
            #</style>
            #""",
            #unsafe_allow_html=True
        #)

    #add_bg_from_url() 
  

    link = "https://raw.githubusercontent.com/Chang-data-0816/projet-Film/main/df_merge_fin.tsv"
    df_production_select = pd.read_csv(link, sep="\t", na_values=['NA','\\N'],
            dtype={'startYear':'int16' ,'runtimeMinutes':'int16','numVotes':'int32','averageRating': 'float16'})
    st.title("Movie - Recommandation")
    #search = st.text_input("Saisir le nom ou l'identifiant du film : ", key="film").capitalize()
    movie_list = df_production_select['primaryTitle'].values
    search = st.selectbox("Saisir le nom du film : ", movie_list )

    #if len(df_production_select[df_production_select.values == search]) != 0 :
    @st.experimental_memo
    def algo_KNN(df,n):
        X = df.select_dtypes(include=['int64']).columns
        distanceKNN = NearestNeighbors(n_neighbors=n).fit(df[X])
        model = df[df.values == search][X]
        neighbors = distanceKNN.kneighbors(model)
        resultat_df = df.iloc[neighbors[1][0]].sort_values('numVotes', ascending=False)
        return resultat_df

    resultat_df = algo_KNN(df_production_select,100)
    # Selectionner tout les autres films du même réalisateur
    search_pro = df_production_select[df_production_select.values == search].nconst
    production_pro = df_production_select[df_production_select.nconst.isin(search_pro)]
    # Concatenate le resultat de 'nconst' & 100 NearestNeighbors
    df_new = pd.concat([resultat_df,production_pro], ignore_index=True)
    # "Get_dummies()" à la colonne 'nconst' pour le nouveau dataFrame df_NearestNeighbors
    #df_NearestNeighbors = df_NearestNeighbors.join(pd.get_dummies(df_NearestNeighbors.nconst, drop_first=True))
    df_getDummies = df_new.nconst.str.get_dummies(',')
    df_NearestNeighbors = pd.concat([df_new, df_getDummies],axis=1)
    resultat_df_F = algo_KNN(df_NearestNeighbors,30)
        
    # Créer un dict pour 'tconst' & 'originalTitle'
    resultat_id = resultat_df_F.tconst.unique()
    resultat_name = resultat_df_F.primaryTitle.unique()
    dict_resultat = dict(zip(resultat_id,resultat_name))
    resultat = []

    for i, j in dict_resultat.items():
        if len(resultat) == 10:
            break
        elif i != search and j != search and not j in resultat:
            resultat.append(j)    
    
    df_resultat = resultat_df_F[resultat_df_F.primaryTitle.isin(resultat)].iloc[:,1:7]
    df_resultat.drop_duplicates(subset=['primaryTitle'], inplace=True)
    df_resultat.averageRating = df_resultat.averageRating.apply(lambda x: format(x, '.2f'))
    # Montrer un dataframe des films recommandés
    st.write("Les 10 films recommandés:") 
    st.dataframe(df_resultat)

       
    #else : 
    #st.write("Attention! Saisir un nom/id correct!")
        
app = MultiApp()

app.add_app("Kpis", kpi)
app.add_app("Système de recommandation", system)

app.run()



    
    

 